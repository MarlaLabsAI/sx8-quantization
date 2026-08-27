"""S-X8 Flash v4.3 — FULL GPU Quantization (V6 + PCA-POST) — Qwen3.8-27B shard-a-shard

Derivado de quantize_qwen35_4b_sx8_v43_gpu.py (el del 4B). Cambios clave:
1. NO se carga el modelo completo (from_pretrained elimina) — se leen los shards
   safetensors uno a uno con safe_open (el 27B no cabe en RAM).
2. Verificación CosSim/MSE en TODOS los tensores (no solo 8 capas).
3. Escritura streaming: un body file por shard (formato de registro idéntico
   al .sx8v43) + state.json reanudable. El ensamblaje final es otro script.

Las funciones de cuantización (v6_encode_6bit_gpu, chunked_v6_encode,
pca_correct_gpu, pack_levels_6bit, sx8_v6_kernel) están copiadas SIN CAMBIOS.
"""
import torch, numpy as np, time, os, struct, pickle, gc, sys, collections, json, argparse
from safetensors import safe_open

MODEL_PATH = "/mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/qwen_27b"
B = 32; L63 = 63; N_BASES = 4   # v4.3: PCA calcula 4 bases, guarda 2 (coeff 1 byte)
MAX_BLOCKS_PER_CHUNK = 3_000_000
DEV = torch.device("cuda")
OUT_DIR = "/home/blackpanter/qwen_27b_sx8"
STATE_DIR = "/tmp/opencode/qwen27b_state"
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(STATE_DIR, exist_ok=True)

EXPECTED_TENSORS = 666
EXPECTED_PARAMS = 27_780_313_088

props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} ({props.total_memory//1024**2} MB) | S-X8 Flash v4.3 shard-a-shard | {MODEL_PATH}", flush=True)


# ============================================================================
# V6 Encode with L=63 (GPU) — copiada SIN CAMBIOS del script 4B
# ============================================================================
def v6_encode_6bit_gpu(blocks):
    N = blocks.shape[0]
    sub = blocks.reshape(N, 4, 8)
    dmin = blocks.min(dim=1).values
    dmax = blocks.max(dim=1).values
    d = (dmax - dmin).clamp(min=1e-10)
    q = d / 4.0
    slo = [dmin, dmin, dmax - q, dmin + q]
    shi = [dmax, dmin + q, dmax, dmax - q]
    best_mse = torch.full((N, 4), float('inf'), device=DEV)
    best_lev = torch.zeros(N, 4, 8, dtype=torch.int32, device=DEV)
    best_str = torch.zeros(N, 4, dtype=torch.int32, device=DEV)
    for ri in range(4):
        lo = slo[ri][:, None, None]
        hi = shi[ri][:, None, None]
        step = ((hi - lo) / L63).clamp(min=1e-10)
        cand = ((sub - lo) / step).round().clamp(0, L63).int()
        rec = lo + step * cand.float()
        mse = ((sub - rec) ** 2).mean(dim=-1)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_lev = torch.where(better.unsqueeze(-1), cand, best_lev)
        best_str = torch.where(better, ri, best_str)
    config = (best_str[:, 0] | (best_str[:, 1] << 2) |
              (best_str[:, 2] << 4) | (best_str[:, 3] << 6))
    return best_lev.to(torch.uint8).reshape(N, 32), dmin, dmax, config, best_str


def chunked_v6_encode(blocks_gpu):
    """GPU V6 encode chunkiendo si es necesario"""
    n_blk = blocks_gpu.shape[0]
    if n_blk <= MAX_BLOCKS_PER_CHUNK:
        lv, dm, dx, cf, st = v6_encode_6bit_gpu(blocks_gpu)
        return lv, dm, dx, cf
    lv_parts, dm_parts, dx_parts, cf_parts = [], [], [], []
    for start in range(0, n_blk, MAX_BLOCKS_PER_CHUNK):
        end = min(start + MAX_BLOCKS_PER_CHUNK, n_blk)
        chunk = blocks_gpu[start:end].clone()
        lv, dm, dx, cf, st = v6_encode_6bit_gpu(chunk)
        lv_parts.append(lv)
        dm_parts.append(dm)
        dx_parts.append(dx)
        cf_parts.append(cf)
        del chunk, st
        torch.cuda.empty_cache()
    return (torch.cat(lv_parts), torch.cat(dm_parts),
            torch.cat(dx_parts), torch.cat(cf_parts))


# ============================================================================
# GPU PCA Correction (por slice de columna) — copiada SIN CAMBIOS
# ============================================================================
def pca_correct_gpu(levels, dmin, dmax, config, blocks_gpu, n_cb):
    n_blk = blocks_gpu.shape[0]
    d = (dmax - dmin).clamp(min=1e-10)
    Q = d / 4.0
    _L = float(L63)

    lo_j = torch.stack([dmin, dmin, dmax - Q, dmin + Q], dim=0)
    st_j = torch.stack([d / _L, Q / _L, Q / _L,
                         (d - 2 * Q).clamp(min=1e-10) / _L], dim=0)

    lv_f = levels.float()
    coeff_bytes = torch.zeros(n_blk, 1, dtype=torch.uint8, device=DEV)
    bases_list = []
    scales_list = []

    for cb in range(n_cb):
        si = torch.where((torch.arange(n_blk, device=DEV) % n_cb) == cb)[0]
        n_slice = si.shape[0]

        if n_slice < 4:
            bases_list.append(np.zeros((2, 32), dtype=np.float32))
            scales_list.append(np.zeros(2, dtype=np.float32))
            continue

        v6_rec = torch.zeros(n_slice, 32, dtype=torch.float32, device=DEV)
        cfg_s = config[si].long()

        for sb in range(4):
            strat_sb = (cfg_s >> (sb * 2)) & 3
            lo_sb = lo_j[strat_sb, si]
            st_sb = st_j[strat_sb, si]
            sbi = slice(sb * 8, (sb + 1) * 8)
            v6_rec[:, sbi] = lo_sb[:, None] + st_sb[:, None] * lv_f[si][:, sbi]

        error = blocks_gpu[si] - v6_rec

        _, S, Vh = torch.linalg.svd(error, full_matrices=False)
        basis = Vh[:N_BASES]

        coeffs_raw = torch.mm(basis, error.T)
        scales = torch.median(coeffs_raw.abs(), dim=1).values / 3.0
        scales = scales.clamp(min=1e-10)

        c_quant = (coeffs_raw / scales[:, None]).round().clamp(-8, 7).to(torch.int8)

        byte0 = ((c_quant[0].byte() & 0xF) | ((c_quant[1].byte() & 0xF) << 4))
        coeff_bytes[si, 0] = byte0

        bases_list.append(basis[:2].cpu().numpy().astype(np.float32))
        scales_list.append(scales[:2].cpu().numpy().astype(np.float32))

        del si, v6_rec, error, basis, coeffs_raw, c_quant, byte0, S, Vh

    return coeff_bytes.cpu().numpy().astype(np.uint8), bases_list, scales_list


# ============================================================================
# Packing 6-bit → 4+2 — copiada SIN CAMBIOS
# ============================================================================
def pack_levels_6bit(levels_cpu):
    """levels_cpu: (n_blk, 32) uint8 with values 0-63"""
    lv_hi = (levels_cpu & 0x3F) >> 2
    lv_lo = levels_cpu & 0x03

    lvf_hi = lv_hi.flatten()
    lo_hi = lvf_hi[0::2] & 0xF
    hi_hi = (lvf_hi[1::2] << 4) & 0xF0
    packed_hi = (lo_hi | hi_hi).astype(np.uint8)

    lvf_lo = lv_lo.flatten()
    groups = lvf_lo.reshape(-1, 4)
    packed_lo = ((groups[:,0].astype(np.uint8)&0x3)<<6) | \
                ((groups[:,1].astype(np.uint8)&0x3)<<4) | \
                ((groups[:,2].astype(np.uint8)&0x3)<<2) | \
                (groups[:,3].astype(np.uint8)&0x3)

    return packed_hi, packed_lo


# ============================================================================
# Numba decode (para verificación) — copiado SIN CAMBIOS
# ============================================================================
from numba import cuda as nbcuda

@nbcuda.jit
def sx8_v6_kernel(dmin_f, dmax_f, config, hi_arr, lo_arr, weights, n_blocks):
    tid = nbcuda.threadIdx.x & 31; wid = nbcuda.threadIdx.x >> 5
    bid = nbcuda.blockIdx.x * 4 + wid
    if bid >= n_blocks: return
    hi = (hi_arr[bid * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
    lo = (lo_arr[bid * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
    lv = (hi << 2) | lo
    lo_f, hi_f = dmin_f[bid], dmax_f[bid]; q = (hi_f - lo_f) * 0.25
    cfg = config[bid]; sb = tid >> 3; strat = (cfg >> (sb * 2)) & 3
    if strat == 0: rlo, rhi = lo_f, hi_f
    elif strat == 1: rlo, rhi = lo_f, lo_f + q
    elif strat == 2: rlo, rhi = hi_f - q, hi_f
    else: rlo, rhi = lo_f + q, hi_f - q
    step = (rhi - rlo) * 0.015873
    if step < 1e-10: step = 1e-10
    weights[bid * 32 + tid] = rlo + step * lv


# ============================================================================
# Escritura de registro de tensor (formato idéntico a write_sx43)
# ============================================================================
def write_record(f, name, qt, bi):
    nb_bytes = name.encode()
    f.write(struct.pack('<I', len(nb_bytes))); f.write(nb_bytes)
    sh = qt['shape']
    f.write(struct.pack('<II', sh[0], sh[1]))
    os_ = qt.get('orig_shape', sh)
    f.write(struct.pack('<B', len(os_)))
    f.write(struct.pack('<%dI' % len(os_), *os_))
    f.write(struct.pack('<II', qt['n_blocks'], qt['n_cb']))
    f.write(qt['dmin'].tobytes())
    f.write(qt['dmax'].tobytes())
    f.write(qt['config'].tobytes())
    f.write(qt['levels_hi'].tobytes())
    f.write(qt['levels_lo'].tobytes())
    f.write(qt['coeff'].tobytes())
    f.write(struct.pack('<I', bi['n_cb']))
    f.write(bi['data'].tobytes())
    f.write(bi['scales'].tobytes())


def read_record_headers(f, max_records=None):
    """Recorre los registros de un body: devuelve lista de (name, sh, os_, nb, ncb)."""
    recs = []
    for _ in range(max_records if max_records else 10**9):
        start = f.tell()
        head = f.read(4)
        if not head:
            break
        (nl,) = struct.unpack('<I', head)
        name = f.read(nl).decode()
        sh = struct.unpack('<II', f.read(8))
        (n_os,) = struct.unpack('<B', f.read(1))
        os_ = struct.unpack('<%dI' % n_os, f.read(4 * n_os))
        nb, ncb = struct.unpack('<II', f.read(8))
        data_bytes = nb * 30 + 4 + ncb * 256 + ncb * 8
        f.seek(data_bytes, 1)
        recs.append((name, sh, os_, nb, ncb, start))
    return recs


# ============================================================================
# Verificación de calidad de UN tensor (decode numba + PCA, corrección in-place)
# ============================================================================
def verify_tensor(qt, bi, W_orig_np):
    nblk = qt['n_blocks']; ncb = qt['n_cb']
    in_f = qt['shape'][1]; out_f = qt['shape'][0]

    cfg = qt['config'].astype(np.uint8)
    dmin_g = nbcuda.to_device(qt['dmin'].astype(np.float32))
    dmax_g = nbcuda.to_device(qt['dmax'].astype(np.float32))
    config_g = nbcuda.to_device(cfg)
    hi_g = nbcuda.to_device(qt['levels_hi'])
    lo_g = nbcuda.to_device(qt['levels_lo'])
    w_g = nbcuda.device_array(nblk * 32, dtype=np.float32)

    grid = (nblk + 3) // 4
    sx8_v6_kernel[grid, 128](dmin_g, dmax_g, config_g, hi_g, lo_g, w_g, nblk)
    nbcuda.synchronize()
    del dmin_g, dmax_g, config_g, hi_g, lo_g

    Wdec = w_g.copy_to_host().reshape(nblk, 32)
    del w_g

    for cb in range(ncb):
        si = np.arange(cb, nblk, ncb)
        if si.size < 4: continue
        basis = bi['basis'][cb]
        scales = bi['scales'][cb]
        coeff = qt['coeff'][si]
        b0 = coeff[:, 0].astype(np.int32)
        c_raw = np.stack([b0 & 0xF, (b0 >> 4) & 0xF], axis=1)
        c_quant = np.where(c_raw >= 8, c_raw - 16, c_raw).astype(np.float32)
        correction = np.dot(c_quant * scales[np.newaxis, :], basis)
        Wdec[si] += correction

    r = Wdec.reshape(out_f, ncb, 32).reshape(out_f, ncb * 32)[:, :in_f].flatten()
    del Wdec
    o = W_orig_np.flatten()
    cs = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-12))
    mse = float(np.mean((o - r) ** 2))
    del o, r
    return cs, mse


# ============================================================================
# Main
# ============================================================================
def build_plan():
    """Enumerar tensores cuantizables por shard (filtro: 'weight' and ndim>=2)."""
    shards = sorted(f for f in os.listdir(MODEL_PATH)
                    if f.endswith(".safetensors") and not f.startswith("model.safetensors.index"))
    plan = {}
    for sh in shards:
        tens = []
        with safe_open(os.path.join(MODEL_PATH, sh), framework="pt") as f:
            for name in f.keys():
                if 'weight' in name:
                    shape = tuple(f.get_slice(name).get_shape())
                    if len(shape) >= 2:
                        tens.append((name, shape))
        plan[sh] = tens
        print(f"  {sh}: {len(tens)} tensores cuantizables", flush=True)
    n_tot = sum(len(v) for v in plan.values())
    p_tot = sum(int(np.prod(s)) for v in plan.values() for _, s in v)
    print(f"TOTAL: {n_tot} tensores | {p_tot:,} params", flush=True)
    assert n_tot == EXPECTED_TENSORS, f"TENSORES: {n_tot} != {EXPECTED_TENSORS}"
    assert p_tot == EXPECTED_PARAMS, f"PARAMS: {p_tot} != {EXPECTED_PARAMS}"
    return plan


def load_state():
    p = os.path.join(STATE_DIR, "quant_state.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"done": {}, "totals": {}}


def save_state(st):
    p = os.path.join(STATE_DIR, "quant_state.json")
    tmp = p + ".tmp"
    json.dump(st, open(tmp, "w"))
    os.replace(tmp, p)


def run_smoke_tensor(plan, target_name):
    """ST2: cuantizar UN tensor end-to-end (pipeline completo idéntico a main),
    verificar CosSim, escribir el registro, releerlo y comprobar roundtrip."""
    shard = None; shape = None
    for sh, tens in plan.items():
        for name, shp in tens:
            if name == target_name:
                shard, shape = sh, shp
                break
        if shard: break
    assert shard, f"Tensor no encontrado: {target_name}"
    print(f"ST2: {target_name} -> {shard} {shape}", flush=True)

    t0 = time.time()
    with safe_open(os.path.join(MODEL_PATH, shard), framework="pt") as f:
        t = f.get_tensor(target_name)
        Wf = t.float(); del t
    out_f, in_f = shape[0], int(np.prod(shape[1:]))
    W_orig_np = Wf.numpy()

    W = Wf.to(DEV)
    n_cb = (in_f + B - 1) // B
    pad_cols = (B - in_f % B) % B
    if pad_cols > 0:
        W = torch.nn.functional.pad(W, (0, pad_cols))
    blocks = W.reshape(out_f, n_cb, B).reshape(-1, B).contiguous()
    del W
    n_blk = blocks.shape[0]

    lv_gpu, dm_gpu, dx_gpu, cf_gpu = chunked_v6_encode(blocks)
    coeff_cpu, bases_list, scales_list = pca_correct_gpu(lv_gpu, dm_gpu, dx_gpu, cf_gpu, blocks, n_cb)
    del blocks
    torch.cuda.empty_cache()

    lv_cpu = lv_gpu.cpu().numpy().astype(np.uint8).reshape(n_blk, 32)
    dm_cpu = dm_gpu.cpu().numpy().astype(np.float16)
    dx_cpu = dx_gpu.cpu().numpy().astype(np.float16)
    cf_cpu_sp = cf_gpu.cpu().numpy().astype(np.uint8)
    del lv_gpu, dm_gpu, dx_gpu, cf_gpu
    torch.cuda.empty_cache()
    packed_hi, packed_lo = pack_levels_6bit(lv_cpu)
    del lv_cpu

    qt = {'name': target_name, 'shape': (out_f, in_f), 'orig_shape': shape,
          'n_blocks': n_blk, 'n_cb': n_cb,
          'dmin': dm_cpu, 'dmax': dx_cpu, 'config': cf_cpu_sp,
          'levels_hi': packed_hi, 'levels_lo': packed_lo,
          'coeff': coeff_cpu}
    bi = {'n_cb': n_cb, 'basis': bases_list,
          'scales': np.array(scales_list, dtype=np.float32),
          'data': np.array([b.flatten() for b in bases_list], dtype=np.float32)}
    cs, mse = verify_tensor(qt, bi, W_orig_np)
    print(f"  ✅ ST2 {target_name}: n_blk={n_blk:,} CosSim={cs:.6f} MSE={mse:.3e} "
          f"({time.time()-t0:.0f}s)", flush=True)
    assert cs > 0.999, f"CosSim ST2 demasiado bajo: {cs}"

    tmp = os.path.join(OUT_DIR, "smoke_st2.bin")
    with open(tmp, "wb") as f:
        write_record(f, target_name, qt, bi)
    with open(tmp, "rb") as f:
        recs = read_record_headers(f)
    assert len(recs) == 1, f"roundtrip: {len(recs)} registros, esperado 1"
    rn, rsh, ros, rnb, rncb, _ = recs[0]
    assert rn == target_name and rsh == (out_f, in_f) and rnb == n_blk and rncb == n_cb
    os.remove(tmp)
    print(f"  ✅ ST2 roundtrip OK: registro leído con {rnb:,} bloques, forma {rsh}", flush=True)
    print("ST2 PASS", flush=True)
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="procesar solo un shard (nombre exacto)")
    ap.add_argument("--smoke-tensor", help="ST2: cuantizar SOLO este tensor (nombre exacto) y salir")
    args = ap.parse_args()

    plan = build_plan()
    if args.smoke_tensor:
        run_smoke_tensor(plan, args.smoke_tensor)
        return

    st = load_state()
    t_global = time.time()
    total_blk = 0; n_params = 0; cs_list = []; mse_list = []

    for si, (shard, tensors) in enumerate(plan.items()):
        if args.only and shard != args.only:
            continue
        if shard in st["done"]:
            d = st["done"][shard]
            total_blk += d["blocks"]; n_params += d["params"]
            cs_list.extend(d.get("cs_list", [])); mse_list.extend(d.get("mse_list", []))
            print(f"[SKIP] {shard} (hecho: {d['tensors']} tensores, {d['blocks']:,} bloques)", flush=True)
            continue

        body_tmp = os.path.join(OUT_DIR, f"body_{shard.replace('.safetensors','')}.bin.tmp")
        body_final = os.path.join(OUT_DIR, f"body_{shard.replace('.safetensors','')}.bin")
        if os.path.exists(body_tmp):
            os.remove(body_tmp)

        t_shard = time.time()
        sh_blocks = 0; sh_params = 0; sh_cs = []; sh_mse = []
        print(f"\n=== [{si+1}/{len(plan)}] {shard} — {len(tensors)} tensores ===", flush=True)
        with safe_open(os.path.join(MODEL_PATH, shard), framework="pt") as f:
            with open(body_tmp, "wb") as bf:
                for li, (name, shape) in enumerate(tensors):
                    t0 = time.time()
                    t = f.get_tensor(name)
                    Wf = t.float()
                    del t
                    out_f, in_f = shape[0], int(np.prod(shape[1:]))
                    W_orig_np = Wf.numpy()

                    W = Wf.to(DEV)
                    n_cb = (in_f + B - 1) // B
                    pad_cols = (B - in_f % B) % B
                    if pad_cols > 0:
                        W = torch.nn.functional.pad(W, (0, pad_cols))
                    blocks = W.reshape(out_f, n_cb, B).reshape(-1, B).contiguous()
                    del W
                    n_blk = blocks.shape[0]

                    try:
                        lv_gpu, dm_gpu, dx_gpu, cf_gpu = chunked_v6_encode(blocks)
                        coeff_cpu, bases_list, scales_list = pca_correct_gpu(
                            lv_gpu, dm_gpu, dx_gpu, cf_gpu, blocks, n_cb)
                    except torch.cuda.OutOfMemoryError:
                        print(f"    OOM en {name} — liberando y reintentando...", flush=True)
                        for v in ('blocks', 'lv_gpu', 'dm_gpu', 'dx_gpu', 'cf_gpu'):
                            if v in locals():
                                try: del locals()[v]
                                except Exception: pass
                        torch.cuda.empty_cache(); gc.collect()
                        lv_gpu, dm_gpu, dx_gpu, cf_gpu = chunked_v6_encode(blocks)
                        coeff_cpu, bases_list, scales_list = pca_correct_gpu(
                            lv_gpu, dm_gpu, dx_gpu, cf_gpu, blocks, n_cb)

                    lv_cpu = lv_gpu.cpu().numpy().astype(np.uint8).reshape(n_blk, 32)
                    dm_cpu = dm_gpu.cpu().numpy().astype(np.float16)
                    dx_cpu = dx_gpu.cpu().numpy().astype(np.float16)
                    cf_cpu_sp = cf_gpu.cpu().numpy().astype(np.uint8)
                    del blocks, lv_gpu, dm_gpu, dx_gpu, cf_gpu
                    torch.cuda.empty_cache()

                    packed_hi, packed_lo = pack_levels_6bit(lv_cpu)
                    del lv_cpu

                    qt = {
                        'name': name, 'shape': (out_f, in_f),
                        'orig_shape': shape,
                        'n_blocks': n_blk, 'n_cb': n_cb,
                        'dmin': dm_cpu, 'dmax': dx_cpu, 'config': cf_cpu_sp,
                        'levels_hi': packed_hi, 'levels_lo': packed_lo, 'coeff': coeff_cpu,
                    }
                    bi = {'n_cb': n_cb,
                          'basis': bases_list,
                          'scales': np.array(scales_list, dtype=np.float32),
                          'data': np.array([b.flatten() for b in bases_list], dtype=np.float32)}

                    cs, mse = verify_tensor(qt, bi, W_orig_np)
                    cs_list.append(cs); mse_list.append(mse)
                    sh_cs.append(cs); sh_mse.append(mse)

                    write_record(bf, name, qt, bi)

                    sh_blocks += n_blk; sh_params += out_f * in_f
                    total_blk += n_blk; n_params += out_f * in_f
                    vram = torch.cuda.memory_allocated() / 1024**3
                    dt = time.time() - t0
                    eta = (time.time() - t_global) / max(total_blk, 1) * (868_134_478 - total_blk) if total_blk else 0
                    print(f"  [{li+1:3d}/{len(tensors)}] {name[:70]:<72} blk={n_blk:>9,d} CosSim={cs:.6f} "
                          f"{dt:.0f}s VRAM={vram:.1f}GB ETA={eta/60:.0f}min", flush=True)
                    if vram > 13.0:
                        print(f"    ⚠️ VRAM alta ({vram:.1f} GB)", flush=True)

                    del Wf, W_orig_np, qt, bi, dm_cpu, dx_cpu, cf_cpu_sp, packed_hi, packed_lo, coeff_cpu
                    gc.collect()
                    torch.cuda.empty_cache()

        os.replace(body_tmp, body_final)
        st["done"][shard] = {"tensors": len(tensors), "blocks": sh_blocks, "params": sh_params,
                             "time_s": time.time() - t_shard,
                             "avg_cs": float(np.mean(sh_cs)) if sh_cs else 0.0,
                             "cs_list": sh_cs, "mse_list": sh_mse}
        save_state(st)
        print(f"  ✅ {shard} completado en {time.time()-t_shard:.0f}s | {sh_blocks:,} bloques | "
              f"avg CosSim={np.mean(sh_cs):.6f}", flush=True)

    # Resumen global
    bpp = total_blk * 30 * 8 / n_params
    ratio = n_params * 2 / (total_blk * 30)
    avg_cs = float(np.mean(cs_list)) if cs_list else 0.0
    avg_mse = float(np.mean(mse_list)) if mse_list else 0.0
    print(f"\n=== RESUMEN GLOBAL ===", flush=True)
    print(f"Tiempo total: {(time.time()-t_global)/60:.1f} min | Bloques: {total_blk:,} | Params: {n_params:,}", flush=True)
    print(f"bpp: {bpp:.4f} (esperado ~7.50) | ratio: {ratio:.3f} (esperado ~2.13) | "
          f"CosSim medio: {avg_cs:.6f} (referencia 4B: 0.999796) | MSE medio: {avg_mse:.3e}", flush=True)
    assert 7.48 <= bpp <= 7.52, f"bpp fuera de rango: {bpp}"
    assert avg_cs >= 0.9995, f"CosSim demasiado bajo: {avg_cs}"
    meta = {'bpp': float(bpp), 'total_params': int(n_params), 'total_blocks': int(total_blk),
            'ratio': float(ratio), 'quality_cossim': float(avg_cs), 'quality_mse': float(avg_mse),
            'model': MODEL_PATH}
    st["meta"] = meta
    save_state(st)
    print(f"META guardada en state.json: {meta}", flush=True)
    print("SIGUIENTE: ejecutar ensamblaje (assemble_sx43.py) para generar el .sx8v43 final.", flush=True)


if __name__ == "__main__":
    main()
