"""S-X8 Flash v4 — FULL GPU Quantization (V6 + PCA-POST) — Qwen3.5-4B COMPLETO (texto+vision+MTP+conv, solo normas 1D FP32)"""
import torch, numpy as np, time, os, struct, pickle, gc, sys, collections
from safetensors import safe_open

MODEL_PATH = "/mnt/Data_3TB/Qwen3.5-4B"
B = 32; L63 = 63; N_BASES = 4   # v4.3: PCA se CALCULA con 4 bases (calibración idéntica al v4)
                               # pero el formato GUARDA solo 2 (coeff 1 byte) — el esquema que dio PPL 10.2289
MAX_BLOCKS_PER_CHUNK = 3_000_000
DEV = torch.device("cuda")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = "/mnt/Data_3TB/project Marla/quant-paper/models"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_NAME = "qwen35_4b_sx8_flash_v4_3.pkl"

props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} ({props.total_memory//1024**2} MB) | S-X8 Flash v4 FULL GPU (PCA-POST)", flush=True)
print(f"Modelo: {MODEL_PATH}\n", flush=True)

# ============================================================================
# V6 Encode with L=63 (GPU, identical logic to L=15 but 6-bit levels)
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
# GPU PCA Correction — per column slice
# ============================================================================
def pca_correct_gpu(levels, dmin, dmax, config, blocks_gpu, n_cb):
    """
    GPU PCA correction: SVD per column slice, compute basis + coeff.
    levels: (n_blk, 32) uint8 on GPU
    dmin, dmax: (n_blk,) float32
    config: (n_blk,) int32
    blocks_gpu: (n_blk, 32) float32
    n_cb: number of column blocks
    Returns: coeff_cpu (n_blk, 2) uint8, bases_list, scales_list
    """
    n_blk = blocks_gpu.shape[0]
    d = (dmax - dmin).clamp(min=1e-10)
    Q = d / 4.0
    _L = float(L63)

    # Precompute lo, step per strategy (4, n_blk)
    lo_j = torch.stack([dmin, dmin, dmax - Q, dmin + Q], dim=0)
    st_j = torch.stack([d / _L, Q / _L, Q / _L,
                         (d - 2 * Q).clamp(min=1e-10) / _L], dim=0)

    lv_f = levels.float()  # (n_blk, 32) float32
    coeff_bytes = torch.zeros(n_blk, 1, dtype=torch.uint8, device=DEV)  # 1 byte: c0|c1<<4
    bases_list = []
    scales_list = []

    for cb in range(n_cb):
        si = torch.where((torch.arange(n_blk, device=DEV) % n_cb) == cb)[0]
        n_slice = si.shape[0]

        if n_slice < 4:
            bases_list.append(np.zeros((2, 32), dtype=np.float32))
            scales_list.append(np.zeros(2, dtype=np.float32))
            continue

        # V6 reconstruction for this slice
        v6_rec = torch.zeros(n_slice, 32, dtype=torch.float32, device=DEV)
        cfg_s = config[si].long()

        for sb in range(4):
            strat_sb = (cfg_s >> (sb * 2)) & 3
            lo_sb = lo_j[strat_sb, si]
            st_sb = st_j[strat_sb, si]
            sbi = slice(sb * 8, (sb + 1) * 8)
            v6_rec[:, sbi] = lo_sb[:, None] + st_sb[:, None] * lv_f[si][:, sbi]

        error = blocks_gpu[si] - v6_rec  # (n_slice, 32)

        # GPU SVD
        _, S, Vh = torch.linalg.svd(error, full_matrices=False)
        basis = Vh[:N_BASES]  # (2, 32)

        # Coefficients and scales
        coeffs_raw = torch.mm(basis, error.T)  # (2, n_slice)
        scales = torch.median(coeffs_raw.abs(), dim=1).values / 3.0
        scales = scales.clamp(min=1e-10)

        # Quantize to 4-bit signed
        c_quant = (coeffs_raw / scales[:, None]).round().clamp(-8, 7).to(torch.int8)

        byte0 = ((c_quant[0].byte() & 0xF) | ((c_quant[1].byte() & 0xF) << 4))
        coeff_bytes[si, 0] = byte0

        bases_list.append(basis[:2].cpu().numpy().astype(np.float32))   # guardar solo 2 bases
        scales_list.append(scales[:2].cpu().numpy().astype(np.float32)) # y sus escalas

        del si, v6_rec, error, basis, coeffs_raw, c_quant, byte0, S, Vh

    return coeff_bytes.cpu().numpy().astype(np.uint8), bases_list, scales_list


# ============================================================================
# Packing: 6-bit levels → 4+2 split
# ============================================================================
def pack_levels_6bit(levels_cpu):
    """levels_cpu: (n_blk, 32) uint8 with values 0-63"""
    lv_hi = (levels_cpu & 0x3F) >> 2  # high 4 bits
    lv_lo = levels_cpu & 0x03          # low 2 bits

    # Nibble-pack hi
    lvf_hi = lv_hi.flatten()
    lo_hi = lvf_hi[0::2] & 0xF
    hi_hi = (lvf_hi[1::2] << 4) & 0xF0
    packed_hi = (lo_hi | hi_hi).astype(np.uint8)

    # Quad-pack lo
    lvf_lo = lv_lo.flatten()
    groups = lvf_lo.reshape(-1, 4)
    packed_lo = ((groups[:,0].astype(np.uint8)&0x3)<<6) | \
                ((groups[:,1].astype(np.uint8)&0x3)<<4) | \
                ((groups[:,2].astype(np.uint8)&0x3)<<2) | \
                (groups[:,3].astype(np.uint8)&0x3)

    return packed_hi, packed_lo


def hide_config(dm_cpu, dx_cpu, cf_cpu):
    u32_low = dm_cpu.view(np.uint32).copy()
    u32_high = dx_cpu.view(np.uint32).copy()
    CLEAR = np.uint32(0xFFFFFFFF ^ (0xF << 13))
    u32_low[:] = (u32_low & CLEAR) | ((cf_cpu.astype(np.uint32) & 0x0F) << 13)
    u32_high[:] = (u32_high & CLEAR) | (((cf_cpu.astype(np.uint32) >> 4) & 0x0F) << 13)
    return u32_low.view(np.float32), u32_high.view(np.float32)


# ============================================================================
# Load model
# ============================================================================
print("Cargando modelo...", flush=True)
t0 = time.time()
from transformers import Qwen3_5ForConditionalGeneration
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.float16,
    device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()

linear_info = []
orig_shapes = {}
for name, param in model.named_parameters():
    if 'weight' in name and param.ndim >= 2:
        flat = (param.shape[0], int(np.prod(param.shape[1:])))
        linear_info.append((name, flat))
        orig_shapes[name] = tuple(param.shape)
del model; gc.collect()
n_total = sum(s[0]*s[1] for _, s in linear_info)

prefs = collections.Counter('.'.join(n.split('.')[:2]) for n, _ in linear_info)
print(f"  {time.time()-t0:.0f}s | {len(linear_info)} tensores | {n_total:,} params | por prefijo:")
for p, c in prefs.most_common(15):
    print(f"    {p}: {c}", flush=True)
print(flush=True)

# ============================================================================
# Quantize (FULL GPU: V6 encode + PCA correction)
# ============================================================================
print("Cuantizando S-X8 v4 (V6 6-bit + PCA-POST GPU)...", flush=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.float16,
    device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
model.eval()
param_dict = dict(model.named_parameters())

# MTP no se instancia en la clase — leerlo directo de los safetensors (simetría total con GGUF)
for shard in sorted(os.listdir(MODEL_PATH)):
    if not shard.endswith(".safetensors") or shard.startswith("model.safetensors.index"):
        continue
    with safe_open(os.path.join(MODEL_PATH, shard), framework="pt") as f:
        for name in f.keys():
            if name.startswith("mtp.") and "weight" in name:
                t = f.get_tensor(name).float()
                if t.dim() >= 2 and name not in param_dict:
                    param_dict[name] = torch.nn.Parameter(t)
                    flat = (t.shape[0], int(np.prod(t.shape[1:])))
                    linear_info.append((name, flat))
                    orig_shapes[name] = tuple(t.shape)
                    print(f"  MTP inyectado: {name} {tuple(t.shape)}", flush=True)

n_total = sum(s[0]*s[1] for _, s in linear_info)
print(f"  TOTAL tras MTP: {n_total:,} params | {len(linear_info)} tensores", flush=True)

all_qt = []; all_bases = {}; total_blk = 0; t_start = time.time()
max_name_len = max(len(name) for name, _ in linear_info)

for li, (name, (out_f, in_f)) in enumerate(linear_info):
    param = param_dict[name]
    W = param.data.float()
    if W.dim() > 2:
        W = W.reshape(W.shape[0], -1)
    W = W.to(DEV)
    n_cb = (in_f + B - 1) // B
    pad_cols = (B - in_f % B) % B
    if pad_cols > 0:
        W = torch.nn.functional.pad(W, (0, pad_cols))
    blocks = W.reshape(out_f, n_cb, B).reshape(-1, B).contiguous()
    del W
    n_blk = blocks.shape[0]

    # Step 1: Chunked V6 encode (L=63)
    lv_gpu, dm_gpu, dx_gpu, cf_gpu = chunked_v6_encode(blocks)

    # Step 2: PCA correction on GPU
    coeff_cpu, bases_list, scales_list = pca_correct_gpu(
        lv_gpu, dm_gpu, dx_gpu, cf_gpu, blocks, n_cb)

    # Step 3: Move to CPU, pack
    lv_cpu = lv_gpu.cpu().numpy().astype(np.uint8).reshape(n_blk, 32)
    dm_cpu = dm_gpu.cpu().numpy().astype(np.float16)   # v4.3: escala FP16 (2 B, exacta)
    dx_cpu = dx_gpu.cpu().numpy().astype(np.float16)
    cf_cpu_sp = cf_gpu.cpu().numpy().astype(np.uint8)  # v4.3: config EXPLÍCITA (array aparte)
    del blocks, lv_gpu, dm_gpu, dx_gpu, cf_gpu
    torch.cuda.empty_cache()

    packed_hi, packed_lo = pack_levels_6bit(lv_cpu)

    all_qt.append({
        'name': name, 'shape': (out_f, in_f),
        'orig_shape': orig_shapes.get(name, (out_f, in_f)),
        'n_blocks': n_blk, 'n_cb': n_cb,
        'dmin': dm_cpu, 'dmax': dx_cpu, 'config': cf_cpu_sp,
        'levels_hi': packed_hi, 'levels_lo': packed_lo, 'coeff': coeff_cpu,
    })
    all_bases[name] = {'basis': bases_list, 'scales': scales_list, 'n_cb': n_cb}
    total_blk += n_blk

    if li < 5 or (li+1) % 30 == 0 or (li+1) >= len(linear_info):
        elapsed = time.time() - t_start
        bps = total_blk / max(elapsed, 0.1)
        print(f"  [{li+1:3d}/{len(linear_info)}] {name:<{max_name_len+2}} blk={n_blk:>8d}  {elapsed:.0f}s  {bps/1e6:.1f}M blk/s", flush=True)

    del lv_cpu, dm_cpu, dx_cpu, cf_cpu_sp, packed_hi, packed_lo, coeff_cpu
    gc.collect()

t_total = time.time() - t_start
bpp = total_blk * 30 * 8 / n_total   # v4.3: 30 B/bloque reales
ratio = n_total * 2 / (total_blk * 30)
print(f"\n  TOTAL: {t_total:.0f}s | Blk: {total_blk:,} | {total_blk*30/1024**2:.0f} MB")
print(f"  bpp: {bpp:.2f} | Ratio: {ratio:.2f}x")
print(f"  Velocidad: {total_blk/t_total/1e6:.1f}M blk/s\n", flush=True)

# ============================================================================
# Quality check — Numba CUDA V6 decode + GPU PCA correction
# ============================================================================
print("Verificando calidad (8 capas, Numba V6 + GPU PCA)...", flush=True)
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
    if step < 1e-10: step = 0.015873
    weights[bid * 32 + tid] = rlo + step * lv

sum_cs, sum_mse, n_chk = 0.0, 0.0, 0

for nch, (nm, (out_f, in_f)) in enumerate(linear_info):
    if nch >= 8: break
    qt_all = [q for q in all_qt if q['name'] == nm]
    if not qt_all: continue
    qt = qt_all[0]

    t0 = time.time()
    param = param_dict[nm]
    Worig = param.data.float()
    if Worig.dim() > 2:
        Worig = Worig.reshape(Worig.shape[0], -1)
    Worig = Worig.numpy()
    nblk = qt['n_blocks']; ncb = qt['n_cb']

    # v4.3: config EXPLÍCITA (array aparte)
    cfg = qt['config'].astype(np.uint8)

    # Numba decode V6
    dmin_g = nbcuda.to_device(qt['dmin'].astype(np.float32))
    dmax_g = nbcuda.to_device(qt['dmax'].astype(np.float32))
    config_g = nbcuda.to_device(cfg)
    hi_g = nbcuda.to_device(qt['levels_hi'])
    lo_g = nbcuda.to_device(qt['levels_lo'])
    w_g = nbcuda.device_array(nblk * 32, dtype=np.float32)

    grid = (nblk + 3) // 4
    sx8_v6_kernel[grid, 128](dmin_g, dmax_g, config_g, hi_g, lo_g, w_g, nblk)
    nbcuda.synchronize()

    Wv6 = w_g.copy_to_host().reshape(nblk, 32)

    # PCA correction (2 bases)
    bases_info = all_bases[nm]
    Wdec = Wv6.copy()
    for cb in range(ncb):
        si = np.arange(nblk)[np.arange(nblk) % ncb == cb]
        if si.size < 4: continue
        basis = bases_info['basis'][cb]           # (2, 32)
        scales = bases_info['scales'][cb]         # (2,)
        coeff = qt['coeff'][si]                   # (n_slice, 1)
        b0 = coeff[:, 0].astype(np.int32)
        c_raw = np.stack([b0 & 0xF, (b0 >> 4) & 0xF], axis=1)
        c_quant = np.where(c_raw >= 8, c_raw - 16, c_raw).astype(np.float32)  # (n_slice, 2)
        correction = np.dot(c_quant * scales[np.newaxis, :], basis)  # (n_slice, 32)
        Wdec[si] += correction

    Wdec = Wdec.reshape(out_f, ncb, 32).reshape(out_f, ncb*32)[:, :in_f]

    o = Worig.flatten(); r = Wdec.flatten()
    cs = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-12))
    mse = float(np.mean((o - r) ** 2))
    dt = time.time() - t0
    print(f"  {nm:<{max_name_len+2}} CosSim={cs:.6f} MSE={mse:.6f}  ({dt:.1f}s)", flush=True)
    sum_cs += cs; sum_mse += mse; n_chk += 1

avg_cs = sum_cs / n_chk if n_chk else 0
avg_mse = sum_mse / n_chk if n_chk else 0
print(f"\n  PROMEDIO: CosSim={avg_cs:.6f} MSE={avg_mse:.6f}\n", flush=True)

del model; gc.collect()

# ============================================================================
# Save
# ============================================================================
print("Guardando...", flush=True)
wd = {}
for qt in all_qt:
    wd[qt['name']] = {k: qt[k] for k in ['shape', 'orig_shape', 'n_blocks', 'n_cb',
                       'dmin', 'dmax', 'config',
                       'levels_hi', 'levels_lo', 'coeff']}

bases_serialized = {}
for nm, bd in all_bases.items():
    bl = bd['basis']
    bases_serialized[nm] = {
        'n_cb': bd['n_cb'],
        'data': np.array([b.flatten() for b in bl], dtype=np.float32) if bl else np.zeros((0, 128), dtype=np.float32),
        'scales': np.array(bd['scales'], dtype=np.float32),
    }

meta = {'bpp': float(bpp), 'total_params': n_total, 'total_blocks': total_blk,
        'ratio': float(ratio), 'quality_cossim': float(avg_cs), 'quality_mse': float(avg_mse),
        'model': MODEL_PATH}
out_path = os.path.join(OUT_DIR, OUT_NAME)
with open(out_path, 'wb') as f:
    pickle.dump({'weights': wd, 'bases': bases_serialized, 'meta': meta}, f)
mb = os.path.getsize(out_path) / 1024**2
print(f"  {mb:.0f} MB | CosSim={avg_cs:.6f} MSE={avg_mse:.6f} | {out_path}", flush=True)
