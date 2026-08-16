"""eval_common.py v2 — S-X8 v4.3 runtime compartido (STANDALONE).

v1.1 del contenedor (config + 2D cuantizados + 1D dentro) → modelo completo
sin depender del modelo base. Sin rutas hardcodeadas.

Fix v1.1 (2026-08-16): kernel de decode con fallback correcto para bloques
degenerados (min==max) — `step = 1e-10` en vez de `step = 0.015873`.
"""
import torch, numpy as np, os, gc, time
from numba import cuda as nbcuda
from kernel_sx8_v4 import decode_tensor_gpu
from kernel_sx8_v43 import decode_tensor_gpu_fast as decode_tensor_gpu_v43

DEV = torch.device("cuda")
N_BASES = 4


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


def decode_tensor(qt, bases_info):
    """Decodifica un tensor S-X8 v4 → FP16 numpy (out_f, in_f)"""
    out_f, in_f = qt['shape']
    nblk = qt['n_blocks']; ncb = qt['n_cb']
    dm_u32 = qt['dmin'].view(np.uint32); dx_u32 = qt['dmax'].view(np.uint32)
    cfg = ((dm_u32 >> 13) & 0xF | ((dx_u32 >> 13) & 0xF) << 4).astype(np.uint8)

    dmin_g = nbcuda.to_device(qt['dmin'].astype(np.float32))
    dmax_g = nbcuda.to_device(qt['dmax'].astype(np.float32))
    config_g = nbcuda.to_device(cfg)
    hi_g = nbcuda.to_device(qt['levels_hi'])
    lo_g = nbcuda.to_device(qt['levels_lo'])
    w_g = nbcuda.device_array(nblk * 32, dtype=np.float32)
    sx8_v6_kernel[(nblk + 3) // 4, 128](dmin_g, dmax_g, config_g, hi_g, lo_g, w_g, nblk)
    nbcuda.synchronize()
    Wv6 = w_g.copy_to_host().reshape(nblk, 32)
    del dmin_g, dmax_g, config_g, hi_g, lo_g, w_g

    Wdec = Wv6.copy()
    data = bases_info['data']; scales = bases_info['scales']
    coeff = qt['coeff']
    if data.size > 0:
        for cb in range(ncb):
            si = np.arange(nblk)[np.arange(nblk) % ncb == cb]
            if si.size < 4: continue
            basis = data[cb].reshape(N_BASES, 32).astype(np.float32)
            sc = scales[cb].astype(np.float32)
            cs = coeff[si].astype(np.int32)
            raw = np.stack([cs[:,0] & 0xF, (cs[:,0] >> 4) & 0xF,
                            cs[:,1] & 0xF, (cs[:,1] >> 4) & 0xF], axis=1)
            cq = np.where(raw >= 8, raw - 16, raw).astype(np.float32)
            Wdec[si] += cq @ (basis * sc[:, None])

    Wrec = Wdec.reshape(out_f, ncb, 32).reshape(out_f, ncb * 32)[:, :in_f]
    orig = qt.get('orig_shape')
    if orig is not None and tuple(orig) != (out_f, in_f):
        Wrec = Wrec.reshape(orig)
    return Wrec.astype(np.float16)


def load_model_standalone(container_path, tokenizer_dir=None, device="cuda",
                          verify=False, decode_fn=None):
    """Carga el modelo SOLO desde el contenedor .sx8v43 v1.1 (autosuficiente).

    - container_path: .sx8v43 v1.1 (config + 2D cuantizados + 1D dentro)
    - tokenizer_dir: carpeta con tokenizer.json
    El modelo se construye en CPU (estructura + buffers correctos) y los pesos
    se sustituyen: 1D del contenedor + decode 2D con kernel propio. CERO
    dependencia del modelo base.
    """
    from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5Config
    from sx8_container_v43 import read_all_v11
    wd, bd, meta, config, small = read_all_v11(container_path)
    if config is None:
        raise ValueError("Contenedor v1.0 sin sección v1.1 (config+1D) — usa el v2")

    dec = decode_fn or decode_tensor_gpu_v43
    t0 = time.time()
    cfg_obj = Qwen3_5Config(**config)
    model = Qwen3_5ForConditionalGeneration(cfg_obj)
    model.eval()
    print(f"Estructura creada desde config del contenedor ({(time.time()-t0):.0f}s)", flush=True)

    params = dict(model.named_parameters())
    n_small = 0
    for name, st in sorted(small.items()):
        if name not in params:
            continue
        arr = st['data']
        dt = torch.float16 if st['dtype'] == 'fp16' else torch.float32
        params[name].data = torch.from_numpy(arr.copy()).to(dt)
        n_small += 1
    print(f"Tensores 1D materializados: {n_small}/{len(small)}", flush=True)

    matched = 0
    for name, qt in wd.items():
        if name not in params:
            continue
        W = dec(qt, bd[name])
        params[name].data = W.to(torch.float16).cpu()
        matched += 1
        del W
        if (matched + 1) % 50 == 0:
            torch.cuda.empty_cache()
        if (matched + 1) % 100 == 0 or matched == len(wd):
            print(f"  [{matched}/{len(wd)}] tensores 2D decodeados ({time.time()-t0:.0f}s)", flush=True)
    print(f"SX8 v4.3 standalone: {matched}/{len(wd)} tensores 2D aplicados", flush=True)
    import kernel_sx8_v43 as _k43
    _k43._BUFFER_POOL.clear()
    _k43._BUFFER_POOL_ORDER.clear()
    torch.cuda.empty_cache()
    model = model.to(device)

    if verify:
        sum_cs, n = 0.0, 0
        import random
        sample = random.sample(list(wd.keys()), min(10, len(wd)))
        for name in sample:
            qt = wd[name]
            W = dec(qt, bd[name]).to(torch.float16)
            o = params[name].data.float()
            r = W.float()
            cs = float((o * r).sum() / (o.norm() * r.norm() + 1e-12))
            sum_cs += cs; n += 1
        print(f"  ✅ CosSim muestra ({n} tensores): {sum_cs/n:.6f}", flush=True)

    tok = None
    if tokenizer_dir is None:
        tokenizer_dir = os.path.dirname(container_path)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Tokenizer no cargado: {e}", flush=True)
    return model, tok, meta


def load_model(quantized=True, use_cache=True, verify=False, mode="v4",
               source_file=None, container=None):
    """Compatibilidad con scripts v1: carga standalone desde el contenedor v1.1.

    El contenedor .sx8v43 v1.1 es COMPLETO (config + 2D cuantizados + 1D), así que
    no existe ya un "modelo base" separado: el modelo ES el contenedor. Los modos
    v4/v42/v43 eran del pkl; el contenedor es siempre v4.3.
    """
    if quantized is False:
        raise NotImplementedError(
            "Modo FP16-base eliminado en v1.1: el modelo standalone se evalúa "
            "desde el contenedor (.sx8v43). Usa quantized=True (por defecto).")
    c = container or source_file or os.environ.get(
        "SX8_CONTAINER", "Qwen3.5-4B-SX8v43.sx8")
    if not os.path.exists(c):
        raise FileNotFoundError(
            f"Contenedor no encontrado: {c}. Pasa source_file=<ruta.al.sx8v43> "
            f"o define SX8_CONTAINER.")
    return load_model_standalone(c, tokenizer_dir=None, verify=verify, device="cuda")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python3 eval_common.py <contenedor.sx8> [tokenizer_dir]")
        sys.exit(1)
    m, t, meta = load_model_standalone(sys.argv[1],
                                       sys.argv[2] if len(sys.argv) > 2 else None,
                                       verify=True)
    n = sum(v.numel() for v in m.parameters())
    print(f"OK — modelo cargado standalone, {n/1e9:.2f}B params")
