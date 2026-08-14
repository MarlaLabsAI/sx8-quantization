"""S-X8 v4 decode compartido — kernel propio Numba CUDA (V6 + PCA inline, 100% GPU)"""
import torch, numpy as np, pickle, os, gc, time
from numba import cuda as nbcuda
from kernel_sx8_v4 import decode_tensor_gpu
from kernel_sx8_v43 import decode_tensor_gpu_fast as decode_tensor_gpu_v43

PKL_PATH = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_new.pkl"
PKL_PATH_V43 = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
MODEL_PATH = "/mnt/Data_3TB/Qwen3.5-4B"
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
    if step < 1e-10: step = 0.015873
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


def load_model(quantized=True, use_cache=True, verify=False, mode="v4", source_file=None):
    """Carga el modelo FP16 (clase multimodal Qwen3_5ForConditionalGeneration) y
    (si quantized) sustituye los pesos con el decode del kernel propio v4 (100% GPU).
    mode="v4" = 4 bases (formato v4) · mode="v42" = 2 bases (v4.2) · mode="v43" = v4.3 (config aparte + escala FP16)"""
    from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cuda",
        low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    if not quantized:
        return model, tok, None

    pkl_path = PKL_PATH_V43 if mode == "v43" else PKL_PATH
    decode_fn = decode_tensor_gpu_v43 if mode == "v43" else decode_tensor_gpu
    if source_file:
        from sx8_container_v43 import read_all
        wd, bd, meta = read_all(source_file)
        print(f"📄 Cargado desde archivo: {source_file.split('/')[-1]}", flush=True)
    else:
        d = pickle.load(open(pkl_path, 'rb'))
        wd, bd, meta = d['weights'], d['bases'], d['meta']
    params = dict(model.named_parameters())

    missing = [k for k in wd if k not in params]
    if missing:
        print(f"⚠️ {len(missing)} tensores del pkl NO están en el modelo (se ignoran): {missing[:5]}")
    matched = 0
    sumsq_o = 0.0; sumsq_r = 0.0; dot = 0.0
    t_start = time.time()
    for idx, (name, qt) in enumerate(wd.items()):
        if name not in params:
            continue
        qt_use = qt
        if mode == "v42":
            qt_use = dict(qt)
            c = np.zeros_like(qt['coeff']); c[:, 0] = qt['coeff'][:, 0]
            qt_use['coeff'] = c                       # solo bases 0-1 (coeff 1 byte)
        W = decode_fn(qt_use, bd[name])               # kernel propio (v4 o v4.3)
        p = params[name]
        p.data.copy_(W)
        matched += 1
        if verify:
            o = p.data.float().cpu().numpy().ravel().astype(np.float64)
            r = W.cpu().numpy().ravel().astype(np.float64)
            dot += float(o @ r)
            sumsq_o += float((o * o).sum()); sumsq_r += float((r * r).sum())
        del W
        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (idx + 1) % 50 == 0 or (idx + 1) == len(wd):
            print(f"  [{idx+1}/{len(wd)}] tensores decodeados con kernel v{'4.3' if mode=='v43' else '4'} ({time.time()-t_start:.0f}s)", flush=True)
    print(f"SX8 {mode} ({pkl_path.split('/')[-1]}): {matched}/{len(wd)} tensores aplicados", flush=True)
    if verify:
        cs = dot / (np.sqrt(sumsq_o) * np.sqrt(sumsq_r) + 1e-12)
        print(f"  ✅ CosSim GLOBAL (todos los tensores): {cs:.6f}")
    return model, tok, meta


if __name__ == "__main__":
    import sys
    m, t, meta = load_model(quantized=(len(sys.argv) < 2 or sys.argv[1] != 'fp16'))
    n = sum(v.numel() for v in m.parameters())
    print(f"OK — modelo cargado, {n/1e9:.2f}B params")
