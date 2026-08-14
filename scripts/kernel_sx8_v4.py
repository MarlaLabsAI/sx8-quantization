"""kernel_sx8_v4.py — S-X8 Flash v4: kernel de decode COMPLETO en GPU (V6 + PCA inline)
Inferencia del formato con kernel propio: niveles → V6 → corrección PCA (4 FMAs) → pesos FP16 en GPU.
Sin round-trip a CPU para la corrección (a diferencia de la vía numpy del prototipo).
"""
import numpy as np, time
import torch
from numba import cuda

DEV = torch.device("cuda")
N_BASES = 4
BYTES_PER_BLOCK = 34  # hi(16) + lo(8) + coeff(2) + dmin(4) + dmax(4)

@cuda.jit
def sx8_v4_kernel(dmin, dmax, config, coeff, hi_arr, lo_arr, basis_buf, scales_buf,
                  weights, n_blocks, n_cb):
    tid = cuda.threadIdx.x & 31
    wid = cuda.threadIdx.x >> 5
    bid = cuda.blockIdx.x * 4 + wid
    if bid >= n_blocks: return

    # 6-bit level: hi nibble (4 bits) + lo quad (2 bits)
    hi = (hi_arr[bid * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
    lo = (lo_arr[bid * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
    lv = (hi << 2) | lo

    # V6 decode (estrategias de rango adaptativas por sub-bloque)
    lo_f = dmin[bid]; hi_f = dmax[bid]
    q = (hi_f - lo_f) * 0.25
    cfg = config[bid]
    sb = tid >> 3
    strat = (cfg >> (sb * 2)) & 3
    if strat == 0: rlo, rhi = lo_f, hi_f
    elif strat == 1: rlo, rhi = lo_f, lo_f + q
    elif strat == 2: rlo, rhi = hi_f - q, hi_f
    else: rlo, rhi = lo_f + q, hi_f - q
    step = (rhi - rlo) * 0.015873
    if step < 1e-10: step = 0.015873
    w = rlo + step * lv

    # Corrección PCA inline (4 FMAs) — bases por columna-slice (n_cb)
    b0 = coeff[bid * 2]; b1 = coeff[bid * 2 + 1]
    c0 = b0 & 0xF; c0 = c0 if c0 < 8 else c0 - 16
    c1 = (b0 >> 4) & 0xF; c1 = c1 if c1 < 8 else c1 - 16
    c2 = b1 & 0xF; c2 = c2 if c2 < 8 else c2 - 16
    c3 = (b1 >> 4) & 0xF; c3 = c3 if c3 < 8 else c3 - 16

    cb = bid % n_cb
    bo = cb * 128; so = cb * 4
    w = w + 1.0 * c0 * scales_buf[so + 0] * basis_buf[bo + tid]
    w = w + 1.0 * c1 * scales_buf[so + 1] * basis_buf[bo + 32 + tid]
    w = w + 1.0 * c2 * scales_buf[so + 2] * basis_buf[bo + 64 + tid]
    w = w + 1.0 * c3 * scales_buf[so + 3] * basis_buf[bo + 96 + tid]

    weights[bid * 32 + tid] = w


def decode_tensor_gpu(qt, bases_info):
    """Decode completo de un tensor S-X8 v4 → torch tensor FP16 en GPU (sin CPU)."""
    out_f, in_f = qt['shape']
    nb = qt['n_blocks']; n_cb = qt['n_cb']

    dm_u = qt['dmin'].view(np.uint32); dx_u = qt['dmax'].view(np.uint32)
    cfg = ((dm_u >> 13) & 0xF | ((dx_u >> 13) & 0xF) << 4).astype(np.uint8)
    coeff_f = qt['coeff'].flatten().astype(np.uint8)

    dm_g = cuda.to_device(qt['dmin'].astype(np.float32))
    dx_g = cuda.to_device(qt['dmax'].astype(np.float32))
    cfg_g = cuda.to_device(cfg)
    coeff_g = cuda.to_device(coeff_f)
    hi_g = cuda.to_device(qt['levels_hi'])
    lo_g = cuda.to_device(qt['levels_lo'])
    basis_g = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
    scales_g = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))
    w_g = cuda.device_array(nb * 32, dtype=np.float32)

    grid = (nb + 3) // 4
    sx8_v4_kernel[grid, 128](dm_g, dx_g, cfg_g, coeff_g, hi_g, lo_g,
                             basis_g, scales_g, w_g, nb, n_cb)
    cuda.synchronize()

    W = torch.as_tensor(w_g.copy_to_host(), dtype=torch.float32, device='cpu')
    W = W.reshape(out_f, n_cb, 32)[:, :, :32].reshape(out_f, n_cb * 32)[:, :in_f]
    del dm_g, dx_g, cfg_g, coeff_g, hi_g, lo_g, basis_g, scales_g, w_g
    orig = qt.get('orig_shape')
    if orig is not None and tuple(orig) != (out_f, in_f):
        W = W.reshape(orig)
    return W.to(DEV).half()


def decode_tensor_gpu_pinned(qt, bases_info):
    """Idéntico a decode_tensor_gpu pero conserva buffer device para reuso (benchmark)."""
    out_f, in_f = qt['shape']
    nb = qt['n_blocks']; n_cb = qt['n_cb']
    dm_u = qt['dmin'].view(np.uint32); dx_u = qt['dmax'].view(np.uint32)
    cfg = ((dm_u >> 13) & 0xF | ((dx_u >> 13) & 0xF) << 4).astype(np.uint8)
    coeff_f = qt['coeff'].flatten().astype(np.uint8)

    dm_g = cuda.to_device(qt['dmin'].astype(np.float32))
    dx_g = cuda.to_device(qt['dmax'].astype(np.float32))
    cfg_g = cuda.to_device(cfg)
    coeff_g = cuda.to_device(coeff_f)
    hi_g = cuda.to_device(qt['levels_hi'])
    lo_g = cuda.to_device(qt['levels_lo'])
    basis_g = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
    scales_g = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))
    w_g = cuda.device_array(nb * 32, dtype=np.float32)
    grid = (nb + 3) // 4
    return dict(dm_g=dm_g, dx_g=dx_g, cfg_g=cfg_g, coeff_g=coeff_g, hi_g=hi_g,
                lo_g=lo_g, basis_g=basis_g, scales_g=scales_g, w_g=w_g, grid=grid,
                nb=nb, n_cb=n_cb, shape=(out_f, in_f), orig=qt.get('orig_shape'))


def run_kernel(b):
    sx8_v4_kernel[b['grid'], 128](b['dm_g'], b['dx_g'], b['cfg_g'], b['coeff_g'],
                                  b['hi_g'], b['lo_g'], b['basis_g'], b['scales_g'],
                                  b['w_g'], b['nb'], b['n_cb'])
    cuda.synchronize()


def benchmark(pkl_path, top_n=10, iters=50):
    """Benchmark de velocidad del kernel v4 sobre los tensores más grandes del pkl."""
    import pickle
    d = pickle.load(open(pkl_path, 'rb'))
    wd, bd = d['weights'], d['bases']
    names = sorted(wd, key=lambda n: -wd[n]['n_blocks'])[:top_n]

    print(f"{'Tensor':<62} {'Bloques':>10} {'ms':>8} {'M p/s':>8} {'GB/s':>7}")
    t_blk = 0.0; t_all = 0.0
    for name in names:
        qt, bi = wd[name], bd[name]
        b = decode_tensor_gpu_pinned(qt, bi)
        for _ in range(5): run_kernel(b)
        times = []
        for _ in range(iters):
            t0 = time.time(); run_kernel(b); times.append(time.time() - t0)
        dt = float(np.median(times))
        wps = b['nb'] * 32 / dt
        bw = b['nb'] * BYTES_PER_BLOCK / dt / 1e9
        print(f"{name[:60]:<62} {b['nb']:>10,} {dt*1e3:8.3f} {wps/1e6:8.1f} {bw:7.2f}")
        t_blk += b['nb'] * 32; t_all += dt
    wps_avg = t_blk / t_all
    print(f"\nMedia ponderada: {wps_avg/1e6:.1f} M pesos/s | "
          f"{t_blk*BYTES_PER_BLOCK/32/t_all/1e9:.2f} GB/s efectivos")
    return wps_avg


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    benchmark("/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_new.pkl")
