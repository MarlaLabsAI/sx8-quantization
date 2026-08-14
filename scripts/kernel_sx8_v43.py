"""kernel_sx8_v43.py — S-X8 v4.3: decode GPU (config EXPLÍCITA + PCA de 2 bases)
Formato 30 B/bloque: dmin(fp16,2) + dmax(fp16,2) + config(1) + hi(16) + lo(8) + coeff(1, 2 bases)
La escala es FP16 EXACTA (10 bits de mantissa — +4 bits vs v4) porque la config ya no
se esconde en la mantissa. PCA: 2 FMAs (bases 0-1).
"""
import numpy as np, time
import torch
from numba import cuda

DEV = torch.device("cuda")
N_BASES = 2
BYTES_PER_BLOCK = 30  # 2+2+1+16+8+1

@cuda.jit
def sx8_v43_kernel(dmin, dmax, config, coeff, hi_arr, lo_arr, basis_buf, scales_buf,
                   weights, n_blocks, n_cb):
    tid = cuda.threadIdx.x & 31
    wid = cuda.threadIdx.x >> 5
    bid = cuda.blockIdx.x * 4 + wid
    if bid >= n_blocks: return

    # 6-bit level: hi nibble (4 bits) + lo quad (2 bits)
    hi = (hi_arr[bid * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
    lo = (lo_arr[bid * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
    lv = (hi << 2) | lo

    # V6 decode — config EXPLÍCITA (array aparte, 2 bits por sub-bloque)
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

    # Corrección PCA con 2 bases (2 FMAs)
    b0 = coeff[bid]
    c0 = b0 & 0xF; c0 = c0 if c0 < 8 else c0 - 16
    c1 = (b0 >> 4) & 0xF; c1 = c1 if c1 < 8 else c1 - 16

    cb = bid % n_cb
    bo = cb * 64; so = cb * 2
    w = w + 1.0 * c0 * scales_buf[so + 0] * basis_buf[bo + tid]
    w = w + 1.0 * c1 * scales_buf[so + 1] * basis_buf[bo + 32 + tid]

    weights[bid * 32 + tid] = w


def decode_tensor_gpu(qt, bases_info):
    """Decode de un tensor S-X8 v4.3 → torch tensor FP16 en GPU.
    Entrada: qt con dmin/dmax float16, config uint8, coeff uint8 (n_blk,), bases (n_cb, 64)."""
    out_f, in_f = qt['shape']
    nb = qt['n_blocks']; n_cb = qt['n_cb']

    dm_g = cuda.to_device(qt['dmin'].astype(np.float32))
    dx_g = cuda.to_device(qt['dmax'].astype(np.float32))
    cfg_g = cuda.to_device(qt['config'].astype(np.uint8))
    coeff_g = cuda.to_device(qt['coeff'].reshape(-1).astype(np.uint8))
    hi_g = cuda.to_device(qt['levels_hi'])
    lo_g = cuda.to_device(qt['levels_lo'])
    basis_g = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
    scales_g = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))
    w_g = cuda.device_array(nb * 32, dtype=np.float32)

    grid = (nb + 3) // 4
    sx8_v43_kernel[grid, 128](dm_g, dx_g, cfg_g, coeff_g, hi_g, lo_g,
                              basis_g, scales_g, w_g, nb, n_cb)
    cuda.synchronize()

    W = torch.as_tensor(w_g.copy_to_host(), dtype=torch.float32, device='cpu')
    W = W.reshape(out_f, n_cb, 32).reshape(out_f, n_cb * 32)[:, :in_f]
    del dm_g, dx_g, cfg_g, coeff_g, hi_g, lo_g, basis_g, scales_g, w_g
    orig = qt.get('orig_shape')
    if orig is not None and tuple(orig) != (out_f, in_f):
        W = W.reshape(orig)
    return W.to(DEV).half()


_BUFFER_POOL = {}
_BUFFER_POOL_ORDER = []
_BUFFER_POOL_MAX = 3


def _get_buffer(n):
    """Pool de buffers CUDA fp32 (máx 3, LRU) — evita alloc/free repetido
    y la fragmentación del pool de torch. Los decodes son síncronos, así que
    reutilizar el buffer es seguro."""
    n_round = (n + 4095) & ~4095
    for k in _BUFFER_POOL_ORDER:
        if k >= n_round:
            buf = _BUFFER_POOL[k]
            _BUFFER_POOL_ORDER.remove(k)
            _BUFFER_POOL_ORDER.append(k)
            return buf
    if len(_BUFFER_POOL_ORDER) >= _BUFFER_POOL_MAX:
        evict = _BUFFER_POOL_ORDER.pop(0)
        del _BUFFER_POOL[evict]
    buf = torch.empty(n_round, dtype=torch.float32, device=DEV)
    _BUFFER_POOL[n_round] = buf
    _BUFFER_POOL_ORDER.append(n_round)
    return buf


def decode_tensor_gpu_fast(qt, bases_info, out=None):
    """Decode S-X8 v4.3 -> torch FP16 CUDA SIN round-trip CPU (FASE 2.0).
    torch es el dueno de la memoria (torch.empty CUDA); numba escribe via
    as_cuda_array (vista). Cero copias GPU<->CPU. Devuelve tensor CUDA fp16.
    Si out se da (tensor CUDA fp16 del shape final), copia en el y lo devuelve."""
    out_f, in_f = qt['shape']
    nb = qt['n_blocks']; n_cb = qt['n_cb']

    if out is not None and tuple(out.shape) != (out_f, in_f):
        raise ValueError(f"out shape {tuple(out.shape)} != {(out_f, in_f)}")

    W = _get_buffer(nb * 32)[:nb * 32]
    w_g = cuda.as_cuda_array(W)

    dm_g = cuda.to_device(qt['dmin'].astype(np.float32))
    dx_g = cuda.to_device(qt['dmax'].astype(np.float32))
    cfg_g = cuda.to_device(qt['config'].astype(np.uint8))
    coeff_g = cuda.to_device(qt['coeff'].reshape(-1).astype(np.uint8))
    hi_g = cuda.to_device(qt['levels_hi'])
    lo_g = cuda.to_device(qt['levels_lo'])
    basis_g = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
    scales_g = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))

    grid = (nb + 3) // 4
    sx8_v43_kernel[grid, 128](dm_g, dx_g, cfg_g, coeff_g, hi_g, lo_g,
                              basis_g, scales_g, w_g, nb, n_cb)
    cuda.synchronize()

    Wf = W.reshape(out_f, n_cb, 32).reshape(out_f, n_cb * 32)[:, :in_f]
    orig = qt.get('orig_shape')
    if orig is not None and tuple(orig) != (out_f, in_f):
        Wf = Wf.reshape(orig)
    Wh = Wf.half()
    if out is not None:
        out.copy_(Wh)
        return out
    return Wh


def benchmark(pkl_path, top_n=10, iters=50):
    """Benchmark de velocidad del kernel v4.3."""
    import pickle
    d = pickle.load(open(pkl_path, 'rb'))
    wd, bd = d['weights'], d['bases']
    names = sorted(wd, key=lambda n: -wd[n]['n_blocks'])[:top_n]
    print(f"{'Tensor':<62} {'Bloques':>10} {'ms':>8} {'M p/s':>8} {'GB/s':>7}")
    t_blk = 0.0; t_all = 0.0
    for name in names:
        qt, bi = wd[name], bd[name]
        out_f, in_f = qt['shape']; nb, ncb = qt['n_blocks'], qt['n_cb']
        dm_g = cuda.to_device(qt['dmin'].astype(np.float32))
        dx_g = cuda.to_device(qt['dmax'].astype(np.float32))
        cfg_g = cuda.to_device(qt['config'].astype(np.uint8))
        coeff_g = cuda.to_device(qt['coeff'].reshape(-1).astype(np.uint8))
        hi_g = cuda.to_device(qt['levels_hi']); lo_g = cuda.to_device(qt['levels_lo'])
        basis_g = cuda.to_device(bi['data'].reshape(-1).astype(np.float32))
        scales_g = cuda.to_device(bi['scales'].reshape(-1).astype(np.float32))
        w_g = cuda.device_array(nb * 32, dtype=np.float32)
        grid = (nb + 3) // 4
        for _ in range(5):
            sx8_v43_kernel[grid, 128](dm_g, dx_g, cfg_g, coeff_g, hi_g, lo_g,
                                      basis_g, scales_g, w_g, nb, ncb)
        cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.time()
            sx8_v43_kernel[grid, 128](dm_g, dx_g, cfg_g, coeff_g, hi_g, lo_g,
                                      basis_g, scales_g, w_g, nb, ncb)
            cuda.synchronize(); times.append(time.time() - t0)
        dt = float(np.median(times))
        wps = nb * 32 / dt; bw = nb * BYTES_PER_BLOCK / dt / 1e9
        print(f"{name[:60]:<62} {nb:>10,} {dt*1e3:8.3f} {wps/1e6:8.1f} {bw:7.2f}")
        t_blk += nb * 32; t_all += dt
    print(f"\nMedia ponderada: {t_blk/t_all/1e6:.1f} M pesos/s | {t_blk*BYTES_PER_BLOCK/32/t_all/1e9:.2f} GB/s")
    return t_blk / t_all


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    benchmark("/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl")
