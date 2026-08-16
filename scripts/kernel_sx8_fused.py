"""kernel_sx8_fused.py — Kernel FUSED dequant+GEMM para S-X8 v4.3 (FASE 2.1)

Objetivo: Y = X @ W con W SX8 v4.3 viviendo COMPRIMIDA en VRAM (~4.4 GB).
Cada thread decodifica en vuelo los bytes que necesita (niveles -> V6 -> PCA)
dentro del matmul. Nunca existe la matriz FP16 completa.

Layout W en VRAM (mismo que kernel_sx8_v43): W plano (out_f=n_salidas, in_f=K)
con bloques de 32 pesos a lo largo de K. bid = n*n_cb + kb  (n = columna salida,
kb = bloque de K). base codebook compartida por fila: cb = kb.

V1 naive: 1 thread por elemento Y[m,n]; decode inline; primero CORRECTO.
"""
import numpy as np
import torch
from numba import cuda

DEV = torch.device("cuda")


def to_kbmajor(qt, bases_info):
    """Convierte los arrays de metadatos del pkl (n-major: bid=n*n_cb+kb) a
    kb-major (bid=kb*N+n) — los kernels fused usan kb-major (acceso secuencial
    y coalescente: los 8 warps de un bloque leen bloques adyacentes)."""
    out_f, in_f = qt['shape']
    n_cb = qt['n_cb']
    N = out_f

    def r1(a):
        return np.asarray(a).reshape(N, n_cb).T.reshape(-1)

    def r2(a, per):
        return np.asarray(a).reshape(N, n_cb, per).transpose(1, 0, 2).reshape(-1)

    qt2 = dict(qt)
    qt2['dmin'] = r1(qt['dmin']).astype(np.float16).copy()
    qt2['dmax'] = r1(qt['dmax']).astype(np.float16).copy()
    qt2['config'] = r1(qt['config']).astype(np.uint8).copy()
    qt2['coeff'] = r1(qt['coeff'].reshape(-1)).astype(np.uint8).copy()
    qt2['levels_hi'] = r2(qt['levels_hi'], 16).astype(np.uint8).copy()
    qt2['levels_lo'] = r2(qt['levels_lo'], 8).astype(np.uint8).copy()
    return qt2, bases_info


@cuda.jit
def sx8_fused_v1_kernel(X, dmin, dmax, config, coeff, hi_arr, lo_arr, basis, scales,
                        Y, M, K, N, n_cb, k_guard):
    """Y[M,N] = X[M,K] @ W[K,N], W en SX8 v4.3 (bloques 32, V6 + PCA 2 bases).
    1 thread por elemento de salida. X es fp16, acumulacion fp32.
    k_guard = K % 32 (bloque final parcial: no leer X mas alla de K)."""
    idx = cuda.grid(1)
    if idx >= M * N:
        return
    m = idx // N
    n = idx - m * N

    acc = 0.0
    x_row = X[m]
    for kb in range(n_cb):
        bid = kb * N + n
        lo_f = float(dmin[bid])
        hi_f = float(dmax[bid])
        q = (hi_f - lo_f) * 0.25
        cfg = config[bid]
        cb = kb  # bid % n_cb
        b_off = cb * 64
        s_off = cb * 2
        c0_raw = coeff[bid] & 0xF
        c0 = c0_raw if c0_raw < 8 else c0_raw - 16
        c1_raw = (coeff[bid] >> 4) & 0xF
        c1 = c1_raw if c1_raw < 8 else c1_raw - 16
        s0 = scales[s_off]
        s1 = scales[s_off + 1]
        k0 = kb * 32

        # bucle de 32 pesos del bloque (guard solo en el ultimo bloque parcial)
        t_max = 32
        if k_guard > 0 and kb == n_cb - 1:
            t_max = k_guard
        for t in range(t_max):
            hi = (hi_arr[bid * 16 + (t >> 1)] >> ((t & 1) * 4)) & 0xF
            lo = (lo_arr[bid * 8 + (t >> 2)] >> ((3 - (t & 3)) * 2)) & 0x3
            lv = (hi << 2) | lo
            strat = (cfg >> ((t >> 3) * 2)) & 3
            if strat == 0:
                rlo, rhi = lo_f, hi_f
            elif strat == 1:
                rlo, rhi = lo_f, lo_f + q
            elif strat == 2:
                rlo, rhi = hi_f - q, hi_f
            else:
                rlo, rhi = lo_f + q, hi_f - q
            step = (rhi - rlo) * 0.015873
            if step < 1e-10:
                step = 1e-10
            w = rlo + step * lv
            w = w + c0 * s0 * basis[b_off + t]
            w = w + c1 * s1 * basis[b_off + 32 + t]
            acc = acc + float(x_row[k0 + t]) * w

    Y[idx] = acc


def gemm_sx8_v1(X, qt, bases_info, device=DEV):
    """Y = X @ W (SX8 v4.3). X: (M,K) fp16 CUDA. Devuelve Y (M,N) fp32 CUDA."""
    out_f, in_f = qt['shape']
    nb = qt['n_blocks']
    n_cb = qt['n_cb']
    if in_f != X.shape[1]:
        raise ValueError(f"X K={X.shape[1]} != in_f={in_f}")
    M = X.shape[0]
    N = out_f
    K = in_f

    Xc = X.contiguous()
    qk, _ = to_kbmajor(qt, bases_info)
    dm = cuda.to_device(qk['dmin'])
    dx = cuda.to_device(qk['dmax'])
    cfg = cuda.to_device(qk['config'])
    co = cuda.to_device(qk['coeff'])
    hi = cuda.to_device(qk['levels_hi'])
    lo = cuda.to_device(qk['levels_lo'])
    bs = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
    sc = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))

    Y = cuda.device_array(M * N, dtype=np.float32)

    threads = 256
    grid = (M * N + threads - 1) // threads
    k_guard = K % 32
    sx8_fused_v1_kernel[grid, threads](cuda.as_cuda_array(Xc), dm, dx, cfg, co, hi, lo,
                                       bs, sc, Y, M, K, N, n_cb, k_guard)
    cuda.synchronize()
    return torch.as_tensor(Y, device=device).reshape(M, N)


@cuda.jit
def sx8_fused_v2_kernel(X, dmin, dmax, config, coeff, hi_arr, lo_arr, basis, scales,
                        Y, M, K, N, n_cb, k_guard):
    """v2 tiles: 1 warp = 1 columna de salida n; los 32 threads decodifican el
    bloque W[n, kb] cooperativamente en registros (acceso coalescente a los bytes
    del bloque), X en shared memory (tile 8x32 por kb), warp-reduce por fila.
    Block = 256 threads (8 warps) x 8 filas m por bloque."""
    tid = cuda.threadIdx.x & 31
    wid = cuda.threadIdx.x >> 5
    n = cuda.blockIdx.x * 8 + wid
    m0 = cuda.blockIdx.y * 8

    sX = cuda.shared.array(8 * 32, dtype=np.float16)
    sB = cuda.shared.array(64, dtype=np.float32)
    sS = cuda.shared.array(2, dtype=np.float32)

    if n >= N:
        return
    acc = cuda.local.array(8, dtype=np.float32)
    for m in range(8):
        acc[m] = 0.0

    for kb in range(n_cb):
        k0 = kb * 32
        # cargar tile X (8x32) y bases a shared (256 threads -> 8x32=256 valores)
        gidx = cuda.threadIdx.x
        if gidx < 256:
            row = gidx >> 5          # 0..7
            col = gidx & 31          # 0..31
            if m0 + row < M and k0 + col < K:
                sX[row * 32 + col] = X[(m0 + row) * K + k0 + col]
            else:
                sX[row * 32 + col] = 0.0  # padding (ultimo bloque parcial / filas fuera)
        # bases (64 floats + 2 scales): 66 valores por bloque -> threads 0..65
        if gidx < 64:
            sB[gidx] = basis[kb * 64 + gidx]
        if gidx < 2:
            sS[gidx] = scales[kb * 2 + gidx]
        cuda.syncthreads()

        # decodificar el peso (n, kb, tid) en registros
        bid = kb * N + n
        hi = (hi_arr[bid * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
        lo = (lo_arr[bid * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
        lv = (hi << 2) | lo
        lo_f = float(dmin[bid]); hi_f = float(dmax[bid])
        q = (hi_f - lo_f) * 0.25
        strat = (config[bid] >> ((tid >> 3) * 2)) & 3
        if strat == 0:
            rlo, rhi = lo_f, hi_f
        elif strat == 1:
            rlo, rhi = lo_f, lo_f + q
        elif strat == 2:
            rlo, rhi = hi_f - q, hi_f
        else:
            rlo, rhi = lo_f + q, hi_f - q
        step = (rhi - rlo) * 0.015873
        if step < 1e-10:
            step = 1e-10
        c0_raw = coeff[bid] & 0xF
        c0 = c0_raw if c0_raw < 8 else c0_raw - 16
        c1_raw = (coeff[bid] >> 4) & 0xF
        c1 = c1_raw if c1_raw < 8 else c1_raw - 16
        w = rlo + step * lv + c0 * sS[0] * sB[tid] + c1 * sS[1] * sB[32 + tid]

        # acumular: thread tid contribuye con el peso en la posicion k de cada fila m
        for m in range(8):
            if m0 + m < M:
                acc[m] += float(sX[m * 32 + tid]) * w

        # BARRERA CRITICA: evitar que un warp veloz sobreescriba sX/sB/sS
        # del siguiente kb mientras otro warp aun los esta usando (race)
        cuda.syncthreads()

    # warp-reduce UNA vez al final (5 shuffles por fila)
    for m in range(8):
        v = acc[m]
        v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 16)
        v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 8)
        v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 4)
        v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 2)
        v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 1)
        acc[m] = v

    # escribir: solo el thread 0 de cada warp (tiene la suma completa)
    if tid == 0 and n < N:
        for m in range(8):
            if m0 + m < M:
                Y[(m0 + m) * N + n] = acc[m]


def gemm_sx8_v2(X, qt, bases_info, device=DEV):
    """Y = X @ W (SX8 v4.3) con kernel por tiles (v2)."""
    out_f, in_f = qt['shape']
    nb = qt['n_blocks']
    n_cb = qt['n_cb']
    if in_f != X.shape[1]:
        raise ValueError(f"X K={X.shape[1]} != in_f={in_f}")
    M = X.shape[0]
    N = out_f
    K = in_f

    Xc = X.contiguous()
    qk, _ = to_kbmajor(qt, bases_info)
    dm = cuda.to_device(qk['dmin'])
    dx = cuda.to_device(qk['dmax'])
    cfg = cuda.to_device(qk['config'])
    co = cuda.to_device(qk['coeff'])
    hi = cuda.to_device(qk['levels_hi'])
    lo = cuda.to_device(qk['levels_lo'])
    bs = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
    sc = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))

    Y = cuda.device_array(M * N, dtype=np.float32)
    k_guard = K % 32

    threads = 256
    grid = ((N + 7) // 8, (M + 7) // 8)
    sx8_fused_v2_kernel[grid, threads](cuda.as_cuda_array(Xc.reshape(-1)), dm, dx, cfg, co, hi, lo,
                                       bs, sc, Y, M, K, N, n_cb, k_guard)
    cuda.synchronize()
    return torch.as_tensor(Y, device=device).reshape(M, N)


def test_v2_vs_cublas(qt, bases_info, M=8, seed=0):
    """Test de igualdad v2 vs cuBLAS."""
    from kernel_sx8_v43 import decode_tensor_gpu_fast
    rng = np.random.default_rng(seed)
    out_f, in_f = qt['shape']
    X_np = rng.standard_normal((M, in_f)).astype(np.float32) * 0.02
    X = torch.tensor(X_np, device=DEV, dtype=torch.float16)

    W = decode_tensor_gpu_fast(qt, bases_info)
    W2 = W.reshape(out_f, in_f)
    Y_ref = X.float() @ W2.T.float()
    Y_fus = gemm_sx8_v2(X, qt, bases_info)

    diff = (Y_ref - Y_fus).abs()
    denom = Y_ref.abs().max().item() + 1e-9
    maxd = diff.max().item()
    mae = diff.mean().item()
    ok = maxd / denom < 5e-3
    print(f"  [v2] shape W=({out_f},{in_f})  M={M}  maxdiff={maxd:.3e}  rel={maxd/denom:.3e}  mae={mae:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


@cuda.jit
def sx8_fused_v3_kernel(X, dmin, dmax, config, coeff, hi_arr, lo_arr, basis, scales,
                        Y, M, K, N, n_cb, k_guard):
    """v3 FINAL: decode cooperativo -> shared -> GEMM por thread.
    - 256 threads (8 warps); warp w decodifica la columna n = bx*8+w (32 pesos
      del bloque en registros) y los escribe en sW[w*32+tid] (shared).
    - sX en shared TRANSPUESTA (sX[t*256 + m]): threads leen sin conflictos
      de banco (cada thread su fila m, stride 1 entre threads).
    - Cada thread (fila m = by*256 + tid) hace el dot product contra las 8
      columnas: 32 FMA por columna. Decode amortizado ~4% del compute.
    - Block = 256 threads; grid = (N/8, M/256)."""
    tid = cuda.threadIdx.x & 31
    wid = cuda.threadIdx.x >> 5
    n = cuda.blockIdx.x * 8 + wid
    m0 = cuda.blockIdx.y * 256

    sX = cuda.shared.array(256 * 32, dtype=np.float16)   # transpuesta: sX[t*256 + m]
    sW = cuda.shared.array(8 * 32, dtype=np.float16)     # TRANSPUESTA: sW[t*8 + w]
    sB = cuda.shared.array(64, dtype=np.float32)
    sS = cuda.shared.array(2, dtype=np.float32)

    if n >= N:
        return
    acc = cuda.local.array(8, dtype=np.float32)
    for w in range(8):
        acc[w] = 0.0

    for kb in range(n_cb):
        k0 = kb * 32
        # cargar tile X (256x32) transpuesto a shared: 256 threads x 32 iteraciones
        gidx = cuda.threadIdx.x
        for it in range(32):
            row = it * 8 + (gidx >> 5)          # 0..255
            col = gidx & 31                     # 0..31
            if m0 + row < M and k0 + col < K:
                sX[col * 256 + row] = X[(m0 + row) * K + k0 + col]
            else:
                sX[col * 256 + row] = 0.0
        if gidx < 64:
            sB[gidx] = basis[kb * 64 + gidx]
        if gidx < 2:
            sS[gidx] = scales[kb * 2 + gidx]
        cuda.syncthreads()

        # fase 1: cada warp decodifica su columna n (cooperativo, coalescente)
        bid = kb * N + n
        hi = (hi_arr[bid * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
        lo = (lo_arr[bid * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
        lv = (hi << 2) | lo
        lo_f = float(dmin[bid]); hi_f = float(dmax[bid])
        q = (hi_f - lo_f) * 0.25
        strat = (config[bid] >> ((tid >> 3) * 2)) & 3
        if strat == 0:
            rlo, rhi = lo_f, hi_f
        elif strat == 1:
            rlo, rhi = lo_f, lo_f + q
        elif strat == 2:
            rlo, rhi = hi_f - q, hi_f
        else:
            rlo, rhi = lo_f + q, hi_f - q
        step = (rhi - rlo) * 0.015873
        if step < 1e-10:
            step = 1e-10
        c0_raw = coeff[bid] & 0xF
        c0 = c0_raw if c0_raw < 8 else c0_raw - 16
        c1_raw = (coeff[bid] >> 4) & 0xF
        c1 = c1_raw if c1_raw < 8 else c1_raw - 16
        w = rlo + step * lv + c0 * sS[0] * sB[tid] + c1 * sS[1] * sB[32 + tid]
        sW[tid * 8 + wid] = np.float16(w)
        cuda.syncthreads()

        # fase 2: GEMM por thread — fila m = m0 + tid, contra las 8 columnas
        # x en registro (reuso 8x), sW TRANSPUESTA (sin bank conflicts)
        m = m0 + tid
        if m < M:
            for t in range(32):
                x = float(sX[t * 256 + tid])
                for w in range(8):
                    acc[w] += x * float(sW[t * 8 + w])

        cuda.syncthreads()  # barrera: no pisar sX/sW mientras otros warps la usan

    # escribir Y: thread tid escribe su fila m para las 8 columnas n del bloque
    m = m0 + tid
    if m < M:
        for w in range(8):
            nn = cuda.blockIdx.x * 8 + w
            if nn < N:
                Y[m * N + nn] = acc[w]


def gemm_sx8_v3(X, qt, bases_info, device=DEV):
    """Y = X @ W (SX8 v4.3) con kernel v3 (decode cooperativo, BM=256)."""
    out_f, in_f = qt['shape']
    n_cb = qt['n_cb']
    if in_f != X.shape[1]:
        raise ValueError(f"X K={X.shape[1]} != in_f={in_f}")
    M = X.shape[0]
    N = out_f
    K = in_f

    Xc = X.contiguous()
    qk, _ = to_kbmajor(qt, bases_info)
    dm = cuda.to_device(qk['dmin'])
    dx = cuda.to_device(qk['dmax'])
    cfg = cuda.to_device(qk['config'])
    co = cuda.to_device(qk['coeff'])
    hi = cuda.to_device(qk['levels_hi'])
    lo = cuda.to_device(qk['levels_lo'])
    bs = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
    sc = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))

    Y = cuda.device_array(M * N, dtype=np.float32)
    k_guard = K % 32

    threads = 256
    grid = ((N + 7) // 8, (M + 255) // 256)
    sx8_fused_v3_kernel[grid, threads](cuda.as_cuda_array(Xc.reshape(-1)), dm, dx, cfg, co,
                                       hi, lo, bs, sc, Y, M, K, N, n_cb, k_guard)
    cuda.synchronize()
    return torch.as_tensor(Y, device=device).reshape(M, N)


def test_v3_vs_cublas(qt, bases_info, M=8, seed=0):
    """Test de igualdad v3 vs cuBLAS."""
    from kernel_sx8_v43 import decode_tensor_gpu_fast
    rng = np.random.default_rng(seed)
    out_f, in_f = qt['shape']
    X_np = rng.standard_normal((M, in_f)).astype(np.float32) * 0.02
    X = torch.tensor(X_np, device=DEV, dtype=torch.float16)

    W = decode_tensor_gpu_fast(qt, bases_info)
    W2 = W.reshape(out_f, in_f)
    Y_ref = X.float() @ W2.T.float()
    Y_fus = gemm_sx8_v3(X, qt, bases_info)

    diff = (Y_ref - Y_fus).abs()
    denom = Y_ref.abs().max().item() + 1e-9
    maxd = diff.max().item()
    mae = diff.mean().item()
    ok = maxd / denom < 5e-3
    print(f"  [v3] shape W=({out_f},{in_f})  M={M}  maxdiff={maxd:.3e}  rel={maxd/denom:.3e}  mae={mae:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def test_v1_vs_cublas(qt, bases_info, M=8, seed=0):
    """Test de igualdad: Y fused vs Y = X @ W_fp16 (cuBLAS)."""
    from kernel_sx8_v43 import decode_tensor_gpu_fast
    rng = np.random.default_rng(seed)
    out_f, in_f = qt['shape']
    X_np = rng.standard_normal((M, in_f)).astype(np.float32) * 0.02
    X = torch.tensor(X_np, device=DEV, dtype=torch.float16)

    W = decode_tensor_gpu_fast(qt, bases_info)          # W fp16 CUDA (vía decode)
    W2 = W.reshape(out_f, in_f)                          # aplanar (conv1d tiene orig 3D)
    Y_ref = X.float() @ W2.T.float()                     # cuBLAS fp32 acumulación
    Y_fus = gemm_sx8_v1(X, qt, bases_info)              # fused

    diff = (Y_ref - Y_fus).abs()
    denom = Y_ref.abs().max().item() + 1e-9
    maxd = diff.max().item()
    mae = diff.mean().item()
    # tolerancia: fp16 -> 5e-3 relativo es generoso; el decode fp16 ya introduce ULP
    ok = maxd / denom < 5e-3
    print(f"  [v1] shape W=({out_f},{in_f})  M={M}  maxdiff={maxd:.3e}  rel={maxd/denom:.3e}  mae={mae:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


@cuda.jit
def sx8_decode_dot_kernel(X, dmin, dmax, config, coeff, hi_arr, lo_arr, basis, scales,
                          Y, K, N, n_cb):
    """DECODE (M=1): Y[n] = sum_k X[k] * W[n,k] con W SX8 v4.3 en VRAM.
    1 warp = 1 columna n (8 por bloque); thread tid decodifica el peso (n, kb, tid)
    coalescentemente, FMA con X[k0+tid] (broadcast L2), warp-reduce final.
    Unroll 2 explicito para romper la latencia de memoria del loop kb."""
    tid = cuda.threadIdx.x & 31
    wid = cuda.threadIdx.x >> 5
    n = cuda.blockIdx.x * 8 + wid
    if n >= N:
        return
    acc = 0.0
    for kb0 in range(0, n_cb - 1, 2):
        # ---- bloque par
        b0 = kb0 * N + n
        x = float(X[kb0 * 32 + tid])
        hi = (hi_arr[b0 * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
        lo = (lo_arr[b0 * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
        lv = (hi << 2) | lo
        lo_f = float(dmin[b0]); hi_f = float(dmax[b0])
        q = (hi_f - lo_f) * 0.25
        strat = (config[b0] >> ((tid >> 3) * 2)) & 3
        if strat == 0:
            rlo, rhi = lo_f, hi_f
        elif strat == 1:
            rlo, rhi = lo_f, lo_f + q
        elif strat == 2:
            rlo, rhi = hi_f - q, hi_f
        else:
            rlo, rhi = lo_f + q, hi_f - q
        step = (rhi - rlo) * 0.015873
        if step < 1e-10:
            step = 1e-10
        c0_raw = coeff[b0] & 0xF
        c0 = c0_raw if c0_raw < 8 else c0_raw - 16
        c1_raw = (coeff[b0] >> 4) & 0xF
        c1 = c1_raw if c1_raw < 8 else c1_raw - 16
        w = rlo + step * lv + c0 * scales[kb0 * 2] * basis[kb0 * 64 + tid] \
            + c1 * scales[kb0 * 2 + 1] * basis[kb0 * 64 + 32 + tid]
        acc += x * w
        # ---- bloque impar
        b1 = (kb0 + 1) * N + n
        x = float(X[(kb0 + 1) * 32 + tid])
        hi = (hi_arr[b1 * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
        lo = (lo_arr[b1 * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
        lv = (hi << 2) | lo
        lo_f = dmin[b1]; hi_f = dmax[b1]
        q = (hi_f - lo_f) * 0.25
        strat = (config[b1] >> ((tid >> 3) * 2)) & 3
        if strat == 0:
            rlo, rhi = lo_f, hi_f
        elif strat == 1:
            rlo, rhi = lo_f, lo_f + q
        elif strat == 2:
            rlo, rhi = hi_f - q, hi_f
        else:
            rlo, rhi = lo_f + q, hi_f - q
        step = (rhi - rlo) * 0.015873
        if step < 1e-10:
            step = 1e-10
        c0_raw = coeff[b1] & 0xF
        c0 = c0_raw if c0_raw < 8 else c0_raw - 16
        c1_raw = (coeff[b1] >> 4) & 0xF
        c1 = c1_raw if c1_raw < 8 else c1_raw - 16
        w = rlo + step * lv + c0 * scales[(kb0 + 1) * 2] * basis[(kb0 + 1) * 64 + tid] \
            + c1 * scales[(kb0 + 1) * 2 + 1] * basis[(kb0 + 1) * 64 + 32 + tid]
        acc += x * w
    # resto (n_cb impar)
    for kb in range(n_cb - (n_cb % 2), n_cb):
        b = kb * N + n
        x = float(X[kb * 32 + tid])
        hi = (hi_arr[b * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
        lo = (lo_arr[b * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
        lv = (hi << 2) | lo
        lo_f = float(dmin[b]); hi_f = float(dmax[b])
        q = (hi_f - lo_f) * 0.25
        strat = (config[b] >> ((tid >> 3) * 2)) & 3
        if strat == 0:
            rlo, rhi = lo_f, hi_f
        elif strat == 1:
            rlo, rhi = lo_f, lo_f + q
        elif strat == 2:
            rlo, rhi = hi_f - q, hi_f
        else:
            rlo, rhi = lo_f + q, hi_f - q
        step = (rhi - rlo) * 0.015873
        if step < 1e-10:
            step = 1e-10
        c0_raw = coeff[b] & 0xF
        c0 = c0_raw if c0_raw < 8 else c0_raw - 16
        c1_raw = (coeff[b] >> 4) & 0xF
        c1 = c1_raw if c1_raw < 8 else c1_raw - 16
        w = rlo + step * lv + c0 * scales[kb * 2] * basis[kb * 64 + tid] \
            + c1 * scales[kb * 2 + 1] * basis[kb * 64 + 32 + tid]
        acc += x * w
    v = acc
    v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 16)
    v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 8)
    v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 4)
    v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 2)
    v += cuda.shfl_xor_sync(0xFFFFFFFF, v, 1)
    if tid == 0:
        Y[n] = v


def decode_dot_sx8(X, qt, bases_info, device=DEV):
    """Y[n] = X @ W[:,n] con W SX8 v4.3 (decode, M=1). X: (K,) fp16 CUDA."""
    out_f, in_f = qt['shape']
    n_cb = qt['n_cb']
    K = in_f
    N = out_f
    Xc = X.reshape(-1).contiguous()
    qk, _ = to_kbmajor(qt, bases_info)
    dm = cuda.to_device(qk['dmin'])
    dx = cuda.to_device(qk['dmax'])
    cfg = cuda.to_device(qk['config'])
    co = cuda.to_device(qk['coeff'])
    hi = cuda.to_device(qk['levels_hi'])
    lo = cuda.to_device(qk['levels_lo'])
    bs = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
    sc = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))
    Y = torch.empty(N, dtype=torch.float32, device=device)
    grid = (N + 7) // 8
    sx8_decode_dot_kernel[grid, 256](cuda.as_cuda_array(Xc), dm, dx, cfg, co, hi, lo,
                                     bs, sc, cuda.as_cuda_array(Y), K, N, n_cb)
    cuda.synchronize()
    return Y


def _launch_v3(xflat, n_cb, K, M, N, d, out_f, in_f):
    """Lanzamiento v3 con device arrays cacheados (d = objeto con _dm,_dx,...).
    Y = X@W es M x out_f (el padding de K lo gestiona k_guard en el kernel)."""
    Y = torch.empty(M * out_f, dtype=torch.float32, device=DEV)
    k_guard = K % 32
    grid = ((out_f + 7) // 8, (M + 255) // 256)
    sx8_fused_v3_kernel[grid, 256](xflat, d._dm, d._dx, d._cfg, d._co,
                                   d._hi, d._lo, d._bs, d._sc,
                                   cuda.as_cuda_array(Y), M, K, out_f, n_cb, k_guard)
    cuda.synchronize()
    return Y.reshape(M, out_f)


def _launch_decode(xflat, n_cb, K, N, d, out_f, in_f):
    """Lanzamiento decode M=1 con device arrays cacheados."""
    Y = torch.empty(out_f, dtype=torch.float32, device=DEV)
    grid = (out_f + 7) // 8
    sx8_decode_dot_kernel[grid, 256](xflat, d._dm, d._dx, d._cfg, d._co,
                                     d._hi, d._lo, d._bs, d._sc,
                                     cuda.as_cuda_array(Y), K, out_f, n_cb)
    cuda.synchronize()
    return Y


def gemm_sx8_cached(X, d, out_f, in_f):
    """Y = X @ W con device arrays SX8 YA en VRAM (d = SX8Linear).
    Dispatcher: M==1 decode dedicado; resto v3 tiles."""
    M = X.shape[0]
    K = X.shape[1]
    n_cb = d.qt['n_cb']
    xflat = cuda.as_cuda_array(X.reshape(-1))
    if M == 1:
        return _launch_decode(xflat, n_cb, K, out_f, d, out_f, in_f).unsqueeze(0)
    return _launch_v3(xflat, n_cb, K, M, out_f, d, out_f, in_f)


def decode_dot_sx8_cached(X, qt, tensors, device=DEV):
    """Y[n] = X @ W[:,n] con W SX8 v4.3 (decode M=1), tensores GPU cacheados
    (dict con dmin/dmax/config/coeff/hi/lo/basis/scales en kb-major)."""
    out_f, in_f = qt['shape']
    n_cb = qt['n_cb']
    K = in_f
    N = out_f
    Xc = X.reshape(-1).contiguous()
    Y = torch.empty(N, dtype=torch.float32, device=device)
    grid = (N + 7) // 8
    sx8_decode_dot_kernel[grid, 256](cuda.as_cuda_array(Xc),
                                     cuda.as_cuda_array(tensors['dmin']),
                                     cuda.as_cuda_array(tensors['dmax']),
                                     cuda.as_cuda_array(tensors['config']),
                                     cuda.as_cuda_array(tensors['coeff']),
                                     cuda.as_cuda_array(tensors['hi']),
                                     cuda.as_cuda_array(tensors['lo']),
                                     cuda.as_cuda_array(tensors['basis']),
                                     cuda.as_cuda_array(tensors['scales']),
                                     cuda.as_cuda_array(Y), K, N, n_cb)
    cuda.synchronize()
    return Y


def gemm_sx8_auto(X, qt, bases_info, device=DEV):
    """Dispatcher: M==1 -> decode dot dedicado; M>=32 -> v3 (tiles, 900 GF/s); else v2."""
    M = X.shape[0]
    if M == 1:
        return decode_dot_sx8(X[0], qt, bases_info, device=device).unsqueeze(0)
    if M >= 32:
        return gemm_sx8_v3(X, qt, bases_info, device=device)
    return gemm_sx8_v2(X, qt, bases_info, device=device)


if __name__ == "__main__":
    import sys, pickle
    sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")
    pkl = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
    d = pickle.load(open(pkl, 'rb'))
    wd, bd = d['weights'], d['bases']
    names = sorted(wd, key=lambda n: -wd[n]['n_blocks'])[:5]
    names += [n for n in wd if wd[n]['shape'][1] % 32 != 0][:3]  # convs con K%32!=0
    all_ok = True
    for name in names:
        all_ok &= test_v1_vs_cublas(wd[name], bd[name])
    print("V1 global:", "PASS" if all_ok else "FAIL")
    all_ok = True
    for name in names[:4]:
        all_ok &= test_v2_vs_cublas(wd[name], bd[name])
    print("V2 global:", "PASS" if all_ok else "FAIL")
    all_ok = True
    for name in names[:4]:
        all_ok &= test_v3_vs_cublas(wd[name], bd[name])
    print("V3 global:", "PASS" if all_ok else "FAIL")
    # test decode (M=1) y dispatcher
    from kernel_sx8_v43 import decode_tensor_gpu_fast
    rng = np.random.default_rng(1)
    all_ok = True
    for name in names[:4]:
        qt2, bi2 = wd[name], bd[name]
        out_f, in_f = qt2['shape']
        X1 = torch.tensor(rng.standard_normal((1, in_f)).astype(np.float32) * 0.02,
                          device=DEV, dtype=torch.float16)
        W2 = decode_tensor_gpu_fast(qt2, bi2).reshape(out_f, in_f)
        Y_ref = X1.float() @ W2.T.float()
        Y_fus = gemm_sx8_auto(X1, qt2, bi2)
        d = (Y_ref - Y_fus).abs().max().item() / (Y_ref.abs().max().item() + 1e-9)
        ok = d < 5e-3
        all_ok &= ok
        print(f"  [auto M=1] {name[:50]:<52} rel={d:.2e}  {'PASS' if ok else 'FAIL'}")
    print("AUTO(M=1) global:", "PASS" if all_ok else "FAIL")
