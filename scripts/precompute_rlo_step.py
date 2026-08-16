"""Precomputación rlo[4]/step[4] FP16 por bloque (lección v8 aplicada a S-X8 v4.3).

El kernel anterior (v8, S-X6) demostró: precomputar offset/scale FP16 en host
elimina las operaciones FP32 por peso (decode de 10 -> 2 ops). Para S-X8 v4.3
NO podemos forzar estrategia 0 (perderíamos el claim de calidad), pero SÍ podemos
precomputar los 4 pares (rlo, step) por bloque (uno por sub-bloque) con la
fórmula sin branches ya validada:
    rlo = lo + q*(3*(s==2) + (s==3))
    rhi = hi - q*(3*(s==1) + (s==3))
    step = (rhi - rlo) * 0.015873  (clamp < 1e-10 -> 1e-10, bloques degenerados)

Layout de inferencia (en VRAM): rlo (n_blk,4) fp16 + step (n_blk,4) fp16
= 16 B/bloque, reemplazando dmin+dmax+config (5 B) -> 41 B/bloque en runtime.
El archivo .sx8v43 (30 B/bloque) NO cambia: es el formato del paper; la tabla
es una transformación de inferencia (como hace llama.cpp con sus kernels).

Decode por peso: w = rlo[sb] + step[sb]*lv + PCA (2 FMAs)  -> ~4 ops/peso.
"""
import pickle, sys
import numpy as np

sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")

PKL = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
S = 0.015873  # 1/63


def precompute_rlo_step(qt):
    """Genera (rlo_f16, step_f16) de shape (n_blk, 4) desde un qt v4.3."""
    lo = qt['dmin'].astype(np.float32)
    hi = qt['dmax'].astype(np.float32)
    q = (hi - lo) * 0.25
    cfg = qt['config'].astype(np.int32)          # (n_blk,)
    s = np.stack([(cfg >> (i * 2)) & 3 for i in range(4)], axis=1)  # (n_blk, 4)
    a = (3 * (s == 2) + (s == 3)).astype(np.float32)
    b = (3 * (s == 1) + (s == 3)).astype(np.float32)
    rlo = lo[:, None] + q[:, None] * a
    rhi = hi[:, None] - q[:, None] * b
    step = (rhi - rlo) * S
    step = np.where(step < 1e-10, S, step)
    return rlo.astype(np.float16), step.astype(np.float16)


def decode_reference(qt, bases_info, seed=0, n=2000):
    """Decode de referencia en numpy (exacto, float32) para verificar."""
    from kernel_sx8_fused import to_kbmajor
    qk, _ = to_kbmajor(qt, bases_info)
    out_f, in_f = qt['shape']
    n_cb = qt['n_cb']
    nb = qt['n_blocks']
    rng = np.random.default_rng(seed)
    # decodificar n bloques aleatorios
    bids = rng.integers(0, nb, size=n)
    W = np.zeros(n * 32, dtype=np.float32)
    for i, bid in enumerate(bids):
        kb = bid // out_f          # kb-major: bid = kb*N + n
        n_col = bid % out_f
        lo_f = float(qk['dmin'][bid]); hi_f = float(qk['dmax'][bid])
        q = (hi_f - lo_f) * 0.25
        cfg = int(qk['config'][bid])
        co = int(qk['coeff'][bid])
        c0 = co & 0xF; c0 = c0 if c0 < 8 else c0 - 16
        c1 = (co >> 4) & 0xF; c1 = c1 if c1 < 8 else c1 - 16
        s0 = float(bases_info['scales'][kb][0]); s1 = float(bases_info['scales'][kb][1])
        for t in range(32):
            hi = (qk['levels_hi'][bid * 16 + (t >> 1)] >> ((t & 1) * 4)) & 0xF
            lo = (qk['levels_lo'][bid * 8 + (t >> 2)] >> ((3 - (t & 3)) * 2)) & 0x3
            lv = (hi << 2) | lo
            sb = t >> 3
            s_ = (cfg >> (sb * 2)) & 3
            a = 3 * (s_ == 2) + (s_ == 3)
            b = 3 * (s_ == 1) + (s_ == 3)
            rlo = lo_f + q * a
            rhi = hi_f - q * b
            stp = (rhi - rlo) * S
            if stp < 1e-10:
                stp = S
            w = rlo + stp * lv + c0 * s0 * float(bases_info['data'][kb][t]) \
                + c1 * s1 * float(bases_info['data'][kb][32 + t])
            W[i * 32 + t] = w
    return bids, W


def decode_fast(qt, bases_info, bids, seed=0):
    """Decode con la tabla precomputada (rlo/step fp16) — lo que hará el kernel."""
    from kernel_sx8_fused import to_kbmajor
    qk, _ = to_kbmajor(qt, bases_info)
    out_f, in_f = qt['shape']
    rlo_f, step_f = precompute_rlo_step(qk)
    nb = qt['n_blocks']
    W = np.zeros(bids.shape[0] * 32, dtype=np.float32)
    for i, bid in enumerate(bids):
        kb = bid // out_f
        co = int(qk['coeff'][bid])
        c0 = co & 0xF; c0 = c0 if c0 < 8 else c0 - 16
        c1 = (co >> 4) & 0xF; c1 = c1 if c1 < 8 else c1 - 16
        s0 = float(bases_info['scales'][kb][0]); s1 = float(bases_info['scales'][kb][1])
        for t in range(32):
            hi = (qk['levels_hi'][bid * 16 + (t >> 1)] >> ((t & 1) * 4)) & 0xF
            lo = (qk['levels_lo'][bid * 8 + (t >> 2)] >> ((3 - (t & 3)) * 2)) & 0x3
            lv = (hi << 2) | lo
            sb = t >> 3
            w = float(rlo_f[bid, sb]) + float(step_f[bid, sb]) * lv \
                + c0 * s0 * float(bases_info['data'][kb][t]) \
                + c1 * s1 * float(bases_info['data'][kb][32 + t])
            W[i * 32 + t] = w
    return W


if __name__ == "__main__":
    d = pickle.load(open(PKL, 'rb'))
    wd, bd = d['weights'], d['bases']
    name = "model.language_model.layers.3.self_attn.q_proj.weight"
    qt, bi = wd[name], bd[name]
    bids, Wref = decode_reference(qt, bi)
    Wfast = decode_fast(qt, bi, bids)
    diff = np.abs(Wref - Wfast)
    print(f"precompute rlo/step fp16 vs decode exacto:")
    print(f"  maxdiff = {diff.max():.3e}  mean = {diff.mean():.3e}")
    print(f"  rel max = {diff.max() / (np.abs(Wref).max() + 1e-9):.3e}")
    ok = diff.max() < 1e-3
    print(f"  {'PASS (error < 1e-3, dentro de ruido fp16)' if ok else 'FAIL'}")

    # tamaño del layout de inferencia
    from kernel_sx8_fused import to_kbmajor
    qk, _ = to_kbmajor(qt, bi)
    rlo_f, step_f = precompute_rlo_step(qk)
    bytes_bloque = 2 + 2 + 1 + 1 + 16 + 8 + 2 + 2  # dmin,dmax,cfg,coeff,hi,lo,rlo,step
    print(f"\n  layout inferencia: {bytes_bloque} B/bloque (vs 30 del archivo)")
    print(f"  rlo_f16[0]: {rlo_f[0]}, step_f16[0]: {step_f[0]}")
