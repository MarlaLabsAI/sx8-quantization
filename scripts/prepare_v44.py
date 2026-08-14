"""prepare_v44.py — Transformación de carga al LAYOUT DE INFERENCIA v4.4 (FASE A2).

Principio ("el orden de los factores no altera el producto"):
  El archivo .sx8v43 (30 B/bloque, 7.50 bpp) NO cambia. Al cargar se transforma
  a un layout en VRAM PENSADO PARA EL KERNEL con el MISMO tamaño (30 B/bloque):

  v4.3 (archivo, n-major):  dmin2 + dmax2 + config1 + hi16 + lo8 + coeff1
  v4.4 (VRAM, kb-major):    hdr6 + [hi4+lo2 por sub-bloque]x4  = 30 B EXACTOS

  hdr (6 B): dmin(fp16) + dmax(fp16) + config(u8) + coeff(u8)   [por bloque]
  levels (24 B): por sub-bloque sb=0..3: hi[4] + lo[2]           [8 pesos c/u]

  Ventajas para el kernel decode M=1:
  - 1 thread = 1 sub-bloque (8 pesos): 1 load 32-bit (hi) + 1 load 16-bit (lo),
    contiguos y alineados (vs bytes escalados leídos por 2-4 threads)
  - hdr contiguo de 6 B por bloque (1 load 32-bit dmin/dmax + 16-bit config/coeff)
  - kb-major (bid = kb*N + n): el warp lee bloques adyacentes → coalescente

  Además: las bases PCA SON POR BLOQUE DE K (n_cb, 64) y compartidas entre
  columnas → la corrección PCA se reformula como Z0[kb], Z1[kb] por capa
  (2 FMA por bloque en vez de 2 FMA + 2 loads fp32 por peso). Ver
  validate_pca_reform.py.

  Este script SOLO genera el layout y lo verifica (byte-exacto en levels,
  numéricamente idéntico en el decode). Los kernels v3 (decode1_v3.cu) lo consumen.
"""
import os
import sys
import numpy as np
import pickle

PKL = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"


def to_v44(qt, bases_info, hdr_aligned=False):
    """Transforma un tensor S-X8 v4.3 (pkl) al layout de inferencia v4.4.
    Devuelve dict con: levels (n_blk,24) u8, hdr (n_blk,6 u 8) u8, n_cb, shape,
    bases (n_cb,64) f32, scales (n_cb,2) f32.
    hdr_aligned=True -> hdr de 8 B (pad 2): load 64-bit alineado en el kernel,
    VRAM +2 B/bloque (+0.26 GB total). hdr_aligned=False -> 6 B exactos (4.373 GB)."""
    out_f, in_f = qt['shape']
    n_cb = qt['n_cb']
    n_blk = qt['n_blocks']

    hi = np.asarray(qt['levels_hi']).reshape(out_f, n_cb, 16)  # n-major (pkl)
    lo = np.asarray(qt['levels_lo']).reshape(out_f, n_cb, 8)
    dmin = np.asarray(qt['dmin']).reshape(out_f, n_cb).astype(np.float16)
    dmax = np.asarray(qt['dmax']).reshape(out_f, n_cb).astype(np.float16)
    config = np.asarray(qt['config']).reshape(out_f, n_cb).astype(np.uint8)
    coeff = np.asarray(qt['coeff']).reshape(out_f, n_cb).astype(np.uint8)

    # levels v4.4: [hi_sb0(4), hi_sb1(4), hi_sb2(4), hi_sb3(4), lo_sb0(2), lo_sb1(2), lo_sb2(2), lo_sb3(2)]
    # hi del sub-bloque sb: bytes 4*sb..4*sb+3 (nibbles de los pesos 8*sb..8*sb+7)
    # lo del sub-bloque sb: bytes 2*sb..2*sb+1 (quads de los pesos 8*sb..8*sb+7)
    levels44 = np.zeros((out_f, n_cb, 24), dtype=np.uint8)
    for sb in range(4):
        levels44[:, :, sb * 4:sb * 4 + 4] = hi[:, :, 4 * sb:4 * sb + 4]
        levels44[:, :, 16 + sb * 2:16 + sb * 2 + 2] = lo[:, :, 2 * sb:2 * sb + 2]

    # hdr v4.4: dmin(2) + dmax(2) + config(1) + coeff(1) = 6 B
    # hdr_aligned: +2 B de pad -> 8 B (load 64-bit alineado en el kernel)
    hdr = np.zeros((out_f, n_cb, 8 if hdr_aligned else 6), dtype=np.uint8)
    hdr[:, :, 0:2] = dmin.view(np.uint8).reshape(out_f, n_cb, 2)
    hdr[:, :, 2:4] = dmax.view(np.uint8).reshape(out_f, n_cb, 2)
    hdr[:, :, 4] = config
    hdr[:, :, 5] = coeff

    # reordenar n-major -> kb-major (bid = kb*N + n), como integrate_fused.reorder
    levels_kb = levels44.transpose(1, 0, 2).reshape(-1, 24)   # (n_blk, 24)
    hdr_kb = hdr.transpose(1, 0, 2).reshape(-1, hdr.shape[2])  # (n_blk, 6|8)

    return {
        'levels': levels_kb,         # (n_blk, 24) u8, kb-major
        'hdr': hdr_kb,               # (n_blk, 6) u8, kb-major
        'n_cb': n_cb,
        'shape': (out_f, in_f),
        'bases': np.asarray(bases_info['data']).reshape(-1, 64).astype(np.float32),
        'scales': np.asarray(bases_info['scales']).reshape(-1, 2).astype(np.float32),
    }


def extract_levels_v43(qt):
    """Extrae los 32 niveles (6-bit) por bloque del layout v4.3 — referencia."""
    n_blk = qt['n_blocks']
    hi = np.asarray(qt['levels_hi']).reshape(n_blk, 16)
    lo = np.asarray(qt['levels_lo']).reshape(n_blk, 8)
    lv = np.zeros((n_blk, 32), dtype=np.uint8)
    for t in range(32):
        nib = (hi[:, t >> 1] >> ((t & 1) * 4)) & 0xF
        qua = (lo[:, t >> 2] >> ((3 - (t & 3)) * 2)) & 0x3
        lv[:, t] = (nib << 2) | qua
    return lv


def extract_levels_v44(v44):
    """Extrae los 32 niveles del layout v4.4 (kb-major) — referencia para comparar."""
    n_blk = v44['levels'].shape[0]
    lv = np.zeros((n_blk, 32), dtype=np.uint8)
    for sb in range(4):
        hi4 = v44['levels'][:, sb * 4:sb * 4 + 4]
        lo2 = v44['levels'][:, 16 + sb * 2:16 + sb * 2 + 2]
        for i in range(8):  # peso local del sub-bloque
            nib = (hi4[:, i >> 1] >> ((i & 1) * 4)) & 0xF
            qua = (lo2[:, i >> 2] >> ((3 - (i & 3)) * 2)) & 0x3
            lv[:, sb * 8 + i] = (nib << 2) | qua
    return lv


def decode_v44_numpy(v44, X, n):
    """Decode REFORMULADO de UNA columna (Z0/Z1 por bloque de K) sobre el layout v4.4.
    Es el contrato numérico que debe cumplir el kernel v3 (misma fórmula sin branches).
    X: (K,) fp32. Devuelve Y[n]."""
    out_f, in_f = v44['shape']
    n_cb = v44['n_cb']
    levels = v44['levels']
    hdr = v44['hdr']
    basis = v44['bases']
    scales = v44['scales']

    # Z0/Z1 por bloque de K (compartidos entre columnas)
    Z0 = np.zeros(n_cb, dtype=np.float32)
    Z1 = np.zeros(n_cb, dtype=np.float32)
    for kb in range(n_cb):
        z0 = 0.0; z1 = 0.0
        for t in range(32):
            z0 += X[kb * 32 + t] * basis[kb, t]
            z1 += X[kb * 32 + t] * basis[kb, 32 + t]
        Z0[kb] = z0; Z1[kb] = z1

    acc = 0.0
    for kb in range(n_cb):
        bid = kb * out_f + n  # kb-major
        dm = hdr[bid, 0:2].view(np.float16)[0]
        dx = hdr[bid, 2:4].view(np.float16)[0]
        cfg = int(hdr[bid, 4])
        co = int(hdr[bid, 5])
        lo_f = float(dm); hi_f = float(dx)
        q = (hi_f - lo_f) * 0.25
        c0 = co & 0xF; c0 = c0 if c0 < 8 else c0 - 16
        c1 = (co >> 4) & 0xF; c1 = c1 if c1 < 8 else c1 - 16
        for sb in range(4):
            s = (cfg >> (sb * 2)) & 3
            rlo = lo_f + q * (3 * (s == 2) + (s == 3))
            rhi = hi_f - q * (3 * (s == 1) + (s == 3))
            step = (rhi - rlo) * 0.015873
            if step < 1e-10:
                step = 0.015873
            hi4 = v44['levels'][bid, sb * 4:sb * 4 + 4]
            lo2 = v44['levels'][bid, 16 + sb * 2:16 + sb * 2 + 2]
            for i in range(8):  # peso local del sub-bloque
                t = sb * 8 + i
                nb = (int(hi4[i >> 1]) >> ((i & 1) * 4)) & 0xF
                qu = (int(lo2[i >> 2]) >> ((3 - (i & 3)) * 2)) & 0x3
                lv = (nb << 2) | qu
                w = rlo + step * lv
                acc += X[kb * 32 + t] * w
        acc += c0 * scales[kb, 0] * Z0[kb] + c1 * scales[kb, 1] * Z1[kb]
    return acc


def verify(qt, bases_info, name):
    """Verificación completa del layout v4.4 para un tensor."""
    ok = True
    v44 = to_v44(qt, bases_info)
    n_blk = qt['n_blocks']

    # 1) bytes por bloque: 30 exactos
    n_bytes = v44['levels'].nbytes + v44['hdr'].nbytes
    if n_bytes != n_blk * 30:
        print(f"  [FAIL] bytes/bloque = {n_bytes/n_blk:.2f} (esperado 30)"); ok = False
    else:
        print(f"  [OK]   {n_bytes/1e9:.3f} GB para {n_blk:,} bloques = 30 B/bloque exacto")

    # 2) niveles byte-exactos vs v4.3 (reordenado a kb-major)
    lv43 = extract_levels_v43(qt)
    out_f, in_f = qt['shape']
    lv43_kb = lv43.reshape(out_f, n_blk // out_f, 32).transpose(1, 0, 2).reshape(-1, 32)
    lv44 = extract_levels_v44(v44)
    if np.array_equal(lv43_kb, lv44):
        print("  [OK]   niveles byte-exactos vs v4.3 (biyectivo)")
    else:
        nd = int(np.sum(lv43 != lv44))
        print(f"  [FAIL] niveles difieren en {nd} pesos"); ok = False

    # 3) decode reformulado vs decode exacto (muestra de 64 columnas)
    rng = np.random.default_rng(7)
    out_f, in_f = qt['shape']
    X = rng.standard_normal(in_f).astype(np.float32) * 0.5
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from validate_pca_reform import decode_exact_one
    md = 0.0
    for n in range(0, out_f, max(1, out_f // 64)):
        y_exact = decode_exact_one(qt, bases_info, X, n)
        y_new = decode_v44_numpy(v44, X, n)
        md = max(md, abs(float(y_exact - y_new)))
    print(f"  [{'OK' if md < 1e-3 else 'FAIL'}]   decode v4.4 vs exacto: maxdiff={md:.3e}")
    if md >= 1e-3:
        ok = False

    # 4) tamaño del pkl en memoria (VRAM pesos) del layout v4.4
    total = n_bytes / 1e9
    print(f"  pesos layout v4.4: {total:.3f} GB (vs archivo 4.38 GB)")
    return ok


def main():
    d = pickle.load(open(PKL, "rb"))
    wd, bd = d['weights'], d['bases']
    names = ["model.language_model.layers.0.mlp.gate_proj.weight",
             "model.language_model.layers.0.mlp.down_proj.weight"]
    ok_all = True
    for name in names:
        print(f"=== {name[:60]}")
        if name not in wd:
            print("  [SKIP] no en pkl")
            continue
        if not verify(wd[name], bd[name], name):
            ok_all = False
    # total de pesos del modelo completo
    total_b = sum(wd[n]['n_blocks'] * 30 for n in wd)
    print(f"\nVRAM pesos SX8 (modelo completo, layout v4.4): {total_b/1e9:.3f} GB")
    print("RESULTADO GLOBAL:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
