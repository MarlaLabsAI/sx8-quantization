"""sx8_container_v43.py — Formato de archivo S-X8 v4.3 (.sx8v43)
30 bytes/bloque: dmin(fp16,2) + dmax(fp16,2) + config(1) + levels_hi(16) + levels_lo(8) + coeff(1, 2 bases)
Los campos del pkl v4.3 YA están en su forma final (escala FP16 exacta, config explícita, coeff 1 byte)
→ el contenedor es una copia plana, sin re-empaquetado.
"""
import numpy as np, os, struct, time, pickle

MAGIC = b"SX43FILE"
VERSION = 1
BYTES_PER_BLOCK = 30

def write_sx43(pkl_path, out_path, verbose=True):
    d = pickle.load(open(pkl_path, 'rb'))
    wd, bd, meta = d['weights'], d['bases'], d['meta']
    t0 = time.time(); n_blk_tot = 0
    with open(out_path, 'wb') as f:
        f.write(MAGIC); f.write(struct.pack('<B', VERSION))
        mb = repr(sorted(meta.items())).encode()
        f.write(struct.pack('<I', len(mb))); f.write(mb)
        f.write(struct.pack('<I', len(wd)))
        for name in sorted(wd):
            qt, bi = wd[name], bd[name]
            nb, ncb = qt['n_blocks'], qt['n_cb']
            nb_bytes = name.encode(); f.write(struct.pack('<I', len(nb_bytes))); f.write(nb_bytes)
            sh = qt['shape']; os_ = qt.get('orig_shape', sh)
            f.write(struct.pack('<II', sh[0], sh[1]))
            f.write(struct.pack('<B', len(os_)))
            f.write(struct.pack('<%dI' % len(os_), *os_))
            f.write(struct.pack('<II', nb, ncb))
            f.write(qt['dmin'].tobytes())          # nb × 2 (fp16)
            f.write(qt['dmax'].tobytes())          # nb × 2
            f.write(qt['config'].tobytes())        # nb × 1
            f.write(qt['levels_hi'].tobytes())     # nb × 16
            f.write(qt['levels_lo'].tobytes())     # nb × 8
            f.write(qt['coeff'].tobytes())         # nb × 1
            f.write(struct.pack('<I', bi['n_cb']))
            f.write(bi['data'].tobytes())          # n_cb × 64 × 4 (2 bases)
            f.write(bi['scales'].tobytes())        # n_cb × 2 × 4
            n_blk_tot += nb
    sz = os.path.getsize(out_path)
    if verbose:
        print(f"✅ .sx8v43 escrito: {out_path}")
        print(f"   {len(wd)} tensores | {n_blk_tot:,} bloques | {sz/1e9:.2f} GB "
              f"({n_blk_tot*BYTES_PER_BLOCK/1e9:.2f} GB teor. 30 B/bloque) | {time.time()-t0:.0f}s")
    return out_path

def verify_sx43(pkl_path, sx43_path):
    """Verificación byte-exacta: arrays del archivo == arrays del pkl (vectorizado)."""
    d = pickle.load(open(pkl_path, 'rb'))
    wd, bd = d['weights'], d['bases']
    f = open(sx43_path, 'rb')
    assert f.read(8) == MAGIC
    f.read(1); (ml,) = struct.unpack('<I', f.read(4)); f.read(ml)
    (nt,) = struct.unpack('<I', f.read(4))
    ok_all = True; checked = 0
    t0 = time.time()
    for _ in range(nt):
        (nl,) = struct.unpack('<I', f.read(4)); name = f.read(nl).decode()
        sh = struct.unpack('<II', f.read(8))
        (n_os,) = struct.unpack('<B', f.read(1))
        os_ = struct.unpack('<%dI' % n_os, f.read(4 * n_os))
        nb, ncb = struct.unpack('<II', f.read(8))
        dmin = np.frombuffer(f.read(nb*2), np.float16)
        dmax = np.frombuffer(f.read(nb*2), np.float16)
        config = np.frombuffer(f.read(nb), np.uint8)
        hi = np.frombuffer(f.read(nb*16), np.uint8)
        lo = np.frombuffer(f.read(nb*8), np.uint8)
        coeff = np.frombuffer(f.read(nb), np.uint8)
        (ncb2,) = struct.unpack('<I', f.read(4))
        data = np.frombuffer(f.read(ncb2*64*4), np.float32).reshape(ncb2, 64)
        scales = np.frombuffer(f.read(ncb2*2*4), np.float32).reshape(ncb2, 2)
        qt, bi = wd[name], bd[name]
        ok = (np.array_equal(np.array(dmin), np.array(qt['dmin'])) and
              np.array_equal(np.array(dmax), np.array(qt['dmax'])) and
              np.array_equal(np.array(config), np.array(qt['config'])) and
              np.array_equal(np.array(hi), np.array(qt['levels_hi'])) and
              np.array_equal(np.array(lo), np.array(qt['levels_lo'])) and
              np.array_equal(np.asarray(coeff).reshape(-1), np.asarray(qt['coeff']).reshape(-1)) and
              np.array_equal(np.array(data), np.array(bi['data'])) and
              np.array_equal(np.array(scales), np.array(bi['scales'])))
        ok_all &= ok; checked += 1
        if not ok or checked <= 2:
            print(f"  {'✅' if ok else '❌'} {name[:52]:<54} ({nb:,} bloques)")
    f.close()
    print(f"\nVERIFICACIÓN v4.3: {checked}/{nt} tensores | "
          f"{'TODO IDÉNTICO ✅' if ok_all else 'DIFERENCIAS ❌'} | {time.time()-t0:.0f}s")
    return ok_all

def read_all(sx43_path):
    """Lee el .sx8v43 completo → (wd, bd, meta) compatible con kernel_sx8_v43.decode_tensor_gpu"""
    f = open(sx43_path, 'rb')
    assert f.read(8) == MAGIC
    f.read(1); (ml,) = struct.unpack('<I', f.read(4)); meta = eval(f.read(ml))
    (nt,) = struct.unpack('<I', f.read(4))
    wd = {}; bd = {}
    for _ in range(nt):
        (nl,) = struct.unpack('<I', f.read(4)); name = f.read(nl).decode()
        sh = struct.unpack('<II', f.read(8))
        (n_os,) = struct.unpack('<B', f.read(1))
        os_ = struct.unpack('<%dI' % n_os, f.read(4 * n_os))
        nb, ncb = struct.unpack('<II', f.read(8))
        dmin = np.frombuffer(f.read(nb*2), np.float16)
        dmax = np.frombuffer(f.read(nb*2), np.float16)
        config = np.frombuffer(f.read(nb), np.uint8)
        hi = np.frombuffer(f.read(nb*16), np.uint8)
        lo = np.frombuffer(f.read(nb*8), np.uint8)
        coeff = np.frombuffer(f.read(nb), np.uint8)
        (ncb2,) = struct.unpack('<I', f.read(4))
        data = np.frombuffer(f.read(ncb2*64*4), np.float32).reshape(ncb2, 64)
        scales = np.frombuffer(f.read(ncb2*2*4), np.float32).reshape(ncb2, 2)
        os_final = os_ if tuple(os_) != (sh[0], sh[1]) else None
        wd[name] = {'shape': (sh[0], sh[1]), 'orig_shape': os_final, 'n_blocks': nb, 'n_cb': ncb,
                    'dmin': dmin, 'dmax': dmax, 'config': config,
                    'levels_hi': hi, 'levels_lo': lo, 'coeff': coeff}
        bd[name] = {'n_cb': ncb2, 'data': data, 'scales': scales}
    f.close()
    return wd, bd, dict(meta)

if __name__ == "__main__":
    pkl = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
    out = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-SX8v43.sx8"
    write_sx43(pkl, out)
    verify_sx43(pkl, out)


# ============================================================================
# v1.1 — sección SXT1 (retrocompatible): config JSON + tensores pequeños (1D)
# Los lectores v1.0 leen los registros de tensor y se detienen; el bloque SXT1
# queda como datos finales ignorados. read_all_v11 lo lee completo.
# dtype: 0 = float16 · 1 = float32
# ============================================================================
SMALL_MAGIC = b"SXT1"


def write_small_section(f, config, small_tensors):
    """Escribe la sección v1.1 al final de un archivo abierto en modo append.
    small_tensors: dict {name: {'shape': tuple, 'dtype': 'fp16'|'fp32', 'data': np.ndarray}}
    """
    import json as _json
    cb = _json.dumps(config).encode()
    f.write(SMALL_MAGIC)
    f.write(struct.pack('<I', len(cb)))
    f.write(cb)
    f.write(struct.pack('<I', len(small_tensors)))
    for name in sorted(small_tensors):
        st = small_tensors[name]
        nb = name.encode()
        f.write(struct.pack('<I', len(nb))); f.write(nb)
        f.write(struct.pack('<B', len(st['shape'])))
        f.write(struct.pack('<%dI' % len(st['shape']), *st['shape']))
        dt = 0 if st['dtype'] == 'fp16' else 1
        f.write(struct.pack('<B', dt))
        f.write(st['data'].tobytes())


def read_small_section(f):
    """Lee la sección SXT1 desde la posición actual. Devuelve (config, small) o (None, None)."""
    import json as _json
    pos = f.tell()
    magic = f.read(4)
    if magic != SMALL_MAGIC:
        f.seek(pos)
        return None, None
    (cl,) = struct.unpack('<I', f.read(4))
    config = _json.loads(f.read(cl))
    (n,) = struct.unpack('<I', f.read(4))
    small = {}
    for _ in range(n):
        (nl,) = struct.unpack('<I', f.read(4))
        name = f.read(nl).decode()
        (nd,) = struct.unpack('<B', f.read(1))
        shape = struct.unpack('<%dI' % nd, f.read(4 * nd))
        (dt,) = struct.unpack('<B', f.read(1))
        nbytes = int(np.prod(shape)) * (2 if dt == 0 else 4)
        data = np.frombuffer(f.read(nbytes), np.float16 if dt == 0 else np.float32).reshape(shape)
        small[name] = {'shape': tuple(shape), 'dtype': 'fp16' if dt == 0 else 'fp32', 'data': data}
    return config, small


def read_all_v11(sx43_path):
    """Lee el .sx8v43 v1.1 completo → (wd, bd, meta, config, small)."""
    f = open(sx43_path, 'rb')
    assert f.read(8) == MAGIC
    f.read(1); (ml,) = struct.unpack('<I', f.read(4)); meta = eval(f.read(ml))
    (nt,) = struct.unpack('<I', f.read(4))
    wd = {}; bd = {}
    for _ in range(nt):
        (nl,) = struct.unpack('<I', f.read(4)); name = f.read(nl).decode()
        sh = struct.unpack('<II', f.read(8))
        (n_os,) = struct.unpack('<B', f.read(1))
        os_ = struct.unpack('<%dI' % n_os, f.read(4 * n_os))
        nb, ncb = struct.unpack('<II', f.read(8))
        dmin = np.frombuffer(f.read(nb*2), np.float16)
        dmax = np.frombuffer(f.read(nb*2), np.float16)
        config_ = np.frombuffer(f.read(nb), np.uint8)
        hi = np.frombuffer(f.read(nb*16), np.uint8)
        lo = np.frombuffer(f.read(nb*8), np.uint8)
        coeff = np.frombuffer(f.read(nb), np.uint8)
        (ncb2,) = struct.unpack('<I', f.read(4))
        data = np.frombuffer(f.read(ncb2*64*4), np.float32).reshape(ncb2, 64)
        scales = np.frombuffer(f.read(ncb2*2*4), np.float32).reshape(ncb2, 2)
        os_final = os_ if tuple(os_) != (sh[0], sh[1]) else None
        wd[name] = {'shape': (sh[0], sh[1]), 'orig_shape': os_final, 'n_blocks': nb, 'n_cb': ncb,
                    'dmin': dmin, 'dmax': dmax, 'config': config_,
                    'levels_hi': hi, 'levels_lo': lo, 'coeff': coeff}
        bd[name] = {'n_cb': ncb2, 'data': data, 'scales': scales}
    cfg, small = read_small_section(f)
    f.close()
    return wd, bd, dict(meta), cfg, small
