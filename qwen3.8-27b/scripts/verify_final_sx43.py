"""verify_final_sx43.py — Verificación STREAMING del .sx8v43 final (sin cargar a RAM):
1) Estructura: cabecera + 666 registros + tamaños coherentes.
2) Spot-check CosSim: 2 tensores por shard (leídos del contenedor, decode con kernel corregido
   vs tensor original del shard).
3) Verifica que el contenedor == concatenación de los bodies (byte a byte).
"""
import struct, sys, os, time
import numpy as np
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")
import quantize_qwen38_27b_sx8_v43_shard as q
from safetensors import safe_open

CTR = "/home/blackpanter/qwen_27b_sx8/Qwen3.8-27B-SX8v43.sx8"
OUT_DIR = "/home/blackpanter/qwen_27b_sx8"
MODEL = "/mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/qwen_27b"
MAGIC = b"SX43FILE"
BYTES_PER_BLOCK = 30

def stream_records(path, start_offset=0):
    f = open(path, 'rb')
    if start_offset:
        f.seek(start_offset)
    pos = start_offset
    while True:
        head = f.read(4)
        if not head:
            break
        start = pos
        (nl,) = struct.unpack('<I', head)
        name = f.read(nl).decode()
        sh = struct.unpack('<II', f.read(8))
        (n_os,) = struct.unpack('<B', f.read(1))
        os_ = struct.unpack('<%dI' % n_os, f.read(4 * n_os))
        nb, ncb = struct.unpack('<II', f.read(8))
        data_bytes = nb * 30 + 4 + ncb * 256 + ncb * 8
        f.seek(data_bytes, 1)
        pos = f.tell()
        yield start, name, sh, os_, nb, ncb
    f.close()

def read_record_at(path, start):
    f = open(path, 'rb')
    f.seek(start)
    (nl,) = struct.unpack('<I', f.read(4))
    name = f.read(nl).decode()
    sh = struct.unpack('<II', f.read(8))
    (n_os,) = struct.unpack('<B', f.read(1))
    os_ = struct.unpack('<%dI' % n_os, f.read(4 * n_os))
    nb, ncb = struct.unpack('<II', f.read(8))
    dmin = np.frombuffer(f.read(nb * 2), np.float16)
    dmax = np.frombuffer(f.read(nb * 2), np.float16)
    config = np.frombuffer(f.read(nb), np.uint8)
    hi = np.frombuffer(f.read(nb * 16), np.uint8)
    lo = np.frombuffer(f.read(nb * 8), np.uint8)
    coeff = np.frombuffer(f.read(nb), np.uint8)
    (ncb2,) = struct.unpack('<I', f.read(4))
    data = np.frombuffer(f.read(ncb2 * 64 * 4), np.float32).reshape(ncb2, 64)
    scales = np.frombuffer(f.read(ncb2 * 2 * 4), np.float32).reshape(ncb2, 2)
    f.close()
    return name, sh, nb, ncb, dmin, dmax, config, hi, lo, coeff, data, scales

def main():
    t0 = time.time()
    
    # 1) Cabecera
    f = open(CTR, 'rb')
    assert f.read(8) == MAGIC, "MAGIC incorrecto"
    f.read(1)
    (ml,) = struct.unpack('<I', f.read(4))
    meta = eval(f.read(ml))
    (nt,) = struct.unpack('<I', f.read(4))
    print(f"Cabecera OK: meta={meta}")
    print(f"Registros declarados: {nt}")
    assert nt == 666, f"nt={nt}"
    head_end = f.tell()
    f.close()
    
    # 2) Recorrido streaming + comparación byte a byte con bodies
    shards = sorted(x for x in os.listdir(MODEL)
                    if x.endswith(".safetensors") and not x.startswith("model.safetensors.index"))
    recs = list(stream_records(CTR, head_end))
    print(f"Registros leídos en streaming: {len(recs)} | {time.time()-t0:.0f}s")
    assert len(recs) == 666
    
    # comparar con bodies (contenedor == cabecera + concat bodies, segmento a segmento)
    off = head_end
    n_match = 0
    with open(CTR, 'rb') as fc:
        for sh in shards:
            body = os.path.join(OUT_DIR, f"body_{sh.replace('.safetensors','')}.bin")
            blen = os.path.getsize(body)
            fc.seek(off)
            cseg = fc.read(blen)
            bseg = open(body, 'rb').read()
            if cseg != bseg:
                for i in range(blen):
                    if cseg[i] != bseg[i]:
                        print(f"❌ DIVERGENCIA en {sh} offset {i}")
                        return False
            off += blen
            n_match += 1
    print(f"Contenedor == bodies: {n_match}/18 shards byte-idénticos ✅")
    
    # 3) Spot-check CosSim (2 tensores por shard, desde el CONTENEDOR)
    spot_ok = True
    seg_off = head_end
    for sh in shards:
        body = os.path.join(OUT_DIR, f"body_{sh.replace('.safetensors','')}.bin")
        brecs = list(stream_records(body))
        picks = [len(brecs)//2, len(brecs)-1] if len(brecs) > 1 else [0]
        for p in picks:
            start_r, name, shp, os_, nb, ncb = brecs[p]
            name_c, sh_c, nb_c, ncb_c, dmin, dmax, config, hi, lo, coeff, data, scales = read_record_at(CTR, seg_off + start_r)
            print(f"    {sh.split('-')[1]}: {name_c[:50]} (nb_c={nb_c}) vs {name[:50]} (nb={nb})")
            assert name_c == name and nb_c == nb
            with safe_open(os.path.join(MODEL, sh), framework="pt") as sf:
                t = sf.get_tensor(name)
                Wf = t.float(); del t
            qt = {'shape': shp, 'n_blocks': nb, 'n_cb': ncb, 'dmin': dmin, 'dmax': dmax,
                  'config': config, 'levels_hi': hi, 'levels_lo': lo, 'coeff': coeff.reshape(-1, 1)}
            bi = {'n_cb': ncb, 'basis': [data[cb].reshape(2, 32) for cb in range(ncb)], 'scales': scales}
            cs, mse = q.verify_tensor(qt, bi, Wf.numpy())
            print(f"  shard {sh.split('-')[1]}: {name[:60]:<62} CosSim={cs:.6f}")
            spot_ok &= cs > 0.9995
            del Wf, qt, bi
        seg_off += os.path.getsize(body)
    
    print(f"\nSPOT-CHECK: {'PASS ✅' if spot_ok else 'FAIL ❌'} | {time.time()-t0:.0f}s total")
    print(f"Tamaño final: {os.path.getsize(CTR)/1e9:.3f} GB decimal ({os.path.getsize(CTR)/2**30:.3f} GiB)")
    

if __name__ == "__main__":
    main()
