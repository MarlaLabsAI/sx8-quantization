"""reverify_sx43.py — Re-verifica TODOS los tensores con el kernel CORREGIDO
(step fallback 1e-10). Lee los bodies ya escritos + los shards originales,
recalcula CosSim/MSE reales, actualiza state.json y guarda la meta final.
"""
import sys, os, struct, json, time
import numpy as np
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")
import quantize_qwen38_27b_sx8_v43_shard as q
from safetensors import safe_open

MODEL = "/mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/qwen_27b"
OUT_DIR = "/home/blackpanter/qwen_27b_sx8"
STATE_FILE = "/tmp/opencode/qwen27b_state/quant_state.json"

def read_record_by_name(path, target):
    f = open(path, 'rb')
    while True:
        head = f.read(4)
        if not head:
            f.close(); return None
        (nl,) = struct.unpack('<I', head)
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
        if name == target:
            f.close()
            return dict(name=name, shape=sh, nb=nb, ncb=ncb, dmin=dmin, dmax=dmax,
                        config=config, hi=hi, lo=lo, coeff=coeff, data=data, scales=scales)

def read_all_records(path):
    f = open(path, 'rb')
    recs = []
    while True:
        head = f.read(4)
        if not head:
            break
        (nl,) = struct.unpack('<I', head)
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
        recs.append(dict(name=name, shape=sh, nb=nb, ncb=ncb, dmin=dmin, dmax=dmax,
                         config=config, hi=hi, lo=lo, coeff=coeff, data=data, scales=scales))
    f.close()
    return recs

def main():
    shards = sorted(f for f in os.listdir(MODEL)
                    if f.endswith(".safetensors") and not f.startswith("model.safetensors.index"))
    st = json.load(open(STATE_FILE))
    cs_all = []; mse_all = []
    t0 = time.time()

    for sh in shards:
        if sh not in st["done"]:
            print(f"[SKIP-INC] {sh}: shard no terminado aún", flush=True)
            continue
        if "deg_blocks" in st["done"][sh]:
            print(f"[SKIP] {sh}: ya re-verificado (avg {st['done'][sh]['avg_cs']:.6f})", flush=True)
            cs_all.extend(st["done"][sh]["cs_list"])
            mse_all.extend(st["done"][sh]["mse_list"])
            continue
        body = os.path.join(OUT_DIR, f"body_{sh.replace('.safetensors','')}.bin")
        recs = read_all_records(body)
        sh_cs = []; sh_mse = []
        n_deg = 0; n_deg_blocks = 0
        with safe_open(os.path.join(MODEL, sh), framework="pt") as f:
            for ri, rec in enumerate(recs):
                name = rec['name']
                t = f.get_tensor(name)
                Wf = t.float(); del t
                shape = rec['shape']
                out_f, in_f = shape[0], shape[1]
                qt = {'shape': shape, 'n_blocks': rec['nb'], 'n_cb': rec['ncb'],
                      'dmin': rec['dmin'], 'dmax': rec['dmax'], 'config': rec['config'],
                      'levels_hi': rec['hi'], 'levels_lo': rec['lo'],
                      'coeff': rec['coeff'].reshape(-1, 1)}
                bi = {'n_cb': rec['ncb'],
                      'basis': [rec['data'][cb].reshape(2, 32) for cb in range(rec['ncb'])],
                      'scales': rec['scales']}
                cs, mse = q.verify_tensor(qt, bi, Wf.numpy())
                # bloques degenerados (rango < 1e-4 en fp16)
                rg = rec['dmax'].astype(np.float32) - rec['dmin'].astype(np.float32)
                nd = int((np.abs(rg) < 1e-4).sum())
                n_deg += nd; n_deg_blocks += rec['nb']
                sh_cs.append(cs); sh_mse.append(mse)
                del Wf, qt, bi
                if ri % 20 == 0 or cs < 0.999:
                    print(f"  [{ri+1}/{len(recs)}] {name[:70]:<72} CosSim={cs:.6f} (deg {nd}/{rec['nb']})", flush=True)
        cs_all.extend(sh_cs); mse_all.extend(sh_mse)
        st["done"][sh]["cs_list"] = sh_cs
        st["done"][sh]["mse_list"] = sh_mse
        st["done"][sh]["avg_cs"] = float(np.mean(sh_cs))
        st["done"][sh]["deg_blocks"] = n_deg
        json.dump(st, open(STATE_FILE, "w"))
        print(f"  ✅ {sh}: avg CosSim={np.mean(sh_cs):.6f} | bloques degenerados: {n_deg}/{n_deg_blocks} "
              f"({100*n_deg/max(n_deg_blocks,1):.3f}%) | {time.time()-t0:.0f}s", flush=True)

    # meta final
    total_blocks = sum(d["blocks"] for d in st["done"].values())
    total_params = sum(d["params"] for d in st["done"].values())
    avg_cs = float(np.mean(cs_all)); avg_mse = float(np.mean(mse_all))
    bpp = total_blocks * 30 * 8 / total_params
    ratio = total_params * 2 / (total_blocks * 30)
    meta = {'bpp': float(bpp), 'total_params': int(total_params), 'total_blocks': int(total_blocks),
            'ratio': float(ratio), 'quality_cossim': float(avg_cs), 'quality_mse': float(avg_mse),
            'model': MODEL}
    st["meta"] = meta
    json.dump(st, open(STATE_FILE, "w"))
    print(f"\n=== META FINAL (con kernel corregido) ===", flush=True)
    print(f"bpp={bpp:.4f} | ratio={ratio:.3f} | CosSim medio={avg_cs:.6f} | MSE={avg_mse:.3e}", flush=True)
    print(f"Tiempo total re-verificación: {(time.time()-t0)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
