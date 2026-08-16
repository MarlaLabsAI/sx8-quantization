"""add_small_tensors_4b.py — Construye el .sx8v43 v2 (v1.1) del 4B: copia el v1
y anexa la sección SXT1 (config JSON + tensores 1D del modelo base). El resultado
es un contenedor COMPLETO y autosuficiente (sin depender del modelo base).

Seguridad: el v1 original NO se toca (se copia a v2); verificación final
(lector v1.0 sigue leyendo 381 registros + lector v1.1 lee 381 + 1D).
"""
import os, sys, json, shutil, struct, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sx8_container_v43 import read_all, write_small_section

BASE = "/mnt/Data_3TB/Qwen3.5-4B"
V1 = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-SX8v43.sx8"
V2 = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-SX8v43-v2.sx8"

def main():
    wd, bd, meta = read_all(V1)
    print(f"v1: {len(wd)} tensores cuantizados | meta ok", flush=True)

    config = json.load(open(f"{BASE}/config.json"))
    print(f"Config: {config.get('architectures')} | {config.get('model_type')}", flush=True)

    idx = json.load(open(f"{BASE}/model.safetensors.index.json"))
    wm = idx['weight_map']
    small = {}
    for name, shard in sorted(wm.items()):
        if name == "__metadata__" or name in wd:
            continue
        path = f"{BASE}/{shard}"
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            if name not in f.keys():
                continue
            t = f.get_tensor(name)
            dt = str(t.dtype)
            if dt in ("torch.float16", "torch.bfloat16"):
                arr = t.float().numpy().astype(np.float16)
                small[name] = {'shape': tuple(arr.shape), 'dtype': 'fp16', 'data': arr}
            else:
                arr = t.float().numpy().astype(np.float32)
                small[name] = {'shape': tuple(arr.shape), 'dtype': 'fp32', 'data': arr}
            del t, arr
        if len(small) % 100 == 0:
            print(f"  {len(small)} tensores 1D recogidos...", flush=True)

    print(f"Tensores 1D: {len(small)} | params: "
          f"{sum(int(np.prod(v['shape'])) for v in small.values()):,}", flush=True)

    with open(V1, 'rb') as src, open(V2, 'wb') as dst:
        shutil.copyfileobj(src, dst, 1 << 26)
        write_small_section(dst, config, small)
    sz = os.path.getsize(V2)
    print(f"v2 escrito: {V2} ({sz/1e9:.3f} GB + {sz - os.path.getsize(V1):,} B)", flush=True)

    # --- Verificación 1: lector v1.0 sigue leyendo 381 ---
    wd2, bd2, meta2 = read_all(V2)
    assert len(wd2) == len(wd), "lector v1.0 roto"
    print(f"✅ Lector v1.0 (publicado): {len(wd2)} registros — retrocompat OK", flush=True)

    # --- Verificación 2: lector v1.1 lee 381 + config + 1D ---
    from sx8_container_v43 import read_all_v11
    w, b, m, cfg, sm = read_all_v11(V2)
    assert len(w) == len(wd) and cfg is not None and len(sm) == len(small)
    print(f"✅ Lector v1.1: {len(w)} 2D + {len(sm)} 1D + config OK", flush=True)

    # --- Verificación 3: el prefijo v2 es byte-idéntico al v1 (registros intactos) ---
    h1 = hashlib.sha256()
    with open(V1, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 26), b''):
            h1.update(blk)
    h2 = hashlib.sha256()
    v1_size = os.path.getsize(V1)
    remaining = v1_size
    with open(V2, 'rb') as f:
        while remaining > 0:
            blk = f.read(min(1 << 26, remaining))
            if not blk:
                break
            h2.update(blk)
            remaining -= len(blk)
    assert h1.hexdigest() == h2.hexdigest(), "prefijo no idéntico"
    print(f"✅ Prefijo v2 == v1 (SHA256 {h1.hexdigest()[:16]}...) — registros intactos", flush=True)
    print("CONTENEDOR 4B v2 COMPLETO Y VERIFICADO", flush=True)

if __name__ == "__main__":
    main()
