"""add_small_tensors_27b.py — Construye el .sx8v43 v1.1 del 27B (autosuficiente):
copia el v1 por streaming y anexa la seccion SXT1 (config JSON + tensores 1D).

El 27B NO cabe en RAM (26 GB vs 16 GB) -> misma estrategia que el quantizador:
lectura/escritura por chunks y shard-a-shard con safe_open. No se materializa
nunca el contenedor completo. El v1 original NO se toca (se copia a v2).

Verificacion ligera (sin materializar pesos):
  1. scan de cabecera + 666 registros (saltando los datos)
  2. seccion SXT1 leida (config + 1D) == lo recolectado de los shards
  3. prefijo SHA-256 byte-identico al v1
"""
import os, sys, json, struct, hashlib
import numpy as np
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sx8_container_v43 import SMALL_MAGIC, write_small_section, read_small_section

BASE = "/mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/qwen_27b"
V1 = "/home/blackpanter/qwen_27b_sx8/Qwen3.8-27B-SX8v43.sx8"
V2 = "/mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/qwen_27b_sx8_v11/Qwen3.8-27B-SX8v43-v2.sx8"

CHUNK = 1 << 26  # 64 MiB


def scan_tensor_records(path, max_records=None):
    """Recorre registros de tensor del .sx8v43 sin materializar datos.
    Devuelve: lista de (name, sh, os_, nb, ncb), bytes de cabecera+registros.
    """
    recs = []
    n_bytes = 0
    with open(path, "rb") as f:
        assert f.read(8) == b"SX43FILE"
        f.read(1)  # version
        (ml,) = struct.unpack('<I', f.read(4)); f.read(ml)  # meta
        (nt,) = struct.unpack('<I', f.read(4))
        n_bytes = 8 + 1 + 4 + ml + 4
        for _ in range(max_records if max_records else nt):
            (nl,) = struct.unpack('<I', f.read(4))
            name = f.read(nl).decode()
            sh = struct.unpack('<II', f.read(8))
            (n_os,) = struct.unpack('<B', f.read(1))
            os_ = struct.unpack('<%dI' % n_os, f.read(4 * n_os))
            nb, ncb = struct.unpack('<II', f.read(8))
            data_bytes = nb * 30 + 4 + ncb * 256 + ncb * 8
            recs.append((name, sh, os_, nb, ncb))
            n_bytes += 4 + nl + 8 + 1 + 4 * n_os + 8 + data_bytes
            f.seek(data_bytes, 1)
    return recs, n_bytes


def copy_and_append_sxt1(src, dst, config, small):
    """Copia src -> dst por chunks y anexa la seccion SXT1 (config + 1D)."""
    with open(src, "rb") as s, open(dst, "wb") as d:
        while True:
            chunk = s.read(CHUNK)
            if not chunk:
                break
            d.write(chunk)
        write_small_section(d, config, small)


def sha256_prefix(path, size):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = size
        while remaining > 0:
            chunk = f.read(min(CHUNK, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def main():
    # --- 0. V1 intacto: escanear cabecera y nombres de los 666 registros (streaming) ---
    recs_v1, n_bytes_v1 = scan_tensor_records(V1)
    assert len(recs_v1) == 666, f"V1: {len(recs_v1)} registros != 666"
    quantized_names = {r[0] for r in recs_v1}
    print(f"V1: {len(recs_v1)} tensores cuantizados | {n_bytes_v1/1e9:.3f} GB | meta ok", flush=True)

    #2. Recolectar config + tensores 1D de los shards (sin los 666 cuantizados)
    config = json.load(open(f"{BASE}/config.json"))
    print(f"Config: {config.get('architectures')} | {config.get('model_type')}", flush=True)

    idx = json.load(open(f"{BASE}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    by_shard = {}
    for name, shard in wm.items():
        if name == "__metadata__" or name in quantized_names:
            continue
        by_shard.setdefault(shard, []).append(name)

    small = {}
    for shard, names in sorted(by_shard.items()):
        path = os.path.join(BASE, shard)
        with safe_open(path, framework="pt") as f:
            for name in names:
                if name not in f.keys():
                    continue
                t = f.get_tensor(name)
                dt = str(t.dtype)
                if dt in ("torch.float16", "torch.bfloat16"):
                    arr = t.float().numpy().astype(np.float16)
                    small[name] = {"shape": tuple(arr.shape), "dtype": "fp16", "data": arr}
                else:
                    arr = t.float().numpy().astype(np.float32)
                    small[name] = {"shape": tuple(arr.shape), "dtype": "fp32", "data": arr}
                del t, arr
        print(f"  {shard}: {len(names)} 1D recogidos (total {len(small)})", flush=True)

    n_params = sum(int(np.prod(v["shape"])) for v in small.values())
    print(f"Tensores 1D: {len(small)} | params: {n_params:,} | "
          f"bytes: {sum(v['data'].nbytes for v in small.values()):,}", flush=True)

    #3. Copiar v1 -> v2 + anexar SXT1
    copy_and_append_sxt1(V1, V2, config, small)
    v1_size = os.path.getsize(V1)
    print(f"V1.1 escrito: {V2} ({os.path.getsize(V2)/1e9:.3f} GB = v1 {v1_size/1e9:.3f} + "
          f"{os.path.getsize(V2)-v1_size:,} B SXT1)", flush=True)

    #4. Verificacion ligera
    recs_v2, n_bytes_v2 = scan_tensor_records(V2)
    assert len(recs_v2) == 666, f"scan v1.1: {len(recs_v2)} != 666"
    with open(V2, "rb") as f:
        f.seek(n_bytes_v2)
        cfg2, sm2 = read_small_section(f)
    assert cfg2 == config, "config no coincide"
    assert set(sm2.keys()) == set(small.keys()), "nombres 1D no coinciden"
    for k, v in small.items():
        s = sm2[k]
        assert s["shape"] == v["shape"] and s["dtype"] == v["dtype"]
        assert np.array_equal(s["data"], v["data"]), f"datos 1D difieren: {k}"
    print(f"OK Lector v1.1: 666 2D + {len(sm2)} 1D + config", flush=True)

    #5. Prefijo SHA-256 byte-identico
    h1 = sha256_prefix(V1, v1_size)
    h2 = sha256_prefix(V2, v1_size)
    assert h1 == h2, "prefijo no identico"
    print(f"OK Prefijo v1.1 == v1 (SHA256 {h1[:16]}...) — registros intactos", flush=True)
    print("CONTENEDOR 27B v1.1 COMPLETO Y VERIFICADO", flush=True)


if __name__ == "__main__":
    main()