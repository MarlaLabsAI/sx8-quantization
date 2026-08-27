"""assemble_sx43.py — Ensambla los bodies de los shards en el .sx8v43 final.

Cabecera: MAGIC(8) + VERSION(1) + meta_len(4) + meta + n_tensors(4) + registros
(registro por tensor, idéntico a write_sx43). Los bodies se copian tal cual
(byte-idénticos a lo escrito por el quantizador). Verificación estructural al
ensamblar: nº registros, longitudes, tamaño total esperado.
"""
import os, struct, sys, json

MAGIC = b"SX43FILE"
VERSION = 1
BYTES_PER_BLOCK = 30

MODEL = "/mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/qwen_27b"
OUT_DIR = "/home/blackpanter/qwen_27b_sx8"
STATE_FILE = "/tmp/opencode/qwen27b_state/quant_state.json"
OUT_NAME = "Qwen3.8-27B-SX8v43.sx8"

shards = sorted(f for f in os.listdir(MODEL)
                if f.endswith(".safetensors") and not f.startswith("model.safetensors.index"))

st = json.load(open(STATE_FILE))
meta = st["meta"]
print(f"META: {meta}", flush=True)

def scan_body(path):
    """Recorre registros: (name, sh, os_, nb, ncb). Devuelve lista + bytes totales."""
    recs = []
    n_bytes = 0
    with open(path, "rb") as f:
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
            data_bytes = nb * BYTES_PER_BLOCK + 4 + ncb * 256 + ncb * 8
            n_bytes += 4 + nl + 8 + 1 + 4 * n_os + 8 + data_bytes
            f.seek(data_bytes, 1)
            recs.append((name, sh, os_, nb, ncb))
    return recs, n_bytes

total_recs = 0
total_bytes = 0
all_recs = []
for shard in shards:
    body = os.path.join(OUT_DIR, f"body_{shard.replace('.safetensors','')}.bin")
    if not os.path.exists(body):
        print(f"❌ FALTA body: {body}", flush=True)
        sys.exit(1)
    recs, nb = scan_body(body)
    n_expected = st["done"][shard]["tensors"]
    if len(recs) != n_expected:
        print(f"❌ {shard}: {len(recs)} registros, esperados {n_expected}", flush=True)
        sys.exit(1)
    print(f"  {shard}: {len(recs)} registros | {nb/1e9:.3f} GB", flush=True)
    total_recs += len(recs)
    total_bytes += nb
    all_recs.append((shard, body))

print(f"\nTOTAL: {total_recs} registros | {total_bytes/1e9:.3f} GB cuerpo", flush=True)
if total_recs != 666:
    print(f"❌ Registros esperados: 666", flush=True)
    sys.exit(1)

mb = repr(sorted(meta.items())).encode()
out_path = os.path.join(OUT_DIR, OUT_NAME)
if os.path.exists(out_path):
    os.remove(out_path)
with open(out_path, "wb") as f:
    f.write(MAGIC)
    f.write(struct.pack('<B', VERSION))
    f.write(struct.pack('<I', len(mb)))
    f.write(mb)
    f.write(struct.pack('<I', total_recs))
    for _, body in all_recs:
        with open(body, "rb") as bf:
            while True:
                chunk = bf.read(1 << 26)
                if not chunk:
                    break
                f.write(chunk)

fsz = os.path.getsize(out_path)
exp = 8 + 1 + 4 + len(mb) + 4 + total_bytes
print(f"\n✅ Ensamblado: {out_path}")
print(f"   Tamaño: {fsz/1e9:.3f} GB decimal ({fsz/2**30:.3f} GiB) | esperado {exp/1e9:.3f} GB")
assert fsz == exp, "Tamaño no coincide"
print(f"   bpp real sobre params: {fsz*8/meta['total_params']:.4f} | ratio: {meta['total_params']*2/fsz:.3f}x")
print("   Estructura verificada: cabecera + 666 registros + tamaño coherente.")
