"""container_to_gguf_sx8_27b.py — Reemplaza los tensores 1:1 del Q8_0 del 27B
por S-X8 del contenedor (mismo pipeline que el 4B: gguf_replace_weights_sx8.py).

Solo se reemplazan los tensores 1:1 (token_embd, output, ffn_gate/up/down) que
tienen el MISMO shape en GGUF y en el contenedor (byte-compatible 30 B/bloque).
El resto (attn_qkv, attn_gate, ssm_*, attn_output, conv1d, normas) se copia
tal cual del Q8_0 (llama.cpp los transforma — doc 43 §9-6).
"""
import sys, os, re, struct
import numpy as np

sys.path.insert(0, "/mnt/Data_3TB/llama-cpp-sx8/gguf-py")
from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf.gguf_reader import GGUFReader

SRC = "/mnt/Data_3TB/qwen_27b_gguf/Qwen3.8-27B-Q8_0.gguf"
CONTAINER = "/mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/qwen_27b_sx8_v11/Qwen3.8-27B-SX8v43-v2.sx8"
OUT = "/mnt/Data_3TB/qwen_27b_gguf/Qwen3.8-27B-SX8v43.gguf"

ALIGN = 32


def build_container_index():
    """Contenedor: {name: (start, nb, ncb, shape)} con skip streaming."""
    f = open(CONTAINER, 'rb')
    assert f.read(8) == b'SX43FILE'
    f.read(1)
    (ml,) = struct.unpack('<I', f.read(4)); f.read(ml)
    (nt,) = struct.unpack('<I', f.read(4))
    idx = {}
    for _ in range(nt):
        start = f.tell()
        (nl,) = struct.unpack('<I', f.read(4)); name = f.read(nl).decode()
        sh = struct.unpack('<II', f.read(8))
        (n_os,) = struct.unpack('<B', f.read(1)); f.read(4 * n_os)
        nb, ncb = struct.unpack('<II', f.read(8))
        idx[name] = (start, nb, ncb, sh)
        f.seek(nb * 30 + 4 + ncb * 256 + ncb * 8, 1)
    f.close()
    return idx


def read_blocks(name, idx):
    """Lee el tensor SX del contenedor (campos SEPARADOS: dmin, dmax, config,
    levels_hi, levels_lo, coeff) y lo reordena a bloques interleaved (nb, 30)
    con blockify_vec — el mismo layout byte-compatible que el pkl del 4B.
    """
    start, nb, ncb, sh = idx[name]
    f = open(CONTAINER, 'rb')
    f.seek(start)
    (nl,) = struct.unpack('<I', f.read(4)); f.read(nl)
    f.read(8)  # shape 2x uint32
    (n_os,) = struct.unpack('<B', f.read(1)); f.read(4 * n_os)
    f.read(8)  # nb, ncb
    dmin = np.frombuffer(f.read(nb * 2), np.float16)
    dmax = np.frombuffer(f.read(nb * 2), np.float16)
    config = np.frombuffer(f.read(nb), np.uint8)
    hi = np.frombuffer(f.read(nb * 16), np.uint8).reshape(nb, 16)
    lo = np.frombuffer(f.read(nb * 8), np.uint8).reshape(nb, 8)
    coeff = np.frombuffer(f.read(nb), np.uint8)
    f.close()
    out = np.zeros((nb, 30), dtype=np.uint8)
    out[:, 0:2] = dmin.view(np.uint8).reshape(nb, 2)
    out[:, 2:4] = dmax.view(np.uint8).reshape(nb, 2)
    out[:, 4] = config
    out[:, 5:21] = hi
    out[:, 21:29] = lo
    out[:, 29] = coeff
    return out


def read_str(f):
    return bytes(f.parts[-1]).decode()


def main():
    r = GGUFReader(SRC)
    idx = build_container_index()
    print(f"Contenedor indexado: {len(idx)} tensores", flush=True)

    if os.path.exists(OUT):
        os.remove(OUT)
    w = GGUFWriter(OUT, "qwen35")

    # 1) copiar TODOS los metadatos del Q8_0 (método correcto: f.parts,
    #    NUNCA f.data que devuelve offsets — doc 43 bugs #7-#9)
    for name, f in sorted(r.fields.items()):
        t = f.types[0]
        if name.startswith("GGUF.") or name == "general.architecture":
            continue  # ya los escribe el GGUFWriter automáticamente
        if t == GGUFValueType.STRING:
            s = read_str(f)
            if name == 'general.name':
                s = 'Qwen3.8-27B-SX8'
            w.add_string(name, s)
        elif t == GGUFValueType.UINT32:
            w.add_uint32(name, int(np.asarray(f.parts[-1]).reshape(-1)[0]))
        elif t == GGUFValueType.INT32:
            w.add_int32(name, int(np.asarray(f.parts[-1]).reshape(-1)[0]))
        elif t == GGUFValueType.FLOAT32:
            w.add_float32(name, float(np.asarray(f.parts[-1]).reshape(-1)[0]))
        elif t == GGUFValueType.BOOL:
            w.add_bool(name, bool(np.asarray(f.parts[-1]).reshape(-1)[0]))
        elif t == GGUFValueType.ARRAY:
            arr_t = f.types[1]
            if arr_t == GGUFValueType.STRING:
                vals = [bytes(f.parts[i]).decode(errors='replace') for i in f.data]
                w.add_array(name, vals)
                continue
            # array numérico: estructura parts [0]=off [1]=name [2]=ARRAY
            # [3]=tipo [4]=len [5..]=valores (doc 43 bug #8)
            if arr_t in (GGUFValueType.UINT32, GGUFValueType.INT32):
                vals = [int(np.asarray(p).reshape(-1)[0]) for p in f.parts[5:]]
            else:
                vals = [float(np.asarray(p).reshape(-1)[0]) for p in f.parts[5:]]
            w.add_array(name, vals)
        else:
            print(f"  [skip] {name} tipo {t}")

    # 2) mapping GGUF name -> container name (1:1 reemplazables)
    def ct_name(gguf_name):
        if gguf_name == "token_embd.weight":
            return "model.language_model.embed_tokens.weight"
        if gguf_name == "output.weight":
            return "lm_head.weight"
        m = re.match(r"blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.weight", gguf_name)
        if m:
            kind = {"ffn_gate": "gate_proj", "ffn_up": "up_proj", "ffn_down": "down_proj"}[m.group(2)]
            return f"model.language_model.layers.{m.group(1)}.mlp.{kind}.weight"
        return None

    # 3) reescribir tensores: SX8 para los 1:1 mapeados; el resto copia del Q8_0
    n_sx8 = 0
    n_copy = 0
    total_sx8 = 0
    for ti in r.tensors:
        name = ti.name.decode() if isinstance(ti.name, bytes) else str(ti.name)
        pname = ct_name(name)
        arr = ti.data
        if pname is not None and pname in idx:
            blocks = read_blocks(pname, idx)
            out_f, in_f = idx[pname][3]
            w.add_tensor(name, blocks.reshape(out_f, idx[pname][2] * 30),
                         raw_dtype=GGMLQuantizationType(41))
            n_sx8 += 1
            total_sx8 += blocks.nbytes
            del blocks
        else:
            if arr.dtype == np.uint8:
                w.add_tensor(name, np.ascontiguousarray(arr).view(np.uint8),
                             raw_dtype=GGMLQuantizationType(8))
            else:
                w.add_tensor(name, arr)
            n_copy += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"SX8: {n_sx8} tensores ({total_sx8/1e9:.2f} GB) | copiados Q8_0: {n_copy} | total: {n_sx8 + n_copy}")


if __name__ == "__main__":
    main()