"""convert_pkl_to_gguf_sx8.py — convierte el pkl S-X8 v4.3 a GGUF con tipo SX8 (v2 vectorizado).

Escribe los tensores SX8 del pkl como bloques de 30 B (block_sx8) en el GGUF
con raw_dtype=GGMLQuantizationType.SX8(41). El layout de cada bloque es
idéntico al .sx8v43 (dmin dmax config qh[16] ql[8] coeff) → byte-compatible.

Blockify VECTORIZADO: construye el buffer (n_blk, 30) con reordenamiento de
arrays completos (sin loop por bloque).
"""
import sys, os, pickle
import numpy as np

sys.path.insert(0, "/mnt/Data_3TB/llama-cpp-sx8/gguf-py")
from gguf import GGUFWriter, GGMLQuantizationType

PKL = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
OUT = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-SX8v43.gguf"


def blockify_vec(qt):
    """(n_blk, 30) uint8: dmin(2) dmax(2) config(1) qh(16) ql(8) coeff(1)."""
    n_blk = qt['n_blocks']
    dm = qt['dmin'].astype(np.float16).reshape(-1)
    dx = qt['dmax'].astype(np.float16).reshape(-1)
    cfg = qt['config'].reshape(-1).astype(np.uint8)
    co = qt['coeff'].reshape(-1).astype(np.uint8)
    hi = qt['levels_hi'].reshape(n_blk, 16)
    lo = qt['levels_lo'].reshape(n_blk, 8)

    out = np.zeros((n_blk, 30), dtype=np.uint8)
    # vistas: cada columna de 30 B se llena con el campo correspondiente
    out[:, 0:2] = dm.view(np.uint8).reshape(n_blk, 2)
    out[:, 2:4] = dx.view(np.uint8).reshape(n_blk, 2)
    out[:, 4] = cfg
    out[:, 5:21] = hi
    out[:, 21:29] = lo
    out[:, 29] = co
    return out


def main():
    d = pickle.load(open(PKL, "rb"))
    wd = d['weights']
    print(f"Pkl: {len(wd)} tensores", flush=True)

    if os.path.exists(OUT):
        os.remove(OUT)
    w = GGUFWriter(OUT, "qwen35")
    w.add_name("Qwen3.5-4B-SX8")
    w.add_quantization_version(2)
    w.add_context_length(262144)
    w.add_embedding_length(2560)
    w.add_block_count(32)
    w.add_feed_forward_length(9216)
    w.add_full_attention_interval(4)
    w.add_head_count(16)
    w.add_head_count_kv(4)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_dimension_count(64)
    w.add_rope_freq_base(10000000.0)
    w.add_rope_dimension_sections([11, 11, 10, 0])
    w.add_ssm_conv_kernel(4)
    w.add_ssm_state_size(128)
    w.add_ssm_group_count(16)
    w.add_ssm_time_step_rank(32)
    w.add_ssm_inner_size(4096)
    w.add_file_type(7)

    total = 0
    n_lin = 0
    for name, qt in wd.items():
        if len(qt['shape']) != 2:
            continue
        out_f, in_f = qt['shape']
        n_cb = qt['n_cb']
        blocks = blockify_vec(qt)
        byte_shape = (out_f, n_cb * 30)
        w.add_tensor(name, blocks.reshape(byte_shape),
                     raw_dtype=GGMLQuantizationType(41))
        total += blocks.nbytes
        n_lin += 1
        del blocks
        if n_lin % 50 == 0:
            print(f"  {n_lin} tensores... ({total/1e9:.2f} GB)", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"GGUF SX8 escrito: {OUT} ({total/1e9:.2f} GB, {n_lin} lineales)")


if __name__ == "__main__":
    main()
