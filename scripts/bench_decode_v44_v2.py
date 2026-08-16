"""bench_decode_v44_v2.py — Benchmark decode M=1 del contenedor v2 (STANDALONE).

Réplica del protocolo de bench_decode_v44.py (VRAM 3.955 GB < GGUF Q8_0):
decodifica cada tensor 2D con el kernel decode1_v44 (M=1, compacto) y mide
velocidad + VRAM pico. Carga SOLO desde el .sx8v43 v1.1.
"""
import sys, time, json, gc
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")

import torch
from sx8_container_v43 import read_all_v11
from sx8_decode1_v3 import make_tensors_v44, decode1_v44, best_split_k

CONTAINER = None


def main():
    global CONTAINER
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default=os.environ.get("SX8_CONTAINER", "Qwen3.5-4B-SX8v43.sx8"))
    args = ap.parse_args()
    CONTAINER = args.container
    wd, bd, meta, cfg, small = read_all_v11(CONTAINER)
    print(f"Contenedor v2: {len(wd)} tensores 2D", flush=True)

    # seleccionar los tensores de TEXTO (como el bench original: lineales del LM)
    names = [n for n in wd if n.startswith("model.language_model")]
    print(f"Tensores de texto: {len(names)}", flush=True)

    # Protocolo del bench original: TODOS los tensores compactos v4.4 en GPU
    # simultáneamente (layout kb-major) + workspace → VRAM ~3.955 GB.
    torch.cuda.reset_peak_memory_stats()
    t_all = []
    t0 = time.time()
    prep = []
    for name in names:
        qt, bi = wd[name], bd[name]
        if len(qt['shape']) != 2:
            continue
        out_f, in_f = qt['shape']
        if in_f < 32:
            continue
        hdr_t, lvl_t, bas_t, sca_t = make_tensors_v44(qt, bi, hdr_aligned=False)
        prep.append((name, qt, bi, out_f, in_f, hdr_t, lvl_t, bas_t, sca_t))
    vram_prep = torch.cuda.max_memory_allocated() / 2**30
    print(f"  VRAM tras cargar TODOS los compactos: {vram_prep:.3f} GB", flush=True)
    for name, qt, bi, out_f, in_f, hdr_t, lvl_t, bas_t, sca_t in prep:
        X = torch.randn(in_f, dtype=torch.float16, device="cuda") * 0.5
        sk = best_split_k(out_f, qt['n_cb'])
        t1 = time.time()
        Y, _ = decode1_v44(X, qt, bi, hdr_t, lvl_t, bas_t, sca_t, 6, sk)
        torch.cuda.synchronize()
        dt = time.time() - t1
        wps = out_f * in_f / dt
        t_all.append((name, out_f * in_f, dt, wps))
        del X, Y
    torch.cuda.empty_cache()

    tot = time.time() - t0
    tot_w = sum(w for _, w, _, _ in t_all)
    tot_dt = sum(dt for _, _, dt, _ in t_all)
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nDecode M=1 (v4.4, standalone v2): {len(t_all)} tensores en {tot:.0f}s")
    print(f"  {tot_w/tot_dt/1e9:.1f} G pesos/s | {tot_w/tot_dt/2**30:.2f} GB/s")
    print(f"  VRAM PICO: {vram:.3f} GB (referencia publicada: 3.955 GB)")
    print(f"  Velocidad media por tensor: {tot_dt/len(t_all):.4f} s")
    json.dump({"mode": "decode1_v44_v2_standalone", "vram_gb": vram,
               "G_wps": tot_w/tot_dt/1e9, "n_tensors": len(t_all),
               "referencia_vram": 3.955},
              open("/mnt/Data_3TB/project Marla/quant-paper/results/bench_decode_v44_v2.json", "w"),
              indent=2)
    print("Guardado: results/bench_decode_v44_v2.json")


if __name__ == "__main__":
    main()
