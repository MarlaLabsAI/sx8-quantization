"""ppl_fused_revalidate_v2.py — PPL wikitext-2 del contenedor 4B v2 (STANDALONE + FUSED).

MISMO protocolo que ppl_fused_v44.py (estrellas publicadas: PPL 10.2267 con PCA,
VRAM 3.955 GB < Q8_0): runtime fused v4.4 (SX8LinearV44, decode on-the-fly,
pesos compactos — NO materializa FP16 en VRAM). Carga SOLO desde el contenedor
v1.1 (sin modelo base). Mide VRAM pico.
"""
import torch, math, sys, os, time, json, gc
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")

from integrate_fused_v44 import clone_model_fused_v44
from eval_common import load_model_standalone
from ppl_wikitext import get_text, run_ppl
from sx8_container_v43 import read_all_v11

CONTAINER = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-SX8v43-v2.sx8"
TOK_DIR = "/tmp/opencode/standalone_test/tokenizer"


def main():
    wd, bd, meta, cfg, small = read_all_v11(CONTAINER)
    torch.manual_seed(3)
    model, tok, _ = load_model_standalone(CONTAINER, tokenizer_dir=TOK_DIR, verify=False)
    m_fused, n_rep = clone_model_fused_v44(model, wd, bd, hdr_bytes=6)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Lineales sustituidos (compacto): {n_rep}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ppl = run_ppl(m_fused, tok, "FUSED v4.4 STANDALONE v2")
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nPPL wikitext-2 (fused v4.4 standalone): {ppl:.4f}  [{dt:.0f}s]")
    print(f"Referencia publicada (fused v4.4): 10.2267")
    print(f"Referencia publicada (numba v43): 10.2358")
    print(f"VRAM pico durante PPL: {vram:.3f} GB (referencia decode puro: 3.955 GB)")
    delta = ppl - 10.2267
    ok = abs(delta) < 0.1
    print(f"VERIFICACIÓN: {'PASS' if ok else 'FAIL'} (delta {delta:+.4f})")
    json.dump({"mode": "fused_v44_v2_standalone", "ppl_wikitext2": ppl, "time_s": dt,
               "vram_gb": vram, "ref_fused": 10.2267, "ref_numba": 10.2358,
               "delta": delta},
              open("/mnt/Data_3TB/project Marla/quant-paper/results/ppl_fused_v44_v2_revalidate.json", "w"),
              indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
