"""ppl_fused_v44.py — PPL wikitext-2 con el modelo fused v4.4 (FASE C).

MISMO protocolo que ppl_wikitext.py (validado en batch 1: 10.2358):
ventanas 512, stride 128, batch 2, padding + attention_mask, KV fresca.
Solo cambia la carga: modelo fused v4.4 (SX8LinearV44) en vez de numba.

Esperado: ~10.2358 (la reformulación PCA + kernel v3 no deben alterar logits
de forma medible; el test de igualdad dio rel ~1.8e-2, lo que se traduce en
PPL casi idéntica).
"""
import torch, math, sys, os, time, json, pickle, gc
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")

from integrate_fused_v44 import clone_model_fused_v44
from eval_common import load_model, DEV
from ppl_wikitext import get_text, run_ppl

PKL = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"


def main():
    d = pickle.load(open(PKL, "rb"))
    wd, bd = d['weights'], d['bases']
    torch.manual_seed(3)
    model, tok, _ = load_model(quantized=False, use_cache=False)
    model.eval()
    m_fused, n_rep = clone_model_fused_v44(model, wd, bd, hdr_bytes=6)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Lineales sustituidos: {n_rep}", flush=True)
    t0 = time.time()
    ppl = run_ppl(m_fused, tok, "FUSED v4.4")
    dt = time.time() - t0
    print(f"\nPPL wikitext-2 (fused v4.4): {ppl:.4f}  [{dt:.0f}s]")
    print(f"Referencia batch 1 (numba v43): 10.2358")
    print(f"Delta: {ppl - 10.2358:+.4f}")
    ok = abs(ppl - 10.2358) < 0.1
    print(f"VERIFICACIÓN CALIDAD: {'PASS' if ok else 'FAIL'}")
    json.dump({"mode": "fused_v44", "ppl_wikitext2": ppl, "time_s": dt,
               "ref_numba": 10.2358, "delta": ppl - 10.2358},
              open("/mnt/Data_3TB/project Marla/quant-paper/results/ppl_fused_v44.json", "w"),
              indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
