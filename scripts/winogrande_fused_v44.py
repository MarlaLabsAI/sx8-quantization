"""winogrande_fused_v44.py — Winogrande_s con el modelo fused v4.4 (FASE C2).

Mismo protocolo que winogrande_sx.py (validation, 1267 muestras, log-likelihood).
Esperado: ~0.5722 (batch 1 con numba v43).
"""
import sys, time, json, pickle, gc
import torch
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")

from integrate_fused_v44 import clone_model_fused_v44
from eval_common import load_model, DEV
from winogrande_sx import score_options, run_winogrande
from datasets import load_dataset

PKL = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"


def main():
    d = pickle.load(open(PKL, "rb"))
    wd, bd = d['weights'], d['bases']
    torch.manual_seed(5)
    model, tok, _ = load_model(quantized=False, use_cache=False)
    model.eval()
    m_fused, n_rep = clone_model_fused_v44(model, wd, bd, hdr_bytes=6)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Lineales sustituidos: {n_rep}", flush=True)

    # spot-check del score_options en 2 muestras (rápido, sin dataset completo)
    from itertools import islice
    ds = load_dataset("allenai/winogrande", "winogrande_s", split="validation")
    for ex in islice(ds, 2):
        parts = ex["sentence"].split("_")
        pred = score_options(m_fused, tok, parts[0], parts[1] if len(parts) > 1 else "",
                             ex["option1"], ex["option2"])
        print(f"  spot pred={pred} answer={ex['answer']}", flush=True)

    t0 = time.time()
    acc = run_winogrande(m_fused, tok, "FUSED v4.4")
    print(f"\nWinogrande_s (fused v4.4): {acc:.4f}  [{time.time()-t0:.0f}s]")
    print(f"Referencia batch 1 (numba v43): 0.5722")
    json.dump({"mode": "fused_v44", "winogrande_s_acc": acc,
               "ref_numba": 0.5722},
              open("/mnt/Data_3TB/project Marla/quant-paper/results/winogrande_fused_v44.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
