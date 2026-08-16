"""winogrande_revalidate_v2.py — Winogrande_s del contenedor 4B v2 (STANDALONE).

Mismo protocolo que winogrande_sx.py (winogrande_s validation, 0-shot,
continuation log-likelihood). Carga SOLO desde el contenedor v1.1.
"""
import torch, sys, time, json
from datasets import load_dataset
from eval_common import load_model_standalone, DEV

CONTAINER = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-SX8v43-v2.sx8"
TOK_DIR = "/tmp/opencode/standalone_test/tokenizer"


def score_options(model, tok, prefix, suffix, opt1, opt2):
    opts = [opt1, opt2]
    maxlen = 0
    seqs = []
    for o in opts:
        pids = tok(prefix, add_special_tokens=False).input_ids
        cids = tok(o + suffix, add_special_tokens=False).input_ids
        seqs.append((pids, cids, len(pids)))
        maxlen = max(maxlen, len(pids) + len(cids))
    scores = []
    with torch.no_grad():
        for pids, cids, lp in seqs:
            full = torch.tensor([pids + cids], dtype=torch.long, device=DEV)
            out = model(full)
            logp = torch.log_softmax(out.logits.float()[0, :-1], dim=-1)
            c = torch.tensor(cids, dtype=torch.long, device=DEV)
            s = logp[lp-1:lp-1+len(c)].gather(-1, c.unsqueeze(-1)).sum().item()
            scores.append(s)
            del full, out, logp
    return 1 if scores[0] >= scores[1] else 2


def run_winogrande(model, tok, label):
    ds = load_dataset("allenai/winogrande", "winogrande_s", split="validation",
                      cache_dir="/mnt/Data_3TB/hf_datasets_cache")
    correct = 0; total = 0
    t0 = time.time()
    for ex in ds:
        sent = ex["sentence"]
        parts = sent.split("_")
        prefix = parts[0]; suffix = parts[1] if len(parts) > 1 else ""
        pred = score_options(model, tok, prefix, suffix, ex["option1"], ex["option2"])
        if pred == int(ex["answer"]):
            correct += 1
        total += 1
        if total % 500 == 0:
            print(f"  {total}/{len(ds)}  acc={correct/total:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    acc = correct / total
    print(f"Winogrande_s {label}: {acc:.4f}  ({correct}/{total})", flush=True)
    return acc


if __name__ == "__main__":
    model, tok, meta = load_model_standalone(CONTAINER, tokenizer_dir=TOK_DIR, verify=False)
    acc = run_winogrande(model, tok, "SX8 v4.3 v2 STANDALONE")
    out = "/mnt/Data_3TB/project Marla/quant-paper/results/winogrande_s_v2_revalidate.json"
    json.dump({"mode": "sx8v43_v2_standalone", "winogrande_s_acc": acc,
               "referencia_v1": 0.5722}, open(out, "w"), indent=2)
    print(f"Guardado: {out}")
