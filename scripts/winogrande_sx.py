"""Winogrande (winogrande_s test, 0-shot, continuation log-likelihood) — FP16 y SX8 v4"""
import torch, sys, time, json
from datasets import load_dataset
from eval_common import load_model, DEV

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
    ds = load_dataset("allenai/winogrande", "winogrande_s", split="validation")
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "sx8"
    model, tok, meta = load_model(quantized=(mode != "fp16"), mode=("v43" if mode == "sx8v43" else "v42" if mode == "sx8v42" else "v4"))
    acc = run_winogrande(model, tok, f"SX8 v4" if mode != "fp16" else "FP16")
    json.dump({"mode": mode, "winogrande_s_acc": acc}, open("/mnt/Data_3TB/project Marla/quant-paper/results/winogrande_s.json", "w"), indent=2)
