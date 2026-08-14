"""mc_eval_gguf.py — Evaluación multiple-choice del GGUF Q8_0 (unsloth) vía llama-cpp-python.

FASE 5 batch 2: HellaSwag (0-shot, acc_norm) + ARC-Challenge (0-shot) + MMLU (5-shot).
Mismo protocolo que mc_eval_sx.py (mismos prompts/separadores/datasets de mc_common);
la única diferencia es el motor (llama-cpp-python CUDA, activaciones Q8_1).

Uso:
    python3 mc_eval_gguf.py [--gguf PATH] [--max N] [--tag spot] [--bench NAME]

JSONs: results/mc_{hellaswag,arc,mmlu}_q8_0[_{tag}].json
"""
import sys, time, json, argparse
import numpy as np
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")

import mc_common as mc

GGUF = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-Q8_0.gguf"
OUT = "/mnt/Data_3TB/project Marla/quant-paper/results"


def make_llm(path, n_ctx=4096):
    from llama_cpp import Llama
    return Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=99, logits_all=True, verbose=False)


def score_cont_gguf(llm, prefix, cont):
    """Log-likelihood de `cont` dado `prefix`. add_bos=False para ser consistente
    con el camino transformers (add_special_tokens=False)."""
    pids = llm.tokenize(prefix.encode(), add_bos=False)
    cids = llm.tokenize(cont.encode(), add_bos=False)
    if not cids:
        return -1e18, 0
    llm.reset()  # KV fresca por opción (protocolo)
    ids = pids + cids
    llm.eval(ids)
    logits = llm.scores[: len(ids)].astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    lse = np.log(np.sum(np.exp(logits), axis=1))
    base = len(pids) - 1
    s = 0.0
    for k, tokid in enumerate(cids):
        s += logits[base + k, tokid] - lse[base + k]
    return s, len(cids)


FULL_SIZES = {"hellaswag": 10042, "arc": 1172, "mmlu": 1531}


def save_checkpoint(outpath, name, mode, tag, correct, total, t0):
    res = {"bench": name, "mode": mode, "tag": tag, "acc": round(correct / total, 4),
           "correct": correct, "total": total, "seconds": round(time.time() - t0, 1),
           "partial": True}
    with open(outpath, "w") as f:
        json.dump(res, f, indent=2)


def run_benchmark(llm, name, max_n, tag, mode):
    t0 = time.time()
    suffix = f"_{tag}" if tag else ""
    outpath = f"{OUT}/mc_{name}_{mode}{suffix}.json"
    if name == "hellaswag":
        ds = mc.load_hellaswag()
        correct = total = 0
        for prefix, endings, label in mc.hellaswag_items(ds, max_n):
            scores = []
            for e in endings:
                s, n = score_cont_gguf(llm, prefix, e)
                scores.append((s, n))
            best = max(range(4), key=lambda i: scores[i][0] / max(scores[i][1], 1))
            if best == label:
                correct += 1
            total += 1
            if total % 500 == 0:
                print(f"  hellaswag {total}/{max_n or len(ds)} acc={correct/total:.4f} ({time.time()-t0:.0f}s)", flush=True)
                save_checkpoint(outpath, name, mode, tag, correct, total, t0)
        acc = correct / total
    elif name == "arc":
        ds = mc.load_arc()
        correct = total = 0
        for prefix, labels, answer_key in mc.arc_items(ds, max_n):
            scores = mc.score_letters_gguf(llm, prefix, labels)
            best = max(range(len(labels)), key=lambda i: scores[i])
            if labels[best] == answer_key:
                correct += 1
            total += 1
            if total % 200 == 0:
                print(f"  arc {total}/{max_n or len(ds)} acc={correct/total:.4f} ({time.time()-t0:.0f}s)", flush=True)
                save_checkpoint(outpath, name, mode, tag, correct, total, t0)
        acc = correct / total
    elif name == "mmlu":
        ds, dev_by_subject = mc.load_mmlu()
        correct = total = 0
        for prefix, letters, answer in mc.mmlu_items(ds, dev_by_subject, max_n):
            scores = mc.score_letters_gguf(llm, prefix, letters)
            best = max(range(len(letters)), key=lambda i: scores[i])
            if best == answer:
                correct += 1
            total += 1
            if total % 200 == 0:
                print(f"  mmlu {total}/{max_n or len(ds)} acc={correct/total:.4f} ({time.time()-t0:.0f}s)", flush=True)
                save_checkpoint(outpath, name, mode, tag, correct, total, t0)
        acc = correct / total
    else:
        raise ValueError(name)
    res = {"bench": name, "mode": mode, "tag": tag, "acc": round(acc, 4),
           "correct": correct, "total": total, "seconds": round(time.time() - t0, 1),
           "partial": (max_n is None and total < FULL_SIZES[name])}
    with open(outpath, "w") as f:
        json.dump(res, f, indent=2)
    print(f"RESULTADO {name} [{mode}]: {acc:.4f} ({correct}/{total}) en {res['seconds']}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bench", default=None, choices=["hellaswag", "arc", "mmlu"])
    args = ap.parse_args()

    llm = make_llm(args.gguf)
    print(f"GGUF cargado: {args.gguf.split('/')[-1]}", flush=True)
    benchnames = [args.bench] if args.bench else ["hellaswag", "arc", "mmlu"]
    results = {}
    for b in benchnames:
        results[b] = run_benchmark(llm, b, args.max, args.tag, "q8_0")
    print("TODOS:", results, flush=True)


if __name__ == "__main__":
    main()
