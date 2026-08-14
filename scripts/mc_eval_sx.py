"""mc_eval_sx.py — Evaluación multiple-choice FP16 y S-X8 v4.3 (fused v4.4, con PCA).

FASE 5 batch 2: HellaSwag (0-shot, acc_norm) + ARC-Challenge (0-shot) + MMLU (5-shot).
Camino de calidad del paper: el modelo SX8 se carga con el kernel fused v4.4
(kernel v3 + PCA) — el mismo que validó PPL 10.2267 y Winogrande 0.5722.

Uso:
    python3 mc_eval_sx.py --mode fp16 [--max N] [--tag spot]
    python3 mc_eval_sx.py --mode sx8v43 [--max N] [--tag spot]

JSONs: results/mc_{hellaswag,arc,mmlu}_{mode}[_{tag}].json
"""
import sys, time, json, gc, argparse, pickle
import torch
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")

import mc_common as mc
from eval_common import load_model, DEV

PKL_V43 = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
OUT = "/mnt/Data_3TB/project Marla/quant-paper/results"


def load_for_mode(mode):
    if mode == "fp16":
        model, tok, _ = load_model(quantized=False, use_cache=False)
        return model, tok
    if mode == "sx8v43":
        d = pickle.load(open(PKL_V43, "rb"))
        wd, bd = d["weights"], d["bases"]
        model, tok, _ = load_model(quantized=False, use_cache=False)
        from integrate_fused_v44 import clone_model_fused_v44
        m_fused, n_rep = clone_model_fused_v44(model, wd, bd, hdr_bytes=6)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"SX8 fused v4.4: {n_rep} lineales sustituidos", flush=True)
        return m_fused, tok
    raise ValueError(mode)


FULL_SIZES = {"hellaswag": 10042, "arc": 1172, "mmlu": 1531}


def save_checkpoint(outpath, name, mode, tag, correct, total, t0):
    res = {"bench": name, "mode": mode, "tag": tag, "acc": round(correct / total, 4),
           "correct": correct, "total": total, "seconds": round(time.time() - t0, 1),
           "partial": True}
    with open(outpath, "w") as f:
        json.dump(res, f, indent=2)


def run_benchmark(model, tok, name, max_n, tag, mode):
    t0 = time.time()
    suffix = f"_{tag}" if tag else ""
    outpath = f"{OUT}/mc_{name}_{mode}{suffix}.json"
    if name == "hellaswag":
        ds = mc.load_hellaswag()
        correct = total = 0
        for prefix, endings, label in mc.hellaswag_items(ds, max_n):
            scores = mc.best_choice_pt(model, tok, prefix, endings)
            # acc_norm: normalizar por longitud de la continuación
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
            scores = mc.score_letters_pt(model, tok, prefix, labels)
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
            scores = mc.score_letters_pt(model, tok, prefix, letters)
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
    ap.add_argument("--mode", default="fp16", choices=["fp16", "sx8v43"])
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--bench", default=None, choices=["hellaswag", "arc", "mmlu"])
    args = ap.parse_args()

    torch.manual_seed(5)
    model, tok = load_for_mode(args.mode)
    model.eval()
    print(f"Modelo {args.mode} cargado", flush=True)

    benchnames = [args.bench] if args.bench else ["hellaswag", "arc", "mmlu"]
    results = {}
    for b in benchnames:
        results[b] = run_benchmark(model, tok, b, args.max, args.tag, args.mode)
    print("TODOS:", results, flush=True)


if __name__ == "__main__":
    main()
