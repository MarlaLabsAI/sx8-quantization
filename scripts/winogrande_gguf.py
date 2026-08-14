"""Winogrande_s sobre GGUF vía llama-cpp-python (CUDA) — continuation log-likelihood"""
import sys, time, json
import numpy as np
from datasets import load_dataset
from llama_cpp import Llama

GGUF = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-Q8_0.gguf"

def make_llm(path, n_gpu=99):
    return Llama(model_path=path, n_ctx=2048, n_gpu_layers=n_gpu, logits_all=True, verbose=False)

def score_option(llm, prefix, cont):
    pids = llm.tokenize(prefix.encode())
    cids = llm.tokenize(cont.encode())
    llm.reset()                       # KV cache fresca por muestra (requisito del protocolo)
    ids = pids + cids
    llm.eval(ids)
    logits = llm.scores[:len(ids)]  # (n_tokens, n_vocab) con logits_all
    logits = logits.astype(np.float64)
    lse = np.log(np.sum(np.exp(logits - logits.max(axis=1, keepdims=True)), axis=1)) + logits.max(axis=1)
    base = len(pids) - 1
    s = 0.0
    for k, tok in enumerate(cids):
        s += logits[base + k, tok] - lse[base + k]
    return s

def run(path, label):
    llm = make_llm(path)
    ds = load_dataset("allenai/winogrande", "winogrande_s", split="validation")
    correct = 0; t0 = time.time()
    for i, ex in enumerate(ds):
        sent = ex["sentence"]
        parts = sent.split("_")
        prefix = parts[0]; suffix = parts[1] if len(parts) > 1 else ""
        s1 = score_option(llm, prefix, ex["option1"] + suffix)
        s2 = score_option(llm, prefix, ex["option2"] + suffix)
        pred = 1 if s1 >= s2 else 2
        if pred == int(ex["answer"]):
            correct += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ds)} acc={correct/(i+1):.4f} ({time.time()-t0:.0f}s)", flush=True)
    acc = correct / len(ds)
    print(f"Winogrande_s GGUF {label}: {acc:.4f} ({correct}/{len(ds)})", flush=True)
    return acc

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else GGUF
    label = sys.argv[2] if len(sys.argv) > 2 else "Q8_0"
    acc = run(path, label)
    json.dump({"mode": "gguf", "label": label, "winogrande_s_acc": acc},
              open("/mnt/Data_3TB/project Marla/quant-paper/results/winogrande_gguf.json", "w"), indent=2)
