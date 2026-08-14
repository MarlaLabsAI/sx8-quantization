"""PPL wikitext-2 test sobre GGUF vía llama-cpp-python (CUDA)
MISMO protocolo que ppl_wikitext.py (comparable con FP16/SX8):
  - ventanas de 512 tokens de contexto, stride 128, se puntúan los últimos 128
  - KV cache reseteada entre ventanas (llm.reset())
  - greedy, sin chat template, sin thinking
"""
import sys, time, math, json
import numpy as np
from datasets import load_dataset
from llama_cpp import Llama

CTX = 512
STRIDE = 128

def make_llm(path, n_gpu=99):
    return Llama(model_path=path, n_ctx=4096, n_gpu_layers=n_gpu,
                 logits_all=True, verbose=False, use_mlock=False)

def window_ppl(llm, ids):
    """Puntúa ventanas de 512: nll de los últimos 128 tokens de cada ventana (última corta incluida)"""
    nll = 0.0; cnt = 0
    for s in range(0, len(ids) - 1, STRIDE):
        e = min(s + CTX, len(ids))
        L = e - s
        n_score = min(L - 1, STRIDE)
        if n_score <= 0:
            break
        chunk = ids[s:e]
        llm.reset()
        llm.eval(chunk)
        logits = llm.scores[:len(chunk)]      # (len, vocab) float32
        cs = L - n_score
        lg = logits[cs - 1:L - 1]
        toks = np.array(chunk[cs:], dtype=np.int64)
        lse = np.log(np.sum(np.exp(lg - lg.max(axis=1, keepdims=True)), axis=1)) + lg.max(axis=1)
        nll += float(np.sum(lg[np.arange(n_score), toks] - lse))
        cnt += n_score
    return nll, cnt

def run(path, label):
    llm = make_llm(path)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(ds["text"])
    ids = llm.tokenize(text.encode())
    print(f"{label}: {len(ids)} tokens wikitext-2 test", flush=True)
    t0 = time.time()
    nll, cnt = window_ppl(llm, ids)
    ppl = math.exp(nll / max(cnt, 1))
    print(f"PPL {label}: {ppl:.4f}  (tokens contados: {cnt}, {time.time()-t0:.0f}s)", flush=True)
    return ppl

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-Q8_0.gguf"
    label = sys.argv[2] if len(sys.argv) > 2 else "Q8_0 (llama-cpp)"
    ppl = run(path, label)
    json.dump({"mode": "gguf_llamacpp", "label": label, "ppl_wikitext2": ppl},
              open("/mnt/Data_3TB/project Marla/quant-paper/results/ppl_gguf.json", "w"), indent=2)
