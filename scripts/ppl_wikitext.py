"""PPL wikitext-2-raw test (completo) — FP16 y SX8 v4
Protocolo estándar comunitario (llama.cpp / HF tutorial / Unsloth):
  - wikitext-2 test, ventanas de 512 tokens de contexto ("Wiki-test with 512 context windows" — Unsloth)
  - stride 128 (misma convención HF tutorial), se puntúan los últimos 128 tokens de cada ventana
  - cada token puntuado tiene >=384 tokens de contexto previo (equivalente al sliding window de llama.cpp -c 512)
  - sin chat template, sin thinking (Qwen3.5 Small: disabled por defecto), greedy (no sampling)
  - cache de KV fresca por ventana (sin reuso entre muestras)
"""
import torch, math, sys, os, time, json
from datasets import load_dataset
from eval_common import load_model, DEV

CTX = 512      # contexto estándar comunitario
STRIDE = 128   # conteo por ventana (convención HF tutorial)
BATCH = 2      # vocab 248K → logits enormes; batch bajo para caber en 16GB

def get_text():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n".join(ds["text"])

def run_ppl(model, tok, label):
    text = get_text()
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel()
    print(f"{label}: {n} tokens wikitext-2 test", flush=True)
    nll_sum = 0.0; nll_cnt = 0
    windows = []
    for i in range(0, n, STRIDE):
        begin = max(i, 0)
        end = min(i + CTX, n)
        windows.append((begin, end))
    with torch.no_grad():
        for bi in range(0, len(windows), BATCH):
            batch = windows[bi:bi+BATCH]
            if len(batch) < BATCH:
                # último batch impar: duplicar la última ventana en el slot vacío
                # (evita filas de solo padding que el fused expone como NaN)
                batch = batch + [batch[-1]]
            seqs = [ids[s:e] for s, e in batch]
            pad_len = max(len(s) for s in seqs)
            inp = torch.full((len(seqs), pad_len), tok.pad_token_id if tok.pad_token_id is not None else 0, dtype=torch.long)
            for j, s in enumerate(seqs):
                inp[j, :len(s)] = s
            inp = inp.to(DEV)
            attn = inp.ne(tok.pad_token_id if tok.pad_token_id is not None else 0).long()
            out = model(input_ids=inp, attention_mask=attn)
            logits = out.logits.float().cpu()   # mover a CPU inmediatamente
            del out, inp, attn
            torch.cuda.empty_cache()
            for j, (s, e) in enumerate(batch):
                L = e - s
                n_score = min(L - 1, STRIDE)      # última ventana corta → puntúa lo que haya
                if n_score <= 0:
                    continue
                cs = L - n_score
                toks = ids[s + cs:e]
                lg = logits[j, cs - 1:L - 1]
                nll = torch.nn.functional.cross_entropy(lg, toks, reduction="sum")
                nll_sum += nll.item()
                nll_cnt += n_score
            del logits
            if (bi // BATCH) % 25 == 0:
                print(f"  ventanas {bi+BATCH}/{len(windows)} | nll={nll_sum:.1f} tok={nll_cnt}", flush=True)
    ppl = math.exp(nll_sum / max(nll_cnt, 1))
    print(f"PPL {label}: {ppl:.4f}  (tokens contados: {nll_cnt})", flush=True)
    return ppl

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sx8"
    t0 = time.time()
    model, tok, meta = load_model(quantized=(mode != "fp16"), mode=("v43" if mode == "sx8v43" else "v42" if mode == "sx8v42" else "v4"))
    label = "SX8 v4 (new)" if mode != "fp16" else "FP16"
    ppl = run_ppl(model, tok, label)
    out = {"mode": mode, "label": label, "ppl_wikitext2": ppl, "time_s": time.time() - t0}
    json.dump(out, open(f"/mnt/Data_3TB/project Marla/quant-paper/results/ppl_{mode}.json", "w"), indent=2)
    print(f"TOTAL {time.time()-t0:.0f}s → results/ppl_{mode}.json")
