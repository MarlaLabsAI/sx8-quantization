"""ppl_revalidate_v2.py — PPL wikitext-2 del contenedor 4B v2 (STANDALONE).

MISMO protocolo que ppl_wikitext.py (validado en batch 1: 10.2358):
ventanas 512, stride 128, batch 2, padding + attention_mask, KV fresca.
Carga SOLO desde el contenedor v1.1 (sin modelo base).
"""
import torch, math, sys, os, time, json
from datasets import load_dataset
from eval_common import load_model_standalone, DEV

CTX = 512
STRIDE = 128
BATCH = 2
CONTAINER = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-SX8v43-v2.sx8"
TOK_DIR = "/tmp/opencode/standalone_test/tokenizer"


def get_text():
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                      cache_dir="/mnt/Data_3TB/hf_datasets_cache")
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
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, len(windows), BATCH):
            batch = windows[bi:bi+BATCH]
            if len(batch) < BATCH:
                batch = batch + [batch[-1]]
            seqs = [ids[s:e] for s, e in batch]
            pad_len = max(len(s) for s in seqs)
            inp = torch.full((len(seqs), pad_len),
                             tok.pad_token_id if tok.pad_token_id is not None else 0,
                             dtype=torch.long)
            for j, s in enumerate(seqs):
                inp[j, :len(s)] = s
            inp = inp.to(DEV)
            attn = inp.ne(tok.pad_token_id if tok.pad_token_id is not None else 0).long()
            out = model(input_ids=inp, attention_mask=attn)
            logits = out.logits.float().cpu()
            del out, inp, attn
            torch.cuda.empty_cache()
            for j, (s, e) in enumerate(batch):
                sl = e - s - 1
                lg = logits[j, :sl].log_softmax(-1)
                tgt = ids[s+1:e]
                nll = -lg.gather(1, tgt.unsqueeze(1)).sum().item()
                nll_sum += nll
                nll_cnt += sl
            if (bi // BATCH) % 40 == 0:
                print(f"  ventana {bi//BATCH}/{len(windows)//BATCH} ({time.time()-t0:.0f}s)", flush=True)
    ppl = math.exp(nll_sum / nll_cnt)
    return ppl, time.time() - t0


def main():
    model, tok, meta = load_model_standalone(CONTAINER, tokenizer_dir=TOK_DIR, verify=False)
    ppl, dt = run_ppl(model, tok, "SX8 v4.3 v2 STANDALONE")
    print(f"\nPPL wikitext-2 (v2 standalone): {ppl:.4f}  [{dt:.0f}s]")
    print(f"Referencia publicada (v1, mismo protocolo): 10.2358")
    out = "/mnt/Data_3TB/project Marla/quant-paper/results/ppl_sx8v43_v2_revalidate.json"
    json.dump({"mode": "sx8v43_v2_standalone", "ppl_wikitext2": ppl, "time_s": dt,
               "referencia_v1": 10.2358}, open(out, "w"), indent=2)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
