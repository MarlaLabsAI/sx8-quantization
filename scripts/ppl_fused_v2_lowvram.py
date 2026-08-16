"""ppl_fused_v2_lowvram.py — PPL wikitext-2 del contenedor 4B v2 (STANDALONE + FUSED,
VRAM BAJA). BATCH=1 → M=1 → kernel decode1_v44 (compacto on-the-fly, ~4.4 GB).

Mismo protocolo que ppl_wikitext.py (CTX 512, STRIDE 128, scoring cross-entropy
sobre los últimos min(L-1, STRIDE) tokens de cada ventana). Mide VRAM pico.
"""
import torch, math, sys, os, time, json, gc
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")

from datasets import load_dataset
from integrate_fused_v44 import clone_model_fused_v44
from eval_common import load_model_standalone, DEV
from ppl_wikitext import get_text

CTX = 512
STRIDE = 128
BATCH = 1
CONTAINER = "/mnt/Data_3TB/project Marla/quant-paper/models/Qwen3.5-4B-SX8v43-v2.sx8"
TOK_DIR = "/tmp/opencode/standalone_test/tokenizer"


def run_ppl_b1(model, tok, label):
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
            s, e = batch[0]
            seqs = [ids[s:e]]
            inp = seqs[0].unsqueeze(0).to(DEV)
            attn = torch.ones_like(inp)
            out = model(input_ids=inp, attention_mask=attn)
            logits = out.logits.float().cpu()
            del out, inp, attn
            torch.cuda.empty_cache()
            L = e - s
            n_score = min(L - 1, STRIDE)
            if n_score > 0:
                cs = L - n_score
                toks = ids[s + cs:e]
                lg = logits[0, cs - 1:L - 1]
                nll = torch.nn.functional.cross_entropy(lg, toks, reduction="sum")
                nll_sum += nll.item()
                nll_cnt += n_score
            del logits
            if (bi // BATCH) % 25 == 0:
                print(f"  ventanas {bi+BATCH}/{len(windows)} | nll={nll_sum:.1f} tok={nll_cnt} "
                      f"| VRAM={torch.cuda.memory_allocated()/2**30:.2f} GB ({time.time()-t0:.0f}s)", flush=True)
    ppl = math.exp(nll_sum / max(nll_cnt, 1))
    print(f"PPL {label}: {ppl:.4f}  (tokens contados: {nll_cnt})", flush=True)
    return ppl, time.time() - t0


def main():
    from sx8_container_v43 import read_all_v11
    from integrate_fused_v44 import SX8LinearV44
    wd, bd, meta, cfg, small = read_all_v11(CONTAINER)
    torch.manual_seed(3)
    model, tok, _ = load_model_standalone(CONTAINER, tokenizer_dir=TOK_DIR,
                                          device="cpu", verify=False)
    m_fused, n_rep = clone_model_fused_v44(model, wd, bd, hdr_bytes=6)
    emb_name = "model.language_model.embed_tokens.weight"
    lm = None
    for mn, mod in model.named_modules():
        if mn.endswith("lm_head"):
            lm = mod
            break
    if lm is not None and emb_name in wd:
        qt_e, bi_e = wd[emb_name], bd[emb_name]
        new_lin = SX8LinearV44(qt_e, bi_e, lm.bias, lm.out_features,
                               lm.in_features, hdr_bytes=6)
        parent = None
        parts = [p for p in (lm_name for lm_name in ()) ]
        # reemplazar en el padre correcto
        for mn, mod in model.named_modules():
            if mn.endswith("lm_head"):
                pp = mn.rsplit(".", 1)
                parent = model.get_submodule(pp[0]) if len(pp) > 1 else model
                setattr(parent, pp[-1], new_lin)
                n_rep += 1
                break
        gc.collect()
        print(f"lm_head sustituido (atado a embed) → total sustituidos: {n_rep}", flush=True)
    gc.collect()
    m_fused = m_fused.to("cuda")
    torch.cuda.empty_cache()
    print(f"Lineales sustituidos (compacto): {n_rep}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ppl, dt = run_ppl_b1(m_fused, tok, "FUSED v4.4 STANDALONE v2 (M=1, VRAM baja)")
    dt = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"\nPPL wikitext-2 (fused v4.4 standalone, M=1): {ppl:.4f}  [{dt:.0f}s]")
    print(f"Referencia publicada (fused v4.4): 10.2267 | (numba v43): 10.2358")
    print(f"VRAM PICO: {vram:.3f} GB (referencia kernel decode1 puro: 3.955 GB)")
    delta = ppl - 10.2267
    ok = abs(delta) < 0.1
    print(f"VERIFICACIÓN: {'PASS' if ok else 'FAIL'} (delta {delta:+.4f})")
    json.dump({"mode": "fused_v44_v2_standalone_m1", "ppl_wikitext2": ppl, "time_s": dt,
               "vram_gb": vram, "ref_fused": 10.2267, "ref_numba": 10.2358,
               "delta": delta},
              open("/mnt/Data_3TB/project Marla/quant-paper/results/ppl_fused_v44_v2_revalidate_m1.json", "w"),
              indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
