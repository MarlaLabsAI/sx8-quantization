"""winogrande_fused_revalidate_v2.py — Winogrande_s del contenedor 4B v2
(STANDALONE + FUSED). Mismo protocolo que winogrande_sx.py; runtime fused v4.4.
"""
import torch, sys, time, json, gc
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")
from datasets import load_dataset
from integrate_fused_v44 import clone_model_fused_v44
from eval_common import load_model_standalone, DEV
from sx8_container_v43 import read_all_v11

CONTAINER = None
TOK_DIR = "/tmp/opencode/standalone_test/tokenizer"


def score_options(model, tok, prefix, suffix, opt1, opt2):
    opts = [opt1, opt2]
    seqs = []
    for o in opts:
        pids = tok(prefix, add_special_tokens=False).input_ids
        cids = tok(o + suffix, add_special_tokens=False).input_ids
        seqs.append((pids, cids, len(pids)))
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
        if total % 400 == 0:
            print(f"  {total}/{len(ds)}  acc={correct/total:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    acc = correct / total
    print(f"Winogrande_s {label}: {acc:.4f}  ({correct}/{total})", flush=True)
    return acc


def main():
    global CONTAINER
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default=os.environ.get("SX8_CONTAINER", "Qwen3.5-4B-SX8v43.sx8"))
    ap.add_argument("--tokenizer-dir", default=None)
    args = ap.parse_args()
    CONTAINER = args.container
    TOK_DIR = args.tokenizer_dir or os.path.dirname(os.path.abspath(CONTAINER))
    from sx8_container_v43 import read_all_v11
    from integrate_fused_v44 import SX8LinearV44
    wd, bd, meta, cfg, small = read_all_v11(CONTAINER)
    torch.manual_seed(3)
    model, tok, _ = load_model_standalone(CONTAINER, tokenizer_dir=TOK_DIR,
                                          device="cpu", verify=False)
    m_fused, n_rep = clone_model_fused_v44(model, wd, bd, hdr_bytes=6)
    emb_name = "model.language_model.embed_tokens.weight"
    for mn, mod in model.named_modules():
        if mn.endswith("lm_head"):
            pp = mn.rsplit(".", 1)
            parent = model.get_submodule(pp[0]) if len(pp) > 1 else model
            setattr(parent, pp[-1], SX8LinearV44(wd[emb_name], bd[emb_name], mod.bias,
                                                 mod.out_features, mod.in_features, hdr_bytes=6))
            n_rep += 1
            break
    gc.collect()
    m_fused = m_fused.to("cuda")
    torch.cuda.empty_cache()
    print(f"Lineales sustituidos (compacto): {n_rep}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    acc = run_winogrande(m_fused, tok, "SX8 v4.3 v2 FUSED")
    vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"Referencia publicada: 0.5722 | VRAM pico: {vram:.3f} GB")
    json.dump({"mode": "fused_v44_v2_standalone", "winogrande_s_acc": acc,
               "vram_gb": vram, "referencia_v1": 0.5722},
              open("/mnt/Data_3TB/project Marla/quant-paper/results/winogrande_s_v2_fused_revalidate.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
