"""infer_sx8.py — S-X8 v4.3 interactive chat (standalone container, one command).

Usage:
    python3 infer_sx8.py [--container Qwen3.5-4B-SX8v43.sx8] [options]

Loads the .sx8v43 v1.1 container (COMPLETE model: config + quantized 2D + 1D) and
starts a chat loop with the S-X8 runtime. No base model needed.

Memory (the declared paper numbers):
  - Default --prefill token: prompt AND generation are processed token by token
    (M=1, compact decode1 kernel) -> VRAM stays at the compact level
    (3.720 GB measured for Qwen3.5-4B, less than Q8_0). Prompt runs at decode
    speed (~80 tok/s) instead of batch speed.
  - --prefill batch: the full prompt in one pass (wmma tables, faster prompt
    ~1411 tok/s) -> VRAM ~9 GB for the 4B.
  - If the CUDA extension cannot be built (no nvcc), falls back to the standard
    runtime (weights materialized, ~9.3 GB) — still works.

For the llama.cpp fork path (63.79 tok/s decode, 1877 tok/s prompt, ~4-5 GB):
see the README (build the fork + use the .gguf).
"""
import argparse, os, sys, time, json

import torch
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="S-X8 v4.3 chat (standalone)")
    ap.add_argument("--container", default=os.environ.get("SX8_CONTAINER", "Qwen3.5-4B-SX8v43.sx8"))
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--prefill", choices=["token", "batch"], default="token",
                    help="token: compact (~3.7 GB, prompt a velocidad decode) | batch: wmma (~9 GB, prompt rápido)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repeat-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--ctx", type=int, default=4096, help="max input tokens per question")
    ap.add_argument("--no-thinking", action="store_true", help="disable thinking mode")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "xhigh"], default="medium")
    ap.add_argument("--system", default=None)
    ap.add_argument("--one-shot", default=None, help="ask ONE question and exit (scripting/tests)")
    return ap.parse_args()


def load_engine(args):
    """Returns (model, tok, engine_name). Tries fused (compact) first, then standard."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    tok_dir = args.tokenizer_dir or os.path.dirname(os.path.abspath(args.container)) or here
    from eval_common import load_model_standalone
    from sx8_container_v43 import read_all_v11

    wd, bd, meta, cfg, small = read_all_v11(args.container)

    try:
        from integrate_fused_v44 import clone_model_fused_v44, SX8LinearV44
        model, tok, _ = load_model_standalone(args.container, tokenizer_dir=tok_dir,
                                              device="cpu", verify=False)
        m_fused, n_rep = clone_model_fused_v44(model, wd, bd, hdr_bytes=6)
        emb_name = "model.language_model.embed_tokens.weight"
        for mn, mod in model.named_modules():
            if mn.endswith("lm_head"):
                pp = mn.rsplit(".", 1)
                parent = model.get_submodule(pp[0]) if len(pp) > 1 else model
                setattr(parent, pp[-1], SX8LinearV44(wd[emb_name], bd[emb_name], mod.bias,
                                                     mod.out_features, mod.in_features,
                                                     hdr_bytes=6))
                n_rep += 1
                break
        import gc
        gc.collect()
        model = model.to("cuda")
        torch.cuda.empty_cache()
        engine = "fused-compact" if args.prefill == "token" else "fused-batch"
        print(f"[S-X8] Engine: fused v4.4 ({n_rep} lineales compactos) | prefill: {args.prefill}",
              flush=True)
        return model, tok, engine
    except Exception as e:
        print(f"[S-X8] Fused runtime no disponible ({str(e)[:100]}) — usando runtime estándar "
              f"(pesos materializados, ~9.3 GB).", flush=True)
        model, tok, _ = load_model_standalone(args.container, tokenizer_dir=tok_dir,
                                              device="cuda", verify=False)
        return model, tok, "standard"


def sample_token(logits, temp, top_p, repeat_penalty, gen_ids, eos_id):
    logits = logits.float()
    if repeat_penalty != 1.0 and gen_ids:
        for g in gen_ids[-64:]:
            logits[g] /= repeat_penalty
    if temp <= 0:
        return int(logits.argmax())
    logits = logits / temp
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = cum <= top_p
        keep[0] = True
        probs = probs.clone()
        probs[~keep.scatter(0, sorted_i, keep)] = 0.0
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1).item())


def run_question(model, tok, messages, args, engine):
    gen_kwargs = {}
    if not args.no_thinking:
        gen_kwargs["enable_thinking"] = True
        gen_kwargs["reasoning_effort"] = args.reasoning_effort
    else:
        gen_kwargs["enable_thinking"] = False

    enc = tok.apply_chat_template(messages, tokenize=True,
                                  add_generation_prompt=True,
                                  return_tensors="pt", **gen_kwargs)
    prompt_ids = enc["input_ids"].to("cuda")
    prompt_ids = prompt_ids[:, -args.ctx:] if prompt_ids.shape[1] > args.ctx else prompt_ids
    eos_id = tok.eos_token_id or 248044

    t0 = time.time()
    if engine == "fused-compact":
        cache = None
        n_in = prompt_ids.shape[1]
        with torch.no_grad():
            for i in range(n_in):
                out = model(input_ids=prompt_ids[:, i:i+1], past_key_values=cache,
                            use_cache=True,
                            attention_mask=torch.ones(1, i+1, dtype=torch.long, device="cuda"))
                cache = out.past_key_values
                del out
            gen_ids = []
            n_gen = 0
            while n_gen < args.max_new_tokens:
                cur = torch.tensor([[gen_ids[-1]]] if gen_ids else prompt_ids[:, -1:],
                                   dtype=torch.long, device="cuda")
                out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                            attention_mask=torch.ones(1, n_in + n_gen, dtype=torch.long,
                                                      device="cuda"))
                cache = out.past_key_values
                logits = out.logits[0, -1]
                del out
                tok_id = sample_token(logits, args.temp, args.top_p,
                                      args.repeat_penalty, gen_ids, eos_id)
                gen_ids.append(tok_id)
                n_gen += 1
                if tok_id == eos_id:
                    break
                if n_gen % 20 == 0:
                    torch.cuda.empty_cache()
        dt = time.time() - t0
        out_text = tok.decode(gen_ids, skip_special_tokens=True)
        tok_s = n_gen / max(dt, 1e-6)
    else:
        gen_cfg = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temp > 0,
                   "temperature": args.temp, "top_p": args.top_p,
                   "repetition_penalty": args.repeat_penalty, "eos_token_id": eos_id}
        if args.temp <= 0:
            gen_cfg["do_sample"] = False
        with torch.no_grad():
            gen = model.generate(prompt_ids, **gen_cfg)
        dt = time.time() - t0
        n_gen = gen.shape[1] - prompt_ids.shape[1]
        out_text = tok.decode(gen[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        tok_s = n_gen / max(dt, 1e-6)

    vram = torch.cuda.max_memory_allocated() / 2**30
    return out_text, tok_s, vram


def main():
    args = parse_args()
    if not os.path.exists(args.container):
        print(f"[S-X8] ERROR: contenedor no encontrado: {args.container}\n"
              f"  Descárgalo de https://huggingface.co/marlalabsAI/Qwen3.5-4B-SX8 "
              f"(Qwen3.5-4B-SX8v43.sx8) o pasa --container <ruta>.")
        sys.exit(1)

    model, tok, engine = load_engine(args)
    n_params = sum(v.numel() for v in model.parameters()) / 1e9
    print(f"[S-X8] Modelo listo: {n_params:.2f}B params | engine: {engine} | "
          f"VRAM inicial: {torch.cuda.memory_allocated()/2**30:.2f} GB", flush=True)
    print("[S-X8] Escribe tu pregunta (exit/bye para salir).", flush=True)

    sys_msg = args.system or ("You are a helpful AI assistant. Answer concisely and correctly.")
    if args.one_shot is not None:
        questions = [args.one_shot]
    else:
        questions = iter([])

    while True:
        if args.one_shot is not None:
            q = questions[0] if isinstance(questions, list) else None
            if q is None:
                break
        else:
            try:
                q = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q.lower() in ("exit", "bye", "quit", "salir"):
                break
        messages = [{"role": "system", "content": sys_msg},
                    {"role": "user", "content": q}]
        torch.cuda.reset_peak_memory_stats()
        out_text, tok_s, vram = run_question(model, tok, messages, args, engine)
        print(f"\nS-X8> {out_text}\n", flush=True)
        print(f"  [~{tok_s:.1f} tok/s · VRAM pico {vram:.2f} GB]", flush=True)
        if args.one_shot is not None:
            print(json.dumps({"answer": out_text, "tok_s": round(tok_s, 1),
                              "vram_gb": round(vram, 2), "engine": engine}))
            break


if __name__ == "__main__":
    main()
