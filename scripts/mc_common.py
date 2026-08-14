"""mc_common.py — Módulo compartido de evaluación multiple-choice (FASE 5 batch 2).

Prompts, separadores y scoring IDÉNTICOS para los dos motores:
  - camino transformers (FP16 y S-X8 v4.3 fused v4.4): mc_eval_sx.py
  - camino llama-cpp-python (GGUF Q8_0 unsloth):           mc_eval_gguf.py

Protocolo (PROTOCOLO.md, batch 1):
  - scoring por log-likelihood de la continuación (greedy, temp 0 = determinista)
  - KV cache fresca por muestra/opción
  - sin chat template (scoring puro)
  - splits: HellaSwag validation (10.042) · ARC-Challenge TEST (1.172, el estándar;
    validation solo tiene 299) · MMLU "all" validation (1.531) + dev (285) para 5-shot

Regla de consistencia CRÍTICA: ambas rutas reciben EXACTAMENTE las mismas cadenas
(prefix, continuation) y concatenan los ids de tokens separados (patrón winogrande).
Nunca tokenizar prefix+cont como un solo string (rompería la consistencia entre
motores si el tokenizador difiere).
"""
import torch
from datasets import load_dataset

LETTERS = "ABCD"

# ---------------------------------------------------------------------------
# Cargadores de datasets (offline, desde caché de HF)
# ---------------------------------------------------------------------------
def load_hellaswag():
    return load_dataset("Rowan/hellaswag", split="validation")

def load_arc():
    return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

def load_mmlu():
    val = load_dataset("cais/mmlu", "all", split="validation")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    dev_by_subject = {}
    for ex in dev:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)
    return val, dev_by_subject

# ---------------------------------------------------------------------------
# Separador de HellaSwag: el ctx termina sin espacio y el ending empieza sin
# espacio ("...roof. he" + "is using...") → añadir " " (igual para los 3 formatos).
# ---------------------------------------------------------------------------
def join_ctx_ending(ctx, ending):
    if ctx.endswith(" ") or ending.startswith(" "):
        return ctx, ending
    return ctx + " ", ending

# ---------------------------------------------------------------------------
# Prompts (estilo lm-eval)
# ---------------------------------------------------------------------------
ARC_TEMPLATE = "Question: {q}\nChoices:\nA. {c0}\nB. {c1}\nC. {c2}\nD. {c3}\nAnswer:"
MMLU_TEMPLATE = "{q}\nA. {c0}\nB. {c1}\nC. {c2}\nD. {c3}\nAnswer:"
MMLU_INTRO = "The following are multiple choice questions (with answers) about {subject}.\n\n"

def arc_prompt(ex):
    """ARC tiene 3 o 4 opciones según la muestra — mapear por la etiqueta real."""
    texts = ex["choices"]["text"]
    labels = ex["choices"]["label"]
    lines = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return f"Question: {ex['question']}\nChoices:\n{lines}\nAnswer:"

def mmlu_shot_block(ex):
    c = ex["choices"]
    return MMLU_TEMPLATE.format(q=ex["question"], c0=c[0], c1=c[1], c2=c[2], c3=c[3])

def mmlu_prompt(q, choices, subject, dev_examples, n_shots=5):
    s = MMLU_INTRO.format(subject=subject)
    for d in dev_examples[:n_shots]:
        s += mmlu_shot_block(d) + " " + LETTERS[int(d["answer"])] + "\n\n"
    s += MMLU_TEMPLATE.format(q=q, c0=choices[0], c1=choices[1], c2=choices[2], c3=choices[3])
    return s

# ---------------------------------------------------------------------------
# Scoring compartido — camino transformers (patrón winogrande_sx.score_options)
# ---------------------------------------------------------------------------
def score_cont_pt(model, tok, prefix, cont, device="cuda"):
    """Log-likelihood (suma de log-probs) de `cont` dado `prefix`, sin chat template."""
    pids = tok(prefix, add_special_tokens=False).input_ids
    cids = tok(cont, add_special_tokens=False).input_ids
    if not cids:
        return -1e18
    full = torch.tensor([pids + cids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(full)
        logp = torch.log_softmax(out.logits.float()[0, :-1], dim=-1)
        c = torch.tensor(cids, dtype=torch.long, device=device)
        s = logp[len(pids) - 1: len(pids) - 1 + len(c)].gather(-1, c.unsqueeze(-1)).sum().item()
    return s

def best_choice_pt(model, tok, prefix, conts):
    """Devuelve (index_mejor, scores). Para acc_norm (HellaSwag) normalizar por len(cont)."""
    scores = []
    for c in conts:
        cids = tok(c, add_special_tokens=False).input_ids
        s = score_cont_pt(model, tok, prefix, c)
        scores.append((s, len(cids)))
    return scores


def score_letters_pt(model, tok, prefix, letters):
    """Log-probs de letras 1-token (ARC/MMLU) con UN SOLO forward del prefijo.

    La logprob de la letra se lee en la última fila de logits del prefijo:
    causalmente no depende del token siguiente, por lo que es matemáticamente
    IDÉNTICO a 4 forwards (prefix+letra) — verificado diff 0.0 (test_gguf_kv).
    Coste: 1 forward por muestra en vez de 4 (4× más rápido en MMLU/ARC).
    """
    pids = tok(prefix, add_special_tokens=False).input_ids
    full = torch.tensor([pids], dtype=torch.long, device="cuda")
    with torch.no_grad():
        out = model(full)
        logp = torch.log_softmax(out.logits.float()[0, -1], dim=-1)
    out_scores = []
    for l in letters:
        cids = tok(l, add_special_tokens=False).input_ids
        out_scores.append(logp[cids[0]].item() if cids else -1e18)
    return out_scores


def score_letters_gguf(llm, prefix, letters):
    """Ídem para llama-cpp-python: eval(prefix) una vez + leer la última fila."""
    import numpy as np
    pids = llm.tokenize(prefix.encode(), add_bos=False)
    llm.reset()
    llm.eval(pids)
    logits = llm.scores[: len(pids)].astype(np.float64)
    row = logits[len(pids) - 1]
    row -= row.max()
    lse = np.log(np.sum(np.exp(row)))
    out_scores = []
    for l in letters:
        cids = llm.tokenize(l.encode(), add_bos=False)
        out_scores.append((row[cids[0]] - lse) if cids else -1e18)
    return out_scores

# ---------------------------------------------------------------------------
# Evaluadores por benchmark (generadores de (prefix, conts) + respuesta correcta)
# ---------------------------------------------------------------------------
def hellaswag_items(ds, max_n=None):
    n = 0
    for ex in ds:
        prefix = join_ctx_ending(ex["ctx"], ex["endings"][0])[0]
        yield prefix, list(ex["endings"]), int(ex["label"])
        n += 1
        if max_n and n >= max_n:
            break

def arc_items(ds, max_n=None):
    n = 0
    for ex in ds:
        yield arc_prompt(ex), list(ex["choices"]["label"]), ex["answerKey"]
        n += 1
        if max_n and n >= max_n:
            break

def mmlu_items(ds, dev_by_subject, max_n=None):
    n = 0
    for ex in ds:
        c = list(ex["choices"])
        yield mmlu_prompt(ex["question"], c, ex["subject"], dev_by_subject.get(ex["subject"], [])), LETTERS, int(ex["answer"])
        n += 1
        if max_n and n >= max_n:
            break
