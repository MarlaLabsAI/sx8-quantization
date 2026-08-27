---
license: apache-2.0
language:
- en
- es
- it
- fr
- zh
tags:
- Text Generation
- GGUF
- quantization
- s-x8
- llm
- compression
- qwen3.8
- 27b
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
---

# Qwen3.8-27B-SX8 — S-X8 v4.3 quantized variant of Qwen3.8-27B

**S-X8 v4.3** is a weight quantization format for large language models: **30 bytes per block = 7.50 bpp**
(fully accounted), FP16-level quality, and a portable byte-aligned decoder. This repository contains the
**Qwen3.8-27B quantized to S-X8 v4.3** — a larger variant of the same quantization applied to
[Qwen3.5-4B](https://huggingface.co/marlalabsAI/Qwen3.5-4B-SX8).

## Format validated on Qwen3.5-4B

The S-X8 v4.3 format is validated on **Qwen3.5-4B** (single RTX 5060 Ti). See the 4B repository for the
full benchmark table and the paper:

| Metric | FP16 | **S-X8 v4.3 (4B)** | Q8_0 |
|---|---|---|---|
| PPL wikitext-2 (PCA runtime) | 10.2090 | **10.2267** (+0.17%) | 10.4540 (+2.40%) |
| Winogrande_s | 0.5746 | **0.5722** | 0.5746 |
| Decode, real-world (llama.cpp fork) | — | **63.79 tok/s** | 40–54 tok/s |
| VRAM decode M=1 | — | **3.720 GB** | 4.48 GB |

**Paper (Zenodo, DOI):** [10.5281/zenodo.21922640](https://doi.org/10.5281/zenodo.21922640)

## This 27B model

Quantized with the **same pipeline** as the 4B: per-tensor S-X8 v4.3, 666 tensors,
CosSim **0.999753** (mean over all tensors), 7.50 bpp, **26.1 GB** container.

> **Validated functionally: the model runs and answers correctly in a local test**
> (same protocol as the 4B model). Formal 27B benchmarks (PPL, Winogrande, ARC, MMLU)
> are **pending** — to be measured on a larger GPU (we will update when measured).
> The larger base model is expected to deliver better absolute quality (same format,
> more capable base).

## Two containers, one model

The S-X8 model exists in **two containers** — same weights inside:

| File | Container | Used by |
|---|---|---|
| `Qwen3.8-27B-SX8v43.sx8` | native S-X8 v4.3 format (complete: config + quantized 2D + 1D + MTP + vision) | our Python runtime (`scripts/infer_sx8.py`, CUDA kernels) |
| `Qwen3.8-27B-SX8v43.gguf` | GGUF (the container llama.cpp reads) with S-X8 weights inside — native type `GGML_TYPE_SX8 = 41` | llama.cpp fork (`run_llama_chat.sh`) |

Think of it like a `.zip`: the container is the same, what's inside can be anything. The GGUF is the
**same S-X8** inside the container llama.cpp reads. MTP lives in the `.sx8v43` (excluded from the GGUF,
same criterion as the 4B).

## Quick start

**Vía A — llama.cpp fork (fast chat):**

```bash
./run_llama_chat.sh build   # one time (clones llama.cpp, applies the patch, compiles)
./run_llama_chat.sh         # chat
```

- GPU ≥ 32 GB: `NGL=999 ./run_llama_chat.sh` (everything in VRAM)
- GPU 16 GB: `NGL=28 ./run_llama_chat.sh` (28 layers GPU + rest CPU), or unified memory:
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 NGL=999 ./run_llama_chat.sh` (100% GPU compute, PCIe spill)

**Vía B — Python runtime with our CUDA kernels (native container):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/infer_sx8.py --container Qwen3.8-27B-SX8v43.sx8
```

Full case-by-case guide: [`QUICK-DEPLOY.md`](QUICK-DEPLOY.md) (GPUs 8–32+ GB, CPU-only, Windows/WSL2).

## Repository layout

```
├── Qwen3.8-27B-SX8v43.sx8      native S-X8 container (complete model)
├── Qwen3.8-27B-SX8v43.gguf      S-X8 inside GGUF (llama.cpp)
├── tokenizer/                  (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, chat_template.jinja)
├── scripts/                    quantization (27B) · containers · runtime (eval_common, infer_sx8) · kernels
├── cuda/                       S-X8 CUDA kernels (decode1_v3, fused_wmma, gemm_compact, embed_v44...)
├── llama-cpp-sx8.patch         llama.cpp fork: S-X8 as native type (MMVQ decode + MMQ prompt + AVX2 CPU)
├── QUICK-DEPLOY.md             deployment guide by GPU case
├── SX8_FLASH_V4_3_SPEC.md      block format spec (30 B/block)
├── SX8_FLASH_V4_3_CONTAINER.md container format spec (v1.1 SXT1)
├── S-X-METHODOLOGY.md          methodology
└── CHANGELOG.md
```

## Transparency

The conceptual seeds of S-X8 come from an independent mathematical analysis of the image of the
Shroud of Turin; the full study is published in the project repo with its own re-verification:
[github.com/MarlaLabsAI/sx8-quantization → shroud-turin-study/](https://github.com/MarlaLabsAI/sx8-quantization/tree/main/shroud-turin-study).
See the paper's Appendix A and [IDEA-PROVENANCE.md](IDEA-PROVENANCE.md). The format itself is
validated empirically in this repository; the study is disclosed only as the source of inspiration,
with full honesty about what can and cannot be corroborated.

## License and attribution

Apache-2.0. The S-X8 format, kernels and scripts in this repository are **original work**
(Copyright (C) 2026 Martí Vidal Leandro), not a Qwen work. The quantized model uses
**Qwen3.8-27B by the Qwen Team** (Alibaba Group), Apache-2.0 (model card) as the base model;
only the weights were quantized to the S-X8 v4.3 format (see [NOTICE](NOTICE)), no other modifications.

## How to cite

If you use this work, please cite:

> Vidal Leandro, M. (2026). **S-X8 v4.3: A 7.50-Bits-Per-Weight Quantization Format with
> FP16-Level Quality and Portable Decoding.** Zenodo. https://doi.org/10.5281/zenodo.21922640

## Contact

Looking for collaborations, integration help or opportunities? Reach out:

- **LinkedIn (profile):** https://www.linkedin.com/in/vidalmarti/
- **LinkedIn (company):** https://www.linkedin.com/company/marlalabs/
- **X / Twitter:** https://x.com/MarlaLabsAI
- **Email:** info@marlalabs.com
- **Web:** https://marlalabs.com

## Support

If this work helps you, consider a donation: [Ko-fi](https://ko-fi.com/) · [GitHub Sponsors](https://github.com/sponsors) — details in FUNDING.yml.

## Deploy with an AI agent

Want another AI (Claude Code, opencode, aider...) to handle the whole deployment
for you — on any OS and hardware, within physical limits? Copy the prompt from
[`DEPLOY-AI.md`](DEPLOY-AI.md) and give it to the agent: it will detect your
system (OS, GPU, RAM), install what's needed, download, run and verify the model
on its own.

## License / NOTICE


Apache-2.0. The S-X8 format is original work of the author; this model is a **quantized variant of
Qwen3.8-27B** (Qwen Team, Alibaba Group, Apache-2.0) — only the weights were quantized to S-X8 v4.3,
no other modifications. See [NOTICE](NOTICE).
