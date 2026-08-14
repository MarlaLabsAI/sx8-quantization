# S-X8 v4.3 — A 7.50-bits-per-weight quantization format

**Paper (Zenodo, DOI):** [10.5281/zenodo.21922640](https://doi.org/10.5281/zenodo.21922640)
**Quantized model (HuggingFace):** [marlalabsAI/Qwen3.5-4B-SX8](https://huggingface.co/marlalabsAI/Qwen3.5-4B-SX8)

**S-X8 v4.3** is a weight quantization format for large language models: **30 bytes per block = 7.50 bpp**
(fully accounted), FP16-level quality, and a portable byte-aligned decoder (~9–10 ALU ops per weight, no
shared memory, no shuffles, no tensor-core dependency).

Validated on **Qwen3.5-4B** (single RTX 5060 Ti):

| Metric | FP16 | **S-X8 v4.3** | Q8_0 |
|---|---|---|---|
| PPL wikitext-2 (PCA runtime) | 10.2090 | **10.2267** (+0.17%) | 10.4540 (+2.40%) |
| Winogrande_s / HellaSwag / ARC / MMLU | 0.5746 / 0.6965 / 0.9172 / 0.7133 | **0.5722 / 0.6964 / 0.9164 / 0.7074** | 0.5746 / 0.6965 / 0.9181 / 0.7087 |
| Text weights size | 9.3 GB | **3.96 GB** (−11.6% vs Q8_0) | 4.48 GB |
| Complete model (vision+MTP) | — | **4.38 GB** (−15% vs Q8_0+mmproj) | 5.15 GB |
| Decode, real-world (llama.cpp fork) | — | **63.79 tok/s** | 40–54 tok/s |
| Prompt pp128 (llama.cpp MMQ) | — | 1,877.85 tok/s | 3,655.61 tok/s |

Multiple-choice benchmarks are statistically indistinguishable across formats; the differentiators are
perplexity (**9× less loss than Q8_0**), size, VRAM and real-world decode speed. Full protocol and data in
[`results/PROTOCOLO.md`](results/PROTOCOLO.md).

## Repository layout

```
├── paper/          paper-sx8.pdf / .html (EN) · paper-sx8-ES.pdf / .html (ES) · paper.tex (arXiv)
├── docs/           IDEA-PROVENANCE (EN/ES) · S-X-METHODOLOGY (EN/ES) · SX8_FLASH_V4_3_SPEC.md · SX8_FLASH_V4_3_CONTAINER.md
├── formats/        .sx8v43 container spec (the "GGUF" of S-X8) + block spec
├── cuda/           kernels: sx8_fused_wmma.cu (prompt) · sx8_decode1_v3.cu (decode) · others
├── scripts/        quantization · evaluation (mc_*, ppl_*, winogrande_*) · containers · validation
├── results/        all raw JSON results · PROTOCOLO.md · VENTAJAS-SX8.md
├── LICENSE         Apache-2.0
└── FUNDING.yml
```

## Quick start

```bash
# Quantize (needs the base model + GPU):
python3 scripts/quantize_qwen35_4b_sx8_v43_gpu.py

# Evaluate (offline mode; datasets cached):
export HF_DATASETS_OFFLINE=1
python3 scripts/mc_eval_sx.py --mode fp16      # FP16 multiple choice
python3 scripts/mc_eval_sx.py --mode sx8v43    # S-X8 (fused runtime, with PCA)
python3 scripts/mc_eval_gguf.py                # Q8_0 reference (llama-cpp-python CUDA)
python3 scripts/verify_arc_method.py           # ARC letter-method validation
```

Environment: PyTorch 2.11, CUDA 13.0, numba 0.65, llama-cpp-python 0.3.34 (CUDA build), datasets 4.8.5.
A llama.cpp fork integrating S-X8 as native type `GGML_TYPE_SX8` (decode MMVQ + prompt MMQ kernels)
is provided as `llama-cpp-sx8.patch` (apply with `git apply` on a clean llama.cpp checkout of commit
7c203670f). The new file `ggml/src/ggml-cuda/template-instances/mmq-instance-sx8.cu` is included in the
patch.

## Transparency

The conceptual seeds of this format came from an independent mathematical analysis of the image of the
Shroud of Turin. The full study (with its own re-verification) is published in this repository:
[`shroud-turin-study/`](shroud-turin-study/) (complete and available, exactly as produced). See the
paper's Appendix A and [`docs/IDEA-PROVENANCE.md`](docs/IDEA-PROVENANCE.md) for the complete statement. The format
itself is validated empirically in this repository; the study is disclosed only as the source of
inspiration, with full honesty about what the author can and cannot corroborate.

## License

Apache-2.0. Base model: Qwen3.5-4B (Apache-2.0). See [LICENSE](LICENSE).

## Cite this work

If you use this work, please cite:

> Vidal Leandro, M. (2026). S-X8 v4.3: A 7.50-Bits-Per-Weight Quantization Format with FP16-Level Quality
> and Portable Decoding. Zenodo. https://doi.org/10.5281/zenodo.21922640

## Support

Marla Labs is an independent AI lab. If this work helps you, consider a donation:
[Ko-fi](https://ko-fi.com/marlalabs) · [GitHub Sponsors](https://github.com/sponsors/MarlaLabs)
· details in `FUNDING.yml`. 🐱
