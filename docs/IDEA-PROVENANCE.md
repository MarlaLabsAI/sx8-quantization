# IDEA PROVENANCE — S-X8 v4.3

*Where the ideas behind the published S-X8 format come from, and how each one is validated.
This document covers only what is published in this repository; research in progress is not
described here.*

---

## 1. Origin of the inspiration

The conceptual seeds of S-X8 come from an independent mathematical analysis of the image of the Shroud of
Turin (2025-2026). That study was an inquiry that found very interesting things; some seem certain and
astonishing, but the author cannot corroborate them, nor does he intend to: that verification belongs to
those with direct access to the Shroud. Some tests and analyses seem well done; others may be closer to
pareidolia. The whole exercise - whether veracious or not - served to extrapolate concepts and ideas for
the format.

The S-X8 format itself has been verified empirically (perplexity and benchmarks on Qwen3.5-4B, see
Section 4 and PROTOCOLO.md); the Shroud study is mentioned only to show where the extrapolation of ideas
came from, in the spirit of full transparency. A complete transparency statement is included in the paper
(Appendix A) and in the study folder (NOTA_TRANSPARENCIA.md).

## 2. What is original in this work

- Exact, fully accounted bit count: 30 bytes/block = 7.50 bpp (every byte justified).
- PCA correction applied at the *output* of the matrix multiply (Z0/Z1 reformulation),
  reducing the per-weight cost to ~0.06 FMAs per weight.
- Portable byte-aligned decoder: ~9–10 ALU ops per weight, no shared memory, no shuffles,
  no tensor-core dependency (validated on SM120 and Maxwell).
- Native integration in a llama.cpp fork as `GGML_TYPE_SX8`, including both a decode kernel
  (MMVQ) and a tensor-core prompt kernel (MMQ).
- Honest evaluation protocol (shared prompts across engines, ARC-method validation) and
  transparent reporting of limitations.

## 3. Validation summary (all numbers in `results/` and `PROTOCOLO.md`)

| Metric | FP16 | S-X8 v4.3 | Q8_0 |
|---|---|---|---|
| PPL wikitext-2 (PCA runtime) | 10.2090 | 10.2267 | 10.4540 |
| Winogrande_s / HellaSwag / ARC / MMLU | 0.5746 / 0.6965 / 0.9172 / 0.7133 | 0.5722 / 0.6964 / 0.9164 / 0.7074 | 0.5746 / 0.6965 / 0.9181 / 0.7087 |
| Text size | 9.3 GB | 3.96 GB | 4.48 GB |

*Multiple-choice benchmarks are statistically indistinguishable across formats; the quality
differentiator is perplexity (9× less loss than Q8_0) plus size, VRAM and real-world decode
speed. Full methodology in PROTOCOLO.md.*
