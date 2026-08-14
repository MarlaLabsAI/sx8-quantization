# S-X8 v4.3 — Format Specification (30 bytes/block = 7.50 bpp)

**Status:** final · **Version:** v4.3 · **Date:** 2026-08 · **License:** Apache-2.0

## 1. Overview

S-X8 v4.3 quantizes weight matrices in **blocks of 32 consecutive weights** of one matrix row.
Each block occupies exactly **30 bytes** (7.50 bits per weight, fully accounted). The decoder is
byte-aligned and uses only ~9–10 ALU operations per weight: byte extraction, a branchless strategy
selection and two FMAs. No shared memory, no warp shuffles, no texture units, no tensor cores —
the same kernel runs on any GPU architecture (validated on SM120 and Maxwell).

## 2. Block layout (30 bytes)

| Offset | Field | Size | Description |
|---|---|---|---|
| 0 | `dmin` | 2 B | FP16 lower range endpoint (exact 10-bit mantissa) |
| 2 | `dmax` | 2 B | FP16 upper range endpoint (exact 10-bit mantissa) |
| 4 | `config` | 1 B | 4 × 2 bits: range strategy per 8-weight sub-block (sb = w >> 3) |
| 5 | `qh` | 16 B | High nibbles of 6-bit levels: `qh[w>>1]` bits `(w&1)*4 .. +3` |
| 21 | `ql` | 8 B | Low 2 bits of 6-bit levels: `ql[w>>2]` bits `(3-(w&3))*2 .. +1` |
| 29 | `coeff` | 1 B | PCA coefficient: `c0 = signed4(coeff & 0xF)`, `c1 = signed4(coeff >> 4)` |
| — | **Total** | **30 B** | **= 7.50 bpp** |

Where `signed4(x) = x < 8 ? x : x - 16` (values −8..7).

## 3. Range strategies (per 8-weight sub-block)

Let `lo = fp16(dmin)`, `hi = fp16(dmax)`, `q = (hi − lo) · 0.25`, and `s = (config >> (sb*2)) & 3`:

| s | Strategy | `rlo` | `rhi` |
|---|---|---|---|
| 0 | full range | `lo` | `hi` |
| 1 | lower quarter | `lo` | `lo + q` |
| 2 | upper quarter | `hi − q` | `hi` |
| 3 | central half | `lo + q` | `hi − q` |

Branchless selection: `rlo = lo + q*(3*(s==2) + (s==3))`, `rhi = hi − q*(3*(s==1) + (s==3))`.

## 4. Decode

Per weight `w` (0..31):
```
lv   = ((qh[w>>1] >> ((w&1)*4)) & 0xF) << 2 | ((ql[w>>2] >> ((3-(w&3))*2)) & 0x3)   # 6-bit level
step = (rhi − rlo) · 0.015873        # 1/63; clamp step to ≥ 1e-10
decoded = rlo + step · lv
```

## 5. PCA correction (optional, applied at the matmul output)

Per block-of-K, the format stores 2 basis vectors `b0, b1` (shape `(2, n_cb, 64)` fp16) and 2 scales
`s0, s1` (`(2, n_cb)` fp16); these live in auxiliary tensors, not in the block. The per-(block, column)
coefficient byte is in the block. Using the linearity of the dot product:

```
Y[n] = Σ_kb core(kb,n) + Σ_kb c0(kb,n)·Z0(kb) + c1(kb,n)·Z1(kb)
Z0(kb) = s0(kb) · Σ_t X[kb·32+t] · b0[kb][t]     # computed once per layer per token
Z1(kb) = s1(kb) · Σ_t X[kb·32+t] · b1[kb][t]
```

This turns the per-weight PCA cost (2 FMA + 2 FP32 loads) into 2 FMAs per block (≈0.06 per weight).

## 6. Container (.sx8v43)

The `.sx8v43` file is byte-aligned, pickle-free, and verified byte-exact round-trip (381/381 tensors,
PPL from file identical to in-memory). See `scripts/sx8_container_v43.py`.

## 7. Validation

| Check | Result |
|---|---|
| CosSim global vs FP16 (float64) | 1.000000 |
| PPL wikitext-2 (reference kernel) | 10.2358 (FP16 10.2090, Q8_0 10.4540) |
| PPL wikitext-2 (fused runtime + PCA) | 10.2267 |
| Multiple choice (4 benchmarks) | ≈ FP16, within noise |
| File round-trip | byte-exact 381/381 |

Full details: `results/PROTOCOLO.md`, `results/VENTAJAS-SX8.md`, paper.
