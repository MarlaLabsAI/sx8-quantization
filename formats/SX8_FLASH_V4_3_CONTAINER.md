# S-X8 v4.3 — Container Format Specification (.sx8v43)

**Status:** final · **License:** Apache-2.0 · **Implementation:** `scripts/sx8_container_v43.py`

The `.sx8v43` container is the file format that packages an S-X8 v4.3 quantized model: a single,
pickle-free, byte-aligned artifact that includes the quantized text transformer AND the vision and
multi-token-prediction towers in the same file (unlike GGUF, which requires a separate FP16 mmproj).

## 1. Global layout

```
┌──────────────┬───────────────────────────────────────────────┐
│ MAGIC        │ 8 bytes: "SX43FILE"                            │
│ VERSION      │ 1 byte (uint8) = 1                             │
│ meta_len     │ uint32 LE — length of the metadata blob        │
│ meta         │ meta_len bytes — repr() of the meta dict (sorted) │
│ n_tensors    │ uint32 LE — number of tensor records           │
│ tensor 1     │ (see below)                                    │
│ ...          │                                                │
│ tensor n     │                                                │
└──────────────┴───────────────────────────────────────────────┘
```

## 2. Tensor record

Per tensor, in order:

| Field | Encoding | Meaning |
|---|---|---|
| `name_len` | uint32 LE | byte length of the tensor name |
| `name` | `name_len` bytes | tensor name (UTF-8) |
| `shape[0]`, `shape[1]` | 2 × uint32 LE | 2D logical shape (out_f, in_f) |
| `n_os` | uint8 | number of original dims (for reshaped tensors) |
| `orig_shape` | `n_os` × uint32 LE | original shape (e.g., conv kernels) |
| `n_blocks` | uint32 LE | number of 32-weight blocks |
| `n_cb` | uint32 LE | number of PCA codebook blocks (per block-of-K) |
| `dmin` | `n_blocks` × 2 B | FP16 range endpoints (little-endian) |
| `dmax` | `n_blocks` × 2 B | FP16 range endpoints |
| `config` | `n_blocks` × 1 B | 4 × 2-bit range strategies |
| `levels_hi` | `n_blocks` × 16 B | high nibbles of 6-bit levels |
| `levels_lo` | `n_blocks` × 8 B | low 2 bits of 6-bit levels |
| `coeff` | `n_blocks` × 1 B | PCA coefficient bytes |
| `bi_n_cb` | uint32 LE | number of PCA codebook blocks (auxiliary) |
| `bases_data` | `bi_n_cb` × 64 × 4 B | FP32 basis vectors (2 bases of 32, concatenated) |
| `scales` | `bi_n_cb` × 2 × 4 B | FP32 PCA scales (s0, s1) |

Every field is little-endian. Blocks are laid out per tensor with the same 30-byte structure defined in
`SX8_FLASH_V4_3_SPEC.md` (dmin + dmax + config + levels_hi + levels_lo + coeff).

## 3. Guarantees

- **Pickle-free** — pure binary, safe to parse.
- **Byte-aligned** — every tensor's block payload is a multiple of 30 bytes; bases/scales are
  FP32-aligned sections.
- **Byte-exact round-trip** — `verify_sx43(pkl, sx43)` asserts every array read from the file is
  bit-identical to the source pkl (verified 381/381 tensors).
- **PPL from file == PPL in memory** — the evaluator (`eval_common.load_model(source_file=...)`)
  reproduces the exact quality numbers.

## 4. Readers

- Reference reader: `scripts/sx8_container_v43.py` → `read_all(path)` returns
  `(weights, bases, meta)` in the same structure as the pkl.
- Evaluators use it via `eval_common.load_model(quantized=True, mode="v43", source_file=...)`.

## 5. Example (Qwen3.5-4B-SX8v43.sx8)

- File size: 4.38 GB · tensors: 381 · blocks: ~146M · theoretical payload at 30 B/block matches
  the file size to <0.1%.
