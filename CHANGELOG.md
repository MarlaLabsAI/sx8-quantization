# Changelog

All notable changes to the S-X8 quantization project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).



## [v27.1] — 2026-08-27 — Qwen3.8-27B-SX8 (new model)

### Added
- **Qwen3.8-27B quantized to S-X8 v4.3** (per-tensor, shard-by-shard, 666 tensors).
  CosSim **0.999753** (mean over all tensors) · bpp 7.5038 · ratio 2.132.
  - Container `Qwen3.8-27B-SX8v43.sx8` (26.1 GB) — complete model (config + 2D + 1D + MTP + vision).
  - GGUF `Qwen3.8-27B-SX8v43.gguf` (26.1 GB) — same S-X8 weights inside GGUF (MTP excluded);
    native type `GGML_TYPE_SX8 = 41` in the llama.cpp fork.
- New directory `qwen3.8-27b/` with the 27B-specific scripts:
  `quantize_qwen38_27b_sx8_v43_shard.py` (streaming shard-by-shard),
  `add_small_tensors_27b.py` (container v1.1 SXT1), `container_to_gguf_sx8_27b.py`
  (container → GGUF with S-X8 blocks), `assemble_sx43.py`, `reverify_sx43.py`,
  `verify_final_sx43.py`.
- **llama.cpp fork**: added an **AVX2 CPU kernel** for `ggml_vec_dot_sx8_q8_1`
  (new variant of the generic decode V6) — improves CPU-side decoding on x86.
- **`qwen3.8-27b/QUICK-DEPLOY.md`** — deployment guide by GPU case
  (8–32+ GB, CPU-only, Windows/WSL2, unified memory).
- **Validation results** (functional local test): `qwen3.8-27b/results/`
  — 20/20 (EN/ES/IT/FR) + 20/20 general.

### Validated
- Functional local test: **50/50 answers correct** across 4 languages
  (EN/ES/IT/FR): arithmetic, general knowledge, logic, sequences, code generation.
- Formal 27B benchmarks (PPL, Winogrande, ARC, MMLU) **pending** — to be measured
  on a larger GPU. The format is formally validated on **Qwen3.5-4B-SX8**
  (PPL 10.2267, Winogrande 0.5722).

### Not changed
- S-X8 v4.3 format spec (30 B/block, kb-major) — same as the 4B.
- CUDA kernels (decode1_v3, fused_wmma, gemm_compact, embed_v44) — same architecture (qwen35).
- License Apache-2.0.

## [v1.3] — 2026-08-17

### Added — Generation speedup for the Python runtime (Option A)

- **`decode1_v44` (M=1) optimized**: `split_k=1` (no reduce_split), reusable buffer pools for
  Z/Y (no per-call allocations), and a combined `compute_z44` + `decode1` CUDA call (one launch
  per tensor instead of two).
  - Per-tensor kernel throughput: **8.8 G p/s → 376 G p/s** (45×).
- **`infer_sx8.py`**: prefill in ONE pass (M≥32 → compact GEMM, ~273 tok/s) + generation M=1 with
  KV cache (decode1 optimized) + static attention mask.
- **Measured generation (RTX 5060 Ti, real chat with KV cache)**: **~10-13 tok/s** ·
  VRAM **4.41 GB** (= loaded compact + KV). GPU-bound (35% util → margin is orchestration
  launch bubbles).
- **Coming soon (Option B)**: CUDA Graphs + static KV cache (subclass of `Qwen3_5Cache` that
  writes in place) — target ~55 tok/s generation for the Python runtime. llama.cpp fork remains
  the fast chat path today (63.79 tok/s).

## [v1.2] — 2026-08-16

### Added — Compact GEMM for M >= 32 (`cuda/sx8_gemm_compact.cu` + `scripts/sx8_gemm_compact.py`)

- **Prompt/PPL processing now decodes on-the-fly from the compact v4.4 layout (30 B/block) — no FP16
  materialization, no precomputed WMMA tables (44 B/block extra).**
- **VRAM during prompt = the loaded compact data + KV**: measured **4.31 GB** during the full
  wikitext-2 PPL run (previously ~9.5 GB with the WMMA tables) — "lo que ocupa cargado + KV".
- **Prompt throughput (steady state, RTX 5060 Ti): ~273 tok/s** (512-token forward in 1.87 s).
- Kernel design: 64-column tiles × 64-row blocks (grid.y over M), M_TILE=16 with amortized decode,
  butterfly reduction; PCA (Z0/Z1) precomputed per row with `compute_z44_batch`.
- **Equality validated** against the numba reference (`decode_tensor_gpu_fast` + matmul) on real
  container tensors (N=8192/9216/2560/248320, M=32..512): maxdiff < 1.5e-3.
- Tensors with N < 64 (e.g., GLA `in_proj_a`, 32×2560) use a numba decode + matmul fallback
  (transient FP16, 164 KB — negligible).
- `SX8_USE_WMMA=1` keeps the old WMMA path as an opt-in (faster prompt, +44 B/block of tables).
- `eval_common.load_model_standalone` now clears the numba decode buffer pool after loading
  (freed ~5 GB of cached workspace).

### Re-validated on the v2 container with the compact GEMM

- **PPL wikitext-2 = 10.2368** (reference 10.2267, Δ +0.0101 — PASS, same protocol: ctx 512, stride 128).
- Prompt VRAM during the run: **4.31 GB**.

## [v1.1] — 2026-08-16

### Added — Container format v1.1 (`.sx8v43`, section `SXT1`)

- The `.sx8v43` container now embeds **everything needed to run the model standalone**:
  1. The **full model config** (JSON) — the architecture can be rebuilt from the container alone.
  2. **All non-quantized 1D tensors** (layer norms, `A_log`, `dt_bias`, attention norms, etc.) — `517,632` params for Qwen3.5-4B (`357` tensors).
- **Backward compatible**: v1.0 readers read the quantized tensor records and ignore the new trailing section; v1.1 readers (`read_all_v11` in `sx8_container_v43.py`) read the complete file.
- The published model file `Qwen3.5-4B-SX8v43.sx8` is now the **v2 file** (v1.1 format). The quantized 2D records are byte-identical to the v1 file (verified by SHA-256 of the v1 prefix: `dc6c88c9…`).

### Changed — Runtime (`eval_common.py` v2)

- **Removed all hardcoded local paths** (`MODEL_PATH`, `PKL_PATH`, `PKL_PATH_V43`).
- New `load_model_standalone()`: builds the architecture from the **container's embedded config**, materializes the 1D tensors from the container, and decodes the 2D quantized tensors with the S-X8 kernels — **runs with only the `.sx8v43` file + tokenizer files. No base model required.**
- Tokenizer files are now shipped alongside the model (Hugging Face repo).

### Fixed — Degenerate-block decode bug (all engines)

- **Bug**: for weight blocks where `min ≈ max` (degenerate/constant blocks — typical of vision towers), the decode fallback forced `step = 1/63`, reconstructing values ≈ `1.0` where the original value was ≈ `0`.
- **Fix**: fallback step is now `1e-10` (matching the encoder's clamp semantics).
- **Files fixed** (all in this release):
  - Python runtime: `kernel_sx8_v43.py`, `kernel_sx8_v4.py`, `kernel_sx8_fused.py` (6 sites), `eval_common.py`, `integrate_fused.py`.
  - Quantization/validation scripts: `quantize_qwen35_4b_sx8_v43_gpu.py`, `quantize_qwen35_4b_sx8_v4_full_gpu.py`, `prepare_v44.py`, `validate_pca_reform.py` (4 sites).
  - CUDA kernels: `sx8_decode1_v3.cu`, `sx8_embed_v44.cu`, `sx8_fused_wmma.cu`, `experiment_texture_decode.cu`.
  - llama.cpp fork: `ggml-cpu/quants.c`, `ggml-quants.c` (2 sites), `ggml-cuda/mmq.cuh` (2 sites), `ggml-cuda/vecdotq.cuh` → **`llama-cpp-sx8.patch` regenerated**.
- **Impact on published results**: the text tensors of Qwen3.5-4B contain **zero** degenerate blocks → all published text benchmarks (PPL, Winogrande, HellaSwag, ARC, MMLU) remain valid unchanged. The vision tower contains `1,933` degenerate blocks (`0.0013%` of blocks, 5 tensors) which are now decoded correctly.
- **Re-validation on the v2 container** (same protocol as v1):
  - `results/ppl_fused_v44_v2_revalidate_m1.json` — **PPL wikitext-2 = 10.2364** (reference 10.2267, Δ +0.0097 — PASS, same protocol: ctx 512, stride 128).
  - `results/bench_decode_v44_v2.json` — **decode M=1 VRAM peak = 3.720 GB** (reference 3.955 GB — the S-X8 < GGUF claim holds on the standalone container).
  - `results/winogrande_s_v2_fused_revalidate.json` — Winogrande_s on the v2 container.
- **Runtime improvement (v1.1)**: `SX8LinearV44.forward` now builds the WMMA tables (`_tensors`) only when `M >= 32`; the M=1 decode path uses only the compact v4.4 tensors (30 B/block) → **decode/generation VRAM reduced** (measured 3.720 GB vs 3.955 GB).

### Notes

- The `mtp.*` tensors (multi-token prediction) are present in the container; the standard inference class does not instantiate the MTP head (it is optional in inference, as in the v1 runtime).
- Qwen3.5-4B-SX8v43 GGUF is unchanged (it already was standalone); the llama.cpp fork patch now includes the degenerate-block fix.

## [v1.0] — 2026-08-13

### Added — Initial public release

- S-X8 v4.3 format (7.50 bits per weight), `.sx8v43` container, CUDA kernels, llama.cpp fork (GGML_TYPE_SX8), Qwen3.5-4B-SX8 quantized model, paper (EN/ES).
- **Known limitation (fixed in v1.1)**: the v1 container contained the quantized 2D tensors (99.99% of the model) but the Python runtime loaded the non-quantized 1D tensors from the base model; `eval_common.py` referenced local paths.
