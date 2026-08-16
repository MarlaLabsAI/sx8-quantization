# Changelog

All notable changes to the S-X8 quantization project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
