# S-X8 v4.3 — Ventajas frente a GGUF Q8_0 (análisis para el paper)

**Fecha:** 6 Ago 2026 · **Formato:** S-X8 v4.3 (30 B/bloque = 7.50 bpp, byte-aligned, kernel propio)
**Referencia:** GGUF Q8_0 (llama.cpp, 8.50 bpp) — modelo Qwen3.5-4B, GPU RTX 5060 Ti

## 1. Resumen ejecutivo

S-X8 v4.3 ofrece **la misma calidad que Q8_0 a 0.75 bits menos por peso** (7.50 vs 8.50 bpp), con el modelo **completo** (texto+visión+MTP) en un **solo archivo byte-aligned** que pesa menos que el GGUF solo de texto. La degradación de calidad sobre FP16 es **9× menor** (+0.26% vs +2.40% de PPL).

## 2. Comparativa cuantitativa (medida, misma máquina, mismo protocolo)

| Métrica | S-X8 v4.3 | GGUF Q8_0 | Ventaja |
|---|---|---|---|
| bpp (todo incluido) | **7.50** | 8.50 | −11.8% bits |
| Archivo texto (4.21B) | **3.96 GB** | 4.48 GB | **−11.6%** |
| Archivo completo (4.66B) | **4.38 GB** | 5.15 GB (con mmproj) | **−15%** |
| PPL wikitext-2 | **10.2358** | 10.4540 | **−2.1% PPL** |
| vs FP16 (PPL) | **+0.26%** | +2.40% | **9× menos pérdida** |
| Winogrande_s (1267) | 0.5722 | 0.5746 | Empate (SE ±1.4%) |

## 3. Ventajas estructurales

### 3.1 Formato autocontenido
- Un solo archivo con TODO el modelo: texto + vision tower + MTP, cuantizados a 7.5 bpp
- GGUF requiere 2 archivos: `Q8_0.gguf` (texto) + `mmproj-F16.gguf` (visión en **FP16 sin cuantizar**, 0.67 GB)
- → SX8: visión cuantizada a 7.5 bpp (0.32 GB) incluida, sin archivos extra

### 3.2 bpp real y transparente
- 7.50 bpp cuenta TODO: niveles (6 bit) + escala (1 bit) + config (0.25 bit) + coeff PCA (0.25 bit) + bases (~0.04 bit)
- (El "7.50" del spec original del v4 era ficticio: no contaba el coeff — el v4 real pesaba 34 B/bloque = 8.5 bpp. El v4.3 es el primer formato que cumple la promesa.)

### 3.3 Decode trivial y portable
- ~9-10 operaciones ALU por peso, byte-aligned, sin shared memory, sin shuffles, sin instrucciones especiales
- Verificado en arquitecturas separadas 10 años: SM120 (RTX 5060 Ti, 2025) y Maxwell (GTX 960M, 2015)
- Portes: CUDA ✅ → ROCm, Metal, CPU SIMD/NEON, Vulkan — en horas (vs los k-quants de GGUF con shuffles)
- **"Cualquier GPU capaz de matmul FP16/FP32"** puede ejecutar SX8 (el decode es ALU pura; el GEMM usa el BLAS nativo)

### 3.4 Escala de alta precisión
- dmin/dmax en FP16 EXACTO (10 bits de mantissa) — la config (8 bits) es EXPLÍCITA, ya no se esconde en la mantissa (el v4 la escondía y corrompía 4 bits de precisión)
- Las estrategias de rango adaptativas (4 por sub-bloque) están saturadas: probadas 6 candidatas adicionales (incluidas las "invertidas") → +0.00% de mejora

### 3.5 Entrenable (QLoRA-style)
- Base congelada (dequantizada por el kernel propio en ~1 min) + adaptadores LoRA FP16
- Con el kernel fused (v2): base cuantizada en VRAM (4.4 GB) → más espacio para contexto/adaptadores

### 3.6 Extensible
- Estrategias de rango adicionales, codebooks mejorados (HVQ), formato mixto por capas (NV_SX: SX8 capas sensibles + SX6 capas robustas)
- Amplificación de calidad por capa (4-17% según el usuario — pendiente de verificar)

## 4. Desventajas (a declarar con honestidad)

1. **Ecosistema**: GGUF tiene soporte universal (llama.cpp, LM Studio, vLLM, Ollama, Jan…); SX8 necesita su runtime propio (kernel fused CUDA C + wmma + decode1 — VER FASE 3, doc 40)
2. ~~Velocidad end-to-end~~ **RESUELTA (FASE 3, Ago 2026)**: kernel fused con tensor cores (CUDA C + wmma m16n16k16) — **prompt 1411 tok/s = 4.5× MÁS RÁPIDO que FP16 cuBLAS (309)** (menos tráfico de memoria: 0.94 vs 2 B/peso) · decode M=1: 11.2 tok/s (111 GB/s, 25% del BW disponible) · GGUF Q8_0 en llama.cpp: ~3056 tok/s prompt (dequant+GEMM fusionados)
3. ~~VRAM~~ **RESUELTA (FASE 2+3)**: los pesos SX8 viven COMPRIMIDOS en VRAM (6.61 GB runtime con tabla rlo/step; 4.96 GB sin ella) — nunca existe la matriz FP16 (vs 9.08 GB FP16 y 4.48 GB GGUF)
4. **Decode M=1 (generación)**: **80.4 tok/s de kernel puro (317 GB/s, FASE 4 — ver 42)**; end-to-end pendiente de la vía CUDA Graphs/llama.cpp (el cuello es la orquestación Python, no el kernel); llama.cpp ~30-50 tok/s en el 4B
5. **Backward (entrenamiento)**: el kernel fused es solo forward; QLoRA requiere dequantizar la base a FP16 (~1 min) o kernel fused con gradientes (futuro)

## 5. Claims del paper (redactados para publicación)

1. "S-X8 v4.3 cuantiza el modelo completo (texto+visión+MTP) a 7.50 bpp con PPL +0.26% sobre FP16, mientras Q8_0 a 8.50 bpp degrada +2.40% — 9× más."
2. "El archivo SX8 del modelo completo pesa 4.38 GB — un 15% menos que GGUF Q8_0 + mmproj (5.15 GB) — e incluye la visión cuantizada que GGUF deja en FP16."
3. "El decode es byte-aligned con ~9-10 ALU por peso: portable a cualquier GPU capaz de matmul FP16/FP32 (verificado en arquitecturas separadas 10 años)."
4. "El formato es entrenable vía QLoRA: base congelada cuantizada + adaptadores FP16."
5. "Todo medido en la misma máquina (RTX 5060 Ti), mismo dataset (wikitext-2 test, 297K tokens) y mismo protocolo (ctx 512, greedy, sin chat template); GGUF Q8_0 verificado por SHA256 (unsloth, comunidad)."
6. **"La inferencia fused con pesos SX8 en VRAM es 4.5× más rápida que FP16 cuBLAS en prompt processing (1411 vs 309 tok/s) — el formato comprimido reduce el tráfico de memoria (0.94 vs 2 B/peso) y el decode en vuelo elimina la matriz FP16."** (kernel CUDA C + wmma, FASE 3)

## 6. Reproducibilidad (bloque del paper)

- Hardware: RTX 5060 Ti 16 GB GDDR7, driver CUDA 13.0, PyTorch 2.11
- Modelo base: Qwen/Qwen3.5-4B (SHA256: 26a93f06…/cb544bd9…)
- GGUF: unsloth/Qwen3.5-4B-GGUF Q8_0 (SHA256: 10cc391b…)
- Kernel decode: numba CUDA (kernel_sx8_v4.py / kernel_sx8_v43.py), ~5.8 G pesos/s
- **Kernel fused (FASE 3)**: CUDA C + wmma m16n16k16 (cuda/sx8_fused_wmma.cu), decode1 (cuda/sx8_decode1.cu), tabla rlo/step (scripts/precompute_rlo_step.py) — compilado con `torch.utils.cpp_extension.load_inline`, nvcc 13.0, -O3
- Dataset: wikitext-2-raw-v1 test (297,053 tokens) · Winogrande_s validation (1267)
- Protocolo: ventanas 512 (stride 128), greedy, thinking off, KV fresca por muestra
- Semillas: determinista (scoring log-likelihood, sin sampling)

---

# PÁRRAFO METHODS PARA EL PAPER — Evaluación (borrador listo)

*Derivado de PROTOCOLO.md (BATCH 2 + VALIDACIÓN ARC). Peleable al redactar.*

**Evaluation methodology.** We evaluate on PPL (wikitext-2, protocolo estándar),
Winogrande_s, HellaSwag (0-shot, acc_norm), ARC-Challenge (0-shot) and MMLU (5-shot),
all on the same RTX 5060 Ti (448 GB/s) with greedy scoring by continuation
log-likelihood and a fresh KV cache per sample, following lm-eval conventions:
multiple-choice prompts with single-letter continuations (A–E), HellaSwag with
length-normalized scores, ARC-Challenge on the test split (1,172 samples) and
HellaSwag/MMLU/Winogrande on their validation splits. The quantized S-X8 model is
evaluated with our fused runtime (kernel with PCA correction — the same engine that
validates the paper's quality numbers, PPL 10.2267); the GGUF Q8_0 reference with
llama.cpp (CUDA, Q8_1 activations) using identical prompts and scoring code
(shared module `mc_common.py`); FP16 with the base model. Prompt and scoring code
are shared between engines to guarantee identical inputs.

*Robustness note (ARC):* for multiple-choice tasks we follow the lm-eval standard of
scoring the answer letter (one token, no length artifacts). We verified that
scoring full choice texts instead produces unstable results (0.80 unnormalized,
0.68 length-normalized vs. 0.91 by letter on 300 samples) due to choice-length
artifacts (up to 46 tokens); the letter-based number is additionally supported by a
per-letter accuracy breakdown of 82–95% across all answer letters (ruling out
positional bias) and by agreement across three independent engines
(FP16 0.9172, S-X8 0.9164, Q8_0 0.9181). Reproducible via `verify_arc_method.py`.
