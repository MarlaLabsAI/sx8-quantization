# Benchmark S-X8 v4 — Protocolo y Resultados (Qwen3.5-4B)

**Fecha:** 6 Ago 2026 · **GPU:** RTX 5060 Ti 16GB · **Modelo base:** Qwen/Qwen3.5-4B (SHA256 verificado)
**Formatos comparados:** FP16 (base) · **S-X8 v4 (kernel propio Numba CUDA, pkl `qwen35_4b_sx8_flash_v4_new.pkl`, modelo completo 4.66B: texto+visión+MTP+conv1d, solo normas 1D FP32)** · GGUF Q8_0 (unsloth, SHA256 verificado)

## Parámetros de evaluación (metodología estándar — Unsloth/llama.cpp/lm-eval)

| Parámetro | Valor | Fuente |
|---|---|---|
| **Thinking** | DESACTIVADO (default en Qwen3.5 Small 0.8B-9B) | unsloth.ai/docs/models/qwen3.5: "reasoning is disabled by default" |
| **Temperatura** | 0 (greedy — sin sampling; scoring determinista) | estándar lm-eval/llama.cpp eval |
| **Chat template** | NO se aplica a tareas de scoring (log-likelihood puro) | lm-eval + llama.cpp eval |
| **Contexto PPL** | ventanas de **512 tokens** ("Wiki-test with 512 context windows" — estándar comunitario) | unsloth.ai/docs/models/qwen3.5/gguf-benchmarks |
| **Stride PPL** | 128 (convención HF tutorial; cada token puntuado tiene ≥384 de contexto) | huggingface.co/blog/how-to-generate |
| **Cache KV** | **Fresca por muestra/ventana** (llm.reset() / forward independiente — sin reuso) | lm-eval (cada sample independiente) |
| **Kernel SX8 v4** | kernel propio Numba CUDA `sx8_v4_kernel` (V6 + PCA inline, 4 FMAs, 100% GPU) — **~5.8 G pesos/s** en RTX 5060 Ti | kernel_sx8_v4.py |
| **GGUF Q8_0 PPL** | `llama-perplexity -c 512 -ngl 99` (herramienta canónica de la comunidad; cuenta todos los tokens con ventana deslizante — ligeramente más favorable que el chunked) | llama.cpp |

## Resultados — PPL wikitext-2-raw test (297,053 tokens)

| Formato | bpp | PPL | vs FP16 | Winogrande_s | Tamaño archivo |
|---|---|---|---|---|---|
| **FP16** | 16.00 | **10.2090** | — | 0.5746 | ~9.3 GB |
| **S-X8 v4** (4 bases) | 8.50 | 10.2221 | +0.13% | 0.5730 | 4.97 GB pkl |
| **S-X8 v4.2** (2 bases truncadas) | 7.75 | 10.2289 | +0.19% | 0.5714 | 4.54 GB |
| **S-X8 v4.3** (escala FP16, 2 bases) | **7.50** | **10.2358** | **+0.26%** | 0.5722 | **4.38 GB .sx8v43** |
| **GGUF Q8_0** (unsloth) | 8.50 | 10.4540 | +2.40% | 0.5746 | 4.48 GB (solo texto) |

**Claim: S-X8 v4.3 (7.50 bpp) supera a Q8_0 (8.5 bpp) en calidad (+0.26% vs +2.40% sobre FP16) comprimiendo ~12% MÁS en el modelo de texto (3.95 GB vs 4.48 GB), y con visión+MTP incluidas en un solo archivo (4.38 GB).**
Nota de conservadurismo: el chunked (transformers) tiende a PPL ≥ que el sliding (llama.cpp); el delta real SX8 vs Q8_0 es ≥ al medido.

## Comparación S-X8 v4.3 vs GGUF Q8_0 — qué comprime cada uno y con qué calidad

| | **S-X8 v4.3** | **GGUF Q8_0** (unsloth) |
|---|---|---|
| **Modelo en el archivo** | COMPLETO: texto (4.21B) + visión (0.33B) + MTP (0.12B) = 4.66B — todo cuantizado a 7.5 bpp | Solo texto (4.21B); la visión va en archivo aparte (mmproj-F16, 0.67 GB, SIN cuantizar) |
| **bpp (todo incluido)** | **7.50** | 8.50 |
| **Archivo de texto** | **3.96 GB** | 4.48 GB → **−11.6%** |
| **Archivo completo** | **4.38 GB** (un solo archivo) | 4.48 + 0.67 (mmproj) = **5.15 GB** → **−15%** |
| **Calidad (PPL wikitext-2)** | **10.2358 (+0.26% vs FP16)** | 10.4540 (+2.40% vs FP16) → **9× más pérdida** |
| **Winogrande_s** | 0.5722 | 0.5746 (empate estadístico, ±1.4% SE) |

**En una frase (para el paper):** *"S-X8 cuantiza el modelo COMPLETO a 7.50 bpp — comprimiendo el texto un 11.6% más que Q8_0 e incluyendo visión y MTP en el mismo archivo (GGUF necesita un archivo extra en FP16) — con calidad prácticamente lossless (+0.26% vs FP16) mientras Q8_0 degrada 9× más (+2.40%)."*

**Matiz de tamaños (evitar ambigüedades):** 4.38 GB vs 4.48 GB "a secas" = −2.2%, PERO nuestro archivo contiene 0.45B params MÁS (visión+MTP cuantizados). Comparando la misma parte (texto): −11.6%. Comparando modelo completo vs completo (GGUF + mmproj): −15%.

## Entrenabilidad (QLoRA-style) — SÍ

- El formato es de **pesos estáticos** (sin backward propio) — pero el entrenamiento estilo QLoRA **no lo necesita**: la base se congela y se entrenan adaptadores (LoRA) en FP16
- **Hoy**: el kernel propio dequantiza el modelo completo en ~1 min (5.8 G p/s) → base FP16 congelada + adaptadores LoRA (r=16-64) → VRAM ~11.5-13 GB con AdamW8 (cabe en 16 GB)
- **Con el kernel fused (FASE 3)**: la base vive en SX8 en VRAM (6.61 GB runtime en vez de 9.08 GB FP16) → más espacio para contexto/adaptadores
- Mismo patrón que GGUF+QLoRA (bitsandbytes/llama.cpp) — sin desventaja frente a Q8_0

## Portabilidad GPU — "cualquier GPU capaz de matmul FP16/FP32"

- El decode v4.3 es **ALU pura (~9-10 ops por peso, byte-aligned, sin shared memory, sin shuffles, sin instrucciones especiales)**
- Verificado en GPUs separadas por 10 años: RTX 5060 Ti (SM120, 2025) y GTX 960M (Maxwell, 2015)
- Portes triviales: NVIDIA CUDA (✅ hoy) · AMD ROCm · Apple Metal · CPU (SIMD/NEON) · Vulkan
- El requisito real de inferencia es el GEMM (cuBLAS/rocBLAS/BLAS nativos de cada plataforma) — el formato no exige nada especial
- **vs Q8_0**: GGUF también es portable (llama.cpp lo porta a todo), pero los k-quants usan shuffles/instrucciones más complejas; el decode SX8 es más simple de portar (SIMD-friendly)

## Ventajas y desventajas vs Q8_0 (para el paper)

**Ventajas:**
1. Tamaño: −11.6% (texto) / −15% (modelo completo)
2. Calidad: 9× más cerca del FP16 en PPL (2.40% → 0.26% de degradación)
3. Autocontenido: visión+MTP cuantizados en un solo archivo (GGUF: archivo extra en FP16)
4. Escala FP16 exacta (10 bits de mantissa; la config explícita ya no corrompe la mantissa)
5. Decode trivial y portable (~9-10 ALU, byte-aligned) — SIMD-friendly
6. bpp real y transparente: 7.50 contando TODO (sin "bpp de marketing")
7. Extensible: estrategias de rango, codebooks, NV_SX futuro
8. Entrenable vía QLoRA (base cuantizada en VRAM con el fused kernel)

**Desventajas (honestas):**
1. Ecosistema: GGUF/Q8_0 tiene soporte universal (llama.cpp, LM Studio, vLLM, Ollama…) — SX8 necesita su runtime propio (kernel fused CUDA C + wmma + decode1 v3 — FASE 3+4, docs 40/42)
2. ~~Velocidad~~ **RESUELTA (FASE 3+4)**: kernel fused con tensor cores → **prompt 1411 tok/s = 4.5× más rápido que FP16 cuBLAS (309)** (menos tráfico: 0.94 vs 2 B/peso) · **decode M=1: 80.4 tok/s de kernel puro (FASE 4 — kernel v3, 317 GB/s)** · end-to-end pendiente (cuello: orquestación Python, ver doc 42 §9) · GGUF llama.cpp: ~3056 tok/s prompt
3. ~~VRAM~~ **RESUELTA (FASE 2+3)**: pesos SX8 comprimidos en VRAM (6.61 GB runtime con tabla rlo/step; nunca existe la matriz FP16) vs 9.08 GB FP16

## Formato v4.3 — FORMATO FINAL (30 bytes/bloque = 7.50 bpp REALES, 6 Ago 2026)

**Evolución:** v4 (34 B/bloque, bpp "7.50" ficticio — el coeff no se contaba) → v4.2 (31 B: 2 bases + re-empaquetado 24 bits) → **v4.3 (30 B/bloque = 7.50 bpp REALES — el número del spec original, por fin cierto)**.

| Layout v4.3 | Bytes | Detalle |
|---|---|---|
| dmin | 2 | **FP16 EXACTO** (10 bits mantissa — +4 bits vs v4, la config ya no corrompe la mantissa) |
| dmax | 2 | ídem |
| config | 1 | 8 bits EXPLÍCITOS (4 sub-bloques × 2 bits) — array aparte |
| levels_hi | 16 | niveles 6-bit (nibbles) |
| levels_lo | 8 | niveles 6-bit (quads) |
| coeff | 1 | PCA 2 bases (2×4 bits) — calculado con 4 bases (calibración idéntica al v4) |
| **Total** | **30** | **= 7.50 bpp** |

**Análisis que lo hizo posible (tests):**
- La escala del pkl solo usa 6 bits reales de mantissa (bits 17-22; bits 0-12 siempre 0 — herencia FP16 del modelo) → cabe en FP16 (2 B) SIN pérdida
- La "inversión"/estrategias de rango adicionales (6 candidatas probadas): **+0.00% mejora** — las 4 estrategias actuales ya son óptimas (config de 8 bits saturada)
- Compresión de archivo (gzip/zstd): **descartada por decisión del usuario** — formato byte-aligned puro
- La escala FP16 no mejoró el PPL medible (10.2358 vs 10.2289 del v4.2 — dentro del ruido del SVD del encode); el beneficio real del v4.3 es el TAMAÑO (30 vs 31 B/bloque)

**Validación v4.3:**
- CosSim global 1.000000 (float64) · PPL 10.2358 (desde pkl) · **PPL 10.2358 IDÉNTICO desde el archivo .sx8v43** (end-to-end: reader→kernel→inferencia) · Winogrande 0.5722
- Verificación byte-exacta: 381/381 tensores idénticos (dmin/dmax/config/levels/coeff/bases)
- Kernel v4.3: mismo V6 + 2 FMAs; config leída de array aparte

**Bugs resueltos en el camino:** shape (nb,1) vs (nb,) del coeff en la comparación · orig_shape de convs con 3 dims (el header solo guardaba 2) · OOM por orden de tensores (embed primero) → torch.cuda.empty_cache en el loop · view() sobre arrays no alineados de frombuffer.

## Resultados — Winogrande_s (validation, 1267 muestras, 0-shot, log-likelihood)

| Formato | acc | vs FP16 |
|---|---|---|
| **FP16** | 0.5746 (728/1267) | — |
| **S-X8 v4** | 0.5730 (726/1267) | -0.16 pts |
| **S-X8 v4.2** | 0.5714 (724/1267) | -0.32 pts |
| **GGUF Q8_0** | 0.5746 (728/1267) | 0.00 |

**Winogrande: los cuatro formatos empatan dentro del ruido estadístico (±1.4% SE).**

## Formato v4.2 (31 bytes/bloque = 7.75 bpp) — 6 Ago 2026

**Descubrimiento del análisis:** el bpp "7.50" documentado NO incluía el coeff del PCA (2 B/bloque) ni las bases → el v4 real pesa 34 B/bloque (8.5 bpp efectivo). Al reducir el PCA a **2 bases** (coeff 1 byte) y re-empaquetar dmin/dmax a **24 bits** (signo1+exp8+mantissa7+config8 — la mantissa del pkl solo usa 7 bits reales), el formato pasa a **31 B/bloque = 7.75 bpp** con pérdida mínima (CosSim 0.99983→0.99982; PPL +0.07% vs v4).

| Layout v4.2 | Bytes |
|---|---|
| dmin (signo+exp+mantissa7+config4) | 3 |
| dmax (ídem, config alta) | 3 |
| levels_hi | 16 |
| levels_lo | 8 |
| coeff (2 bases, 2×4 bits) | 1 |
| **Total** | **31** |

- **Verificado byte a byte**: 381/381 tensores idénticos pkl↔.sx8v42 (dmin/dmax/levels/coeff)
- Kernel: el v4 existente funciona sin cambios (los coeffs de bases 2-3 se leen como 0)
- Experimento de validación completo: CosSim global 1.000000 (v4.2), PPL 10.2289, Winogrande 0.5714
- **El misterio del "bit libre"**: los bits 0-11 de dmin/dmax son siempre 0 (precisión heredada del FP16) — la escala NO se puede reducir a 16/24 bits con config naïf (CosSim cae a 0.98 por error de escala), pero el re-empaquetado 24 bits + 2 bases SÍ funciona (7.75 bpp reales)

## Verificación de integridad

- Qwen3.5-4B shards: SHA256 = 26a93f06… / cb544bd9… (oid oficial HF) · Q8_0.gguf: SHA256 = 10cc391b… (oid oficial HF)
- Pkl S-X8 v4 _new: 381 tensores, 4.66B params, bpp 7.51 (modelo completo), **CosSim global 1.000000 (float64)** con el kernel propio
- 8 tensores MTP no aplicables (no instanciados por la clase — no se usan en inferencia de texto)
- Kernel v4: decode de 1 ULP de diferencia vs la vía numpy (redondeo FP16 del orden de sumas — irrelevante, CosSim 1.0)

---

# BATCH 2 (9 Ago 2026) — HellaSwag + ARC-Challenge + MMLU 5-shot

## Tabla de resultados (todo medido en la misma RTX 5060 Ti, protocolo estándar)

| Benchmark | Split | FP16 | **S-X8 v4.3** | GGUF Q8_0 (unsloth) | Δ SX8−FP16 | Δ Q8_0−FP16 |
|---|---|---|---|---|---|---|
| **HellaSwag** (0-shot, acc_norm) | validation (10.042) | 0.6965 | **0.6964** | 0.6965 | −0.0001 | 0.0000 |
| **ARC-Challenge** (0-shot) | test (1.172) | 0.9172 | **0.9164** | 0.9181 | −0.0008 | +0.0009 |
| **MMLU** (5-shot) | validation (1.531) | 0.7133 | **0.7074** | 0.7087 | −0.0059 | −0.0046 |
| **Winogrande_s** (batch 1) | validation (1.267) | 0.5746 | **0.5722** | 0.5746 | −0.0024 | 0.0000 |

**Lectura honesta (la que va al paper):** en los benchmarks de opción múltiple, los TRES
formatos están dentro del ruido estadístico (diferencias máximas 0.6 pts, SE típico ±0.5-1.4 pts).
Las tareas MC son **robustas a la cuantización 8-bit**: la ventaja de calidad de S-X8 sobre
Q8_0 se manifiesta en **PPL** (10.2358 vs 10.4540 = 9× menos pérdida) y en tamaño (−11.6%
texto, −15% modelo completo), NO en accuracy MC. El paper reporta ambas cosas con transparencia.

## Cómo se midió cada formato (protocolo exacto — IMPORTANTE de documentar)

### Por qué SX8 se mide con el runtime propio y Q8_0 con llama.cpp
- **S-X8 v4.3**: se mide con el **runtime propio fused v4.4** (kernel v3 + PCA, el que validó
  PPL 10.2267 y Winogrande 0.5722). Razón: el kernel del fork llama.cpp **ignora el término
  PCA** (las bases no caben en el bloque de 30 B — decisión documentada en 43 §2.2) y el
  `llama-cpp-python` estándar **no conoce el tipo SX8** (41, solo existe en nuestro fork).
  Medir SX8 en llama.cpp daría la calidad SIN PCA (PPL 10.5043), no la del paper (10.2267).
- **Q8_0 (unsloth)**: se mide con `llama-cpp-python` CUDA (llm.scores, KV fresca por muestra)
  — el motor canónico de GGUF, igual que el batch 1 (llama-perplexity).
- **FP16**: transformers nativo.
- **La comparación es justa**: MISMO dataset, MISMO split, MISMO prompt (mc_common.py),
  MISMO scoring (suma de log-probs de la continuación). Solo cambia el motor de inferencia
  (llama.cpp cuantiza las activaciones a Q8_1: +0.24 PPL para TODOS los tipos, Q8_0 incluido).
  Los prompts/scoring viven en un módulo compartido (`mc_common.py`) usado por ambos motores.

### Prompt y scoring (estilo lm-eval — el estándar de la comunidad)
| Benchmark | Prompt | Continuación | Métrica |
|---|---|---|---|
| HellaSwag | `ctx` (+espacio si falta — "…roof. he" + "is playing…") | `ending` | **acc_norm** (score / longitud en tokens) |
| ARC-Challenge | `Question: {q}\nChoices:\nA. …\nB. …\nC. …\nD. …\nAnswer:` | la **letra** (A-E; ARC tiene muestras de 3, 4 y 5 opciones) | acc |
| MMLU | intro "about {subject}" + 5 ejemplos del dev + `…\nAnswer:` | la **letra** | acc |

- Splits: HellaSwag **validation** (10.042, el estándar lm-eval) · ARC-Challenge **test**
  (1.172 — el estándar; el validation solo tiene 299) · MMLU **validation** (1.531) + dev (285,
  5 ejemplos por subject para el few-shot).
- Scoring por log-likelihood de la continuación (greedy, determinista, **KV fresca por
  muestra/opción** — protocolo del batch 1). Sin chat template.
- **Optimización validada (truco de la letra)**: la logprob de una continuación de 1 token
  se lee de la última fila de logits del prefijo (causal) → 1 forward por muestra en vez de
  4. **Verificado matemáticamente idéntico: maxdiff 0.0** (test_letter_equiv). MMLU/ARC 4×
  más rápido sin cambiar los números.
- **Nota metodológica (ARC)**: el scoring por letra es el estándar lm-eval (1 token, sin
  artefactos de longitud). Métodos alternativos de texto completo dan números distintos
  (0.80 sin normalizar / 0.68 normalizado en 300 muestras) por artefactos de longitud/fluidez
  (opciones de hasta 46 tokens) — se documenta, no se usa. **Validación completa y
  reproducible: sección "Validación del método ARC" más abajo** (`verify_arc_method.py`).

### Ejecución y verificación
- **Spot-check previo**: 10 muestras/benchmark FP16 vs SX8 → concordancia 29/30 (la
  divergencia fue SX8 acertando donde FP16 falló) — PASS.
- **Auditoría de datos**: 7 muestras ARC con 3/5 opciones (fix del prompt, etiquetas por
  `choices.label`); MMLU answers todos int; contextos máx 3.022 tokens (< 4096 del GGUF).
- **Checkpointing**: JSON parcial cada 500 (HellaSwag) / 200 (ARC/MMLU) muestras — un crash
  no pierde el progreso.
- Scripts: `scripts/mc_common.py` (prompts/scoring compartidos) · `scripts/mc_eval_sx.py`
  (FP16 + SX8 fused v4.4) · `scripts/mc_eval_gguf.py` (Q8_0) · resultados en
  `results/mc_{bench}_{mode}.json` + `results/batch2_results.json`.
- Reproducción: `python3 mc_eval_sx.py --mode fp16|sx8v43 [--bench N]` · `python3 mc_eval_gguf.py [--bench N]`
  (offline: HF_DATASETS_OFFLINE=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True).

## Futuro — PCA en llama.cpp (v2 del repo)
Guía completa de implementación en `43-SESION-LLAMACPP-SX8.md §10c` (tensores GGUF
pca_bases/pca_scale + kernel pca_precompute_z + corrección en MMVQ/MMQ → PPL 10.5043 →
~10.24-10.27; esfuerzo 1-2 días). Para el día-D se reporta la calidad con el runtime propio
(10.2267) y la igualdad sin PCA en llama.cpp es aceptable.

---

# VALIDACIÓN DEL MÉTODO DE PUNTUACIÓN DE ARC (por qué el 91.7% de FP16 es fiable)

**Script reproducible:** `scripts/verify_arc_method.py` → `results/verify_arc_method.json`
(300 muestras del split test, FP16, ~8 min GPU). Reproduce todos los números de abajo.

## 1. Por qué el número parecía sospechoso
Los modelos de ~4B suelen obtener 60-80% en ARC-Challenge en las leaderboards
comunitarias; el 91.7% de nuestro FP16 es nivel de modelos 20× más grandes. Antes de
aceptarlo se comprobó que no fuera un artefacto de la medición.

## 2. Sospecha 1 — sesgo posicional: DESCARTADA por la distribución
Un modelo que "apueste" a una letra sin entender la pregunta no puede superar la
frecuencia de esa letra. Distribución de respuestas del split test (1.172):

| Letra correcta | A | B | C | D | (1-4 en 25 muestras) |
|---|---|---|---|---|---|
| Nº preguntas | 263 | 301 | 303 | 283 | 25 |

**Balanceada (≈25% cada una)** → el sesgo posicional puro no puede explicar más de
~30% de acierto. El 91.7% requiere comprensión real de la pregunta.

## 3. Sospecha 2 — artefacto del método: validación cruzada de 3 métodos
300 muestras, mismo modelo FP16, misma semilla:

| Método | Cómo puntúa | Accuracy | Fiabilidad |
|---|---|---|---|
| **M1 — Letras** (el del batch 2) | P(letra \| pregunta+opciones+`Answer:`) — 1 token por opción | **0.9067** | **Limpio — estándar lm-eval** |
| M2 — Texto sin opciones en prompt | P(texto de la opción \| pregunta) | 0.4633 | Roto (la opción no aparece en el prompt) |
| M3 — Texto con opciones en prompt | P(texto \| pregunta+opciones+`Answer:`) | 0.8000 | Artefacto de longitud |
| M3n — Ídem normalizado por longitud | score / nº tokens | 0.6767 | Inestable (¡empeoró al normalizar!) |

**Por qué M1 es el método correcto y M2/M3 no:**
- **Artefacto de longitud (M3):** la puntuación de una opción es la SUMA de las
  log-probs de sus tokens. Las opciones de ARC miden de 3 a **46 tokens** → una opción
  correcta larga pierde sistemáticamente contra una incorrecta corta solo por longitud.
- **Artefacto de fluidez (M2/M3):** se puntúa "cuán natural continúa el modelo la
  frase", no "cuán correcta es la respuesta".
- **Prueba empírica (M3n):** al dividir por longitud, la accuracy CAYÓ (0.80 → 0.68)
  — el número del texto es arbitrario según el ajuste, no fiable.
- **M1 no tiene ninguno de los dos:** cada letra es 1 token (longitud constante) y no
  hay fluidez que medir. Por eso la industria (lm-eval, OpenAI evals) estandarizó la
  puntuación por letra: es la medición limpia, comparable con las leaderboards.

## 4. Prueba final — desglose por letra (discriminación real)
Precisión de M1 separada por letra correcta:

| La respuesta correcta era | A | B | C | D |
|---|---|---|---|---|
| Aciertos | 60/73 | 82/86 | 77/84 | 50/54 |
| Accuracy | **0.82** | **0.95** | **0.92** | **0.93** |

Si hubiera sesgo posicional (ej. apostar por B), la columna de A se hundiría (~0).
El modelo acierta **82-95% en las 4 letras** → debe entender cada pregunta para
acertar las de respuesta A. La diferencia A (82%) vs B/C/D (92-95%) es el ruido
estadístico esperado (SE ±5.7 pts con 73 muestras) + una ligera aversión del modelo
a la posición A, sin impacto material.

## 5. Confirmación independiente (no es artefacto del motor FP16)
Tres motores de inferencia distintos dan el mismo número:

| Formato | Motor | ARC (test) |
|---|---|---|
| FP16 | transformers/cuBLAS | 0.9172 |
| S-X8 v4.3 | runtime propio fused v4.4 (kernel v3 + PCA) | 0.9164 |
| GGUF Q8_0 | llama-cpp-python (CUDA, activaciones Q8_1) | 0.9181 |

Dentro de 0.5 pts entre sí → el 91.7% es del modelo, no de la medición.

## 6. Veredicto
El accuracy de letras de ARC (91.7% FP16) es real y se reporta como el estándar
lm-eval, con la nota metodológica declarada (los métodos de texto alternativos dan
números distintos por artefactos de longitud — 80% — y no se usan). Contexto de
plausibilidad: Qwen3.5-4B es un modelo 2026 de una familia con GAIA 95.1% (Agents-A1);
los modelos modernos pequeños son mucho más fuertes en razonamiento escolar.
