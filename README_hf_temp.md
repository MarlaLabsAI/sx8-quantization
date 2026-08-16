---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
  - quantization
  - s-x8
  - llm
  - compression
pipeline_tag: text-generation
---

# Qwen3.5-4B — S-X8 v4.3 (7.50 bpp)

Qwen3.5-4B quantized with the **S-X8 v4.3** format: 7.50 bits per weight (fully accounted), FP16-level
quality and a portable decoder (no tensor cores, no shared memory — runs on any GPU).

**Paper (Zenodo):** [10.5281/zenodo.21922640](https://doi.org/10.5281/zenodo.21922640) · **Repo:**
[github.com/MarlaLabsAI/sx8-quantization](https://github.com/MarlaLabsAI/sx8-quantization)

**Author:** Martí Vidal Leandro (MarlaLabs) · [LinkedIn](https://www.linkedin.com/in/vidalmarti/) ·
[X / Twitter](https://x.com/MarlaLabsAI) · [info@marlalabs.com](mailto:info@marlalabs.com) ·
[marlalabs.com](https://marlalabs.com)

**Authorship registration (Safe Creative):** [ID 2608136715874](https://www.safecreative.org/work/2608136715874)

## v1.1 (2026-08-16) — Corrections & complete standalone container

**The `.sx8v43` container is now a COMPLETE standalone model.** It embeds the model config and all
non-quantized tensors (section `SXT1`) — **no base model is needed anymore**. The Python runtime
(`eval_common.load_model_standalone`) builds and runs the model from the container + tokenizer only.

- **Fix — degenerate-block decode**: in blocks where `min ≈ max` (typical of vision towers), the
  decode fallback forced `step = 1/63`, reconstructing ≈1.0 where the original was ≈0. Fixed to
  `step = 1e-10` in all kernels (Python, CUDA) and in the llama.cpp fork (`llama-cpp-sx8.patch`
  regenerated). Text tensors contain zero degenerate blocks → **all published text benchmarks
  remain valid**.
- **Re-validated on the v2 container** (same protocols): **PPL wikitext-2 = 10.2364** (ref 10.2267) ·
  **Winogrande_s = 0.5722** (ref 0.5722) · **decode VRAM = 3.720 GB** (ref 3.955 GB).
- **How to use** (Option B below): download this repo (container + tokenizer + scripts) → run — no
  Qwen base model required.
- Full details: `CHANGELOG.md` in this repo and in the GitHub repo.

## Quality (Qwen3.5-4B, RTX 5060 Ti)

| Metric | FP16 | **S-X8 v4.3** | Q8_0 |
|---|---|---|---|
| PPL wikitext-2 (PCA runtime) | 10.2090 | **10.2267** (+0.17%) | 10.4540 (+2.40%) |
| Winogrande_s | 0.5746 | **0.5722** | 0.5746 |
| HellaSwag (0-shot) | 0.6965 | **0.6964** | 0.6965 |
| ARC-Challenge (0-shot) | 0.9172 | **0.9164** | 0.9181 |
| MMLU (5-shot) | 0.7133 | **0.7074** | 0.7087 |

## Size

| | **S-X8 v4.3** | Q8_0 |
|---|---|---|
| Text file | **3.96 GB** (−11.6%) | 4.48 GB |
| Complete model (vision+MTP) | **4.38 GB** (one file) | 5.15 GB (GGUF+mmproj) |
| Weights VRAM (decode M=1) | **3.720 GB** | 4.48 GB |

## Files

- `Qwen3.5-4B-SX8v43.sx8` (4.38 GB) — **complete standalone container** (config + quantized 2D + 1D tensors)
- `Qwen3.5-4B-SX8v43.gguf` (3.83 GiB) — GGUF with native type `GGML_TYPE_SX8` (requires the llama.cpp
  fork with S-X8 support)
- `tokenizer.json` · `tokenizer_config.json` · `chat_template.jinja` · `vocab.json` · `merges.txt` —
  tokenizer (needed for Option B)
- `CHANGELOG.md` — version history (v1.1 corrections)

## How to use the model

The model comes in **two containers** (same S-X8 weights): the native `.sx8v43` (for the S-X8 Python
runtime) and the `.gguf` (S-X8 inside the GGUF envelope, for the llama.cpp fork). The runtime code is
in `scripts/` (kernels in `cuda/` — compiled automatically on first use, requires CUDA toolkit).

### INSTALL (Python runtime)

```bash
pip install -r requirements.txt        # torch, transformers, numba, safetensors
# Requiere: NVIDIA GPU + CUDA toolkit (nvcc) para compilar los kernels fused (automático, 1ª vez)
```

### Chat — two engines (honest table)

| Engine | Prompt | Generation | VRAM | Notes |
|---|---|---|---|---|
| **llama.cpp fork + `.gguf`** (fast chat) | **~1877 tok/s** | **~63.79 tok/s** | **~4-5 GB** | The paper's interactive speeds. Build once (5 commands, see below) |
| **Python runtime** (`infer_sx8.py`) | **~273 tok/s** (compact GEMM) | ~2 tok/s | **4.31 GB** (= loaded compact + KV) | No build needed (only CUDA toolkit). Compact GEMM M≥32 decodes on-the-fly — no FP16 materialization |

**Option A — fast chat at the declared speeds (llama.cpp fork):**
```bash
./run_llama_chat.sh build      # once: clones llama.cpp @7c203670f, applies llama-cpp-sx8.patch, builds CUDA
./run_llama_chat.sh            # interactive chat with Qwen3.5-4B-SX8v43.gguf (--jinja, thinking on)
```
Measured on RTX 5060 Ti: decode 63.79 tok/s (faster than Q8_0 in real use), prompt 1877 tok/s (MMQ),
VRAM ~4-5 GB (less than Q8_0's 4.48 GB weights).

**Option B — Python runtime (no build, works everywhere with CUDA):**
```bash
cd scripts
python3 infer_sx8.py --container ../Qwen3.5-4B-SX8v43.sx8 --tokenizer-dir ..
# chat loop with thinking mode; token-by-token (compact decode1 kernel).
# Honest: per-token Python overhead gives ~2 tok/s; the kernel itself runs at
# 80.4 tok/s (bench_decode_v44_v2.py). A fused prompt kernel (MMQ-equivalent)
# is in development.
```

### Reproducing the paper numbers (RTX 5060 Ti)

| Paper claim | Command |
|---|---|
| Decode kernel 80.4 tok/s · **VRAM 3.720 GB** (< Q8_0) | `cd scripts && python3 bench_decode_v44_v2.py --container ../Qwen3.5-4B-SX8v43.sx8` |
| PPL wikitext-2 **10.2267** (PCA runtime) | `cd scripts && python3 ppl_fused_v2_lowvram.py` |
| PPL **10.5043** (llama.cpp) | `./build/bin/llama-perplexity -m Qwen3.5-4B-SX8v43.gguf -c 512` (fork) |
| Decode **63.79 tok/s** · prompt **1877 tok/s** (llama.cpp) | `./build/bin/llama-bench -m Qwen3.5-4B-SX8v43.gguf -p 128 -n 64` (fork) |
| Winogrande_s **0.5722** | `cd scripts && python3 winogrande_fused_revalidate_v2.py` |

All claims were re-validated on the v2 container (see `CHANGELOG.md`): PPL 10.2364, Winogrande 0.5722,
decode VRAM 3.720 GB.

## How to cite

> Vidal Leandro, M. (2026). S-X8 v4.3: A 7.50-Bits-Per-Weight Quantization Format with FP16-Level Quality
> and Portable Decoding. Zenodo. https://doi.org/10.5281/zenodo.21922640

## Contact

Looking for collaborations, integration help or opportunities? Reach out:

- **LinkedIn (profile):** https://www.linkedin.com/in/vidalmarti/
- **LinkedIn (company):** https://www.linkedin.com/company/marlalabs/
- **X / Twitter:** https://x.com/MarlaLabsAI
- **Email:** info@marlalabs.com
- **Web:** https://marlalabs.com

## Usage files in this repo (self-contained)
- `llama-cpp-sx8.patch` — llama.cpp fork patch (apply on commit 7c203670f)
- `SX8_FLASH_V4_3_SPEC.md` · `SX8_FLASH_V4_3_CONTAINER.md` — format and container specs
- `S-X-METHODOLOGY.md` — methodology authorship
- `eval_common.py` · `kernel_sx8_v43.py` · `kernel_sx8_v4.py` · `sx8_container_v43.py` — runtime for the `.sx8v43`
- `paper-sx8.pdf` · `paper-sx8-ES.pdf` — the paper (EN/ES)

## Transparency

The conceptual seeds of S-X8 come from an independent mathematical analysis of the image of the Shroud of
Turin; the full study is published in the project repo with its own re-verification: github.com/MarlaLabsAI/sx8-quantization → `shroud-turin-study/`. See the paper's Appendix A
and `docs/IDEA-PROVENANCE.md`. The format itself is validated empirically in this repository; the study
is disclosed only as the source of inspiration.

## License and attribution

Apache-2.0. The S-X8 format, kernels and scripts in this repository are **original work** (Copyright
(C) 2026 Martí Vidal Leandro), not a Qwen work. The quantized model uses **Qwen3.5-4B** by the Qwen
Team (Alibaba Group), Apache-2.0 ([model card](https://huggingface.co/Qwen/Qwen3.5-4B)) as the base
model; only the weights were quantized to the S-X8 v4.3 format (see `NOTICE`), no other modifications.

---

# Qwen3.5-4B — S-X8 v4.3 (7,50 bpp) [ES]

Qwen3.5-4B cuantizado con el formato **S-X8 v4.3**: 7,50 bits por peso (contabilizados al completo),
calidad de nivel FP16 y decodificador portable (sin tensor cores, sin memoria compartida — funciona en
cualquier GPU).

**Paper (Zenodo):** [10.5281/zenodo.21922640](https://doi.org/10.5281/zenodo.21922640) · **Repo:**
[github.com/MarlaLabsAI/sx8-quantization](https://github.com/MarlaLabsAI/sx8-quantization)

**Autor:** Martí Vidal Leandro (MarlaLabs) · [LinkedIn](https://www.linkedin.com/in/vidalmarti/) ·
[X / Twitter](https://x.com/MarlaLabsAI) · [info@marlalabs.com](mailto:info@marlalabs.com) ·
[marlalabs.com](https://marlalabs.com)

**Registro de autoría (Safe Creative):** [ID 2608136715874](https://www.safecreative.org/work/2608136715874)

## v1.1 (2026-08-16) — Correcciones y contenedor completo autosuficiente

**El contenedor `.sx8v43` es ahora un MODELO COMPLETO y autosuficiente.** Incluye la config del modelo
y todos los tensores no cuantizados (sección `SXT1`) — **ya no se necesita el modelo base**. El runtime
Python (`eval_common.load_model_standalone`) construye y ejecuta el modelo solo con el contenedor + tokenizer.

- **Fix — decode de bloques degenerados**: en bloques donde `min ≈ max` (típicos de la torre de visión),
  el fallback de decode forzaba `step = 1/63`, reconstruyendo ≈1,0 donde el original era ≈0. Corregido a
  `step = 1e-10` en todos los kernels (Python, CUDA) y en el fork de llama.cpp (`llama-cpp-sx8.patch`
  regenerado). Los tensores de texto tienen cero bloques degenerados → **todos los benchmarks de texto
  publicados siguen siendo válidos**.
- **Re-validado con el contenedor v2** (mismos protocolos): **PPL wikitext-2 = 10,2364** (ref 10,2267) ·
  **Winogrande_s = 0,5722** (ref 0,5722) · **VRAM decode = 3,720 GB** (ref 3,955 GB).
- **Cómo usar** (Opción B abajo): descarga este repo (contenedor + tokenizer + scripts) → ejecuta — sin
  modelo base de Qwen.
- Detalles: `CHANGELOG.md` en este repo y en el repo de GitHub.

## Calidad (Qwen3.5-4B, RTX 5060 Ti)

| Métrica | FP16 | **S-X8 v4.3** | Q8_0 |
|---|---|---|---|
| PPL wikitext-2 (runtime PCA) | 10,2090 | **10,2267** (+0,17%) | 10,4540 (+2,40%) |
| Winogrande_s | 0,5746 | **0,5722** | 0,5746 |
| HellaSwag (0-shot) | 0,6965 | **0,6964** | 0,6965 |
| ARC-Challenge (0-shot) | 0,9172 | **0,9164** | 0,9181 |
| MMLU (5-shot) | 0,7133 | **0,7074** | 0,7087 |

## Tamaño

| | **S-X8 v4.3** | Q8_0 |
|---|---|---|
| Archivo de texto | **3,96 GB** (−11,6%) | 4,48 GB |
| Modelo completo (visión+MTP) | **4,38 GB** (un archivo) | 5,15 GB (GGUF+mmproj) |
| VRAM pesos (decode M=1) | **3,720 GB** | 4,48 GB |

## Cómo usar el modelo [ES — resumen]

Dos motores (mismos pesos S-X8 en dos contenedores): **llama.cpp fork + `.gguf`** = chat rápido a las
velocidades declaradas (decode 63.79 tok/s, prompt 1877 tok/s, VRAM ~4-5 GB; construye el fork con
`./run_llama_chat.sh build`); **runtime Python** (`scripts/infer_sx8.py`) = funciona sin build
(requiere CUDA toolkit), honestamente más lento por token (~2 tok/s). Detalles completos en la parte EN.

## Ficheros

- `Qwen3.5-4B-SX8v43.sx8` (4,38 GB) — contenedor nativo byte-aligned (verificable byte-exacto)
- `Qwen3.5-4B-SX8v43.gguf` (3,83 GiB) — GGUF con tipo nativo `GGML_TYPE_SX8` (requiere el fork de
  llama.cpp con soporte S-X8)

## Cómo usar el modelo

**Opción A — GGUF con el fork de llama.cpp (recomendado):**
1. Descarga el patch: `github.com/MarlaLabsAI/sx8-quantization` → `llama-cpp-sx8.patch`
2. Clona llama.cpp en el commit `7c203670f` y aplica: `git apply llama-cpp-sx8.patch`
3. Compila con CUDA (`-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120`)
4. Ejecuta:
   ```
   ./build/bin/llama-cli -m Qwen3.5-4B-SX8v43.gguf -ngl 99 -p "Hola, ¿cómo estás?"
   ```
   Decode medido: 63,79 tok/s en RTX 5060 Ti (más rápido que Q8_0 en uso real).

**Opción B — Contenedor .sx8v43 con el runtime S-X8 (desarrolladores/investigación):**
1. Clona `github.com/MarlaLabsAI/sx8-quantization` (scripts: `eval_common.py`, `sx8_container_v43.py`, kernels)
2. Carga: `from eval_common import load_model; m, tok, meta = load_model(quantized=True, mode="v43", source_file="Qwen3.5-4B-SX8v43.sx8")`
3. El contenedor es byte-aligned y verificable byte-exacto (381/381 tensores).

**Enlaces:** [Repo GitHub](https://github.com/MarlaLabsAI/sx8-quantization) · [Paper (Zenodo, DOI)](https://doi.org/10.5281/zenodo.21922640)

## Cómo citar

> Vidal Leandro, M. (2026). S-X8 v4.3: Un formato de cuantización a 7,50 bits por peso con calidad de
> FP16 y decodificación portable. Zenodo. https://doi.org/10.5281/zenodo.21922640

## Contacto

¿Buscas colaboración, ayuda con la integración u oportunidades? Contacta:

- **LinkedIn (perfil):** https://www.linkedin.com/in/vidalmarti/
- **LinkedIn (empresa):** https://www.linkedin.com/company/marlalabs/
- **X / Twitter:** https://x.com/MarlaLabsAI
- **Email:** info@marlalabs.com
- **Web:** https://marlalabs.com

## Archivos de uso en este repo (autosuficiente)
- `llama-cpp-sx8.patch` — patch del fork de llama.cpp (aplicar sobre el commit 7c203670f)
- `SX8_FLASH_V4_3_SPEC.md` · `SX8_FLASH_V4_3_CONTAINER.md` — spec del formato y del contenedor
- `S-X-METHODOLOGY.md` — autoría de la metodología
- `eval_common.py` · `kernel_sx8_v43.py` · `kernel_sx8_v4.py` · `sx8_container_v43.py` — runtime para el `.sx8v43`
- `paper-sx8.pdf` · `paper-sx8-ES.pdf` — el paper (EN/ES)

## Transparencia

Las semillas conceptuales de S-X8 provienen de un análisis matemático independiente de la imagen de la
Sábana Santa de Turín; el estudio completo se publica en el repo del proyecto con su propia re-verificación: github.com/MarlaLabsAI/sx8-quantization → `shroud-turin-study/`. Ver el
Apéndice A del paper y `docs/IDEA-PROVENANCE.md` del repo. El formato en sí está validado empíricamente
en este repositorio; el estudio se menciona solo como fuente de inspiración.

## Licencia y atribución

Apache-2.0. El formato S-X8, los kernels y los scripts de este repositorio son **obra original**
(Copyright (C) 2026 Martí Vidal Leandro), no un trabajo de Qwen. El modelo cuantizado usa
**Qwen3.5-4B** de Qwen Team (Alibaba Group), Apache-2.0 ([model card](https://huggingface.co/Qwen/Qwen3.5-4B))
como modelo base; solo se cuantizaron los pesos al formato S-X8 v4.3 (ver `NOTICE`), sin otras
modificaciones.
