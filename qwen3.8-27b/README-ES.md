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

# Qwen3.8-27B-SX8 — Variante cuantizada con S-X8 v4.3 de Qwen3.8-27B

**S-X8 v4.3** es un formato de cuantización de pesos para modelos de lenguaje: **30 bytes por bloque = 7.50 bpp**
(completamente contabilizado), calidad a nivel FP16, y un decodificador portable alineado a byte.
Este repositorio contiene el **Qwen3.8-27B cuantizado a S-X8 v4.3** — una variante más grande de la
misma cuantización aplicada a [Qwen3.5-4B](https://huggingface.co/marlalabsAI/Qwen3.5-4B-SX8).

## Formato validado en Qwen3.5-4B

El formato S-X8 v4.3 está validado en **Qwen3.5-4B** (una RTX 5060 Ti). Consulta el repositorio del 4B
para la tabla completa de benchmarks y el paper:

| Métrica | FP16 | **S-X8 v4.3 (4B)** | Q8_0 |
|---|---|---|---|
| PPL wikitext-2 (runtime PCA) | 10.2090 | **10.2267** (+0.17%) | 10.4540 (+2.40%) |
| Winogrande_s | 0.5746 | **0.5722** | 0.5746 |
| Decode real (fork llama.cpp) | — | **63.79 tok/s** | 40–54 tok/s |
| VRAM decode M=1 | — | **3.720 GB** | 4.48 GB |

**Paper (Zenodo, DOI):** [10.5281/zenodo.21922640](https://doi.org/10.5281/zenodo.21922640)

## Este modelo 27B

Cuantizado con el **mismo pipeline** que el 4B: S-X8 v4.3 por tensor, 666 tensores,
CosSim **0.999753** (media sobre todos los tensores), 7.50 bpp, **26.1 GB** de contenedor.

> **Validado funcionalmente: el modelo se ejecuta y responde correctamente en una prueba local**
> (mismo protocolo que el modelo 4B). Los benchmarks formales del 27B (PPL, Winogrande, ARC, MMLU)
> están **pendientes** — se medirán en una GPU más grande (actualizaremos cuando se midan).
> Se espera que el modelo base más grande ofrezca mejor calidad absoluta (mismo formato,
> base más capaz).

## Dos envases, un modelo

El modelo S-X8 existe en **dos envases** — los mismos pesos dentro:

| Archivo | Envase | Usado por |
|---|---|---|
| `Qwen3.8-27B-SX8v43.sx8` | formato nativo S-X8 v4.3 (completo: config + 2D cuantizado + 1D + MTP + visión) | nuestro runtime Python (`scripts/infer_sx8.py`, kernels CUDA) |
| `Qwen3.8-27B-SX8v43.gguf` | GGUF (el envase que lee llama.cpp) con pesos S-X8 dentro — tipo nativo `GGML_TYPE_SX8 = 41` | fork llama.cpp (`run_llama_chat.sh`) |

Piénsalo como un `.zip`: el envase es el mismo, lo de dentro puede ser lo que sea. El GGUF es el
**mismo S-X8** dentro del envase que lee llama.cpp. El MTP vive en el `.sx8v43` (excluido del GGUF,
mismo criterio que el 4B).

## Inicio rápido

**Vía A — fork llama.cpp (chat rápido):**

```bash
./run_llama_chat.sh build   # una vez (clona llama.cpp, aplica el patch, compila)
./run_llama_chat.sh         # chat
```

- GPU ≥ 32 GB: `NGL=999 ./run_llama_chat.sh` (todo en VRAM)
- GPU 16 GB: `NGL=28 ./run_llama_chat.sh` (28 capas GPU + resto CPU), o memoria unificada:
  `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 NGL=999 ./run_llama_chat.sh` (cómputo 100% GPU, spill por PCIe)

**Vía B — runtime Python con nuestros kernels CUDA (envase nativo):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/infer_sx8.py --container Qwen3.8-27B-SX8v43.sx8
```

Guía completa caso por caso: [`QUICK-DEPLOY.md`](QUICK-DEPLOY.md) (GPUs 8–32+ GB, solo CPU, Windows/WSL2).

## Estructura del repositorio

```
├── Qwen3.8-27B-SX8v43.sx8      contenedor nativo S-X8 (modelo completo)
├── Qwen3.8-27B-SX8v43.gguf      S-X8 dentro de GGUF (llama.cpp)
├── tokenizer/                  (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, chat_template.jinja)
├── scripts/                    cuantización (27B) · contenedores · runtime (eval_common, infer_sx8) · kernels
├── cuda/                       kernels CUDA S-X8 (decode1_v3, fused_wmma, gemm_compact, embed_v44...)
├── llama-cpp-sx8.patch         fork llama.cpp: S-X8 como tipo nativo (decode MMVQ + prompt MMQ + AVX2 CPU)
├── QUICK-DEPLOY.md             guía de despliegue por caso de GPU
├── DEPLOY-AI.md                prompt para que otra IA despliegue todo de forma autónoma
├── SX8_FLASH_V4_3_SPEC.md      especificación del bloque (30 B/bloque)
├── SX8_FLASH_V4_3_CONTAINER.md especificación del contenedor (v1.1 SXT1)
├── S-X-METHODOLOGY.md          metodología
└── CHANGELOG.md
```

## Transparencia

Las semillas conceptuales de S-X8 provienen de un análisis matemático independiente de la imagen de
la Sábana Santa de Turín; el estudio completo está publicado en el repositorio del proyecto con su
propia re-verificación: [github.com/MarlaLabsAI/sx8-quantization → shroud-turin-study/](https://github.com/MarlaLabsAI/sx8-quantization/tree/main/shroud-turin-study).
Ver el Apéndice A del paper y [IDEA-PROVENANCE.md](IDEA-PROVENANCE.md). El formato en sí está
validado empíricamente en este repositorio; el estudio se divulga solo como fuente de inspiración,
con total honestidad sobre lo que se puede y no se puede corroborar.

## Licencia y atribución

Apache-2.0. El formato S-X8, los kernels y los scripts de este repositorio son **obra original**
(Copyright (C) 2026 Martí Vidal Leandro), no un trabajo de Qwen. El modelo cuantizado usa
**Qwen3.8-27B de Qwen Team** (Alibaba Group), Apache-2.0 (model card) como modelo base;
solo se cuantizaron los pesos al formato S-X8 v4.3 (ver [NOTICE](NOTICE)), sin otras modificaciones.

## Cómo citar

Si usas este trabajo, cítalo como:

> Vidal Leandro, M. (2026). **S-X8 v4.3: A 7.50-Bits-Per-Weight Quantization Format with
> FP16-Level Quality and Portable Decoding.** Zenodo. https://doi.org/10.5281/zenodo.21922640

## Contacto

¿Buscas colaboraciones, ayuda de integración u oportunidades? Escríbenos:

- **LinkedIn (perfil):** https://www.linkedin.com/in/vidalmarti/
- **LinkedIn (empresa):** https://www.linkedin.com/company/marlalabs/
- **X / Twitter:** https://x.com/MarlaLabsAI
- **Email:** info@marlalabs.com
- **Web:** https://marlalabs.com

## Apoyo

Si este trabajo te ayuda, considera una donación: [Ko-fi](https://ko-fi.com/) · [GitHub Sponsors](https://github.com/sponsors) — detalles en FUNDING.yml.

## Despliegue con un agente de IA

¿Quieres que otra IA (Claude Code, opencode, aider...) se encargue de todo el despliegue por ti —
en cualquier sistema operativo y hardware, dentro de los límites físicos? Copia el prompt de
[`DEPLOY-AI.md`](DEPLOY-AI.md) y entrégaselo al agente: detectará tu sistema (SO, GPU, RAM),
instalará lo necesario, descargará, ejecutará y verificará el modelo por su cuenta.

## Licencia / NOTICE

Apache-2.0. El formato S-X8 es obra original del autor; este modelo es una **variante cuantizada de
Qwen3.8-27B** (Qwen Team, Alibaba Group, Apache-2.0) — solo se cuantizaron los pesos a S-X8 v4.3,
sin otras modificaciones. Ver [NOTICE](NOTICE).