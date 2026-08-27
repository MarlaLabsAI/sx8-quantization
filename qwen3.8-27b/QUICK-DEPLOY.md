# QUICK-DEPLOY — S-X8 models (Qwen3.8-27B) — guía por casos

Esta guía explica cómo ejecutar el **Qwen3.8-27B-SX8** en distintos tipos de máquina.
El formato S-X8 v4.3 está validado en **Qwen3.5-4B-SX8** (PPL 10.2267, Winogrande 0.5722,
decode 63.79 tok/s, VRAM 3.720 GB) — ver https://huggingface.co/marlalabsAI/Qwen3.5-4B-SX8
para benchmarks y resultados formales. Este modelo 27B usa el MISMO pipeline (CosSim 0.999753
sobre 666 tensores) y está validado funcionalmente en una prueba local.

## Contenido del repositorio

- `Qwen3.8-27B-SX8v43.sx8` — contenedor nativo S-X8 v4.3 (formato propio, completo:
  config + 666 tensores cuantizados + 533 tensores 1D + MTP + visión).
- `Qwen3.8-27B-SX8v43.gguf` — el MISMO modelo S-X8 dentro del contenedor que lee
  llama.cpp (el GGUF es solo el envase; dentro hay pesos S-X8, tipo nativo
  `GGML_TYPE_SX8 = 41`). MTP excluido del GGUF (vive en el .sx8v43).
- `scripts/` — quantización, contenedores, runtime Python con kernels CUDA propios.
- `cuda/` — kernels CUDA S-X8 (decode1_v3, fused_wmma, gemm_compact, embed_v44...).
- `llama-cpp-sx8.patch` — fork de llama.cpp con S-X8 como tipo nativo
  (decode MMVQ + prompt MMQ en GPU, y kernel AVX2 en CPU).

## 1. REQUISITOS

- Ubuntu 22.04/24.04 (Linux) · NVIDIA GPU con CUDA (Blackwell recomendado: RTX 50)
- CUDA toolkit (nvcc) · cmake · gcc · git · python3 ≥ 3.10
- RAM: 32 GB recomendada (el 27B cuantizado son ~26 GB en memoria al cargar)

## 1b. REQUISITOS DE HARDWARE RELATIVOS

Orientación relativa (sin benchmarks formales del 27B todavía — ver §8 NOTAS DE HONESTIDAD):
la experiencia depende de la relación entre el modelo (~24.3 GiB) y tu hardware.

| Perfil de máquina | Cómo corre | Experiencia relativa |
|---|---|---|
| **GPU ≥ 32 GB VRAM** (RTX 5080 Ti, 5090...) | `NGL=999` — modelo entero en VRAM | **Óptima**: máxima velocidad, sin spill |
| **GPU 16 GB** (RTX 5060 Ti, 5080...) | `NGL=28` (default) o memoria unificada | Funcional: 28 capas GPU + resto CPU (o 100% GPU con spill PCIe) |
| **APU / iGPU con ≥ 32 GB RAM unificada** | iGPU vía Vulkan (o CPU AVX2 como fallback) | Funcional: responde, no es fluido |
| **GPU 8–12 GB** | `NGL=10` (ajustar) | Básica: la mayor parte computa en CPU |
| **Windows** | WSL2 + CUDA | No probado nativamente |

- **Punto de referencia (4B validado)**: con una RTX 5060 Ti (16 GB), el Qwen3.5-4B-SX8
  hace decode 63.79 tok/s con solo 3.720 GB de VRAM (ver repo del 4B). El 27B es el
  MISMO formato con una base más grande: esperablemente más lento por tamaño, con la
  misma calidad por bit.
- **Con conocimientos medios**: `./run_llama_chat.sh build` + `./run_llama_chat.sh`
  cubren todo el despliegue; y `DEPLOY-AI.md` permite que otra IA lo haga por ti
  (detecta tu hardware, instala, descarga y verifica).

## 2. DESCARGAR

```bash
huggingface-cli download marlalabsAI/Qwen3.8-27B-SX8 --local-dir ./sx8
# o con HF hub Python:
pip install -U huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('marlalabsAI/Qwen3.8-27B-SX8', local_dir='./sx8')"
```

## 3. INSTALAR (runtime Python — kernels CUDA propios)

```bash
cd sx8
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 4. CHAT — VÍA A: llama.cpp fork + .gguf (la rápida)

```bash
./run_llama_chat.sh build          # una vez (clona llama.cpp + aplica el patch + compila)
./run_llama_chat.sh                # chat interactivo
```

| Tu GPU | Qué usar | Notas |
|---|---|---|
| **≥ 32 GB VRAM** | `NGL=999 ./run_llama_chat.sh` | Todo el modelo a VRAM. Máxima velocidad. |
| **16 GB VRAM (RTX 5060 Ti, 5080...)** | `NGL=28 ./run_llama_chat.sh` (default) | 28 capas GPU + resto CPU. |
| **16 GB + memoria unificada** | `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 NGL=999 ./run_llama_chat.sh` | Cómputo 100% GPU; el spill de VRAM→RAM lo gestiona CUDA (cuello: PCIe, no CPU). |
| **8–12 GB VRAM** | `NGL=10 ./run_llama_chat.sh` (ajusta) | Cuanto menos VRAM, menos capas GPU y más CPU. |
| **Windows** | WSL2 + CUDA | No probado — en Windows usar WSL2 con CUDA toolkit. |

Para configurar en el chat de llama.cpp (template oficial del modelo):

```bash
./run_llama_chat.sh
# parámetros recomendados (oficiales Qwen3.8): temp 1.0 · top_p 0.95 · top_k 20 ·
# min_p 0.0 · repetition_penalty 1.0. El template trae reasoning xhigh por defecto;
# para desactivar el razonamiento: chat_template_kwargs={"enable_thinking": false}
```

## 5. CHAT — VÍA B: runtime Python propio + .sx8v43 (kernel CUDA S-X8)

Usa los kernels CUDA de S-X8 (decode1_v3 / fused_wmma / gemm_compact) sobre el
contenedor nativo `.sx8v43`:

```bash
source .venv/bin/activate
python3 scripts/infer_sx8.py --container Qwen3.8-27B-SX8v43.sx8
```

- Carga standalone (no necesita el modelo base — el contenedor es completo).
- Prefill en una pasada (GEMM compacto) + generación M=1 con KV cache.
- Requiere GPU (kernels CUDA) — para el 27B se recomienda ≥ 32 GB de VRAM o
  suficiente RAM (el contenedor pesa 26 GB en memoria).

## 6. VERIFICACIÓN

- `nvidia-smi` — la VRAM usada debe estar por debajo del máximo según tu caso.
- La CLI imprime `tok/s` al terminar.
- Prueba rápida: `25 + 30` → debe responder `55`.

## 7. SOLUCIÓN DE PROBLEMAS

| Problema | Solución |
|---|---|
| `nvcc: command not found` | `sudo apt install nvidia-cuda-toolkit` (o instala el CUDA toolkit de NVIDIA) |
| CUDA out of memory (OOM) | Baja `NGL` (ej. 28 → 10) o usa memoria unificada (16 GB) |
| `CUDA_ARCHITECTURES` | 120 = Blackwell (RTX 50) · 89 = Ada (RTX 40) · 86 = Ampere (RTX 30) |
| Chat sin respuesta / piensa mucho | El template trae `reasoning xhigh` por defecto → `enable_thinking: false` |
| El patch no aplica | Asegúrate de `git checkout 7c203670f` antes de `git apply` |
| Windows | Usa WSL2 con CUDA (no probado nativamente) |

## 8. NOTAS DE HONESTIDAD

- Este 27B está **validado funcionalmente** (responde correctamente en pruebas locales).
- **Los benchmarks formales (PPL, Winogrande, ARC, MMLU) del 27B están pendientes** —
  se medirán en una GPU grande. El formato S-X8 está validado formalmente en el 4B
  (enlace arriba). Se espera que el 27B tenga mejor calidad absoluta (mismo formato,
  base más capaz).
- Nunca decimos "mejor que FP16": el claim es "≈FP16 dentro del ruido, 9× menos
  pérdida que Q8_0" (validado en el 4B).
- **Sobre CPU**: el modo CPU (kernel AVX2) NO es una vía de uso real — nadie ejecuta
  estos modelos en una CPU de propósito general. En nuestro equipo lo usamos solo
  como iteración de verificación para confirmar la coherencia correcta del modelo
  (prueba de 50 preguntas en 4 idiomas). En producción se usa **GPU o APU**.
- **Sobre el método**: el método de CREACIÓN de este 27B es exactamente el mismo que
  el del Qwen3.5-4B (cuyo formato está validado formalmente: PPL, Winogrande, ARC, MMLU).
  Por tanto se esperan resultados **equivalentes o mejores**: a mayor tamaño del modelo,
  mayor calidad.