# QUICK-DEPLOY — S-X8 models (Qwen3.5-4B) — guía por casos

Esta guía explica cómo ejecutar el **Qwen3.5-4B-SX8** en distintos tipos de máquina.
El formato S-X8 v4.3 está **validado formalmente** en este modelo (PPL 10.2267,
Winogrande 0.5722, decode 63.79 tok/s, VRAM 3.720 GB) — ver el paper y el README.

## Contenido del repositorio

- `Qwen3.5-4B-SX8v43.sx8` — contenedor nativo S-X8 v4.3 (formato propio, completo:
  config + tensores cuantizados + 1D + MTP + visión).
- `Qwen3.5-4B-SX8v43.gguf` — el MISMO modelo S-X8 dentro del contenedor que lee
  llama.cpp (el GGUF es solo el envase; dentro hay pesos S-X8, tipo nativo
  `GGML_TYPE_SX8 = 41`). MTP excluido del GGUF (vive en el .sx8v43).
- `scripts/` — cuantización, contenedores, runtime Python con kernels CUDA propios.
- `cuda/` — kernels CUDA S-X8 (decode1_v3, fused_wmma, gemm_compact, embed_v44...).
- `llama-cpp-sx8.patch` — fork de llama.cpp con S-X8 como tipo nativo
  (decode MMVQ + prompt MMQ en GPU, y kernel AVX2 en CPU).

## 1. REQUISITOS

- Ubuntu 22.04/24.04 (Linux) · NVIDIA GPU con CUDA (Blackwell recomendado: RTX 50)
- CUDA toolkit (nvcc) · cmake · gcc · git · python3 ≥ 3.10
- RAM: 8 GB mínima, 16 GB recomendada (el 4B cuantizado son ~4.4 GB en memoria)

## 1b. REQUISITOS DE HARDWARE RELATIVOS

El 4B cuantizado pesa ~4.4 GB (GGUF 4.12 GB) — es **ligero**: la relación con tu
hardware es favorable en casi cualquier máquina.

| Perfil de máquina | Cómo corre | Experiencia relativa |
|---|---|---|
| **GPU ≥ 8 GB VRAM** (RTX 5060 Ti, 4060, 3050...) | `NGL=99` — modelo entero en VRAM | **Óptima**: decode ~63.79 tok/s, VRAM 3.720 GB |
| **GPU 4–6 GB VRAM** | `NGL=32` (ajustar) | Muy buena: mayoría de capas en GPU |
| **APU / iGPU con ≥ 16 GB RAM unificada** | iGPU vía Vulkan (o CPU AVX2 como fallback) | Funcional: responde a ritmo moderado |
| **Windows** | WSL2 + CUDA | No probado nativamente |

- **Referencia medida (RTX 5060 Ti, 16 GB)**: decode 63.79 tok/s · prompt 1877.85 tok/s
  (fork llama.cpp) · VRAM decode 3.720 GB · prompt GEMM compacto 4.31 GB.
- **Con conocimientos medios**: `./run_llama_chat.sh build` + `./run_llama_chat.sh`
  cubren todo el despliegue.

## 2. DESCARGAR

```bash
huggingface-cli download marlalabsAI/Qwen3.5-4B-SX8 --local-dir ./sx8
# o con HF hub Python:
pip install -U huggingface_hub
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('marlalabsAI/Qwen3.5-4B-SX8', local_dir='./sx8')"
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
| **≥ 8 GB VRAM** | `NGL=99 ./run_llama_chat.sh` | Todo el modelo a VRAM. Máxima velocidad (~63.79 tok/s). |
| **4–6 GB VRAM** | `NGL=32 ./run_llama_chat.sh` | Ajusta según VRAM libre. |
| **APU / iGPU** | iGPU vía Vulkan (o CPU AVX2 como fallback) | Funcional (no medido en APU). |
| **Windows** | WSL2 + CUDA | No probado — en Windows usar WSL2 con CUDA toolkit. |

## 5. CHAT — VÍA B: runtime Python propio + .sx8v43 (kernel CUDA S-X8)

Usa los kernels CUDA de S-X8 (decode1_v3 / fused_wmma / gemm_compact) sobre el
contenedor nativo `.sx8v43`:

```bash
source .venv/bin/activate
python3 scripts/infer_sx8.py --container Qwen3.5-4B-SX8v43.sx8
```

- Carga standalone (no necesita el modelo base — el contenedor es completo).
- Prefill en una pasada (GEMM compacto, ~273 tok/s) + generación M=1 con KV cache
  (~10-13 tok/s, VRAM 4.41 GB).
- Requiere GPU (kernels CUDA).

## 6. VERIFICACIÓN

- `nvidia-smi` — la VRAM usada debe estar por debajo del máximo según tu caso
  (decode: 3.720 GB).
- La CLI imprime `tok/s` al terminar.
- Prueba rápida: `25 + 30` → debe responder `55`.

## 7. SOLUCIÓN DE PROBLEMAS

| Problema | Solución |
|---|---|
| `nvcc: command not found` | `sudo apt install nvidia-cuda-toolkit` (o instala el CUDA toolkit de NVIDIA) |
| CUDA out of memory (OOM) | Baja `NGL` (ej. 99 → 32) |
| `CUDA_ARCHITECTURES` | 120 = Blackwell (RTX 50) · 89 = Ada (RTX 40) · 86 = Ampere (RTX 30) |
| Chat sin respuesta / piensa mucho | El template trae `reasoning xhigh` por defecto → `enable_thinking: false` |
| El patch no aplica | Asegúrate de `git checkout 7c203670f` antes de `git apply` |
| Windows | Usa WSL2 con CUDA (no probado nativamente) |

## 8. NOTAS DE HONESTIDAD

- Este modelo está **validado formalmente**: PPL 10.2267 (≈ FP16 dentro del ruido),
  Winogrande 0.5722, decode 63.79 tok/s (fork llama.cpp) — nunca decimos "mejor que
  FP16": el claim es "≈FP16 dentro del ruido, 9× menos pérdida que Q8_0".
- El GGUF requiere el **fork de llama.cpp** (S-X8 es tipo nativo `GGML_TYPE_SX8 = 41`);
  con llama.cpp estándar no cargará.
- **Sobre CPU**: el modo CPU (kernel AVX2) NO es una vía de uso real — nadie ejecuta
  estos modelos en una CPU de propósito general. En nuestro equipo lo usamos solo como
  iteración de verificación para confirmar la coherencia correcta del modelo
  (prueba de 50 preguntas en 4 idiomas). En producción se usa **GPU o APU**;
  y el método de creación del 27B es exactamente el mismo que el de este 4B
  (validado formalmente), por lo que se esperan resultados equivalentes o mejores:
  a mayor tamaño del modelo, mayor calidad.