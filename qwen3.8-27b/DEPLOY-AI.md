# DEPLOY-AI — Prompt para que otra IA despliegue S-X8 (Qwen3.8-27B)

Este archivo es un **prompt listo para copiar y pegar** a un agente de IA
(Claude Code, opencode, aider, etc.). La IA debe encargarse de TODO el despliegue
de forma autónoma, adaptándose a cualquier sistema operativo y hardware,
siempre dentro de los límites físicos posibles.

> Referencia en el README: este modelo (Qwen3.8-27B-SX8) se despliega igual que
> el 4B (ver https://huggingface.co/marlalabsAI/Qwen3.5-4B-SX8 para benchmarks
> formales del formato). El formato S-X8 v4.3 está validado en el 4B.

---

## PROMPT (copia todo lo que hay entre las dos líneas `===`)

```
===
Tienes que desplegar el modelo cuantizado Qwen3.8-27B-SX8 en la máquina actual.
Hazlo TODO de forma autónoma: detecta el sistema, instala lo necesario, descarga
el modelo, lo ejecuta y verifica que responde. No preguntes al usuario salvo
bloqueo real.

1. DETECCIÓN (primero, siempre):
   - Sistema operativo: Linux (con CUDA? nvcc presente?) / Windows (WSL2 disponible?) / macOS.
   - GPU NVIDIA: `nvidia-smi` → modelo, VRAM total. ¿Es Blackwell (RTX 50 / SM120)?
   - RAM total (`free -g` en Linux, `wmic`/`systeminfo` en Windows).
   - Decide la vía según la tabla de abajo y anuncia tu elección al usuario.

2. VÍAS DE EJECUCIÓN (elige una):
   - GPU >= 32 GB VRAM: llama.cpp fork con todo a VRAM (`-ngl 999`).
   - GPU 16 GB VRAM: `-ngl 28` (28 capas GPU + resto CPU) — o si el SO permite
     memoria unificada CUDA: `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 -ngl 999`
     (cómputo 100% GPU; el spill VRAM->RAM lo gestiona CUDA; cuello = PCIe).
   - GPU 8-12 GB VRAM: `-ngl` bajo (10-20) + resto CPU.
   - Sin GPU NVIDIA: solo CPU (`-ngl 0`), o el runtime Python si no hay nvcc.
   - Windows: usa WSL2 con CUDA toolkit dentro (no nativo).
   - macOS: avisa que el fork CUDA no es compatible; la vía es el runtime Python
     (si hay CUDA vía Metal no soportada) o recomienda una GPU NVIDIA/WSL2.
   - La fuente oficial del modelo: https://huggingface.co/marlalabsAI/Qwen3.8-27B-SX8

3. INSTALACIÓN (según lo detectado):
   - Python 3.10+; crea venv; `pip install -r requirements.txt` (torch,
     transformers, numba, safetensors, datasets).
   - Para llama.cpp: cmake, gcc, CUDA toolkit (nvcc). Compila el fork:
     `./run_llama_chat.sh build` (clona llama.cpp @7c203670f, aplica
     `llama-cpp-sx8.patch`, `-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=<arch>`).
     Archs: 120 Blackwell · 89 Ada · 86 Ampere · 80 A100. Si no hay nvcc,
     avisa que hace falta el CUDA toolkit (no falles silenciosamente).

4. DESCARGA:
   - `huggingface-cli download marlalabsAI/Qwen3.8-27B-SX8 --local-dir ./sx8`
     (o snapshot_download con python). Necesitas el GGUF (`Qwen3.8-27B-SX8v43.gguf`)
     para llama.cpp, o el contenedor (`Qwen3.8-27B-SX8v43.sx8`) para el runtime Python.
   - El contenedor `.sx8v43` es COMPLETO (config + 2D cuantizado + 1D + MTP + visión):
     no necesita el modelo base. El GGUF es el MISMO S-X8 dentro del envase GGUF
     (tipo nativo GGML_TYPE_SX8 = 41), MTP excluido.

5. EJECUCIÓN:
   - llama.cpp: `./run_llama_chat.sh` (o `NGL=<valor> ./run_llama_chat.sh`).
     Parámetros oficiales del modelo: temp 1.0 · top_p 0.95 · top_k 20 · min_p 0.0 ·
     repetition_penalty 1.0. El template trae reasoning `xhigh` por defecto:
     si el usuario quiere respuestas directas, usa `enable_thinking: false`
     (chat_template_kwargs).
   - Runtime Python (contenedor nativo, kernels CUDA S-X8):
     `python3 scripts/infer_sx8.py --container Qwen3.8-27B-SX8v43.sx8`
     (prefill 1 pasada con GEMM compacto + generación M=1).

6. VERIFICACIÓN (obligatoria):
   - Confirma que carga sin errores (llama.cpp imprime "offloaded X layers to GPU").
   - Pregunta de prueba: "25 + 30" → debe responder "55".
   - `nvidia-smi`: la VRAM usada debe estar por debajo del máximo de tu caso.
   - Si la respuesta es basura/`????`: no es culpa del hardware — revisa que el
     GGUF/parámetros son correctos y que el modelo responde con thinking off.

7. LÍMITES Y HONESTIDAD (no los violes):
   - Este 27B está validado FUNCIONALMENTE (prueba local, 50/50 respuestas
     correctas en 4 idiomas). Los benchmarks formales (PPL, Winogrande, ARC,
     MMLU) están PENDIENTES (GPU grande). No inventes números.
   - Nunca digas "mejor que FP16". El claim del formato: "≈FP16 dentro del ruido,
     9× menos pérdida que Q8_0" (validado en el 4B).
   - El 27B es una VARIANTE cuantizada de Qwen3.8-27B (Qwen Team, Apache-2.0);
     solo se cuantizaron los pesos.
   - Si algo no se puede hacer (p. ej. macOS sin CUDA), dímelo y sugiere la
     alternativa, no lo simules.
===
```

---

## Notas para quien entrega este prompt

- La IA debe **elegir una vía** y comunicarla: elige según la detección, no según
  lo que "suene mejor".
- Si la IA se queda sin opciones (p. ej. sin GPU y sin tiempo), la vía CPU es
  siempre el último recurso funcional (kernel AVX2 S-X8 incluido en el fork).
- El `llama-cpp-sx8.patch` contiene el tipo S-X8 nativo: decode MMVQ + prompt MMQ
  en GPU, y kernel AVX2 en CPU — no hace falta tocar el fork a mano.