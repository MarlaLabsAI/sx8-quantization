# PUNTO DE PARTIDA — Continuar Proyecto Shroud FGN v3

**PARA:** Cualquier IA o desarrollador que retome este proyecto en cualquier máquina.
**FECHA:** 13 Junio 2026
**UBICACIÓN ACTUAL:** `/mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/Shroud_Project/`

---

## ARRANQUE RÁPIDO (si estás en la misma máquina)

```bash
# Activar entorno virtual
source /mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/Shroud_Project/.venv/bin/activate

# Ir al directorio de trabajo
cd /mnt/34cede50-39d5-40d3-9f49-4cad55e7c210/Shroud_Project/05_Implementation/tests_experimental

# Verificar que todo funciona
python3 -c "
from fgn_v3_cerebras import cargar_fgn_v3_desde_cerebras
print('FGN v3 import OK')
"
```

---

## SI ESTÁS EN OTRA MÁQUINA (sin nada instalado)

### Requisitos:
```bash
pip install torch transformers datasets accelerate
# GPU recomendada: 16GB+ VRAM (mínimo 8GB para CPU + RAM)
# Python 3.10+
```

### Descargar modelo base:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-590M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-590M", trust_remote_code=True)
# ~4.6 GB de descarga, se cachea automáticamente
```

### Copiar archivos esenciales (solo necesitas estos 3 para empezar):
```
fgn_v3_cerebras.py          ← Modelo FGN v3 completo (626 líneas)
pipeline_fgn_v3_cerebras.py ← Pipeline de entrenamiento FUNCIONAL
SESION_2026-06-13.md        ← Documentación completa de la sesión (leer esto)
```

---

## ¿QUÉ ESTAMOS HACIENDO?

**Objetivo:** Transferir progresivamente un Transformer pre-entrenado (Cerebras-GPT-590M) a la arquitectura FGN v3 (Fractal Grid Network), validando que:
1. La transferencia preserva calidad (PPL no se dispara sin remedio)
2. El fine-tuning recupera la degradación inicial
3. FGN v3 iguala o supera al Transformer original

**Arquitectura FGN v3:** 3 mecanismos clave:
- **Grid Adaptativo:** Celdas de tamaño variable (16-64 tokens) según densidad de información
- **Cruz Fractal Multi-Nivel:** Propagación de información entre celdas (local→regional→global)
- **Atención Dual:** 80% simétrica + 20% direccional en espacio celular

---

## ESTADO ACTUAL (13 Junio 2026)

### Lo que FUNCIONA:
- Transferencia de pesos Cerebras-GPT → FGN v3: CORRECTA (Conv1D transpuesto a Linear)
- Pipeline `pipeline_fgn_v3_cerebras.py`: FUNCIONA
  - Estadio 1 (17% FGN): PPL=36 vs baseline 40.3 (+10% mejora)
  - LR=5e-6, 1024 tokens, batch=2, 500 muestras WikiText-2
  - ~120s por época en GPU

### Lo que NO FUNCIONA:
- Pipeline `pipeline_fgn_v3_rapido.py`: ROTO
  - LR=1e-4 (20× demasiado alto) → sobreajuste extremo
  - PPL=416 (inútil)
  - **No usar este archivo hasta arreglarlo**

### Checkpoints existentes (12 GB total):
```
fgn_v3_pipeline/
├── estadio_1_final.pt    ← Estadio 1 completado (PPL=36, el mejor)
├── estadio_2_final.pt    ← Estadio 2 parcial
└── fgn_estadio_1_*.pt    ← Varios checkpoints intermedios
```

### KV Cache: Concepto diseñado pero NO implementado
- GPT2Layer: añadir 3 líneas para KV cache estándar
- FGNv3Layer: cache de cell_states (64× menos memoria que Transformer)
- Ver sección 7 de `SESION_2026-06-13.md`

---

## QUÉ HACER A CONTINUACIÓN

### Opción A (RECOMENDADA): Continuar pipeline funcional

```bash
cd 05_Implementation/tests_experimental

# Ejecutar el pipeline desde cero o continuar desde checkpoint
python3 pipeline_fgn_v3_cerebras.py
```

Esto ejecuta 5 estadios (17% → 33% → 50% → 67% → 100% FGN).
Tiempo estimado: 2-4 horas en GPU.

Si el estadio 1 ya está completado, el pipeline continúa desde el checkpoint.

### Opción B: Reparar el pipeline rápido

Arreglar `pipeline_fgn_v3_rapido.py`:
1. Cambiar LR: `1e-4` → `5e-6`
2. Eliminar curriculum (o invertir orden: 1024→512→256)
3. Eliminar creación duplicada de modelo en estadio 1
4. Benchmark con max_length=512

### Opción C: Implementar KV cache

Añadir soporte de `past_key_value` a `GPT2Layer.forward()` y `use_cell_cache` a `FGNv3Layer.forward()`. Modificar `generate()` en `FGNv3Model` para usar caché. Esto habilitará generación larga sin O(n²).

### Opción D: Validar SHROUD quantization

Aplicar cuantización a un checkpoint FGN v3 existente:
- Config: 3-bit attn + 5-bit MLP
- Bloques: 32×32 con escalas por bloque  
- Evaluar PPL post-cuantización
- Confirmar que FGN atenúa 3.6× el ruido de cuantización

### Opción E: Cargar y evaluar el mejor checkpoint

```python
from fgn_v3_cerebras import FGNv3Model
import torch

# Cargar checkpoint del estadio 1
ckpt = torch.load('fgn_v3_pipeline/estadio_1_final.pt', map_location='cpu', weights_only=False)
model = FGNv3Model(vocab_size=50257, hidden_size=1536, num_layers=18, num_heads=12, fgn_ratio=0.17)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Evaluar PPL en un texto
# Hacer benchmark de preguntas
```

---

## REGLAS DE ORO (aprendidas con sangre)

1. **LR NUNCA > 5e-6 para FGN**. Con LR=1e-4 el modelo memoriza en 8 épocas.
2. **Capas FGN SIEMPRE al final**. Si se ponen al principio, corrompen el input desde la capa 0.
3. **Degradación post-transferencia es NORMAL**. PPL salta de ~40 a ~100-200. Con fine-tuning suave se recupera en 15-25 épocas.
4. **`is_causal=True` SIEMPRE**. Sin esto, la atención es bidireccional (fatal).
5. **Loss con shift**: `logits[:,:-1]` vs `labels[:,1:]`. Sin shift, el modelo aprende la identidad.
6. **Conv1D → Linear requiere `.t()`**. Los pesos de GPT-2 se almacenan transpuestos.
7. **No usar `pipeline_fgn_v3_rapido.py`** sin arreglarlo primero.
8. **Cerebras-GPT-590M es un modelo base**, no instruction-tuned. No esperes respuestas coherentes a preguntas directas. Evalúa con PPL en texto.

---

## ARCHIVOS CLAVE (solo los importantes)

| Archivo | Para qué sirve | Estado |
|---------|---------------|--------|
| `fgn_v3_cerebras.py` | Modelo FGN v3 completo | ESTABLE, usar |
| `pipeline_fgn_v3_cerebras.py` | Pipeline 5 estadios funcional | ESTABLE, usar |
| `pipeline_fgn_v3_rapido.py` | Pipeline rápido (3 estadios) | ROTO, reparar |
| `SESION_2026-06-13.md` | Documentación completa de la sesión | Leer primero |
| `fgn_v3_pipeline/estadio_1_final.pt` | Mejor checkpoint (PPL=36) | Cargar para evaluar |
| `fgn_v3_model_holografico.py` | FGN v3 original (Qwen) | Solo referencia histórica |
| `STATE.md` | Diario de desarrollo (previo a esta sesión) | Contexto histórico |
| `GRID_QUANTIZATION_LLM_ROADMAP.md` | Plan de cuantización SHROUD | Futuro |

---

## PREGUNTAS FRECUENTES

**P: ¿Por qué no usar Qwen como base?**
R: Qwen usa RoPE (Rotary Position Embeddings). FGN v3 hace pooling de tokens en celdas, lo que destruye la información posicional de RoPE. Cerebras-GPT usa learned positions (wpe), compatibles con pooling.

**P: ¿Por qué el PPL salta tanto al transferir?**
R: Es normal. Las capas FGN operan en celdas (no tokens). Aunque los pesos se transfieren 1:1, el camino computacional es totalmente distinto. El `cell_gate` inicial (12% FGN) amortigua esto. Con fine-tuning se recupera.

**P: ¿Qué es el "cell_gate"?**
R: Un mecanismo de gating que combina la salida FGN con el residual del Transformer original: `gate * cell_out + (1-gate) * residual`. Inicializado conservadoramente (12% FGN, 88% residual). Aprende a confiar más en FGN durante el fine-tuning.

**P: ¿Cuánto tardan los experimentos?**
R: ~120s por época con 1024 tokens, batch=2, 500 muestras en GPU. ~15-25 épocas por estadio. 5 estadios = ~2-4 horas total.

**P: ¿Funciona en CPU?**
R: Sí, pero es muy lento (~20-30 minutos por época). Para desarrollo/testing funciona. El modelo cabe en 8-16 GB RAM.

---

## CONTACTO CON EL CONTEXTO ANTERIOR

Este documento es la continuación de:
- `HANDOFF_CONTINUAR_EN_NUEVO_EQUIPO.md` (transición de Windows GTX 960M a Linux RTX 5060 Ti)
- `STATE.md` en `tests_experimental/` (diario de desarrollo hasta 12 Junio)
- `SESION_2026-06-13.md` (esta sesión, documentación completa)

Leer los 3 en orden para entender toda la historia del proyecto.

---

**FIN DEL HANDOFF. BUENA SUERTE.**
