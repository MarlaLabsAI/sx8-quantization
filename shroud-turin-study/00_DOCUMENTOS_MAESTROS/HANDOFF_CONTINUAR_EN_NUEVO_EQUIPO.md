# HANDOFF: Continuar Proyecto Shroud - Convertidor Teletransportador FGN v2

**Fecha:** 2026-06-11  
**Equipo anterior:** GTX 960M (4GB VRAM, 16GB RAM) - Windows  
**Equipo actual:** RTX 5060 Ti Blackwell (16GB VRAM, 32GB RAM)

---

## ESTADO DEL PROYECTO

### Completado (100%)
- ✅ Análisis de la Sabana Santa de Turín (52 tests)
- ✅ Diseño de arquitectura FGN v2 (Fractal Grid Network)
- ✅ Implementación del Convertidor Teletransportador v2.1
- ✅ Conversión de Qwen2.5-0.5B → FGN v2 (pesos transferidos)
- ✅ Validación de fractales añadidos (D=1.400, cerca del objetivo 1.329)
- ✅ Benchmark Qwen original: 7/10 (70%)
- ✅ Benchmark FGN sin fine-tuning: 0/10 (0%) - **esperado**

### En progreso (75%)
- ⏳ **Fine-tuning de FGN v2** ← AQUÍ ESTAMOS

### Problema en equipo anterior
El equipo anterior (GTX 960M, 4GB VRAM) **crasheaba** durante el fine-tuning. Causa identificada:
- Fine-tuning con 256 tokens + AdamW = 4.82GB VRAM → **excede 4GB**
- TDR de Windows resetea el driver → pantalla se congela

**Soluciones probadas:**
- ✅ SGD + `zero_grad(set_to_none=True)` → VRAM estable 2.38GB
- ✅ Fine-tuning simple 3 epochs funcionó (loss: 16.17 → 11.06)
- ❌ Pero el modelo resultante genera texto incoherente (catastrophic forgetting)

---

## LO QUE DEBES HACER

### Paso 1: Leer documentación (15 min)
```
C:\Shroud_Project\
├── README.md                              ← Punto de entrada
├── QUICKSTART.md                          ← Onboarding rápido
├── STATE.json                             ← Estado actual
├── CHANGELOG.md                           ← Historial (v1.2.0)
├── INDICE_ARCHIVOS.md                     ← Mapa completo de archivos
├── 05_Implementation\
│   ├── INVESTIGACION_FINETUNING_BEST_PRACTICES.md  ← ⭐ CRÍTICO: Leer esto
│   ├── INFORME_BENCHMARK_QWEN_VS_FGN.md
│   └── tests_experimental\
│       ├── fgn_v2_model.py                ← Modelo FGN v2
│       ├── convertidor_teletransportador_v2.py
│       └── results\                       ← Todos los resultados JSON
```

### Paso 2: Entender el problema (10 min)
El Convertidor Teletransportador transfiere pesos de Qwen a FGN v2, pero:
- Los pesos llegan correctamente ✅
- La arquitectura FGN es diferente (atención por celdas, no tokens)
- Sin fine-tuning, el modelo no sabe usar los pesos → texto incoherente
- Con fine-tuning simple, hay **catastrophic forgetting** severo

### Paso 3: Fine-tuning con tu hardware (30-60 min)
Con 16GB VRAM puedes hacer lo que no podíamos:

**Opción A: Knowledge Distillation (RECOMENDADA)**
```python
# Teacher: Qwen 0.5B (en GPU, 1GB)
# Student: FGN v2 (en GPU, 1.18GB)
# Dataset: Alpaca 5K samples
# Loss = 0.7 * KD_loss + 0.3 * CE_loss
# Optimizer: AdamW, lr=1e-5
# Epochs: 10
# VRAM estimada: ~8-10GB (dentro de tu 16GB)
```

**Opción B: Fine-tuning completo con más datos**
```python
# Dataset: Alpaca 5K samples (no wikitext 50)
# Optimizer: AdamW, lr=1e-5
# Epochs: 10 con early stopping
# VRAM estimada: ~6-8GB
```

**Opción C: LoRA/Adapter (más conservadora)**
```python
# Congelar pesos FGN, entrenar solo adapters
# VRAM: ~3-4GB
# Más seguro pero menos efectivo
```

### Paso 4: Evaluar resultado (10 min)
Después del fine-tuning, ejecutar el benchmark de 10 preguntas:
```python
python tests_experimental/test_benchmark_fgn_v2.py
```
Objetivo: >50% de aciertos (actualmente 0%)

---

## INVESTIGACIÓN CLAVE (de Firecrawl)

En `05_Implementation/INVESTIGACION_FINETUNING_BEST_PRACTICES.md` está el análisis completo. Resumen:

### Learning rate para modelos 0.5B
- **Recomendado:** 1e-4 a 1e-5
- **Experiencia real (Reddit):** lr=1e-4, 20 epochs, batch=8

### Cantidad de datos necesarios
| Escenario | Samples necesarios |
|-----------|-------------------|
| Sin distillation | 10,000+ |
| **Con distillation** | **3,000 - 5,000** ← Recomendado |
| Con LoRA | 500 - 2,000 |

### Por qué falló el fine-tuning anterior
1. **Dataset insuficiente:** 50 samples vs 5,000 necesarios (100x menos)
2. **Sin knowledge distillation:** No hay guía del teacher
3. **Catastrophic forgetting:** Cambio radical de arquitectura sin protección

### Estrategia recomendada
**Knowledge Distillation** con:
- Teacher: Qwen 0.5B (preserva conocimiento)
- Student: FGN v2 (aprende nueva arquitectura)
- Dataset: Alpaca 5K samples
- Loss combinado: 70% distillation + 30% task

---

## ARCHIVOS CLAVE

### Scripts principales
```
05_Implementation/tests_experimental/
├── fgn_v2_model.py                        ← Modelo FGN v2 (forward pass)
├── convertidor_teletransportador_v2.py    ← Convierte Qwen → FGN
├── test_benchmark_10_preguntas.py         ← Benchmark Qwen (70%)
├── test_benchmark_fgn_v2.py               ← Benchmark FGN (0% sin fine-tune)
├── fine_tuning_3epochs_512tok.py          ← Fine-tuning que funcionó (pero modelo incoherente)
└── results/
    ├── test_convertidor_fgn_v2.json       ← Configuración del convertidor
    ├── fgn_v2_finetuned_3epochs_512tok.pth ← Modelo fine-tuneado (incoherente)
    └── test_benchmark_qwen_original.json  ← Benchmark Qwen (7/10)
```

### Resultados importantes
```
results/
├── test_exp01_modelo_basico.json          ← Info Qwen 0.5B
├── test_exp05_fractales.json              ← D=1.993 (plano, sin fractales)
├── test_exp07_fractales_convertidor.json  ← D=1.400 (fractales añadidos)
├── test_convertidor_fgn_v2.json           ← FGN v2 convertida
└── test_benchmark_qwen_original.json      ← Qwen: 70% acierto
```

---

## CONFIGURACIÓN DEL EQUIPO ANTERIOR (para referencia)

```
GPU: NVIDIA GTX 960M
VRAM: 4GB
RAM: 16GB
OS: Windows 10/11
Python: 3.13
PyTorch: 2.7.1+cu118
Transformers: 5.11.0
```

### Problemas encontrados
1. **VRAM insuficiente:** 4GB no alcanza para fine-tuning con AdamW
2. **TDR de Windows:** Cuando VRAM > 4GB, driver se resetea → crash
3. **Fuga de VRAM:** `optimizer.zero_grad()` sin `set_to_none=True` no libera gradientes
4. **NaN en loss:** LR muy alto (1e-4 con SGD) causa inestabilidad

### Soluciones aplicadas
1. ✅ `optimizer.zero_grad(set_to_none=True)` → libera gradientes completamente
2. ✅ SGD en vez de AdamW → ahorra 2.36GB VRAM
3. ✅ Monitoreo estricto de VRAM en cada batch
4. ✅ Detección de NaN con abort automático

---

## PRÓXIMOS PASOS DETALLADOS

### 1. Instalar dependencias
```bash
pip install torch transformers datasets accelerate
```

### 2. Verificar hardware
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
```

### 3. Probar modelo FGN v2
```python
cd C:\Shroud_Project\05_Implementation\tests_experimental
python test_carga_fgn.py
```
Debe mostrar: "VRAM usada: 1.18GB"

### 4. Implementar Knowledge Distillation
Crear nuevo script `knowledge_distillation_rtx5060.py` con:
- Teacher en GPU (no CPU, tu VRAM lo permite)
- Dataset Alpaca 5K
- AdamW con lr=1e-5
- 10 epochs con early stopping
- Monitoreo de VRAM cada 50 batches

### 5. Evaluar resultado
```python
python test_benchmark_fgn_v2.py
```
Objetivo: >50% (idealmente >70% como Qwen original)

---

## CONTACTO Y SOPORTE

Si tienes dudas:
1. Revisar `INVESTIGACION_FINETUNING_BEST_PRACTICES.md` (análisis completo de Firecrawl)
2. Revisar `INDICE_ARCHIVOS.md` (mapa de todos los archivos)
3. Revisar `CHANGELOG.md` (historial de cambios)

---

## RESUMEN EJECUTIVO

**Proyecto:** Convertidor Teletransportador para arquitectura FGN v2  
**Estado:** 75% completado, falta fine-tuning efectivo  
**Problema:** Catastrophic forgetting durante fine-tuning  
**Solución:** Knowledge Distillation con dataset Alpaca 5K  
**Hardware necesario:** 8-10GB VRAM (tu RTX 5060 Ti tiene 16GB ✅)  
**Tiempo estimado:** 1-2 horas para completar  
**Probabilidad de éxito:** 70-80% con Knowledge Distillation

---

**Documento creado:** 2026-06-11  
**Autor:** Equipo anterior (GTX 960M)  
**Para:** Equipo actual (RTX 5060 Ti Blackwell)  
**Prioridad:** ALTA - Continuar fine-tuning
