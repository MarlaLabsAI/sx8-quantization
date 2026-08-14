# ÍNDICE DE ARCHIVOS: Shroud Project

**Fecha:** 2026-06-11  
**Ubicación:** `C:\Shroud_Project\`  
**Total archivos:** ~100+ (documentos, scripts, resultados, visualizaciones)

---

## ESTRUCTURA JERÁRQUICA COMPLETA

```
C:\Shroud_Project\
│
├── DOCUMENTOS PRINCIPALES (Raíz)
│   ├── README.md                              ← Punto de entrada del proyecto
│   ├── QUICKSTART.md                          ← Onboarding en 5 minutos
│   ├── STATE.json                             ← Estado actual y próximos pasos
│   ├── CHANGELOG.md                           ← Historial de cambios (v1.2.0)
│   ├── RESUMEN_EJECUTIVO.md                   ← Descubrimientos clave
│   ├── RESUMEN_ANALISIS.txt                   ← Resumen en texto plano
│   ├── DOCUMENTACION_MAESTRA.md               ← Documento maestro completo
│   └── requirements.txt                       ← Dependencias Python
│
├── 01_Discovery/                              ← FASE 1: Descubrimiento ASIC (100%)
│   ├── DOCUMENTACION_TECNICA_COMPLETA.md      ← Metodología detallada
│   ├── INFORME_COMPLETO.md                    ← Informe técnico inicial
│   ├── results/
│   │   ├── analisis_chip.json                 ← Tests CHIP-1 a CHIP-10
│   │   └── TESTS_A1_A6_resultados.json        ← Tests A1-A6 (topología 3D)
│   └── scripts/
│       ├── analisis_chip_profundo.py          ← Script CHIP-1 a CHIP-10
│       └── tests_A1_A6 ASIC_3D.py             ← Script A1-A6
│
├── 02_Deep_Analysis/                          ← FASE 2: Análisis cruz central (100%)
│   ├── ANALISIS_CRUZ_CENTRAL_DIMENSIONAL.md   ← Análisis inicial (D1-D10)
│   ├── ANALISIS_CRUZ_CENTRAL_PROFUNDIZACION.md ← Análisis profundo (D2b-D13)
│   ├── results/
│   │   ├── TESTS_D1_D10_resultados.json       ← Tests dimensionales
│   │   └── TESTS_D2b_D13_profundizacion.json  ← Tests profundización
│   └── scripts/
│       ├── tests_D1_D10_dimensional_cruz.py   ← Script D1-D10
│       └── tests_D2b_D13_profundizacion.py    ← Script D2b-D13
│
├── 03_Sindonologia_16_Tests/                  ← FASE 3: Tests sindonológicos (100%)
│   ├── SINTESIS_COMPLETA_MULTINIVEL.md        ← Síntesis multinivel
│   ├── results/
│   │   ├── sindonologia_16_tests_gpu_results.json
│   │   ├── sindonologia_nivel2_recurrence_matrix.json
│   │   ├── sindonologia_nivel3_process_formation.json
│   │   ├── tests_abc_hidden_mechanisms.json   ← Tests A,B,C (mecanismos ocultos)
│   │   └── test_A2_directional_information.json ← Test A2 (direccional)
│   └── scripts/
│       ├── tests_sindonologia_16_pruebas_gpu.py
│       ├── tests_sindonologia_nivel2_recurrence.py
│       ├── tests_sindonologia_nivel3_process.py
│       ├── tests_abc_hidden_mechanisms.py
│       └── test_A2_directional.py
│
├── 04_FGN_Architecture/                       ← FASE 4: Arquitectura FGN (100%)
│   ├── FGN_V2_ARQUITECTURA_CAMINO.md          ← Roadmap FGN v2
│   ├── FGN_V2_ARQUITECTURA_FINAL.md           ← Diseño final FGN v2
│   ├── FGN_VALIDATION_ANALISIS.md             ← Validación FGN (ventajas/riesgos)
│   ├── GRID_QUANTIZATION_LLM_ROADMAP.md       ← Roadmap cuantización LLMs
│   └── validation/
│       └── FGN_VALIDATION_results.json        ← Resultados validación FGN
│
├── 05_Implementation/                         ← FASE 5: Implementación (75%)
│   ├── CONVERTIDOR_TELETRANSPORTADOR.md       ← Diseño del convertidor
│   ├── convertidor_teletransportador_v2.py    ← Implementación v2.1
│   ├── GUIA_IMPLEMENTACION_CONVERTIDOR.md     ← Guía paso a paso
│   ├── ANALISIS_EXPERIMENTAL_QWEN_VS_SABANA.md ← Análisis comparativo
│   ├── INFORME_TESTS_CONVERTIDOR.md           ← Informe tests convertidor
│   ├── INFORME_BENCHMARK_QWEN_VS_FGN.md       ← Benchmark Qwen vs FGN
│   └── tests_experimental/
│       ├── fgn_v2_model.py                    ← Modelo FGN v2 (forward pass)
│       ├── run_all_tests.py                   ← Script maestro (EXP-01 a EXP-06)
│       ├── test_exp01_cargar_modelo.py        ← Test EXP-01
│       ├── test_exp02_extraccion_pesos.py     ← Test EXP-02
│       ├── test_exp03_correlaciones_atencion.py ← Test EXP-03
│       ├── test_exp04_busqueda_grid.py        ← Test EXP-04
│       ├── test_exp05_fractales.py            ← Test EXP-05
│       ├── test_exp06_recurrence.py           ← Test EXP-06
│       ├── test_exp07_fractales_convertidor.py ← Test EXP-07 (fractales añadidos)
│       ├── test_convertidor.py                ← Test del convertidor
│       ├── test_benchmark_10_preguntas.py     ← Benchmark Qwen original
│       ├── test_benchmark_fgn_v2.py           ← Benchmark FGN v2
│       ├── debug_model.py                     ← Debug estructura modelo
│       ├── debug_fgn.py                       ← Debug FGN
│       ├── debug_params.json                  ← Parámetros debug
│       └── results/
│           ├── RESUMEN_TESTS_EXPERIMENTALES.json ← Resumen comparativo
│           ├── test_exp01_modelo_basico.json
│           ├── test_exp02_pesos_stats.json
│           ├── test_exp02_attention_weights.pt ← Pesos crudos (PyTorch)
│           ├── test_exp03_correlaciones_atencion.json
│           ├── test_exp04_busqueda_grid.json
│           ├── test_exp05_fractales.json
│           ├── test_exp06_recurrence.json
│           ├── test_exp07_fractales_convertidor.json
│           ├── test_convertidor_fgn_v2.json
│           ├── test_convertidor_resumen.json
│           ├── test_benchmark_qwen_original.json
│           └── test_benchmark_fgn_v2.json
│
└── assets/
    ├── images/                                ← Imágenes originales del sudario
    │   ├── imagen1_negativo.jpeg
    │   ├── imagen2_dos_caras.jpeg
    │   └── imagen3_sepia.jpeg
    └── visualizaciones/                       ← Visualizaciones de tests
        ├── COMPARACION_CHIP_VISUAL.png
        ├── VISUALIZACION_CHIP_RESUMEN.png
        ├── imagen1_chip1_simetria.png ... imagen1_chip10_conectividad.png (10 imgs)
        ├── imagen2_chip1_simetria.png ... imagen2_chip10_conectividad.png (9 imgs)
        ├── imagen3_chip1_simetria.png ... imagen3_chip10_conectividad.png (10 imgs)
        ├── TEST_A1_topologia_3D.png + .webp
        ├── TEST_A2_segmentacion_colores.png
        ├── TEST_A3_adyacencia_conectividad.png
        ├── TEST_A4_multiescala_fractal.png
        ├── TEST_A5_componentes_funcionales.png + .webp
        └── TEST_A6_flujo_informacion.png + .webp
```

---

## RESUMEN POR FASES

| Fase | Directorio | Documentos | Scripts | Resultados | Estado |
|------|------------|------------|---------|------------|--------|
| **1. Discovery** | `01_Discovery/` | 2 | 2 | 2 | ✅ 100% |
| **2. Deep Analysis** | `02_Deep_Analysis/` | 2 | 2 | 2 | ✅ 100% |
| **3. Sindonología** | `03_Sindonologia_16_Tests/` | 1 | 5 | 5 | ✅ 100% |
| **4. FGN Architecture** | `04_FGN_Architecture/` | 4 | 0 | 1 | ✅ 100% |
| **5. Implementation** | `05_Implementation/` | 6 | 17 | 13 | ⚠️ 75% |

**Total:** 15 documentos + 26 scripts + 23 resultados + 40 visualizaciones

---

## HALLAZGOS PRINCIPALES POR FASE

### Fase 1: Discovery (CHIP-1 a CHIP-10, A1-A6)
- Grid 14×14 celdas
- Dimensión fractal D=1.642
- 64.8% similitud entre celdas
- Simetría diagonal perfecta (0.000)

### Fase 2: Deep Analysis (D1-D10, D2b-D13)
- Cruz central en (416, 416) = (0.77, 0.77) relativo
- Punto crítico degenerado (gradiente=0, Jacobiano=0)
- Simetría rotacional 90°
- Conexión no-local con celdas

### Fase 3: Sindonología (16 tests + A,B,C,A2)
- Grid adaptativo: ratio central/periferia = 0.511
- Cruz fractal multi-nivel: D=1.329, CV=0.065
- Información direccional sutil: Q2=-0.696, Q3=+0.696
- Matriz de recurrencia: isotropía=0.890, 41K picos espectrales

### Fase 4: FGN Architecture (V1-V6, R1-R5)
- FGN v2: O(n·log n) vs O(n²) de Transformers
- 62x menos memoria
- Sparsity 25% (igual que Sabana Santa)
- Grid jerárquico resuelve riesgo crítico R3

### Fase 5: Implementation (EXP-01 a EXP-07, Benchmark)
- Pesos Qwen: D=1.993 (plano, sin fractales)
- Convertidor AÑADE fractales: D=1.400 (cerca de 1.329)
- Mejora relativa: 29.8% en dimensión fractal
- Qwen 0.5B: 70% acierto en benchmark simple
- FGN v2: 0% acierto (necesita fine-tuning)

---

## PRÓXIMOS PASOS

1. **Fine-tuning de FGN v2** con wikitext-2
2. **Validar preservación** de conocimiento después de fine-tuning
3. **Benchmark más extenso** (MMLU, HellaSwag, etc.)
4. **Optimizar arquitectura** si es necesario

---

## INSTRUCCIONES PARA CONTINUAR EN OTRO EQUIPO

1. Copiar carpeta `C:\Shroud_Project\` completa
2. Instalar dependencias: `pip install -r requirements.txt`
3. Verificar GPU: `python -c "import torch; print(torch.cuda.is_available())"`
4. Leer en orden: README.md → QUICKSTART.md → STATE.json
5. Ejecutar tests: `python 05_Implementation/tests_experimental/run_all_tests.py`

---

**Documento creado:** 2026-06-11  
**Versión:** 1.0  
**Total archivos documentados:** ~100+
