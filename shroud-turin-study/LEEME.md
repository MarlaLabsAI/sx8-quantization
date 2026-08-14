# ESTUDIOS DE LA SABANA SANTA DE TURIN

Copia estructurada y completa de todos los analisis realizados sobre la imagen
de la Sabana Santa de Turin: ingenieria inversa, tests dimensionales, pruebas
sindonologicas y sus resultados numericos.

**Origen:** `/mnt/Data_3TB/shroud_ProjectA/` (fases 1-3 + assets + documentos maestros)
**Fecha de copia:** 2026-08-09

---

## ESTRUCTURA

```
Estudios_Sabana_Santa_Turin/
├── 00_DOCUMENTOS_MAESTROS/        Documentos raiz del proyecto original
│   ├── DOCUMENTACION_MAESTRA.md   Documento maestro completo
│   ├── RESUMEN_EJECUTIVO.md       Descubrimientos clave
│   ├── RESUMEN_ANALISIS.txt       Resumen en texto plano (resultados crudos)
│   ├── INDICE_ARCHIVOS.md         Inventario original de ~100+ archivos
│   ├── README.md / QUICKSTART.md  Puntos de entrada
│   ├── CHANGELOG.md / STATE.json  Historial y estado
│   └── HANDOFF*.md / CONTINUAR_DESDE_AQUI.md
│
├── 01_Discovery/                  FASE 1: Descubrimiento ASIC (CHIP-1..10, A1-A6)
│   ├── DOCUMENTACION_TECNICA_COMPLETA.md
│   ├── INFORME_COMPLETO.md
│   ├── resultados/
│   │   ├── analisis_chip.json             Tests CHIP-1 a CHIP-10
│   │   └── TESTS_A1_A6_resultados.json    Tests A1-A6 (topologia 3D)
│   └── scripts/
│       ├── analisis_chip_profundo.py
│       └── tests_A1_A6 ASIC_3D.py
│
├── 02_Deep_Analysis/              FASE 2: Cruz central (D1-D10, D2b-D13)
│   ├── ANALISIS_CRUZ_CENTRAL_DIMENSIONAL.md
│   ├── ANALISIS_CRUZ_CENTRAL_PROFUNDIZACION.md
│   ├── resultados/
│   │   ├── TESTS_D1_D10_resultados.json
│   │   └── TESTS_D2b_D13_profundizacion.json
│   └── scripts/
│       ├── tests_D1_D10_dimensional_cruz.py
│       └── tests_D2b_D13_profundizacion.py
│
├── 03_Sindonologia_16_Tests/      FASE 3: 16 pruebas GPU + niveles 2/3 + A,B,C,A2
│   ├── SINTESIS_COMPLETA_MULTINIVEL.md
│   ├── resultados/
│   │   ├── sindonologia_16_tests_gpu_results.json
│   │   ├── sindonologia_nivel2_recurrence_matrix.json
│   │   ├── sindonologia_nivel3_process_formation.json
│   │   ├── tests_abc_hidden_mechanisms.json
│   │   └── test_A2_directional_information.json
│   └── scripts/ (5 scripts de tests)
│
├── 04_IMAGENES_ORIGINALES/        Las 3 imagenes del sudario
│   ├── imagen1_negativo.jpeg
│   ├── imagen2_dos_caras.jpeg
│   └── imagen3_sepia.jpeg
│
└── 05_VISUALIZACIONES/            Graficas de todos los tests
    ├── COMPARACION_CHIP_VISUAL.png / VISUALIZACION_CHIP_RESUMEN.png
    ├── imagen{1,2,3}_chip{1..10}_*.png   (simetria, autocorr, espectral, grid,
    │                                     celdas, fractal, info_mutua, cruz,
    │                                     jerarquico, conectividad)
    └── TEST_A{1..6}_*.png/.webp          (topologia 3D, segmentacion, adyacencia,
                                          multiescala, componentes, flujo)
```

---

## HALLAZGOS PRINCIPALES (resumen)

### Fase 1: Discovery (CHIP-1 a CHIP-10, A1-A6)
- Grid 14x14 celdas (196 total, 169 analizadas)
- Dimension fractal D = 1.642
- 64.8% similitud entre celdas
- Simetria diagonal perfecta (0.000)
- Auto-similitud en 6 escalas (1:1 a 1:32)
- Espaciados bimodales ~29px y ~50px

### Fase 2: Deep Analysis (D1-D10, D2b-D13)
- Cruz central en (416, 416) = (0.77, 0.77) relativo
- Punto critico degenerado (gradiente = 0, Jacobiano = 0)
- Espectro multifractal extremadamente ancho (Delta_alpha = 4.7652)
- Simetria de rotacion 90 grados (brazos arriba=izquierda, abajo=derecha)
- Conexion no-local con celdas (MI independiente de distancia)
- Propiedades consistentes con proyeccion 3D->2D (simulacion D12)

### Fase 3: Sindonologia (16 tests + A,B,C,A2)
- Grid adaptativo: ratio central/periferia = 0.511
- Cruz fractal multi-nivel: D = 1.329, CV = 0.065, ratio pico = 7.625
- Informacion direccional sutil: Q2 = -0.696, Q3 = +0.696
- Matriz de recurrencia: isotropia = 0.890, 41K picos espectrales
- La matriz captura el PROCESO de formacion, no el objeto

---

## NOTA

Esta carpeta contiene SOLO el analisis de la imagen y sus resultados.
Las tecnologias derivadas (FGN, compresion, kernels, entrenamientos) NO estan
incluidas; permanecen en `/mnt/Data_3TB/shroud_ProjectA/`.
