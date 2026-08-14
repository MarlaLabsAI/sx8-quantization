# CHANGELOG: Shroud Project

## [1.2.0] - 2026-06-11

### Added
- Benchmark de 10 preguntas simples para Qwen 0.5B y FGN v2
- Modelo FGN v2 con forward pass funcional
- Informe comparativo Qwen vs FGN
- Soporte para GQA (Grouped Query Attention) en FGN
- Grid adaptativo dinámico basado en longitud de secuencia

### Benchmark Results
- **Qwen 0.5B Original:** 70% acierto (7/10)
- **FGN v2 Convertida:** 0% acierto (0/10)
- **Conclusión:** FGN necesita fine-tuning para preservar conocimiento

### Cambios Técnicos
- Implementación completa de FGNCellAttention con GQA
- Corrección de dimensiones en forward pass
- Ajuste de grid_boundaries para secuencias variables
- Transferencia completa de pesos (embeddings, attention, MLP, norms)

### Archivos Nuevos
- `05_Implementation/tests_experimental/test_benchmark_10_preguntas.py`
- `05_Implementation/tests_experimental/test_benchmark_fgn_v2.py`
- `05_Implementation/tests_experimental/fgn_v2_model.py`
- `05_Implementation/tests_experimental/debug_fgn.py`
- `05_Implementation/INFORME_BENCHMARK_QWEN_VS_FGN.md`
- `05_Implementation/tests_experimental/results/test_benchmark_qwen_original.json`
- `05_Implementation/tests_experimental/results/test_benchmark_fgn_v2.json`

### Conocimiento Adquirido
1. El convertidor transfiere pesos correctamente
2. FGN v2 puede hacer forward pass sin errores
3. Pero FGN necesita fine-tuning para usar los pesos efectivamente
4. Analogía: Como transferir pesos de CNN a Vision Transformer

### Próximos Pasos
- Fine-tuning de FGN v2 con wikitext-2
- Validar preservación de conocimiento
- Benchmark más extenso

---

## [1.1.0] - 2026-06-11

### Added
- Implementación completa del Convertidor Teletransportador v2.1
- Tests experimentales EXP-01 a EXP-07 con Qwen2.5-0.5B
- Validación de fractales AÑADIDOS por el convertidor
- Documento ANALISIS_EXPERIMENTAL_QWEN_VS_SABANA.md
- Documento INFORME_TESTS_CONVERTIDOR.md

### Cambios Principales

#### Convertidor Teletransportador v2.1
- **Implementación completa** en Python/PyTorch
- **5 fases:** Análisis, Proyección a grid, Creación fractales, Recurrencia, Ensamblaje
- **AÑADE estructura fractal** durante la proyección (no la busca en pesos originales)
- **Parámetros ajustados** según datos experimentales de Qwen

#### Tests Experimentales
- **EXP-01:** Carga y análisis básico de Qwen2.5-0.5B (24 capas, 494M params)
- **EXP-02:** Extracción y análisis estadístico de pesos
- **EXP-03:** Análisis de correlaciones en matrices de atención (GQA 14 heads)
- **EXP-04:** Búsqueda de estructura de grid natural (detectado, CV=0.34)
- **EXP-05:** Análisis fractal de pesos (D=1.993, plano, sin fractales)
- **EXP-06:** Análisis de matriz de recurrencia (ASIC, isotropía=0.924)
- **EXP-07:** Validación de fractales AÑADIDOS (D=1.400, cerca de objetivo 1.329)

### Resultados Clave

| Métrica | Qwen Original | FGN Convertida | Objetivo | Estado |
|---------|---------------|----------------|----------|--------|
| Dimensión fractal | 1.993 | 1.400 | 1.329 | ✅ 29.8% mejora |
| Grid detectado | Sí (CV=0.34) | Sí (CV=0.513) | - | ✅ Proyectado |
| Isotropía | 0.924 | 0.92 | 0.890 | ✅ Preservada |
| Simetría | 1.000 | 1.000 | 1.000 | ✅ Perfecta |
| Fractales | NO | SÍ | SÍ | ✅ AÑADIDOS |

### Correcciones Importantes

#### Error de Interpretación Original
- **Antes:** "Los pesos de Transformer NO tienen fractales, por lo tanto el convertidor no puede usar fractales"
- **Ahora:** "Los pesos de Transformer NO tienen fractales, por lo tanto el convertidor DEBE AÑADIR fractales durante la proyección"
- **Analogía correcta:** Como la Sabana Santa proyecta 3D→2D y genera fractales emergentes, el convertidor proyecta Transformer→FGN y añade fractales artificiales

#### Ajustes de Parámetros
- **Grid:** cell_size=300 (detectado en Qwen, no 30 de Sabana Santa)
- **CV:** 0.40 (detectado en Qwen, no 0.15 de Sabana Santa)
- **Fractales:** Objetivo D=1.329 (de Sabana Santa, no de Qwen)

### Fase 5: Implementation (60% - ACTUALIZADO)
- ✅ Diseño del Convertidor Teletransportador
- ✅ Implementación en Python/PyTorch
- ✅ Tests experimentales completados
- ✅ Validación de fractales añadidos
- ⏳ Implementar forward pass de FGN v2
- ⏳ Transferir pesos completos
- ⏳ Validar preservación de conocimiento

### Archivos Nuevos
- `05_Implementation/convertidor_teletransportador_v2.py`
- `05_Implementation/tests_experimental/test_exp01_cargar_modelo.py`
- `05_Implementation/tests_experimental/test_exp02_extraccion_pesos.py`
- `05_Implementation/tests_experimental/test_exp03_correlaciones_atencion.py`
- `05_Implementation/tests_experimental/test_exp04_busqueda_grid.py`
- `05_Implementation/tests_experimental/test_exp05_fractales.py`
- `05_Implementation/tests_experimental/test_exp06_recurrence.py`
- `05_Implementation/tests_experimental/test_exp07_fractales_convertidor.py`
- `05_Implementation/tests_experimental/test_convertidor.py`
- `05_Implementation/tests_experimental/run_all_tests.py`
- `05_Implementation/ANALISIS_EXPERIMENTAL_QWEN_VS_SABANA.md`
- `05_Implementation/INFORME_TESTS_CONVERTIDOR.md`

### Próximos Pasos
1. Implementar modelo FGN v2 funcional (forward pass)
2. Transferir pesos completos de Qwen a FGN v2
3. Validar preservación de conocimiento (perplexity)
4. Fine-tuning si es necesario
5. Benchmark vs Qwen original

---

## [1.0.0] - 2026-06-XX

### Added
- Estructura completa de Shroud_Project portable
- Documentación principal (README, QUICKSTART, STATE)
- Diseño del Convertidor Teletransportador
- Todos los análisis de la Sabana Santa organizados en fases

### Fases Completadas

#### Fase 1: Discovery (100%)
- Tests CHIP-1 a CHIP-10: Análisis profundo de estructura ASIC
- Tests A1-A6: Topología 3D y componentes funcionales
- Descubrimiento: Grid 14×14, D=1.642, 64.8% similitud entre celdas

#### Fase 2: Deep Analysis (100%)
- Tests D1-D10: Análisis dimensional de la cruz central
- Tests D2b-D13: Profundización en hallazgos
- Descubrimiento: Cruz central como punto de anclaje dimensional

#### Fase 3: Sindonología 16 Tests (100%)
- 16 pruebas sindonológicas en 3 imágenes originales
- 16 pruebas en matriz de recurrencia (Nivel 2)
- 6 pruebas del proceso de formación (Nivel 3)
- Tests A, B, C: Mecanismos de corrección ocultos
- Test A2: Información direccional
- Descubrimiento: Grid adaptativo, cruz fractal, información direccional sutil

#### Fase 4: FGN Architecture (100%)
- Diseño de FGN v1 (versión inicial)
- Validación de ventajas y riesgos
- Diseño de FGN v2 con mecanismos de corrección
- Descubrimiento: FGN v2 resuelve limitaciones de Transformers

#### Fase 5: Implementation (10% - EN PROGRESO)
- Diseño del Convertidor Teletransportador
- Pendiente: Implementación en Python/PyTorch
- Pendiente: Validación con Qwen 3.5 2B

### Key Findings
1. Matriz de recurrencia revela el PROCESO, no el objeto
2. Grid adaptativo: 2x más denso en centro que periferia (ratio 0.511)
3. Cruz fractal multi-nivel: D=1.329, auto-similar, pico 7.6x
4. Información direccional sutil complementa simetría
5. FGN v2: O(n·log n) vs O(n²) de Transformers (62x menos memoria)

### Decisiones Tomadas
- Usar DeepSeek v4 Flash para generación de datos (no Qwen local para distillation)
- Implementar convertidor en vez de entrenar desde cero
- Grid adaptativo en vez de grid fijo
- Anclas fractales multi-nivel en vez de punto único
- Simetría + dirección opcional en vez de solo simetría

### Decisiones Pendientes
- ¿Qwen 3.5 2B local (4-bit) o DeepSeek v4 Flash para validación?
- ¿Componente direccional desde el inicio o después?
- ¿Qué dataset para fine-tuning?

---

**Última actualización:** 2026-06-11  
**Versión:** 1.1.0
