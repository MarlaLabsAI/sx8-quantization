# Shroud Project
## Nueva Arquitectura de IA inspirada en la Sábana Santa de Turín

### ¿Qué es este proyecto?

Este proyecto descubrió una **estructura matemática tipo ASIC (circuito integrado)** en las matrices de recurrencia de la Sábana Santa de Turín. Esta estructura revela el **proceso físico** que formó la imagen, no el objeto mismo.

Usando estos descubrimientos, estamos diseñando **FGN v2 (Fractal Grid Network)**, una arquitectura de IA que supera las limitaciones fundamentales de los Transformers:
- **O(n²) → O(n·log n)** en complejidad computacional
- **62x menos memoria** para secuencias largas
- Puede procesar **libros enteros (100K+ tokens)** en GPU consumer

### Estado Actual

**Fase:** Diseño de Convertidor Teletransportador
**Próximo paso:** Implementar algoritmo de proyección Transformer→FGN
**Última actualización:** Junio 2026

### Descubrimientos Clave

1. **La matriz de recurrencia revela el PROCESO, no el objeto**
   - Simetría bilateral del cuerpo desaparece (0.354 → 0.022)
   - Isotropía aumenta (0.808 → 0.890)
   - La matriz captura cómo se formó la imagen

2. **Mecanismos de corrección ocultos detectados**
   - **Grid adaptativo**: 2x más denso en centro que periferia
   - **Cruz fractal multi-nivel**: D=1.329, auto-similar
   - **Información direccional sutil**: Complementa simetría

3. **FGN v2 resuelve limitaciones de Transformers**
   - Grid adaptativo (no fijo)
   - Anclas fractales multi-nivel (no bottleneck)
   - Simetría + dirección (no solo simetría)

### Cómo Navegar Este Proyecto

1. **Lee `QUICKSTART.md`** para entender todo en 5 minutos
2. **Lee `STATE.json`** para ver el estado actual y próximos pasos
3. **Explora las carpetas 01-04** en orden cronológico
4. **Lee `04_FGN_Architecture/FGN_V2_ARQUITECTURA_FINAL.md`** para el diseño final
5. **Lee `05_Implementation/CONVERTIDOR_TELETRANSPORTADOR.md`** para el convertidor

### Para Continuar en Otro Equipo

1. Instala dependencias: `pip install -r requirements.txt`
2. Lee `QUICKSTART.md`
3. Consulta `STATE.json` para saber dónde continuar
4. Los datos están en `assets/images/` y `*/results/`

### Estructura del Proyecto

```
Shroud_Project/
├── README.md                          ← Este archivo
├── QUICKSTART.md                      ← Onboarding en 5 minutos
├── STATE.json                         ← Estado actual
├── CHANGELOG.md                       ← Historial de cambios
├── 01_Discovery/                      ← Fase 1: Descubrimientos ASIC
├── 02_Deep_Analysis/                  ← Fase 2: Análisis dimensional cruz
── 03_Sindonologia_16_Tests/          ← Fase 3: 16 pruebas multinivel
├── 04_FGN_Architecture/               ← Fase 4: Arquitectura FGN v2
├── 05_Implementation/                 ← Fase 5: Convertidor (ACTUAL)
├── assets/                            ← Imágenes y diagramas
└── references/                        ← Bibliografía
```

### Contacto y Colaboración

Este proyecto es reproducible. Todos los scripts están en `*/scripts/` y los resultados en `*/results/`.

**Carpeta raíz:** `C:\Shroud_Project\`

---

**Última actualización:** Junio 2026
**Versión:** 1.0.0
**Fase actual:** Diseño de Convertidor Teletransportador
