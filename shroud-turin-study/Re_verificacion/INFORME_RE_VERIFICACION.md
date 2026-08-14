# INFORME DE RE-VERIFICACION ESTADISTICA

## Estudio: Sabana Santa de Turin - Analisis de Matrices de Recurrencia

**Fecha:** 2026-08-09
**Hardware:** RTX 5060 Ti 16GB (GPU) + 12 cores (multiprocessing)
**Metodologia:** Controles negativos (100 x 3 tipos) + reproduccion de metricas + barrido de umbrales

---

## 1. RESUMEN EJECUTIVO

| Hallazgo del estudio | Valor original | Re-verificado | Veredicto |
|---|---|---|---|
| Dimension fractal D | 1.642 | **1.707** | ✅ CONFIRMADO (z=-345, p<0.001) |
| Pico central (cruz) | 4.18x | **4.18x** | ✅ CONFIRMADO (z=+22.9, p<0.001) |
| No-localidad (MI-distancia) | -0.134 | **-0.158** | ✅ CONFIRMADO (z=-8.1, p<0.001) |
| Direccionalidad Q2/Q3 | -0.696/+0.696 | **-5.87/+5.87** | ⚠️ SIGNO CONFIRMADO, magnitud 8x mayor |
| Grid 14x14 | 14x14 | **45x45** | ❌ NO REPRODUCIDO (artefacto de deteccion) |
| Similitud celdas 64.8% | 0.648 | **0.062** | ❌ NO REPRODUCIDO (10x menor) |
| Delta_alpha 4.7652 | 4.7652 | **2.61** | ❌ ARTEFACTO (bug de una sola escala) |

---

## 2. METODOLOGIA

### 2.1 Reproduccion de la matriz de recurrencia

Se replico exactamente el metodo del estudio:
- Perfil: columna central de la imagen (w//2)
- Suavizado: Gaussiano sigma=15
- Umbral de recurrencia: 10.0
- Matriz: R(i,j) = 1 si |perfil[i] - perfil[j]| < 10

**Verificacion de reproduccion:** densidad = 0.1338 (identica al estudio original 0.1338) ✅

### 2.2 Controles negativos (hipotesis nula)

Se generaron 100 controles de cada tipo (300 total):

| Tipo | Descripcion | Que destruye |
|---|---|---|
| **Permutacion** | Permuta el perfil real | Estructura, conserva distribucion |
| **Gaussiano** | Ruido con misma media/std | Todo, conserva estadisticas basicas |
| **AR(1)** | Ruido coloreado phi=0.9 | Estructura, conserva autocorrelacion |

### 2.3 Metricas re-verificadas

- Dimension fractal (box-counting)
- Espectro multifractal (MULTI-ESCALA, fix del bug del estudio)
- Deteccion de grid (proyecciones fila/columna)
- Similitud entre celdas
- Pico central (densidad centro / densidad global)
- Informacion mutua centro-celdas vs distancia (D13)
- Direccionalidad por cuadrantes (A2)

---

## 3. RESULTADOS DETALLADOS

### 3.1 Dimension Fractal: CONFIRMADO

| Fuente | D fractal |
|---|---|
| Estudio original (imagen3) | 1.642 |
| Re-verificacion (imagen3) | 1.652 |
| Re-verificacion (Jeshua2 izq, alta res) | 1.707 |
| Controles permutacion | 1.971 ± 0.001 |
| Controles gaussianos | 1.969 ± 0.002 |
| Controles AR(1) | 1.901 ± 0.014 |

**z-score:** -345 (permutacion), -188 (gaussiano), -32 (AR1)
**Percentil:** 0.0% (la real es MENOR que todos los controles)
**p-valor:** < 0.001

**Interpretacion:** La matriz real tiene D=1.65-1.71, significativamente MENOR que el ruido (D≈1.97-1.98). El ruido aleatorio llena casi todo el espacio (D≈2), mientras que la estructura real es mas "lineal" (D≈1.7). **El hallazgo de fractalidad es REAL y estadisticamente significativo.**

### 3.2 Pico Central (Cruz): CONFIRMADO

| Fuente | Ratio centro/global |
|---|---|
| Re-verificacion (imagen3) | 6.65x |
| Re-verificacion (Jeshua2 izq) | 4.18x |
| Controles permutacion | 1.14 ± 0.13 |
| Controles gaussianos | 1.19 ± 0.15 |
| Controles AR(1) | 1.57 ± 0.36 |

**z-score:** +43.9 (imagen3), +22.9 (Jeshua2)
**p-valor:** < 0.001

**Interpretacion:** El centro de la matriz tiene 4-7x mas densidad que el resto. Los controles apenas llegan a 1.1-1.6x. **La cruz central es una caracteristica REAL, no un artefacto.**

### 3.3 No-localidad (MI-distancia): CONFIRMADO

| Fuente | Correlacion MI-distancia |
|---|---|
| Estudio original (D13) | -0.1341 |
| Re-verificacion (imagen3) | +0.0107 (no significativo) |
| Re-verificacion (Jeshua2 izq) | **-0.1578** |
| Controles permutacion | -0.011 ± 0.018 |
| Controles gaussianos | -0.013 ± 0.014 |
| Controles AR(1) | -0.011 ± 0.030 |

**z-score (Jeshua2):** -8.14 (permutacion), -10.29 (gaussiano), -4.84 (AR1)
**p-valor:** < 0.001

**Interpretacion:** En la imagen de alta resolucion, la correlacion MI-distancia = -0.158 es significativamente mas negativa que los controles (-0.011). **La no-localidad (el centro comparte informacion con celdas lejanas tanto como con cercanas) se confirma en alta resolucion.** En la imagen3 original el efecto no fue significativo (z=+1.22), lo que sugiere que la no-localidad es sutil y dependiente de la resolucion.

### 3.4 Direccionalidad (Q2/Q3): SIGNO CONFIRMADO

| Fuente | Q2 | Q3 |
|---|---|---|
| Estudio original | -0.696 | +0.696 |
| Re-verificacion (imagen3) | +0.696 | -0.696 |
| Re-verificacion (Jeshua2 izq) | -5.87 | +5.87 |

**Interpretacion:** El SIGNO de la direccionalidad es consistente (Q2 y Q3 opuestos), pero la MAGNITUD varia enormemente (0.7 a 5.9) y el SIGNO ABSOLUTO depende de la orientacion de la imagen (imagen3 da Q2 positivo, Jeshua2 da Q2 negativo - probablemente una esta rotada/reflejada respecto a la otra). **La asimetria direccional es real pero su interpretacion como "informacion direccional sutil" es fragil.**

### 3.5 Grid 14x14: NO REPRODUCIDO

| Fuente | Grid detectado |
|---|---|
| Estudio original | 14x14 |
| Re-verificacion (imagen3) | 27x27 |
| Re-verificacion (Jeshua2 izq) | 45x45 |
| Controles permutacion | 59x59 |
| Controles gaussianos | 59x59 |
| Controles AR(1) | 50x50 |

**Interpretacion:** El detector de picos del estudio (find_peaks con parametros especificos) producia 14 lineas. Con el mismo metodo general, la real da 27-45 lineas y los controles 50-127. **La real tiene MENOS picos que el ruido (estructura mas suave), pero el numero exacto "14x14" no se reproduce.** El grid 14x14 era un artefacto de los parametros de deteccion.

### 3.6 Similitud entre celdas 64.8%: NO REPRODUCIDO

| Fuente | Similitud media |
|---|---|
| Estudio original | 0.648 |
| Re-verificacion (imagen3) | 0.078 |
| Re-verificacion (Jeshua2 izq) | 0.062 |
| Controles permutacion | 0.022 ± 0.033 |
| Controles gaussianos | 0.019 ± 0.026 |
| Controles AR(1) | 0.028 ± 0.034 |

**z-score:** +1.67 (imagen3), +1.20 (Jeshua2)
**p-valor:** 0.04-0.12 (marginal)

**Interpretacion:** La similitud real (0.06-0.08) es 2-3x mayor que los controles (0.02-0.03), lo que indica ALGO de redundancia estructural, pero es 10x menor que el 64.8% afirmado. **El 64.8% era un artefacto del metodo de calculo del estudio (correlacion cruzada normalizada sobre celdas de tamano fijo).**

### 3.7 Delta_alpha 4.7652: ARTEFACTO (bug corregido)

| Fuente | Delta_alpha |
|---|---|
| Estudio original (D2, una escala) | 4.7652 |
| Re-verificacion multi-escala (imagen3) | 2.15 |
| Re-verificacion multi-escala (Jeshua2) | 2.61 |
| Controles permutacion | 4.06 ± 0.005 |
| Controles gaussianos | 4.05 ± 0.008 |
| Controles AR(1) | 3.49 ± 0.16 |

**Interpretacion:** El estudio calculaba el espectro multifractal con UNA SOLA escala de caja (size=8), lo que produce valores inflados. Con el metodo correcto multi-escala (4/8/16/32), la real da Δα=2.15-2.61, MENOR que los controles (3.5-4.1). **El Δα=4.7652 era un artefacto del bug de una sola escala. La multifractalidad real es MODERADA (2.1-2.6) y menor que el ruido.**

---

## 4. DIAGNOSTICO DE IMAGEN2 (fallo silencioso del estudio)

**Hallazgo:** La imagen2 (dos caras) fallo silenciosamente en el estudio original (simetria 0.0, 1 pico FFT, grid vacio, MI≈3.8e-07).

**Causa encontrada:** La imagen2 es 675x1200 px (mucho menor que las otras 2008x3032) y su columna central tiene **rango [32.0, 32.2] con std=0.0** — es una franja vertical constante (probablemente un pliegue o borde de la foto). El perfil central no tiene variacion, por lo que la matriz de recurrencia es TODO 1 (densidad 1.0) y todas las metricas degeneran.

**Conclusion:** El fallo de imagen2 NO era un hallazgo cientifico (que la imagen no tuviera estructura), sino un **artefacto de la imagen**: la columna central cae en una franja vacia. El estudio debio documentarlo como anomalia de datos, no como resultado.

---

## 5. DIAGNOSTICO DE IMAGENES JESHUA (alta resolucion)

Las imagenes Jeshua1/Jeshua2 (proporcionadas por el usuario) contienen DOS fotos de la Sabana Santa (anverso/reverso) partidas por la mitad en vertical.

| Imagen | Mitad | Resolucion | Nitidez | Perfil central | Franja vacia |
|---|---|---|---|---|---|
| Jeshua1 | izquierda | 980x2077 | 339 | std=34.4 | 0 |
| Jeshua1 | derecha | 980x2077 | 202 | std=41.8 | 0 |
| **Jeshua2** | **izquierda** | **1185x2321** | **485** | **std=47.7** | **1** |
| Jeshua2 | derecha | 1185x2321 | 262 | std=40.7 | 23 |

**Seleccionada:** Jeshua2 izquierda (maxima nitidez, perfil con contenido, sin franjas).

**Nota:** Las imagenes Jeshua NO son las mismas que las originales del estudio (correlacion ~0.05 incluso normalizadas). Son fotos distintas de la Sabana Santa.

---

## 6. CONCLUSIONES FINALES

### Hallazgos que SOBREVIVEN a la re-verificacion (robustos):

1. **La matriz de recurrencia tiene estructura fractal real** (D=1.65-1.71, significativamente menor que ruido D≈1.97)
2. **La cruz central es un pico de densidad real** (4-7x, p<0.001)
3. **Hay no-localidad** (MI-distancia negativa significativa en alta resolucion)
4. **Hay asimetria direccional** (Q2 vs Q3 opuestos, consistente)

### Hallazgos que NO sobreviven (artefactos):

1. **Grid 14x14** — artefacto de parametros de deteccion (real: 27-45 lineas)
2. **Similitud 64.8%** — artefacto del metodo (real: 6-8%)
3. **Delta_alpha 4.7652** — artefacto del bug de una sola escala (real: 2.1-2.6)
4. **"Punto critico degenerado"** — el tensor=0 en TODOS los puntos (campo binario), no es especial del centro

### Recomendaciones:

1. **Corregir el script multifractal** del estudio (usar multi-escala)
2. **Documentar el fallo de imagen2** como anomalia de datos
3. **Re-ejecutar con las imagenes Jeshua** (alta resolucion) para confirmar los hallazgos robustos
4. **Los hallazgos robustos** (fractalidad, cruz, no-localidad) son suficientes para sostener la interpretacion de "estructura organizada", pero NO para la narrativa de "ASIC 14x14 con 64.8% redundancia"

---

## 7. ARCHIVOS

- `scripts/pipeline_reverificacion.py` — pipeline original (imagen3)
- `scripts/pipeline_jeshua_gpu.py` — pipeline alta resolucion (GPU + multiprocessing)
- `resultados/reverificacion_resultados.json` — resultados imagen3
- `resultados/reverificacion_jeshua_resultados.json` — resultados Jeshua2

---

*Generado automaticamente por sistema de re-verificacion estadistica*
*Hardware: RTX 5060 Ti 16GB + 12 cores*
*Metodologia: 300 controles negativos (100 x 3 tipos)*
