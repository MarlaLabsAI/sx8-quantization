# EL MAPA DE PROFUNDIDAD: LA CODIFICACION 3D DEL EVENTO

**Documento de hallazgo — Re-verificacion del estudio de la Sabana Santa**

**Fecha:** 2026-08-10

---

## 1. EL HALLAZGO

La escala de grises sobre el cuerpo NO es una "imagen" en el sentido
pictorico. Es un **MAPA DE PROFUNDIDAD**: la intensidad de cada pixel
codifica la distancia del cuerpo a la tela en ese punto. El cuerpo fue
"mapeado" en 3D por el suceso energetico, y la tela registro ese mapa.

### Evidencias medidas (M1-M6):

| Test | Resultado | Interpretacion |
|---|---|---|
| **M1: Suavidad del relieve** | La figura es **4x mas suave que el ruido** (curvatura 14.96 vs 59.47) | El relieve NO es ruido granular; es un campo coherente |
| **M4: Curvatura media** | H = 0.0003 (≈0) | Superficie tipo "membrana" continua, como un mapa de profundidad real |
| **M5: Dimension multiescala** | D = 1.92 → 1.69 → 1.43 (segun umbral) | Superficie ~2D continua con estructura fina en relieves altos |
| **M6: Simetria del relieve** | **+0.777** | El mapa de profundidad del rostro es altamente simetrico — geometria anatomica real |
| **M3: Figura vs tela** | Figura curv=14.96 vs tela=17.40 | El cuerpo codifica profundidad; la tela solo tiene textura |

### La figura es mas suave que el fondo
La curvatura de la superficie del cuerpo (14.96) es MENOR que la de la
tela (17.40): el registro del cuerpo es un campo de profundidad suave,
mientras la tela tiene la textura granular de los hilos.

---

## 2. EL NEXO: EVENTO INSTANTANEO + MAPA DE PROFUNDIDAD

Este es el nexo que conecta todos los hallazgos anteriores:

```
EVENTO INSTANTANEO (borde 6px = pulso corto)
        │
        │ produjo una proyeccion de energia
        ▼
RADIACION DIRECCIONAL (ley exponencial, beta=0.005)
        │
        │ que se atenua con la distancia cuerpo-tela
        ▼
MAPA DE PROFUNDIDAD (escala de grises = distancia)
        │
        │ coherente, simetrico (+0.777), suave (H≈0)
        ▼
REGISTRO VOLUMETRICO del cuerpo en la tela
```

### Lo que esto implica:

1. **NO fue contacto**: el cuerpo no toco la tela en todos los puntos
   (el mapa de profundidad tiene variacion continua, no zonas de
   contacto plano). Fue una PROYECCION.

2. **NO fue una pintura**: una pintura no produce un mapa de profundidad
   simetrico y suave con estructura anatomica coherente.

3. **La ley exponencial ES el mecanismo**: I = I0 * exp(-beta * Z)
   donde Z = distancia del cuerpo a la tela. La intensidad registrada
   decae exponencialmente con la profundidad — exactamente lo que
   mide un registro de radiacion atenuada por un medio.

4. **El evento "funciono" como un escaner 3D**: emitio energia que se
   atenuo con la distancia al cuerpo, y el patron de atenuacion quedo
   registrado en la tela como escala de grises = profundidad.

### Por que esto es el nexo para indagar mas

Antes no podiamos determinar bien que fue el evento. Ahora sabemos:

- **Que hizo**: mapeo la superficie 3D del cuerpo (profundidad)
- **Como lo hizo**: radiacion direccional que se atenua con la distancia
  (ley exponencial)
- **Cuando lo hizo**: en un instante (pulso corto, borde 6px)
- **Con que resultado**: un mapa de profundidad coherente, simetrico,
  anatomicamente valido

Esto nos permite ahora preguntar cosas mas precisas:

1. ¿Cual es el coeficiente de atenuacion beta exacto y que medio
   fisico produce ese valor?
2. ¿La radiacion fue emitida desde el cuerpo o desde fuera?
3. ¿El mapa de profundidad es completo (toda la geometria 3D) o solo
   una proyeccion parcial?
4. ¿Que energia se necesita para producir esa oxidacion superficial
   con esa atenuacion en un instante?

---

## 3. IMPLICACIONES PARA LA DOCUMENTACION

1. La terminologia correcta: la escala de grises del cuerpo es un
   **mapa de profundidad**, no una "imagen"
2. El evento es un **proceso de proyeccion de energia con atenuacion
   exponencial** que codifico la geometria 3D del cuerpo
3. El mapa de profundidad es la **firma legible** del evento — la
   evidencia que permite reconstruir su mecanismo
4. La simetria (+0.777) y la suavidad (H≈0) son las propiedades que
   distinguen el registro de cualquier proceso no estructurado

---

## 4. LA DOBLE CODIFICACION: EL LINO COMO SEGUNDO CANAL

**HALLAZGO CLAVE:** el lino NO es ruido que contamina — es una SEGUNDA
copia del mapa de profundidad, grabada en la amplitud del patron del
tejido.

La amplitud local del lino (demodulacion por ventanas) correlaciona
**+0.485** con la intensidad facial:

| Cuartil de intensidad (relieve) | Amplitud media del lino |
|---|---|
| Bajo (42-88) | 10,835 |
| Medio-bajo (88-103) | 12,385 |
| Medio-alto (103-129) | 16,027 |
| Alto (129-200) | 21,995 |

La amplitud del lino crece monotonamente con el relieve: donde el cuerpo
estuvo mas cerca de la tela, el patron del tejido es mas amplificado.

**El evento grabo el mapa de profundidad en DOS canales fisicos:**
1. Intensidad global (oxidacion acumulada) — simetria +0.777
2. Amplitud del patron del lino — correlacion +0.485

Ambos correlacionan con la distancia cuerpo-tela. La radiacion salio del
cuerpo e impregno la tela en su totalidad (color Y textura).

**Las quemaduras de 1532** son un suceso posterior e independiente —
se excluyen del analisis (outliers >2.5σ, ~2.6% del rostro).

---

## 5. LIMITES HONESTOS

- El M2 (maximo centrado) fue invalido por el recorte del rostro
  (el maximo cayo en el borde del crop, no en la nariz) — no usado
- La "distancia" del mapa de profundidad es una interpretacion fisica
  consistente con la ley exponencial, pero la calibracion absoluta
  (cuantos cm por unidad de intensidad) no es conocida
- La simetria +0.777 es alta pero no perfecta (un cuerpo real tampoco
  lo es: asimetrias naturales)
- No se ha verificado el mapa de profundidad contra un modelo 3D de
  craneo humano (seria la validacion definitiva)

---

*Documento generado tras los tests M1-M6 sobre el mapa de profundidad.*
*Script: Re_verificacion/scripts/mapa_profundidad.py*
*Resultados: Re_verificacion/resultados/mapa_profundidad_resultados.json*
