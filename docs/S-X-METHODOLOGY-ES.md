# La metodología de cuantización S-X — Autoría y alcance

**Autor:** Martí Vidal Leandro
**Fecha de la primera versión:** 2025
**Licencia de los artefactos publicados:** Apache-2.0
**Estado:** este documento registra la autoría y el alcance de la metodología S-X. Cubre solo lo
publicado; los miembros de la familia en desarrollo se mencionan solo por nombre, sin detalles técnicos.

## 1. Autoría

La metodología de cuantización S-X —la familia de formatos que incluye S-X8, S-X6 y otros miembros en
desarrollo— fue **concebida y desarrollada por Martí Vidal Leandro** durante 2025–2026, como esfuerzo de
investigación independiente de MarlaLabs. Sus semillas conceptuales provienen de un análisis matemático
independiente de la imagen de la Sábana Santa de Turín; ver el Apéndice A del paper y
`IDEA-PROVENANCE-ES.md` para la declaración de transparencia completa.

## 2. Técnicas núcleo de la metodología (publicadas en S-X8 v4.3)

La metodología se define por cuatro técnicas núcleo, totalmente documentadas y validadas en S-X8 v4.3
(este repositorio):

1. **Estrategias de rango adaptativas** por sub-bloque de 8 pesos, elegidas por error cuadrático medio
   (4 estrategias, 2 bits de metadatos; verificado óptimo para 2 bits).
2. **Niveles jerárquicos de 6 bits** (4 altos + 2 bajos).
3. **Corrección PCA aplicada a la salida de la multiplicación de matrices** (reformulación Z0/Z1), que
   reduce el coste de corrección por peso a ~0,06 FMA por peso.
4. **Recuento de bits exacto y completo** (30 bytes/bloque = 7,50 bpp, cada byte justificado) y
   **decodificador portable alineado a byte** (~9–10 operaciones ALU por peso, sin memoria compartida,
   sin shuffles, sin dependencia de tensor cores).

Estas técnicas se generalizan entre anchos de bit; cada miembro de la familia las instancia con sus
propios parámetros, que no se describen aquí.

## 3. Ejemplo de posibles miembros de la familia

| Miembro de ejemplo | Estado | ¿Publicado? |
|---|---|---|
| **S-X8 v4.3** | Validado (Qwen3.5-4B: PPL +0,26% vs FP16, 9× menos pérdida que Q8_0, −11,6% tamaño de texto) | ✅ Este repositorio |
| **S-X6** | — | 🔒 Registrado, se publicará aparte |
| **S-X4 / S-X3 / S-X2** | — | 🔒 Registrado como dirección de investigación |
| **SX-FP4** | — | 🔒 Registrado, se publicará aparte |

Los miembros marcados con 🔒 son objeto de publicaciones separadas; no se divulga en este repositorio
ningún resultado, parámetro o detalle de implementación sobre ellos.

## 4b. Ampliación de alcance — derivados

Este registro cubre también **cualquier miembro, adaptación o derivado de la metodología S-X
desarrollado por el autor**, presente o futuro, en cualquier ancho de bit y para cualquier plataforma —
incluidos, sin limitación, los formatos basados en estrategias de rango adaptativas por sub-bloque
elegidas por métricas de error, codificaciones de niveles jerárquicos, términos de corrección a la
salida (PCA u otros), recuento exacto de bytes o decodificación portable solo-ALU, así como cualquier
formato que se derive, se base en o sea similar a estas técnicas. Los miembros y derivados de esta
familia se publicarán por separado; este documento evidencia la autoría y fecha de la metodología de la
que derivan.

## 4. Registro de prioridad

Este documento, junto con el paper S-X8 y la especificación del contenedor, se registró con certificados
sellados con fecha (Safe Creative; Registro de la Propiedad Intelectual, España) antes de la publicación
pública, como evidencia de autoría y fecha de la metodología y sus miembros.
