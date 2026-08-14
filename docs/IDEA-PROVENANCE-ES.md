# PROCEDENCIA DE LAS IDEAS — S-X8 v4.3

*De dónde vienen las ideas del formato S-X8 publicado, y cómo se valida cada una.
Este documento cubre solo lo publicado en este repositorio; la investigación en curso no se describe aquí.*

---

## 1. Origen de la inspiración

Las semillas conceptuales de S-X8 provienen de un análisis matemático independiente de la imagen de la
Sábana Santa de Turín (2025–2026). Ese estudio fue un ejercicio de indagación que encontró cosas muy
interesantes; algunas parecen ciertas y asombrosas, pero el autor no puede corroborarlas, ni lo pretende:
esa verificación corresponde a quienes tienen acceso directo a la Sábana. Algunas pruebas y análisis
parecen bien hechos; otros quizá se acerquen más a una pareidolia. Todo ese ejercicio —haya producido
resultados veraces o no— sirvió para extrapolar conceptos e ideas para el formato.

El formato S-X8 en sí ha sido verificado empíricamente (perplexity y benchmarks en Qwen3.5-4B; ver Sección
4 y PROTOCOLO.md); el estudio de la Sábana solo se menciona para mostrar, en gran parte, de dónde vino la
extrapolación de ideas, en un espíritu de transparencia total. La declaración de transparencia completa
está en el paper (Apéndice A) y en la carpeta del estudio (NOTA_TRANSPARENCIA.md).

## 2. Influencias del campo (conocimiento público)

El formato se apoya en trabajo bien establecido y documentado públicamente:

- **GGUF k-quants (llama.cpp)** — cuantización por bloques con escala/punto cero; Q8_0 es la referencia
  de 8 bits usada en toda nuestra evaluación.
- **IQ-quants** — tasas de bits por debajo de 4 bpp con codebooks sensibles a la importancia.
- **AWQ y sucesores** — selección de canales/capas sensibles por calibración.
- **NVFP4 / MXFP4** — formatos de hardware de 4 bits para tensor cores de datacenter; el decodificador
  es deliberadamente solo-ALU y portable, y el silicio Blackwell de consumo (SM120) no expone rutas de
  matmul FP4/FP8 a cuBLAS/PyTorch.
- **S-Quant (ICML 2026)** — los rangos adaptativos por sub-bloque mejoran la calidad por bit; extendemos
  esta dirección con recuento exacto de bytes, corrección PCA a la salida e integración completa en el motor.

## 3. Qué es original en este trabajo

- Recuento de bits exacto y completo: 30 bytes/bloque = 7,50 bpp (cada byte justificado).
- Corrección PCA aplicada a la *salida* de la multiplicación de matrices (reformulación Z0/Z1), que reduce
  el coste por peso a ~0,06 FMA por peso.
- Decodificador portable alineado a byte: ~9–10 operaciones ALU por peso, sin memoria compartida, sin
  shuffles, sin dependencia de tensor cores (validado en SM120 y Maxwell).
- Integración nativa en un fork de llama.cpp como `GGML_TYPE_SX8`, con kernel de decodificación (MMVQ) y
  kernel de prompt con tensor cores (MMQ).
- Protocolo de evaluación honesto (prompts compartidos entre motores, validación del método ARC) y
  reporte transparente de limitaciones.

## 4. Resumen de validación (todos los números en `results/` y `PROTOCOLO.md`)

| Métrica | FP16 | S-X8 v4.3 | Q8_0 |
|---|---|---|---|
| PPL wikitext-2 (runtime PCA) | 10,2090 | 10,2267 | 10,4540 |
| Winogrande_s / HellaSwag / ARC / MMLU | 0,5746 / 0,6965 / 0,9172 / 0,7133 | 0,5722 / 0,6964 / 0,9164 / 0,7074 | 0,5746 / 0,6965 / 0,9181 / 0,7087 |
| Tamaño de texto | 9,3 GB | 3,96 GB | 4,48 GB |

*Los benchmarks de opción múltiple son estadísticamente indistinguibles entre formatos; el diferenciador
de calidad es la perplexity (9× menos pérdida que Q8_0) más el tamaño, la VRAM y la velocidad de
decodificación en uso real. Metodología completa en PROTOCOLO.md.*
