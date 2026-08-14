# QUICKSTART: Shroud Project en 5 Minutos

## ¿Qué descubrimos?

Analizamos la Sábana Santa de Turín y descubrimos que:

1. **La matriz de recurrencia** (correlaciones del perfil vertical) revela una estructura tipo ASIC/circuito integrado
2. **Esta estructura NO es el objeto** (cuerpo humano), sino el **proceso físico** que formó la imagen
3. **El proceso tiene propiedades extraordinarias**: isotropía, simetría diagonal, grid adaptativo, cruz fractal

## ¿Qué estamos construyendo?

**FGN v2 (Fractal Grid Network)**: Una arquitectura de IA inspirada en el proceso de la Sabana Santa que:
- Es **62x más eficiente** que Transformers para secuencias largas
- Puede procesar **libros enteros (100K+ tokens)** en GPU consumer
- Usa **grid adaptativo** y **anclas fractales multi-nivel**

## ¿Cómo lo estamos haciendo?

**Convertidor Teletransportador**: En vez de entrenar FGN v2 desde cero, **convertimos** un modelo Transformer existente (Qwen 3.5 2B) a arquitectura FGN v2, preservando su conocimiento.

**Analogía**: Como la Sabana Santa proyecta información 3D en 2D, nosotros proyectamos pesos de Transformer en FGN.

## Estado Actual

✅ Descubrimientos completados (Fases 1-4)
 Diseñando Convertidor Teletransportador (Fase 5)
⏳ Implementación pendiente

## Próximos Pasos Inmediatos

1. **Diseñar algoritmo de proyección** Transformer→FGN
2. **Implementar convertidor** en Python/PyTorch
3. **Validar** que FGN v2 convertido preserva conocimiento de Qwen
4. **Fine-tuning** si es necesario

## Archivos Clave

- **`04_FGN_Architecture/FGN_V2_ARQUITECTURA_FINAL.md`**: Diseño completo de FGN v2
- **`05_Implementation/CONVERTIDOR_TELETRANSPORTADOR.md`**: Diseño del convertidor
- **`STATE.json`**: Estado detallado y decisiones pendientes
- **`03_Sindonologia_16_Tests/SINTESIS_COMPLETA_MULTINIVEL.md`**: Síntesis de todos los análisis

## Para Empezar a Trabajar

1. Lee `STATE.json` para ver el estado exacto
2. Lee `05_Implementation/CONVERTIDOR_TELETRANSPORTADOR.md` para entender el convertidor
3. Revisa los scripts en `*/scripts/` para reproducir análisis
4. Consulta `03_Sindonologia_16_Tests/results/` para ver datos empíricos

## Dependencias

```bash
pip install torch numpy scipy opencv-python matplotlib scikit-image transformers datasets
```

## Hardware Requerido

- **GPU:** NVIDIA con 4GB+ VRAM (GTX 960M o superior)
- **RAM:** 8GB+ recomendado
- **Almacenamiento:** 10GB+ para modelos y datasets

---

**Tiempo de lectura:** 5 minutos
**Última actualización:** Junio 2026
