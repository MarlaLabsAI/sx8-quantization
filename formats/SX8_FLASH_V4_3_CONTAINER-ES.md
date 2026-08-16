# S-X8 v4.3 — Especificación del contenedor (.sx8v43)

**Estado:** final · **Licencia:** Apache-2.0 · **Implementación:** `scripts/sx8_container_v43.py`

El contenedor `.sx8v43` es el formato de archivo que empaqueta un modelo cuantizado S-X8 v4.3: un único
artefacto, sin pickle, alineado a byte, que incluye el transformer de texto cuantizado Y las torres de
visión y predicción multi-token en el mismo archivo (a diferencia de GGUF, que requiere un mmproj FP16
aparte).

## 1. Layout global

```
┌──────────────┬───────────────────────────────────────────────┐
│ MAGIC        │ 8 bytes: "SX43FILE"                            │
│ VERSION      │ 1 byte (uint8) = 1                             │
│ meta_len     │ uint32 LE — longitud del blob de metadatos     │
│ meta         │ meta_len bytes — repr() del dict de meta (ordenado) │
│ n_tensors    │ uint32 LE — número de registros de tensores    │
│ tensor 1     │ (ver abajo)                                    │
│ ...          │                                                │
│ tensor n     │                                                │
└──────────────┴───────────────────────────────────────────────┘
```

## 2. Registro de tensor

Por tensor, en orden:

| Campo | Codificación | Significado |
|---|---|---|
| `name_len` | uint32 LE | longitud en bytes del nombre |
| `name` | `name_len` bytes | nombre del tensor (UTF-8) |
| `shape[0]`, `shape[1]` | 2 × uint32 LE | forma lógica 2D (out_f, in_f) |
| `n_os` | uint8 | número de dims originales (tensores remoldeados) |
| `orig_shape` | `n_os` × uint32 LE | forma original (p. ej. kernels de conv) |
| `n_blocks` | uint32 LE | número de bloques de 32 pesos |
| `n_cb` | uint32 LE | bloques de codebook PCA (por bloque-de-K) |
| `dmin` | `n_blocks` × 2 B | extremos de rango FP16 (little-endian) |
| `dmax` | `n_blocks` × 2 B | extremos de rango FP16 |
| `config` | `n_blocks` × 1 B | 4 × 2 bits: estrategias de rango |
| `levels_hi` | `n_blocks` × 16 B | nibbles altos de los niveles de 6 bits |
| `levels_lo` | `n_blocks` × 8 B | 2 bits bajos de los niveles de 6 bits |
| `coeff` | `n_blocks` × 1 B | bytes de coeficiente PCA |
| `bi_n_cb` | uint32 LE | bloques de codebook PCA (auxiliar) |
| `bases_data` | `bi_n_cb` × 64 × 4 B | vectores base FP32 (2 bases de 32, concatenadas) |
| `scales` | `bi_n_cb` × 2 × 4 B | escalas PCA FP32 (s0, s1) |

Todos los campos en little-endian. Los bloques usan la estructura de 30 bytes definida en
`SX8_FLASH_V4_3_SPEC.md` (dmin + dmax + config + levels_hi + levels_lo + coeff).

## 3. Garantías

- **Sin pickle** — binario puro, seguro de parsear.
- **Alineado a byte** — el payload de bloques de cada tensor es múltiplo de 30 bytes; bases/escalas en
  secciones alineadas FP32.
- **Round-trip byte-exacto** — `verify_sx43(pkl, sx43)` verifica que cada array leído del archivo es
  bit-idéntico al pkl de origen (verificado 381/381 tensores).
- **PPL desde archivo == PPL en memoria** — el evaluador (`eval_common.load_model(source_file=...)`)
  reproduce exactamente los números de calidad.

## 4. Lectores

- Lector de referencia: `scripts/sx8_container_v43.py` → `read_all(path)` devuelve
  `(weights, bases, meta)` en la misma estructura que el pkl.
- Los evaluadores lo usan vía `eval_common.load_model(quantized=True, mode="v43", source_file=...)`.

## 5. Ejemplo (Qwen3.5-4B-SX8v43.sx8)

- Tamaño: 4,38 GB · tensores: 381 · bloques: ~146M · el payload teórico a 30 B/bloque coincide con el
  tamaño del archivo en <0,1%.

## 6. v1.1 — Small-tensor section (`SXT1`) — complete standalone model

v1.1 appends a trailing section after the tensor records. v1.0 readers read the tensor records
and ignore the trailing bytes (backward compatible); v1.1 readers (`read_all_v11`) read everything.

```
┌──────────────┬───────────────────────────────────────────────┐
│ SMALL_MAGIC  │ 4 bytes: "SXT1"                               │
│ config_len   │ uint32 LE — length of the config JSON         │
│ config       │ config_len bytes — full model config (JSON)   │
│ n_small      │ uint32 LE — number of small tensors           │
│ small 1      │ (see below)                                   │
│ ...          │                                               │
│ small n      │                                               │
└──────────────┴───────────────────────────────────────────────┘
```

Per small tensor (1D/3D non-quantized tensors: layer norms, `A_log`, `dt_bias`, ...):

| Field | Encoding | Meaning |
|---|---|---|
| `name_len` | uint32 LE | byte length of the tensor name |
| `name` | `name_len` bytes | tensor name (UTF-8) |
| `ndim` | uint8 | number of dimensions |
| `shape` | `ndim` × uint32 LE | tensor shape |
| `dtype` | uint8 | 0 = float16 · 1 = float32 |
| `data` | prod(shape) × 2/4 B | tensor payload |

With the config + small tensors embedded, the `.sx8v43` file is a **complete standalone model**:
`eval_common.load_model_standalone(container)` builds the architecture from the embedded config
and materializes every tensor from the file — no base model required.

### v1.1 changes

- `read_all(path)` — unchanged (v1.0 behavior, ignores the trailing section).
- `read_all_v11(path)` — returns `(weights, bases, meta, config, small)`.
- `write_small_section(f, config, small)` — appends the section to an open file.
- Model files published as v1.1 (v2 files): `Qwen3.5-4B-SX8v43.sx8`. The quantized 2D records are
  byte-identical to the v1 file (verified by prefix SHA-256).
