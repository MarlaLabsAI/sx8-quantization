# Shroud Project - Entorno Python

## Activar entorno

```bash
source /mnt/Data_3TB/shroud_ProjectA/.venv/bin/activate
```

## Verificar entorno

```bash
cd /mnt/Data_3TB/shroud_ProjectA/05_Implementation/transferencia_fgn_v3_llama
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')"
```

## Configuración

- **Archivo .env**: `/mnt/Data_3TB/shroud_ProjectA/.env`
- **Pipeline**: `05_Implementation/transferencia_fgn_v3_llama/pipeline_transferencia_qwen35_phase.py`

## Hardware

| Componente | Valor |
|------------|-------|
| GPU | RTX 5060 Ti Blackwell |
| VRAM | 16 GB |
| Compute Capability | 12.0 |
| BF16 nativo | Si |
| PyTorch | 2.11.0+cu128 |
| CUDA driver | 13.0 |

## Modelo base

- **Qwen3.5-0.8B** → FGN V3 (6 capas full-attention convertidas)
- **SmolLM2-360M-Instruct** → FGN V3 (pipeline preparado)
- Formato: `torch_dtype: bfloat16`

## Documentación

Ver `docs/00_INDEX.md` para quick start.
Documentación completa en `docs/`.

