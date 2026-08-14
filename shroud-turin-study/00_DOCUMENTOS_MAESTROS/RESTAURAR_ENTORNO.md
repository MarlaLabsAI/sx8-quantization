# RESTAURAR ENTORNO — Shroud Project

## Archivos NO incluidos (referencias para restaurar)

### 1. .env (API keys)
Las claves de API no se incluyen en este repositorio. Crear el archivo `.env`
localmente en la raíz del proyecto con las claves necesarias (p. ej. token de
Hugging Face) antes de ejecutar los scripts.

### 2. .venv (Python virtual environment)
```bash
cd Shroud_Project
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate peft trl numpy scipy matplotlib huggingface_hub sentencepiece
```

Paquetes principales:
- torch >= 2.5 (con CUDA 13.0)
- transformers >= 4.50
- datasets >= 3.0
- accelerate
- numpy, scipy

### 3. Checkpoints antiguos (Cerebras)
```
05_Implementation/tests_experimental/fgn_v3_pipeline/*.pt  (~12 GB)
05_Implementation/tests_experimental/results/*.pt (~2 GB)
```
NO necesarios. Eran checkpoints de experimentos con Cerebras-GPT-590M.

### 4. Modelos descargados (HuggingFace cache)
Se descargan automáticamente al ejecutar:
- `SmolLM2-360M-Instruct` → `~/.cache/huggingface/hub/`
- WikiText-2 → `~/.cache/huggingface/datasets/`

### 5. .firecrawl/
Resultados de búsqueda web (no esenciales). Se regeneran si se necesita.

---

## Restauración rápida

```bash
# 1. Clonar o copiar proyecto a nueva máquina
cp -r /media/CHARD/Shroud_Project /destination/

# 2. Crear venv
cd /destination/Shroud_Project
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets accelerate numpy scipy matplotlib

# 3. Crear .env (opcional, solo para HF rate limits)
echo 'HF_TOKEN=hf_your_token_here' > .env

# 4. Verificar
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. Ejecutar pipeline
cd 05_Implementation/transferencia_fgn_v3_llama
python3 pipeline_transferencia.py
```

---

## GPU requerida
- NVIDIA RTX 5060 Ti 16GB VRAM (Blackwell, BF16/FP8 nativo)
- Alternativa: cualquier GPU con ≥16GB VRAM y CUDA ≥12.0
