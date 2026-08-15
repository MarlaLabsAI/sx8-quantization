# S-X8 v4.3 — Un formato de cuantización a 7,50 bits por peso

**Paper (Zenodo, DOI):** [10.5281/zenodo.21922640](https://doi.org/10.5281/zenodo.21922640)
**Modelo cuantizado (HuggingFace):** [marlalabsAI/Qwen3.5-4B-SX8](https://huggingface.co/marlalabsAI/Qwen3.5-4B-SX8)

**S-X8 v4.3** es un formato de cuantización de pesos para modelos de lenguaje grandes: **30 bytes por
bloque = 7,50 bpp** (contabilizados al completo), calidad de nivel FP16 y un decodificador portable
alineado a byte (~9–10 operaciones ALU por peso, sin memoria compartida, sin shuffles, sin dependencia de
tensor cores).

Validado en **Qwen3.5-4B** (una RTX 5060 Ti):

| Métrica | FP16 | **S-X8 v4.3** | Q8_0 |
|---|---|---|---|
| PPL wikitext-2 (runtime PCA) | 10,2090 | **10,2267** (+0,17%) | 10,4540 (+2,40%) |
| Winogrande_s / HellaSwag / ARC / MMLU | 0,5746 / 0,6965 / 0,9172 / 0,7133 | **0,5722 / 0,6964 / 0,9164 / 0,7074** | 0,5746 / 0,6965 / 0,9181 / 0,7087 |
| Tamaño de pesos de texto | 9,3 GB | **3,96 GB** (−11,6% vs Q8_0) | 4,48 GB |
| Modelo completo (visión+MTP) | — | **4,38 GB** (−15% vs Q8_0+mmproj) | 5,15 GB |
| Decode, uso real (fork llama.cpp) | — | **63,79 tok/s** | 40–54 tok/s |
| Prompt pp128 (MMQ llama.cpp) | — | 1.877,85 tok/s | 3.655,61 tok/s |

Los benchmarks de opción múltiple son estadísticamente indistinguibles entre formatos; los diferenciadores
son la perplexity (**9× menos pérdida que Q8_0**), el tamaño, la VRAM y la velocidad de decodificación en
uso real. Protocolo y datos completos en [`results/PROTOCOLO.md`](results/PROTOCOLO.md).

## Estructura del repositorio

```
├── paper/          paper-sx8.pdf / .html (EN) · paper-sx8-ES.pdf / .html (ES) · paper.tex (arXiv)
├── docs/           IDEA-PROVENANCE (EN/ES) · S-X-METHODOLOGY (EN/ES) · SX8_FLASH_V4_3_SPEC.md · SX8_FLASH_V4_3_CONTAINER.md
├── formats/        spec del contenedor .sx8v43 (el "GGUF" de S-X8) + spec del bloque
├── cuda/           kernels: sx8_fused_wmma.cu (prompt) · sx8_decode1_v3.cu (decode) · otros
├── scripts/        cuantización · evaluación (mc_*, ppl_*, winogrande_*) · contenedores · validación
├── results/        todos los JSONs en bruto · PROTOCOLO.md · VENTAJAS-SX8.md
├── LICENSE         Apache-2.0
└── FUNDING.yml
```

## Inicio rápido

```bash
# Cuantizar (necesita el modelo base + GPU):
python3 scripts/quantize_qwen35_4b_sx8_v43_gpu.py

# Evaluar (modo offline; datasets en caché):
export HF_DATASETS_OFFLINE=1
python3 scripts/mc_eval_sx.py --mode fp16      # FP16 opción múltiple
python3 scripts/mc_eval_sx.py --mode sx8v43    # S-X8 (runtime fusionado, con PCA)
python3 scripts/mc_eval_gguf.py                # Referencia Q8_0 (llama-cpp-python CUDA)
python3 scripts/verify_arc_method.py           # Validación del método ARC
```

Entorno: PyTorch 2.11, CUDA 13.0, numba 0.65, llama-cpp-python 0.3.34 (build CUDA), datasets 4.8.5.
Un fork de llama.cpp que integra S-X8 como tipo nativo `GGML_TYPE_SX8` (kernels de decode MMVQ y de prompt
MMQ) se incluye como `llama-cpp-sx8.patch` (aplicar con `git apply` sobre un checkout limpio de llama.cpp
en el commit 7c203670f). El archivo nuevo `ggml/src/ggml-cuda/template-instances/mmq-instance-sx8.cu` está
incluido en el patch.

## Transparencia

Las semillas conceptuales de este formato provienen de un análisis matemático independiente de la imagen
de la Sábana Santa de Turín. El estudio completo (con su propia re-verificación) se publica en este
repositorio: [`shroud-turin-study/`](shroud-turin-study/) (íntegro y disponible, tal cual se produjo).
Ver el Apéndice A del paper y [`docs/IDEA-PROVENANCE-ES.md`](docs/IDEA-PROVENANCE-ES.md) para la
declaración completa. El formato en sí se valida empíricamente en este repositorio; el estudio se menciona solo como
fuente de inspiración, con total honestidad sobre lo que el autor puede y no puede corroborar.

## Licencia

Apache-2.0. El formato S-X8 es obra original del autor; el modelo cuantizado usa **Qwen3.5-4B** de
Qwen Team (Alibaba Group), Apache-2.0 ([model card](https://huggingface.co/Qwen/Qwen3.5-4B)) como modelo
base — solo se cuantizaron los pesos al formato S-X8 v4.3, sin otras modificaciones. Ver
[LICENSE](LICENSE) y [NOTICE](NOTICE). Evidencia de autoría: Safe Creative ID
[2608136715874](https://www.safecreative.org/work/2608136715874) · Zenodo DOI
[10.5281/zenodo.21922640](https://doi.org/10.5281/zenodo.21922640).

## Cómo citar este trabajo

Si usas este trabajo, cítalo así:

> Vidal Leandro, M. (2026). S-X8 v4.3: Un formato de cuantización a 7,50 bits por peso con calidad de FP16
> y decodificación portable. Zenodo. https://doi.org/10.5281/zenodo.21922640

## Contacto

¿Buscas colaboración, ayuda con la integración u oportunidades? Contacta:

- **LinkedIn (perfil):** https://www.linkedin.com/in/vidalmarti/
- **LinkedIn (empresa):** https://www.linkedin.com/company/marlalabs/
- **X / Twitter:** https://x.com/MarlaLabsAI
- **Email:** info@marlalabs.com
- **Web:** https://marlalabs.com

## Apoyo

Marla Labs es un laboratorio independiente de IA. Si este trabajo te es útil, considera una donación:
[Ko-fi](https://ko-fi.com/marlalabs) · [GitHub Sponsors](https://github.com/sponsors/MarlaLabs)
· detalles en `FUNDING.yml`. 🐱
