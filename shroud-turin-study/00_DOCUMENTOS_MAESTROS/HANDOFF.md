# HANDOFF — FGN V3 Transfer Project (2026-06-13)

## STATE: Where we left off

### What we're doing
Transferring a pre-trained Transformer (SmolLM2-360M-Instruct) to FGN V3 architecture
via layer-by-layer progressive conversion with minimal fine-tuning.

### Current status: IN PROGRESS — Pipeline running but stuck at converting layer 24→23

We just discovered that converting layers from the END (last→first) works much better
than from the beginning. The pipeline is currently running with:

```
FGN V3 LLaMA pipeline: 64 phases (A/B per layer, 32 layers)
Converting FROM layer 31 DOWN TO layer 0
Status: Reached ~25% FGN (8 layers converted: 31,30,29,28,27,26,25,24)
PPL holding steady at 15-18 (baseline PPL=22 on WikiText)
```

### What we proved today (2026-06-13)

1. **RoPE + mean pooling = MATHEMATICAL IDENTITY**
   - `mean(Q_rope_cell) · mean(K_rope_cell) ≡ mean_pairs(Q_rope[t]·K_rope[s])`
   - Cell-cell attention with pooled Q,K is EXACT for attention scores
   - Verified: cos_sim=1.0, MSE=0.0 for all cell sizes

2. **KV cache for cells is EXACT**
   - Running mean: `new_mean = (old_mean*n + new_token)/(n+1)` with zero error

3. **Transfer from the END works, from the BEGINNING fails**
   - Converting layer 0 first breaks ALL downstream layers
   - Converting layer 31 first only affects the LM head (fine-tuning can fix)
   - Currently at 25% FGN (8 layers from end), PPL stable at 15-18

### Architecture: FGN V3 LLaMA-style

Key files: `05_Implementation/transferencia_fgn_v3_llama/fgn_v3_llama.py`

- **Input**: SmolLM2-360M-Instruct (32L, 960 hidden, 15Q/5KV GQA, RoPE, SwiGLU, RMSNorm)
- **FGNv3Layer**: Pool tokens into 32-token cells → DualAttention (cell-level) → FractalHierarchy → Broadcast with density modulation → Gate with residual
- **Gate**: `hidden = sigmoid(gate) * cell_out + (1-sigmoid(gate)) * residual`
- **Gate init**: bias=-2 → sigmoid=0.12 → 12% FGN initial
- **Density modulation**: Per-token variance-based density weights (Sábana Santa pattern)
- **FractalHierarchy**: 3 levels (local→regional→global), init=eye (near-identity)
- **DualAttention**: 80% symmetric + 20% directional, no o_proj (handled by FGNv3Layer)

### Transfer pipeline

File: `05_Implementation/transferencia_fgn_v3_llama/pipeline_transferencia.py`

- 64 phases (2 per layer: convert + fine-tune)
- Converting from layer 31 DOWN to layer 0
- 120-150 steps per phase, LR=5e-6, AdamW, WikiText-2 (400 chunks of 1024 tokens)
- Gate bias=-2 (12% FGN initial)
- Checkpoints at 50% and 100% FGN

### Bug history (all FIXED)

| Bug | Fix |
|-----|-----|
| RoPE cos/sin dim mismatch | Use `cos.unsqueeze(1)` for [B,1,S,D] shape |
| Position_ids advanced indexing | Use `position_ids.flatten()` then reshape |
| Double o_proj (DualAttention + FGNv3Layer) | Removed o_proj from DualAttention |
| V-through dim mismatch (repeat_kv) | Fixed v shape to [B,n_kv,S,d] |
| Gate mixing pre-LN residual with cell_out → MLP input mismatch | Used gate at attention-output level |
| Converting from beginning breaks downstream layers | Now convert from END backwards |

### Next steps for the AI taking over

1. **Read the docs first**: `05_Implementation/transferencia_fgn_v3_llama/PLAN_TRANSFERENCIA.md`
2. **Continue pipeline**: `python3 05_Implementation/transferencia_fgn_v3_llama/pipeline_transferencia.py`
3. **Monitor PPL**: Should stay in 15-30 range during conversion. If it spikes >100, something broke.
4. **After 50%**: Check Q&A coherence with `05_Implementation/tests_experimental/ask_smollm2.py`
5. **After 100%**: Run full benchmark, compare with baseline

### Key learnings

- The "teletransportador" (Transformer→FGN weight transfer) works ONLY with:
  a) RoPE applied before pooling (mathematically exact)
  b) Uniform pooling (NOT density-weighted)
  c) Converting from last layer backwards
  d) Gate init = -2 (12% FGN, conservative)
  e) Pre-trained Q/K/V/O weights transferred to DualAttention
- FGN V3 has +40% parameters per layer (FractalHierarchy + cell_gate)
- 100% FGN model (32 layers FGN) = ~508M params vs 361M base (+40%)
- FGN KV cache is 29-66× smaller than Transformer
- FGN attention is O(n_cells²) vs Transformer O(n²), 1024× cheaper at 1024 tokens

### Files to restore (not included in copy)

See `RESTAURAR_ENTORNO.md` for:
- `.venv` — Python venv with dependencies
- `.env` — API keys
- `05_Implementation/tests_experimental/fgn_v3_pipeline/*.pt` — Old checkpoints (12GB, not needed)

### Hardware requirements

- GPU: RTX 5060 Ti 16GB VRAM (or any 16GB+)
- RAM: 32GB+
- Python 3.12, PyTorch 2.x, CUDA 13.0
