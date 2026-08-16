"""integrate_fused_v44.py — integración del decode M=1 v4.4 en el modelo fused (FASE A4).

Estrategia SEGURA (sin tocar integrate_fused.py):
  - Subclase SX8LinearV44(SX8Linear): reutiliza el constructor del padre
    (metadatos kb-major para wmma) y SOLO sobreescribe la rama M==1 con el
    kernel v3 (layout v4.4 + PCA reformulado Z0/Z1 + split-K).
  - clone_model_fused_v44: copia de clone_model_fused pero instanciando
    SX8LinearV44 (para el embed se reutiliza SX8Embedding del padre).

Test: test_igualdad_v44 — logits a través de 32 capas vs FP16 (cuBLAS),
mismo protocolo que integrate_fused.test_igualdad (rel < 5e-2).
"""
import sys, pickle, time, gc
import torch
import torch.nn as nn

SCRIPT_DIR = "/mnt/Data_3TB/project Marla/quant-paper/scripts"
sys.path.insert(0, SCRIPT_DIR)

from integrate_fused import SX8Linear, clone_model_fused  # noqa: E402
from sx8_embed_v44 import SX8EmbeddingV44  # noqa: E402
from sx8_decode1_v3 import decode1_v44, best_split_k, make_tensors_v44  # noqa: E402
from eval_common import load_model  # noqa: E402

PKL_V43 = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
DEV = torch.device("cuda")


class SX8LinearV44(SX8Linear):
    """SX8Linear con decode M=1 v4.4 (kernel CUDA v3). M>=32: wmma (padre)."""

    def __init__(self, qt, bases_info, bias=None, out_f=None, in_f=None,
                 hdr_bytes=6):
        super().__init__(qt, bases_info, bias, out_f, in_f)
        self.hdr_bytes = hdr_bytes
        self._t44 = None  # tensores layout v4.4 (lazy)

    def _load_t44(self):
        if self._t44 is None:
            self._t44 = make_tensors_v44(self.qt, self.bases_info,
                                         hdr_aligned=(self.hdr_bytes == 8))

    def forward(self, x):
        x = x.contiguous()
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        M = x.shape[0]
        if M == 1:
            # ---- decode M=1 v4.4 (nuevo kernel CUDA v3) ----
            # Solo se construyen las tablas compactas v4.4 (30 B/bloque); las
            # tablas wmma (_tensors) NO se crean para M==1 (VRAM mínima).
            self._load_t44()
            hdr_t, lvl_t, bas_t, sca_t = self._t44
            split_k = best_split_k(self.out_f, self.qt['n_cb'])
            y, _ = decode1_v44(x[0], self.qt, self.bases_info, hdr_t, lvl_t,
                               bas_t, sca_t, self.hdr_bytes, split_k)
            y = y.unsqueeze(0)
        else:
            # ---- M>=32: wmma (tensor cores, igual que antes) ----
            if self._tensors is None:
                self._tensors = make_tensors(self._kbm, self.bases_info, x.device)
            y = gemm_sx8_wmma_cached(x, self._tensors, self.qt['n_cb'])
        y = y.half()
        if self.bias is not None:
            y = y + self.bias
        if x.dim() != orig_shape:
            y = y.reshape(*orig_shape[:-1], self.out_f)
        return y


# import tardio (evita ciclo)
from sx8_fused_wmma import gemm_sx8_wmma_cached, make_tensors  # noqa: E402


def clone_model_fused_v44(model, wd, bd, hdr_bytes=6):
    """Sustituye IN SITU los lineales por SX8LinearV44 (M==1 -> kernel v4.4).
    Mismo recorrido que clone_model_fused (nombres de parámetros en wd)."""
    m2 = model
    n_replaced = 0
    for name, p in list(m2.named_parameters()):
        if name in wd and p.dim() >= 2:
            qt = wd[name]
            bi = bd[name]
            parts = name.split('.')
            parent = m2
            for part in parts[:-2]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            lin_attr = parts[-2]
            lin = getattr(parent, lin_attr)
            if isinstance(lin, nn.Embedding):
                setattr(parent, lin_attr, SX8EmbeddingV44(qt, bi))
                torch.cuda.empty_cache()
                n_replaced += 1
                continue
            if not isinstance(lin, nn.Linear):
                continue
            new_lin = SX8LinearV44(qt, bi, lin.bias, lin.out_features,
                                   lin.in_features, hdr_bytes=hdr_bytes)
            setattr(parent, lin_attr, new_lin)
            del lin.weight
            torch.cuda.empty_cache()
            n_replaced += 1
    return m2, n_replaced


def test_igualdad_v44(n_tokens=16, seed=42, hdr_bytes=6, verbose=True):
    """Test de igualdad end-to-end: logits FP16 (cuBLAS) vs fused v4.4,
    a través de 32 capas. Mismo protocolo que integrate_fused.test_igualdad."""
    d = pickle.load(open(PKL_V43, "rb"))
    wd, bd = d['weights'], d['bases']

    torch.manual_seed(seed)
    model_fp16, tok, _ = load_model(quantized=True, use_cache=False, mode="v43")
    model_fp16.eval()
    torch.manual_seed(seed + 1)
    ids = torch.randint(0, tok.vocab_size - 10, (1, n_tokens)).to(DEV)
    with torch.no_grad():
        t0 = time.time()
        out_fp16 = model_fp16(input_ids=ids).logits
        t_ref = time.time() - t0
    ref_cpu = out_fp16.float().cpu()
    del model_fp16, out_fp16
    torch.cuda.empty_cache()
    gc.collect()
    import kernel_sx8_v43 as k43
    k43._BUFFER_POOL.clear()
    k43._BUFFER_POOL_ORDER.clear()
    torch.cuda.empty_cache()
    gc.collect()

    torch.manual_seed(seed)
    model_fused, tok2, _ = load_model(quantized=False, use_cache=False)
    model_fused.eval()
    m_fused, n_rep = clone_model_fused_v44(model_fused, wd, bd,
                                           hdr_bytes=hdr_bytes)
    if verbose:
        print(f"Lineales sustituidos por SX8LinearV44: {n_rep}")
    with torch.no_grad():
        t0 = time.time()
        out_fused = m_fused(input_ids=ids).logits
        t_fus = time.time() - t0
    fused_cpu = out_fused.float().cpu()

    diff = (ref_cpu - fused_cpu).abs()
    denom = ref_cpu.abs().max().item() + 1e-9
    maxd = diff.max().item()
    mean_rel = diff.mean().item() / (ref_cpu.abs().mean().item() + 1e-9)
    ok = maxd / denom < 5e-2

    if verbose:
        print(f"logits: maxdiff={maxd:.3e}  rel={maxd/denom:.3e}  mean_rel={mean_rel:.3e}")
        print(f"tiempo forward (16 tok): FP16={t_ref*1e3:.0f}ms  FUSEDv44={t_fus*1e3:.0f}ms")
    print(f"TEST IGUALDAD V44 (32 capas, logits, hdr{hdr_bytes}B): {'PASS' if ok else 'FAIL'}")
    return ok, maxd / denom


if __name__ == "__main__":
    hb = 8 if len(sys.argv) > 1 and sys.argv[1] == "hdr8" else 6
    ok, rel = test_igualdad_v44(hdr_bytes=hb)
    sys.exit(0 if ok else 1)
