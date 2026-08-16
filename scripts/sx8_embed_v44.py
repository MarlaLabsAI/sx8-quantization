"""sx8_embed_v44.py — wrapper del embedding SX8 v4.4 capturable por CUDA Graphs.

Sustituye el SX8Embedding numba (que rompía la captura). Mismo cómputo
(Out[t,:] = W[ids[t],:]) con layout v4.4 y buffer de salida PREASIGNADO.

Test: igualdad vs el SX8Embedding numba original sobre tokens reales.
"""
import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
DEV = torch.device("cuda")

SRC = open(os.path.join(SCRIPT_DIR, "..", "cuda", "sx8_embed_v44.cu")).read()

_ext = None


def ext():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="sx8_embed_v44",
            cpp_sources="""
torch::Tensor sx8_embed_v44(torch::Tensor ids, torch::Tensor hdr, torch::Tensor levels,
                            torch::Tensor basis, torch::Tensor scales,
                            int64_t K, int64_t N, int64_t n_cb, torch::Tensor out);
""",
            cuda_sources=SRC,
            functions=["sx8_embed_v44"],
            extra_cuda_cflags=["-O3"],
            verbose=False)
    return _ext


class SX8EmbeddingV44(nn.Module):
    """Embedding SX8 v4.4 con buffer preasignado (CUDA Graphs friendly)."""

    def __init__(self, qt, bases_info, hdr_bytes=6):
        super().__init__()
        self.qt = qt
        self.bases_info = bases_info
        self.N = qt['shape'][0]
        self.K = qt['shape'][1]
        self.n_cb = qt['n_cb']
        self._t = None      # tensores v4.4 (lazy)
        self._out = None    # buffer de salida (lazy, crece con T)

    def _load(self):
        if self._t is None:
            from sx8_decode1_v3 import make_tensors_v44
            self._t = make_tensors_v44(self.qt, self.bases_info,
                                       hdr_aligned=False)
            self._hdr, self._lvl, self._bas, self._sca = self._t
            self.qt = None
            self.bases_info = None

    def forward(self, ids):
        self._load()
        T = ids.numel()
        if self._out is None or self._out.shape[0] < T:
            self._out = torch.empty(T, self.K, dtype=torch.float16,
                                    device=DEV)
        ids_c = ids.reshape(-1).to(torch.long)
        out = self._out[:T]
        ext().sx8_embed_v44(ids_c, self._hdr, self._lvl, self._bas, self._sca,
                            self.K, self.N, self.n_cb, out)
        return out.reshape(*ids.shape, self.K)


def test_embed(seed=5, n_tok=64):
    """Igualdad SX8EmbeddingV44 vs SX8Embedding numba original."""
    sys.path.insert(0, SCRIPT_DIR)
    from integrate_fused import SX8Embedding
    from eval_common import load_model

    d = pickle.load(open(PKL, "rb"))
    wd, bd = d['weights'], d['bases']
    name = "model.language_model.embed_tokens.weight"
    qt, bi = wd[name], bd[name]

    rng = np.random.default_rng(seed)
    ids = torch.tensor(rng.integers(0, qt['shape'][0], n_tok),
                       dtype=torch.long, device=DEV)

    emb_num = SX8Embedding(qt, bi).to(DEV)
    emb_v44 = SX8EmbeddingV44(qt, bi).to(DEV)

    with torch.no_grad():
        y_num = emb_num(ids)
        y_v44 = emb_v44(ids)
    md = float((y_num.float() - y_v44.float()).abs().max())
    rel = md / float(y_num.float().abs().max() + 1e-9)
    ok = md < 2e-2
    print(f"embed v44 vs numba: maxdiff={md:.3e} rel={rel:.3e} [{'PASS' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    test_embed()
