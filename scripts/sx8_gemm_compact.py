"""sx8_gemm_compact.py — GEMM compacto S-X8 v4.4 para M>=32 (prompt/PPL).

Compila cuda/sx8_gemm_compact.cu y expone:
  - compute_z44_batch(X, basis, scales, n_cb) -> Z (M, n_cb, 2)   [PCA por fila]
  - gemm_compact(X, hdr, levels, Z, n_cb, hdr_bytes) -> Y (M, N)  [GEMM compacto]

El peso se decodifica on-the-fly desde el layout compacto v4.4 (30 B/bloque,
kb-major) — NUNCA se materializa FP16: VRAM = datos compactos + Z + Y.
Misma decodificación que decode1_v44 (M=1) pero con M filas (tiles de 8,
decode amortizado).
"""
import os
import sys
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_SRC = open(os.path.join(SCRIPT_DIR, "..", "cuda", "sx8_gemm_compact.cu")).read()

_cached = None


def _ext():
    global _cached
    if _cached is None:
        _cached = load_inline(
            name="sx8_gemm_compact",
            cpp_sources="""
torch::Tensor compute_z44_batch(torch::Tensor X, torch::Tensor basis, torch::Tensor scales, int64_t n_cb);
torch::Tensor gemm_compact(torch::Tensor X, torch::Tensor hdr, torch::Tensor levels,
                           torch::Tensor Z, int64_t n_cb, int64_t hdr_bytes);
""",
            cuda_sources=CUDA_SRC,
            functions=["compute_z44_batch", "gemm_compact"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _cached


def gemm_sx8_compact(X, qt, bases_info, hdr_t=None, lvl_t=None, bas_t=None, sca_t=None,
                     hdr_bytes=6, in_f=None, n_cb=None):
    """Y = X @ W^T (S-X8 v4.4 compacto). X: (M, K) fp16 CUDA.
    Devuelve Y (M, N) fp32 CUDA. X se rellena con ceros hasta n_cb*32 si K no es múltiplo.
    N < 64 (menos de un tile de columnas): se decodifica el tensor completo a FP16
    (son diminutos: p.ej. in_proj_a 32×2560 = 164 KB) y se hace matmul estándar —
    el kernel GEMM compacto está validado para N >= 64."""
    if in_f is None:
        in_f = qt['shape'][1]
    if n_cb is None:
        n_cb = qt['n_cb']
    out_f = qt['shape'][0]
    if out_f < 64:
        from kernel_sx8_v43 import decode_tensor_gpu_fast
        W = decode_tensor_gpu_fast(qt, bases_info)
        Y = torch.nn.functional.linear(X.contiguous(), W).float()
        del W
        return Y
    if hdr_t is None:
        from sx8_decode1_v3 import make_tensors_v44
        hdr_t, lvl_t, bas_t, sca_t = make_tensors_v44(qt, bases_info,
                                                      hdr_aligned=(hdr_bytes == 8))
    ext = _ext()
    Xc = X.contiguous()
    K = Xc.shape[1]
    Kp = n_cb * 32
    if K < Kp:
        Xp = torch.zeros((Xc.shape[0], Kp), dtype=Xc.dtype, device=Xc.device)
        Xp[:, :K] = Xc
        Xc = Xp
    Z = ext.compute_z44_batch(Xc, bas_t, sca_t, n_cb)
    Y = ext.gemm_compact(Xc, hdr_t, lvl_t, Z, n_cb, hdr_bytes)
    return Y
