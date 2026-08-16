"""sx8_fused_wmma.py — Kernel FUSED S-X8 v4.3 + tensor cores (wmma m16n16k16) — FASE 3.

Patron (llama.cpp-style): decode SX8 en vuelo -> shared -> mma.sync (tensor cores).
- Block 256 threads (8 warps); cada warp = 1 tile de 16 columnas N
- TM=8 tiles de M (128 filas por bloque): el decode de cada bloque de W se
  amortiza sobre TM tiles de M (b_frag reutilizado TM veces)
- Decode SIN branches: estrategia expresada como aritmetica booleana
- Metadatos en kb-major (bid = kb*N + n), dmin/dmax en fp16

Rendimiento medido (RTX 5060 Ti, M=2048): ~17.5 TFLOPs vs 950 GF/s numba v3
(18x) y 44 TFLOPs cuBLAS (2.5x). El decode SX8 cuesta ~50% (mma puro: 43 TFLOPs).
"""
import os
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

_CU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuda", "sx8_fused_wmma.cu")
_CU_DECODE1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuda", "sx8_decode1.cu")

_CPP_SRC = "torch::Tensor fused_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"
_CPP_DECODE1_SRC = "torch::Tensor decode1_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);"

_module = None
_module_decode1 = None


def _get_module():
    global _module
    if _module is None:
        with open(_CU_PATH) as f:
            cu = f.read()
        _module = load_inline(
            name="sx8fwmma_v6",
            cpp_sources=_CPP_SRC,
            cuda_sources=cu,
            functions=["fused_forward"],
            verbose=False,
            extra_cuda_cflags=["-O3"],
        )
    return _module


def _get_decode1_module():
    global _module_decode1
    if _module_decode1 is None:
        with open(_CU_DECODE1_PATH) as f:
            cu = f.read()
        _module_decode1 = load_inline(
            name="sx8decode1fast",
            cpp_sources=_CPP_DECODE1_SRC,
            cuda_sources=cu,
            functions=["decode1_forward"],
            verbose=False,
            extra_cuda_cflags=["-O3"],
        )
    return _module_decode1


def gemm_sx8_wmma_cached(X, tensors, n_cb):
    """Y = X @ W con tensor cores. tensors: dict con los 8 tensores GPU ya creados."""
    mod = _get_module()
    return mod.fused_forward(X.contiguous(),
                             tensors['dmin'], tensors['dmax'], tensors['config'],
                             tensors['coeff'], tensors['hi'], tensors['lo'],
                             tensors['basis'], tensors['scales'], int(n_cb))


def make_tensors(qt_kbmajor, bases_info, device):
    """Crea los tensores GPU de los metadatos SX8 (una vez, para cachear).
    Incluye rlo/step precomputados (leccion v8: decode a 2 ops/peso)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from precompute_rlo_step import precompute_rlo_step
    rlo_f, step_f = precompute_rlo_step(qt_kbmajor)
    return {
        'rlo': torch.as_tensor(rlo_f.reshape(-1), device=device),
        'step': torch.as_tensor(step_f.reshape(-1), device=device),
        'dmin': torch.as_tensor(qt_kbmajor['dmin'], device=device),
        'dmax': torch.as_tensor(qt_kbmajor['dmax'], device=device),
        'config': torch.as_tensor(qt_kbmajor['config'], device=device),
        'coeff': torch.as_tensor(qt_kbmajor['coeff'], device=device),
        'hi': torch.as_tensor(qt_kbmajor['levels_hi'], device=device),
        'lo': torch.as_tensor(qt_kbmajor['levels_lo'], device=device),
        'basis': torch.as_tensor(bases_info['data'].reshape(-1).astype(np.float32), device=device),
        'scales': torch.as_tensor(bases_info['scales'].reshape(-1).astype(np.float32), device=device),
    }


def gemm_sx8_wmma(X, qt_kbmajor, bases_info, n_cb):
    """Y = X @ W con W SX8 v4.3 (metadatos en kb-major) usando tensor cores.
    X: (M, K) fp16 CUDA. qt_kbmajor: dict con dmin/dmax fp16, config/coeff/levels u8.
    Devuelve Y (M, N) fp32 CUDA."""
    return gemm_sx8_wmma_cached(X, make_tensors(qt_kbmajor, bases_info, X.device), n_cb)


def decode1_sx8_cached(X, tensors, n_cb):
    """Decode M=1 (CUDA C, rlo/step precomputados): Y[n] = X @ W[:,n].
    X: (K,) fp16. Devuelve Y (N,) fp32. ~111 GB/s (leccion v8: 2 ops/peso)."""
    mod = _get_decode1_module()
    return mod.decode1_forward(X.contiguous(),
                               tensors['rlo'], tensors['step'],
                               tensors['coeff'], tensors['hi'], tensors['lo'],
                               tensors['basis'], tensors['scales'], int(n_cb))


if __name__ == "__main__":
    # test rapido de correctness
    import sys, pickle, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kernel_sx8_fused import to_kbmajor
    from kernel_sx8_v43 import decode_tensor_gpu_fast

    pkl = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
    d = pickle.load(open(pkl, 'rb'))
    wd, bd = d['weights'], d['bases']
    name = "model.language_model.layers.3.self_attn.q_proj.weight"
    qt, bi = wd[name], bd[name]
    out_f, in_f = qt['shape']
    n_cb = qt['n_cb']
    qk, _ = to_kbmajor(qt, bi)

    rng = np.random.default_rng(0)
    M = 2048
    X = torch.tensor(rng.standard_normal((M, in_f)).astype(np.float32) * 0.02,
                     device='cuda', dtype=torch.float16)
    W = decode_tensor_gpu_fast(qt, bi).reshape(out_f, in_f)
    Y_ref = X.float() @ W.T.float()

    tens = make_tensors(qk, bi, 'cuda')
    Y = gemm_sx8_wmma_cached(X, tens, n_cb)
    rel = (Y.cpu() - Y_ref.cpu()).abs().max().item() / (Y_ref.abs().max().item() + 1e-9)
    print(f"correctness: rel={rel:.3e} {'PASS' if rel < 5e-3 else 'FAIL'}")

    for _ in range(3):
        gemm_sx8_wmma_cached(X, tens, n_cb)
    torch.cuda.synchronize()
    ts = []
    for _ in range(10):
        t0 = time.time(); gemm_sx8_wmma_cached(X, tens, n_cb); torch.cuda.synchronize()
        ts.append(time.time() - t0)
    t = min(ts)
    fl = M * in_f * out_f * 2
    print(f"fused wmma (cached): {t*1e3:.2f} ms -> {fl/(t)/1e9:.0f} GF/s")
