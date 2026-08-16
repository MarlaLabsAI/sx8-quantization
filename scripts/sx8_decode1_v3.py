"""sx8_decode1_v3.py — kernel decode M=1 para el layout v4.4 (FASE A3).

Compila cuda/sx8_decode1_v3.cu (diseño v2 COALESCENTE) y expone:
  - compute_z44(X, basis, scales, n_cb) -> Z (n_cb, 2)   [por capa]
  - decode1_v44(X, hdr, levels, Z, n_cb, hdr_bytes) -> Y (N,)  [GEMV]

DOS variantes de layout:
  - hdr_bytes=6: hdr 6 B/bloque -> VRAM pesos 4.373 GB (EXACTO, objetivo)
  - hdr_bytes=8: hdr 8 B/bloque (pad 2, alineado) -> 4.63 GB (opción rendimiento)

Test de igualdad: decode1_v44 vs decode_tensor_gpu_fast (numba v43, referencia
del pipeline de PPL) sobre tensores reales del pkl — PARA AMBAS VARIANTES.

Benchmark: GB/s efectivos (bytes = (24+hdr_bytes) por bloque leídos de DRAM).
"""
import os
import sys
import time
import pickle
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
DEV = torch.device("cuda")

CUDA_SRC = open(os.path.join(SCRIPT_DIR, "..", "cuda", "sx8_decode1_v3.cu")).read()

_cached = None


def _ext():
    global _cached
    if _cached is None:
        _cached = load_inline(
            name="sx8_decode1_v3",
            cpp_sources="""
torch::Tensor compute_z44(torch::Tensor X, torch::Tensor basis, torch::Tensor scales, int64_t n_cb);
torch::Tensor decode1_v44(torch::Tensor X, torch::Tensor hdr, torch::Tensor levels,
                          torch::Tensor Z, int64_t n_cb, int64_t hdr_bytes, int64_t split_k);
""",
            cuda_sources=CUDA_SRC,
            functions=["compute_z44", "decode1_v44"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _cached


def make_tensors_v44(qt, bases_info, hdr_aligned=False):
    """Layout v4.4 (kb-major) desde el pkl -> tensores CUDA.
    hdr (n_blk,6|8) u8 + levels (n_blk,24) u8 + bases (n_cb,64) f32 + scales (n_cb,2) f32."""
    from prepare_v44 import to_v44
    v44 = to_v44(qt, bases_info, hdr_aligned=hdr_aligned)
    hdr_t = torch.from_numpy(v44['hdr'].copy()).to(DEV)
    lvl_t = torch.from_numpy(v44['levels'].copy()).to(DEV)
    bas_t = torch.from_numpy(v44['bases'].copy()).to(DEV)
    sca_t = torch.from_numpy(v44['scales'].copy()).to(DEV)
    return hdr_t, lvl_t, bas_t, sca_t


def best_split_k(N, n_cb):
    """Split-K calibrado con microbench_v44.py (RTX 5060 Ti):
    - N grande (>=8192): 4-6 splits bastan (0.67 waves base)
    - N pequeño (2560): 6-16 splits (0.19 waves base -> llenar)
    Regla: apuntar a ~216 bloques residentes (36 SM x 6) x 2.5 ondas, máx 16."""
    n_blocks = (N + 63) // 64
    want = 540  # ~2.5 waves
    sk = max(1, (want + n_blocks - 1) // n_blocks)
    return min(sk, 16)


def decode1_v44(X, qt, bases_info, hdr_t=None, lvl_t=None, bas_t=None, sca_t=None,
                hdr_bytes=6, split_k=None):
    """Y = X @ W (SX8 v4.4) con el kernel nuevo. X: (K,) half."""
    if hdr_t is None:
        hdr_t, lvl_t, bas_t, sca_t = make_tensors_v44(qt, bases_info,
                                                      hdr_aligned=(hdr_bytes == 8))
    n_cb = qt['n_cb']
    N = qt['shape'][0]
    if split_k is None:
        split_k = best_split_k(N, n_cb)
    Xc = X.contiguous().half()
    ext = _ext()
    Z = ext.compute_z44(Xc, bas_t, sca_t, n_cb)
    Y = ext.decode1_v44(Xc, hdr_t, lvl_t, Z, n_cb, hdr_bytes, split_k)
    return Y, (hdr_t, lvl_t, bas_t, sca_t)


def test_equality(names, hdr_bytes=6, seed=1):
    """decode1_v44 vs decode_tensor_gpu_fast (numba v43) — mismo X, mismas columnas."""
    sys.path.insert(0, SCRIPT_DIR)
    from kernel_sx8_v43 import decode_tensor_gpu_fast

    d = pickle.load(open(PKL, "rb"))
    wd, bd = d['weights'], d['bases']
    rng = np.random.default_rng(seed)
    ok = True
    for name in names:
        qt, bi = wd[name], bd[name]
        out_f, in_f = qt['shape']
        n_cb = qt['n_cb']
        hdr_t, lvl_t, bas_t, sca_t = make_tensors_v44(qt, bi,
                                                      hdr_aligned=(hdr_bytes == 8))
        X = torch.randn(in_f, dtype=torch.float16, device=DEV) * 0.5
        Y_new, _ = decode1_v44(X, qt, bi, hdr_t, lvl_t, bas_t, sca_t, hdr_bytes)
        W = decode_tensor_gpu_fast(qt, bi)  # (out_f, in_f) half cuda
        Y_ref = (X.float() @ W.float().t()).squeeze(0)  # (N,)
        md = float((Y_new.float() - Y_ref).abs().max())
        rel = md / float(Y_ref.abs().max() + 1e-9)
        status = "PASS" if md < 2e-2 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  hdr{hdr_bytes}B {name[:52]:<54} {in_f:>6}x{out_f:<6} maxdiff={md:.3e} rel={rel:.3e} [{status}]")
    return ok


def bench(names, hdr_bytes=6, iters=30):
    """Benchmark GB/s del kernel decode1_v44 (bytes/peso = (24+hdr_bytes)/32)."""
    d = pickle.load(open(PKL, "rb"))
    wd, bd = d['weights'], d['bases']
    ext = _ext()
    total_bytes = 0.0
    total_time = 0.0
    for name in names:
        qt, bi = wd[name], bd[name]
        out_f, in_f = qt['shape']
        n_cb = qt['n_cb']
        hdr_t, lvl_t, bas_t, sca_t = make_tensors_v44(qt, bi,
                                                      hdr_aligned=(hdr_bytes == 8))
        X = torch.randn(in_f, dtype=torch.float16, device=DEV) * 0.5
        Xc = X.contiguous()
        sk = best_split_k(out_f, n_cb)
        Z = ext.compute_z44(Xc, bas_t, sca_t, n_cb)
        ext.decode1_v44(Xc, hdr_t, lvl_t, Z, n_cb, hdr_bytes, sk)  # warmup
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            Z = ext.compute_z44(Xc, bas_t, sca_t, n_cb)
            ext.decode1_v44(Xc, hdr_t, lvl_t, Z, n_cb, hdr_bytes, sk)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        dt = float(np.median(ts))
        n_bytes = qt['n_blocks'] * (24 + hdr_bytes)
        total_bytes += n_bytes
        total_time += dt
        gbps = n_bytes / dt / 1e9
        print(f"  hdr{hdr_bytes}B {name[:52]:<54} K={in_f:>6} N={out_f:<6} {dt*1e3:7.3f} ms  {gbps:6.1f} GB/s")
    print(f"  TOTAL hdr{hdr_bytes}B: {total_bytes/1e9:.2f} GB en {total_time*1e3:.0f} ms -> {total_bytes/total_time/1e9:.1f} GB/s")
    return total_bytes / total_time / 1e9


def run_all(hdr_bytes=6):
    names = ["model.language_model.layers.0.mlp.gate_proj.weight",
             "model.language_model.layers.0.mlp.down_proj.weight"]
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"=== Variante hdr {hdr_bytes} B ===")
    print("  Test de igualdad vs kernel numba v43 (referencia PPL):")
    test_equality(names, hdr_bytes)
    print("  Benchmark (bytes = 24+hdr por bloque):")
    bench(names, hdr_bytes)
    print()


if __name__ == "__main__":
    run_all(6)
    run_all(8)

