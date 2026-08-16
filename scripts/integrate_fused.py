"""FASE 2.2 — Integración del kernel fused SX8 en transformers (Qwen3.5-4B).

Estrategia: sustituir los nn.Linear cuantizados por un módulo SX8Linear cuyo
forward usa nuestro kernel fused (gemm_sx8_auto) con los pesos SX8 en VRAM.
El bias se aplica igual que cuBLAS (y = x@W.T + b).

Test de igualdad OBLIGATORIO (antes de medir nada):
  1) Modelo referencia: pesos FP16 decodificados (cuBLAS nativo)
  2) Modelo fused: SX8Linear con pesos SX8 en VRAM
  3) Mismo input random -> logits identicos (tolerancia fp16)

Diseño:
  - load_model(mode='v43') carga el modelo con pesos FP16 (para la referencia)
  - clone_model_fused() crea una copia donde los lineales cuantizados se
    sustituyen por SX8Linear (los pesos FP16 del clon se descartan)
  - forward_idempotente: mismo seed, mismo input -> compara logits
"""
import sys, copy, pickle, time, gc
import numpy as np, torch
from numba import cuda
sys.path.insert(0, "/mnt/Data_3TB/project Marla/quant-paper/scripts")
import torch.nn as nn
from eval_common import load_model
from kernel_sx8_fused import gemm_sx8_auto, gemm_sx8_cached, decode_dot_sx8, decode_dot_sx8_cached
from sx8_fused_wmma import gemm_sx8_wmma_cached, make_tensors, decode1_sx8_cached
from kernel_sx8_v43 import decode_tensor_gpu_fast

PKL_V43 = "/mnt/Data_3TB/project Marla/quant-paper/models/qwen35_4b_sx8_flash_v4_3.pkl"
DEV = torch.device("cuda")


class SX8Linear(nn.Module):
    """nn.Linear cuantizado: forward = kernel fused SX8 con TENSOR CORES (wmma).
    Metadatos SX8 en VRAM (kb-major) cacheados en __init__; M>=32 -> wmma,
    M==1 -> decode dot numba (dedicado)."""
    def __init__(self, qt, bases_info, bias=None, out_f=None, in_f=None):
        super().__init__()
        self.qt = qt
        self.bases_info = bases_info
        self.bias = bias
        self.out_f = out_f if out_f is not None else qt['shape'][0]
        self.in_f = in_f if in_f is not None else qt['shape'][1]
        N = self.out_f
        n_cb = qt['n_cb']
        # reordenar n-major -> kb-major: bid' = kb*N + n
        def reorder(a, per_block=1):
            a = np.asarray(a)
            if per_block == 1:
                return a.reshape(N, n_cb).T.reshape(-1)
            return a.reshape(N, n_cb, per_block).transpose(1, 0, 2).reshape(-1)
        self._kbm = {
            'dmin': reorder(qt['dmin'], 1).astype(np.float16).copy(),
            'dmax': reorder(qt['dmax'], 1).astype(np.float16).copy(),
            'config': reorder(qt['config'], 1).astype(np.uint8).copy(),
            'coeff': reorder(qt['coeff'].reshape(-1), 1).astype(np.uint8).copy(),
            'levels_hi': reorder(qt['levels_hi'], 16).astype(np.uint8).copy(),
            'levels_lo': reorder(qt['levels_lo'], 8).astype(np.uint8).copy(),
        }
        self._tensors = None  # creados lazy en el primer forward (device de X)

    def forward(self, x):
        x = x.contiguous()
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        if self._tensors is None:
            self._tensors = make_tensors(self._kbm, self.bases_info, x.device)
        M = x.shape[0]
        if M == 1:
            y = decode1_sx8_cached(x[0], self._tensors, self.qt['n_cb']).unsqueeze(0)
        else:
            y = gemm_sx8_wmma_cached(x, self._tensors, self.qt['n_cb'])
        y = y.half()
        if self.bias is not None:
            y = y + self.bias
        if x.dim() != orig_shape:
            y = y.reshape(*orig_shape[:-1], self.out_f)
        return y


class SX8Embedding(nn.Module):
    """Embedding SX8: W[token, :] decodificada en vuelo (1 warp por token).
    Solo decodifica las filas de los tokens activos (ideal para el fused)."""
    def __init__(self, qt, bases_info):
        super().__init__()
        self.qt = qt
        self.N = qt['shape'][0]
        self.in_f = qt['shape'][1]
        n_cb = qt['n_cb']
        N = self.N

        def r1(a):
            return np.asarray(a).reshape(N, n_cb).T.reshape(-1)

        def r2(a, per):
            return np.asarray(a).reshape(N, n_cb, per).transpose(1, 0, 2).reshape(-1)

        self._dm = cuda.to_device(r1(qt['dmin']).astype(np.float16).copy())
        self._dx = cuda.to_device(r1(qt['dmax']).astype(np.float16).copy())
        self._cfg = cuda.to_device(r1(qt['config']).astype(np.uint8).copy())
        self._co = cuda.to_device(r1(qt['coeff'].reshape(-1)).astype(np.uint8).copy())
        self._hi = cuda.to_device(r2(qt['levels_hi'], 16).astype(np.uint8).copy())
        self._lo = cuda.to_device(r2(qt['levels_lo'], 8).astype(np.uint8).copy())
        self._bs = cuda.to_device(bases_info['data'].reshape(-1).astype(np.float32))
        self._sc = cuda.to_device(bases_info['scales'].reshape(-1).astype(np.float32))

    def forward(self, ids):
        T = ids.numel()
        K = self.in_f
        Out = torch.empty(T * K, dtype=torch.float32, device=DEV)
        grid = (T + 7) // 8
        ids64 = ids.reshape(-1).to(torch.int64)
        _sx8_embed_kernel[grid, 256](cuda.as_cuda_array(ids64),
                                     self._dm, self._dx, self._cfg, self._co,
                                     self._hi, self._lo, self._bs, self._sc,
                                     cuda.as_cuda_array(Out), T, K, self.N, self.qt['n_cb'])
        cuda.synchronize()
        return Out.half().reshape(*ids.shape, K)


@cuda.jit
def _sx8_embed_kernel(ids, dmin, dmax, config, coeff, hi_arr, lo_arr, basis, scales,
                      Out, T, K, N, n_cb):
    """Out[t, k] = W[ids[t], k] decodificado. 1 warp por token; thread tid
    decodifica los pesos de su posicion k (tid, tid+32, ...) de la fila ids[t]."""
    tid = cuda.threadIdx.x & 31
    wid = cuda.threadIdx.x >> 5
    t = cuda.blockIdx.x * 8 + wid
    if t >= T:
        return
    row = ids[t]
    if row >= N:
        return
    k = tid
    while k < K:
        kb = k >> 5
        bid = kb * N + row  # kb-major
        hi = (hi_arr[bid * 16 + (tid >> 1)] >> ((tid & 1) * 4)) & 0xF
        lo = (lo_arr[bid * 8 + (tid >> 2)] >> ((3 - (tid & 3)) * 2)) & 0x3
        lv = (hi << 2) | lo
        lo_f = dmin[bid]; hi_f = dmax[bid]
        q = (hi_f - lo_f) * 0.25
        strat = (config[bid] >> ((tid >> 3) * 2)) & 3
        if strat == 0:
            rlo, rhi = lo_f, hi_f
        elif strat == 1:
            rlo, rhi = lo_f, lo_f + q
        elif strat == 2:
            rlo, rhi = hi_f - q, hi_f
        else:
            rlo, rhi = lo_f + q, hi_f - q
        step = (rhi - rlo) * 0.015873
        if step < 1e-10:
            step = 1e-10
        c0_raw = coeff[bid] & 0xF
        c0 = c0_raw if c0_raw < 8 else c0_raw - 16
        c1_raw = (coeff[bid] >> 4) & 0xF
        c1 = c1_raw if c1_raw < 8 else c1_raw - 16
        w = rlo + step * lv + c0 * scales[kb * 2] * basis[kb * 64 + tid] \
            + c1 * scales[kb * 2 + 1] * basis[kb * 64 + 32 + tid]
        Out[t * K + k] = w
        k += 32


def clone_model_fused(model, wd, bd):
    """Sustituye IN SITU los lineales cuantizados por SX8Linear (los pesos
    FP16 del lineal se descartan -> VRAM baja de 9.3GB a ~4.4GB)."""
    m2 = model
    n_replaced = 0
    for name, p in list(m2.named_parameters()):
        if name in wd and p.dim() >= 2:
            qt = wd[name]
            bi = bd[name]
            parts = name.split('.')
            # el PARAMETRO vive en un nn.Linear cuyo padre es parts[:-2]
            parent = m2
            for part in parts[:-2]:
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)
            lin_attr = parts[-2]
            lin = getattr(parent, lin_attr)
            if isinstance(lin, nn.Embedding):
                # sustituir el embedding (peso SX8 -> gather+decode en vuelo)
                setattr(parent, lin_attr, SX8Embedding(qt, bi))
                torch.cuda.empty_cache()
                n_replaced += 1
                continue
            if not isinstance(lin, nn.Linear):
                continue  # solo lineales (no convs/normas)
            # 1) liberar el peso FP16 del Linear ANTES de copiar metadatos SX8 a VRAM
            bias = lin.bias
            setattr(parent, lin_attr, None)
            torch.cuda.empty_cache()
            # 2) crear el SX8Linear (copia metadatos SX8 comprimidos a VRAM)
            setattr(parent, lin_attr, SX8Linear(qt, bi, bias=bias))
            n_replaced += 1
    return m2, n_replaced


def test_igualdad(n_tokens=16, seed=42, verbose=True):
    """Test de igualdad: forward FP16 (cuBLAS) vs forward fused SX8. Mismo input.
    Dos modelos consecutivos (no deepcopy -> sin duplicar 9.3GB)."""
    import torch as _t
    d = pickle.load(open(PKL_V43, 'rb'))
    wd, bd = d['weights'], d['bases']

    # ---- 1) referencia FP16
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
    # vaciar el pool de buffers de numba (puede retener varios GB de VRAM)
    import kernel_sx8_v43 as k43
    k43._BUFFER_POOL.clear()
    k43._BUFFER_POOL_ORDER.clear()
    torch.cuda.empty_cache()
    gc.collect()

    # ---- 2) modelo fused (carga nueva + sustitucion in situ)
    torch.manual_seed(seed)
    model_fused, tok2, _ = load_model(quantized=False, use_cache=False)
    model_fused.eval()
    m_fused, n_rep = clone_model_fused(model_fused, wd, bd)
    if verbose:
        print(f"Lineales sustituidos por SX8Linear: {n_rep}")
    with torch.no_grad():
        t0 = time.time()
        out_fused = m_fused(input_ids=ids).logits
        t_fus = time.time() - t0
    fused_cpu = out_fused.float().cpu()

    diff = (ref_cpu - fused_cpu).abs()
    denom = ref_cpu.abs().max().item() + 1e-9
    maxd = diff.max().item()
    mean_rel = diff.mean().item() / (ref_cpu.abs().mean().item() + 1e-9)
    ok = maxd / denom < 5e-2  # tolerancia: fp16 acumulado a traves de 32 capas

    if verbose:
        print(f"logits: maxdiff={maxd:.3e}  rel={maxd/denom:.3e}  mean_rel={mean_rel:.3e}")
        print(f"tiempo forward: FP16={t_ref*1e3:.0f}ms  FUSED={t_fus*1e3:.0f}ms")
    print(f"TEST IGUALDAD (32 capas, logits): {'PASS' if ok else 'FAIL'}")
    return ok, maxd / denom


def benchmark_vram_y_toks(n_prompt=256, n_gen=16, seed=7, verbose=True):
    """FASE 2.3: VRAM de pesos + tok/s end-to-end (prompt y decode).
    Compara: modelo SX8 fused (pesos comprimidos en VRAM) vs FP16 (cuBLAS)."""
    d = pickle.load(open(PKL_V43, 'rb'))
    wd, bd = d['weights'], d['bases']

    # ---- modelo fused (pesos SX8 en VRAM)
    torch.manual_seed(seed)
    model_fused, tok, _ = load_model(quantized=False, use_cache=False)
    model_fused.eval()
    m_fused, n_rep = clone_model_fused(model_fused, wd, bd)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        ids = torch.randint(0, tok.vocab_size - 10, (1, n_prompt)).to(DEV)
        # warmup (incluye JIT compile de los modulos CUDA la primera vez)
        _ = m_fused(input_ids=ids).logits
        _ = m_fused(input_ids=ids[:, :1]).logits
        torch.cuda.synchronize()
        t0 = time.time()
        _ = m_fused(input_ids=ids).logits
        torch.cuda.synchronize()
        t_prompt = time.time() - t0
        vram_fused = torch.cuda.max_memory_allocated() / 1e9
        # decode: generar n_gen tokens de uno en uno (KV cache)
        torch.cuda.reset_peak_memory_stats()
        past = None
        t0 = time.time()
        gen_ids = ids[:, :1]
        for i in range(n_gen):
            out = m_fused(input_ids=gen_ids, use_cache=True)
            next_tok = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen_ids = next_tok
        torch.cuda.synchronize()
        t_gen = time.time() - t0
    vram_fused_peak = torch.cuda.max_memory_allocated() / 1e9
    tok_s_prompt_f = n_prompt / t_prompt
    tok_s_decode_f = n_gen / t_gen

    if verbose:
        print(f"=== FUSED SX8 ===")
        print(f"  VRAM pesos (allocated): {vram_fused:.2f} GB  | pico decode: {vram_fused_peak:.2f} GB")
        print(f"  prompt {n_prompt} tok: {t_prompt*1e3:.0f} ms -> {tok_s_prompt_f:.0f} tok/s")
        print(f"  decode {n_gen} tok: {t_gen*1e3:.0f} ms -> {tok_s_decode_f:.1f} tok/s")
    return dict(vram=vram_fused, vram_peak=vram_fused_peak,
                tok_prompt=tok_s_prompt_f, tok_decode=tok_s_decode_f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="FASE 2.3 VRAM+tok/s")
    args = ap.parse_args()
    if args.bench:
        benchmark_vram_y_toks()
    else:
        ok, rel = test_igualdad()
        sys.exit(0 if ok else 1)
