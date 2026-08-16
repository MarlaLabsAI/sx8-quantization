// sx8_embed_v44.cu — Embedding SX8 v4.4 capturable por CUDA Graphs (CUADRO 2)
//
// Sustituye el SX8Embedding numba (que rompía la captura: asigna buffers y
// llama a cuda.synchronize). Mismo cómputo: Out[t, :] = W[ids[t], :] con el
// layout v4.4 (hdr 6 B + levels 24 B, kb-major) + PCA por peso.
//
// CLAVE PARA CUDA GRAPHS: el wrapper recibe `out` PREASIGNADO (tensor half
// (T,K)) y lo rellena — sin torch::empty en el hot path.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float __half_raw_to_float(unsigned short s) {
    return __half2float(__ushort_as_half(s));
}

// 1 warp por token; thread tid decodifica los pesos k = tid, tid+32, ...
// (mismo esquema que el kernel numba original, layout v4.4)
__global__ void sx8_embed_v44_kernel(
        const long long* __restrict__ ids,   // (T,)
        const unsigned char* __restrict__ hdr,    // (n_blk, 6) kb-major
        const unsigned char* __restrict__ levels, // (n_blk, 24) kb-major
        const float* __restrict__ basis,      // (n_cb, 64)
        const float* __restrict__ scales,     // (n_cb, 2)
        __half* __restrict__ Out,             // (T, K) preasignado
        int T, int K, int N, int n_cb) {
    int tid = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int t = blockIdx.x * 8 + wid;
    if (t >= T) return;
    long long row = ids[t];
    if (row >= N) return;

    int k = tid;
    while (k < K) {
        int kb = k >> 5;
        int bid = kb * N + (int)row;   // kb-major
        const unsigned char* h = hdr + bid * 6;
        unsigned short h0, h1;
        memcpy(&h0, h + 0, 2);
        memcpy(&h1, h + 2, 2);
        unsigned char cfg = h[4];
        unsigned char coeff = h[5];
        float lo_f = __half_raw_to_float(h0);
        float hi_f = __half_raw_to_float(h1);

        // rlo/step en vuelo para mi sub-bloque (tid>>3)
        float q = (hi_f - lo_f) * 0.25f;
        int s = (cfg >> ((tid >> 3) * 2)) & 3;
        float rlo = lo_f + q * (3 * (s == 2) + (s == 3));
        float rhi = hi_f - q * (3 * (s == 1) + (s == 3));
        float step = (rhi - rlo) * 0.015873f;
        if (step < 1e-10f) step = 1e-10f;

        int c0 = coeff & 0xF;  if (c0 >= 8) c0 -= 16;
        int c1 = (coeff >> 4) & 0xF;  if (c1 >= 8) c1 -= 16;

        // levels v4.4: hi[4] en sb*4, lo[2] en 16+sb*2  (sb = tid>>3, peso i = tid&7)
        const unsigned char* lv = levels + bid * 24;
        int sb = tid >> 3;
        unsigned int hi4;
        unsigned short lo2;
        memcpy(&hi4, lv + sb * 4, 4);
        memcpy(&lo2, lv + 16 + sb * 2, 2);
        int i = tid & 7;
        int nib = (int)((hi4 >> (i * 4)) & 0xF);
        int qua = (int)((lo2 >> ((i >> 2) * 8 + (3 - (i & 3)) * 2)) & 0x3);
        int lv_i = (nib << 2) | qua;

        float w = rlo + step * (float)lv_i
                + (float)c0 * scales[kb * 2] * basis[kb * 64 + tid]
                + (float)c1 * scales[kb * 2 + 1] * basis[kb * 64 + 32 + tid];
        Out[(long long)t * K + k] = __float2half(w);
        k += 32;
    }
}

torch::Tensor sx8_embed_v44(torch::Tensor ids, torch::Tensor hdr, torch::Tensor levels,
                            torch::Tensor basis, torch::Tensor scales,
                            int64_t K, int64_t N, int64_t n_cb, torch::Tensor out) {
    int T = ids.numel();
    auto ids_c = ids.reshape(-1).contiguous().to(torch::kLong);
    sx8_embed_v44_kernel<<<(T + 7) / 8, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const long long*)ids_c.data_ptr(),
        (const unsigned char*)hdr.data_ptr(),
        (const unsigned char*)levels.data_ptr(),
        (const float*)basis.data_ptr(),
        (const float*)scales.data_ptr(),
        (__half*)out.data_ptr(),
        T, (int)K, (int)N, (int)n_cb);
    return out;
}
