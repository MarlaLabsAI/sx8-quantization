// sx8_gemm_compact.cu — GEMM S-X8 v4.4 para M>=32 (PROMT/PPL) SIN materializar FP16.
//
// Y (M, N) = X (M, K) @ W^T  — W en layout compacto v4.4 kb-major (30 B/bloque):
//   hdr (n_blk, HDR_BYTES): dmin(fp16) dmax(fp16) config(u8) coeff(u8)  (6 u 8 B)
//   levels (n_blk, 24): por sub-bloque sb: hi[4] (sb*4) + lo[2] (16+sb*2)
//   bases (n_cb, 64) fp32, scales (n_cb, 2) fp32  (corrección PCA)
//
// DISEÑO: bloque = 256 threads = 8 warps; warp = 8 columnas consecutivas
// (coalescente en kb-major); la fila m se procesa en tiles de M_TILE=8 con
// decode AMORTIZADO (el peso se decodifica una vez por (kb, col, sb) y se
// aplica a las 8 filas del tile). Z0/Z1 (PCA por fila y kb) se precomputan
// en compute_z44_batch. VRAM: solo los datos compactos (30 B/bloque) + Z.
//
// Igualdad esperada con decode1_v44 (M=1) y con wmma: maxdiff < 1e-3.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float __half_raw_to_float(unsigned short s) {
    return __half2float(__ushort_as_half(s));
}

// ---------------------------------------------------------------------------
// compute_z44_batch: Z (M, n_cb, 2) — Z0/Z1 por fila y kb
// ---------------------------------------------------------------------------
__global__ void compute_z44_batch_kernel(
        const __half* __restrict__ X,     // (M, K) K = n_cb*32
        const float* __restrict__ basis,  // (n_cb, 64)
        const float* __restrict__ scales, // (n_cb, 2)
        float* __restrict__ Z,            // (M, n_cb, 2)
        int K, int n_cb) {
    int m = blockIdx.y;
    int kb = blockIdx.x;
    if (kb >= n_cb) return;
    int t = threadIdx.x;
    float xv = __half2float(X[(long long)m * K + kb * 32 + t]);
    float acc0 = xv * basis[kb * 64 + t];
    float acc1 = xv * basis[kb * 64 + 32 + t];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xFFFFFFFF, acc0, off);
        acc1 += __shfl_xor_sync(0xFFFFFFFF, acc1, off);
    }
    if (t == 0) {
        Z[((long long)m * n_cb + kb) * 2 + 0] = scales[kb * 2 + 0] * acc0;
        Z[((long long)m * n_cb + kb) * 2 + 1] = scales[kb * 2 + 1] * acc1;
    }
}

// ---------------------------------------------------------------------------
// gemm_compact_kernel: tiles de M_TILE filas, decode amortizado
// ---------------------------------------------------------------------------
template <int HDR_BYTES, int M_TILE>
__global__ void gemm_compact_kernel(
        const __half* __restrict__ X,      // (M, K) K = n_cb*32
        const unsigned char* __restrict__ hdr,    // (n_blk, HDR_BYTES) kb-major
        const unsigned char* __restrict__ levels, // (n_blk, 24) kb-major
        const float* __restrict__ Z,        // (M, n_cb, 2)
        float* __restrict__ Y,              // (M, N)
        int M, int N, int n_cb) {
    int t = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;    // 0..7
    int col_local = t >> 2;        // 0..7
    int sb = t & 3;
    int n = blockIdx.x * 64 + wid * 8 + col_local;
    int m_base = blockIdx.y * 64;  // 64 filas por bloque
    int m_top = min(m_base + 64, M);

    for (int m0 = m_base; m0 < m_top; m0 += M_TILE) {
        int mt = min(M_TILE, m_top - m0);
        float acc[M_TILE];
        #pragma unroll
        for (int mm = 0; mm < M_TILE; mm++) acc[mm] = 0.0f;

        for (int kb = 0; kb < n_cb; kb++) {
            int bid = kb * N + n;
            // ---- hdr ----
            const unsigned char* h = hdr + bid * HDR_BYTES;
            unsigned short h0, h1;
            memcpy(&h0, h + 0, 2);
            memcpy(&h1, h + 2, 2);
            unsigned char cfg = h[4];
            unsigned char coeff = h[5];
            float lo_f = __half_raw_to_float(h0);
            float hi_f = __half_raw_to_float(h1);
            // ---- rlo/step del sub-bloque ----
            float q = (hi_f - lo_f) * 0.25f;
            int s = (cfg >> (sb * 2)) & 3;
            float rlo = lo_f + q * (3 * (s == 2) + (s == 3));
            float rhi = hi_f - q * (3 * (s == 1) + (s == 3));
            float step = (rhi - rlo) * 0.015873f;
            if (step < 1e-10f) step = 1e-10f;
            // ---- coeff PCA ----
            int c0 = coeff & 0xF;  if (c0 >= 8) c0 -= 16;
            int c1 = (coeff >> 4) & 0xF;  if (c1 >= 8) c1 -= 16;
            // ---- levels del sub-bloque ----
            const unsigned char* lv = levels + bid * 24;
            unsigned int hi4;
            unsigned short lo2;
            memcpy(&hi4, lv + sb * 4, 4);
            memcpy(&lo2, lv + 16 + sb * 2, 2);

            // ---- aplicar a las filas del tile (decode amortizado) ----
            #pragma unroll
            for (int mm = 0; mm < M_TILE; mm++) {
                if (mm >= mt) break;
                const __half2* X2 = (const __half2*)(X + ((long long)(m0 + mm)) * (long long)(n_cb * 32) + kb * 32 + sb * 8);
                __half2 xa = X2[0];
                __half2 xb = X2[1];
                __half2 xc = X2[2];
                __half2 xd = X2[3];
                float xv[8] = {
                    __low2float(xa), __high2float(xa),
                    __low2float(xb), __high2float(xb),
                    __low2float(xc), __high2float(xc),
                    __low2float(xd), __high2float(xd),
                };
                float pca = 0.0f;
                if (sb == 0)
                    pca = (float)c0 * Z[((long long)(m0 + mm) * n_cb + kb) * 2 + 0]
                        + (float)c1 * Z[((long long)(m0 + mm) * n_cb + kb) * 2 + 1];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int nib = (int)((hi4 >> (i * 4)) & 0xF);
                    int qua = (int)((lo2 >> ((i >> 2) * 8 + (3 - (i & 3)) * 2)) & 0x3);
                    int lv_i = (nib << 2) | qua;
                    float w = rlo + step * (float)lv_i;
                    acc[mm] += xv[i] * w;
                }
                acc[mm] += pca;
            }
        }

        // ---- reducir 4 sub-bloques de la misma columna (butterfly secuencial) ----
        #pragma unroll
        for (int mm = 0; mm < M_TILE; mm++) {
            if (mm >= mt) break;
            #pragma unroll
            for (int off = 2; off > 0; off >>= 1)
                acc[mm] += __shfl_xor_sync(0xFFFFFFFF, acc[mm], off);
        }

        if (sb == 0) {
            #pragma unroll
            for (int mm = 0; mm < M_TILE; mm++)
                if (mm < mt)
                    Y[(long long)(m0 + mm) * N + n] = acc[mm];
        }
    }
}

// ---------------------------------------------------------------------------
// wrappers
// ---------------------------------------------------------------------------
torch::Tensor compute_z44_batch(torch::Tensor X, torch::Tensor basis, torch::Tensor scales,
                                int64_t n_cb) {
    int M = X.size(0);
    auto Xc = X.contiguous();
    auto Z = torch::empty({M, n_cb, 2}, X.options().dtype(torch::kFloat32));
    dim3 grid((int)n_cb, M);
    compute_z44_batch_kernel<<<grid, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)Xc.data_ptr(),
        (const float*)basis.data_ptr(),
        (const float*)scales.data_ptr(),
        (float*)Z.data_ptr(), (int)(Xc.size(1)), (int)n_cb);
    return Z;
}

torch::Tensor gemm_compact(torch::Tensor X, torch::Tensor hdr, torch::Tensor levels,
                           torch::Tensor Z, int64_t n_cb, int64_t hdr_bytes) {
    int M = X.size(0);
    int N = hdr.size(0) / n_cb;   // kb-major: n_blk = n_cb * N
    auto Xc = X.contiguous();
    auto Y = torch::empty({M, N}, X.options().dtype(torch::kFloat32));
    dim3 blocks((N + 63) / 64, (M + 63) / 64);
    if (hdr_bytes == 8) {
        gemm_compact_kernel<8, 16><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            (const __half*)Xc.data_ptr(),
            (const unsigned char*)hdr.data_ptr(),
            (const unsigned char*)levels.data_ptr(),
            (const float*)Z.data_ptr(),
            (float*)Y.data_ptr(), M, N, (int)n_cb);
    } else {
        gemm_compact_kernel<6, 16><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            (const __half*)Xc.data_ptr(),
            (const unsigned char*)hdr.data_ptr(),
            (const unsigned char*)levels.data_ptr(),
            (const float*)Z.data_ptr(),
            (float*)Y.data_ptr(), M, N, (int)n_cb);
    }
    return Y;
}
