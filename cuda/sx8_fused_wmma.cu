#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define TM 8
#define SWARPS 8
#define SWB (SWARPS * 16 * 16)
#define SWX (TM * 16 * 16)

__global__ void sx8_wmma_kernel(
        const __half* __restrict__ X,
        const __half* __restrict__ dmin, const __half* __restrict__ dmax,
        const unsigned char* __restrict__ config, const unsigned char* __restrict__ coeff,
        const unsigned char* __restrict__ hi_arr, const unsigned char* __restrict__ lo_arr,
        const float* __restrict__ basis, const float* __restrict__ scales,
        float* __restrict__ Y,
        int M, int K, int N, int n_cb) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int ltid = tid & 31;
    int m_base = blockIdx.y * (TM * 16);
    int n_base = blockIdx.x * (SWARPS * 16) + warp_id * 16;

    __shared__ __half sX[SWX];
    __shared__ __half sB[SWB];

    if (m_base >= M) return;
    for (int i = tid; i < SWB; i += 256)
        sB[i] = __float2half(0.0f);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[TM];
    #pragma unroll
    for (int tmn = 0; tmn < TM; tmn++)
        wmma::fill_fragment(acc[tmn], 0.0f);

    int ra = ltid >> 2;
    int ca = ltid & 3;

    for (int k0 = 0; k0 < K; k0 += 16) {
        #pragma unroll
        for (int tmn = 0; tmn < TM; tmn++) {
            int r = tid >> 4;
            int c = tid & 15;
            int mm = m_base + tmn * 16;
            if (mm + r < M && k0 + c < K)
                sX[tmn * 256 + r * 16 + c] = X[(mm + r) * K + k0 + c];
            else
                sX[tmn * 256 + r * 16 + c] = __float2half(0.0f);
        }

        for (int j = 0; j < 4; j++) {
            int n = n_base + ca + j * 4;
            if (n >= N) continue;
            int kk1 = k0 + ra;          // fila ra del tile
            int kk2 = k0 + ra + 8;      // fila ra+8
            if (kk1 >= K) continue;     // FIX: K no múltiplo de 16 -> tile B parcial
            int kb = kk1 >> 5;          // mismo bloque de 32 (kk2 < 32k+16 <= 32(kb+1))
            int bid = kb * N + n;
            int t1 = kk1 & 31;
            int t2 = kk2 & 31;
            // niveles (2)
            int hi1 = (hi_arr[bid * 16 + (t1 >> 1)] >> ((t1 & 1) * 4)) & 0xF;
            int lo1 = (lo_arr[bid * 8 + (t1 >> 2)] >> ((3 - (t1 & 3)) * 2)) & 0x3;
            int hi2 = (hi_arr[bid * 16 + (t2 >> 1)] >> ((t2 & 1) * 4)) & 0xF;
            int lo2 = (lo_arr[bid * 8 + (t2 >> 2)] >> ((3 - (t2 & 3)) * 2)) & 0x3;
            int lv1 = (hi1 << 2) | lo1;
            int lv2 = (hi2 << 2) | lo2;
            // metadatos del bloque (UNA vez)
            float lo_f = __half2float(dmin[bid]);
            float hi_f = __half2float(dmax[bid]);
            float q = (hi_f - lo_f) * 0.25f;
            int cfg = config[bid];
            // estrategias sin branches (por sub-bloque de t1 y t2)
            int s1 = (cfg >> ((t1 >> 3) * 2)) & 3;
            int s2 = (cfg >> ((t2 >> 3) * 2)) & 3;
            // rlo = lo + q*(3*(s==2) + (s==3)); rhi = hi - q*(3*(s==1) + (s==3))
            float rlo1 = lo_f + q * (float)(3 * (s1 == 2) + (s1 == 3));
            float rhi1 = hi_f - q * (float)(3 * (s1 == 1) + (s1 == 3));
            float rlo2 = lo_f + q * (float)(3 * (s2 == 2) + (s2 == 3));
            float rhi2 = hi_f - q * (float)(3 * (s2 == 1) + (s2 == 3));
            float stp1 = (rhi1 - rlo1) * 0.015873f;
            if (stp1 < 1e-10f) stp1 = 1e-10f;
            float stp2 = (rhi2 - rlo2) * 0.015873f;
            if (stp2 < 1e-10f) stp2 = 1e-10f;
            // PCA (1 vez por bloque)
            int c0r = coeff[bid] & 0xF; if (c0r >= 8) c0r -= 16;
            int c1r = (coeff[bid] >> 4) & 0xF; if (c1r >= 8) c1r -= 16;
            float s0 = scales[kb * 2];
            float s1v = scales[kb * 2 + 1];
            float p0 = c0r * s0;
            float p1 = c1r * s1v;
            float w1 = rlo1 + stp1 * lv1 + p0 * basis[kb * 64 + t1] + p1 * basis[kb * 64 + 32 + t1];
            float w2 = rlo2 + stp2 * lv2 + p0 * basis[kb * 64 + t2] + p1 * basis[kb * 64 + 32 + t2];
            sB[warp_id * 256 + (ra) * 16 + ca + j * 4] = __float2half(w1);
            sB[warp_id * 256 + (ra + 8) * 16 + ca + j * 4] = __float2half(w2);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(b_frag, sB + warp_id * 256, 16);
        #pragma unroll
        for (int tmn = 0; tmn < TM; tmn++) {
            int mm = m_base + tmn * 16;
            if (mm >= M) continue;   // FIX: procesar tiles parciales (sX=0 en filas inválidas)
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, sX + tmn * 256, 16);
            wmma::mma_sync(acc[tmn], a_frag, b_frag, acc[tmn]);
        }
        __syncthreads();
    }

    __shared__ float sTmp[SWARPS][256];  // store temporal por WARP (epílogo) — sin race
    #pragma unroll
    for (int tmn = 0; tmn < TM; tmn++) {
        int mm = m_base + tmn * 16;
        if (mm >= M) continue;
        int m_ok = min(16, M - mm);          // filas válidas del tile (1..16)
        int n_ok = min(16, N - n_base);      // columnas válidas (1..16)
        if (m_ok == 16 && n_ok == 16) {
            wmma::store_matrix_sync(Y + mm * N + n_base, acc[tmn], N, wmma::mem_row_major);
        } else {
            // FIX: tile parcial (M o N no múltiplo de 16) -> store temporal + copiar válidas
            // Solo los warps del tile parcial usan sTmp; el epílogo no tiene
            // __syncthreads pendientes -> sin race entre warps.
            wmma::store_matrix_sync(sTmp[warp_id], acc[tmn], 16, wmma::mem_row_major);
            for (int r = 0; r < m_ok; r++)
                for (int c = 0; c < n_ok; c++)
                    Y[(mm + r) * N + n_base + c] = sTmp[warp_id][r * 16 + c];
        }
    }
}

torch::Tensor fused_forward(torch::Tensor X, torch::Tensor dmin, torch::Tensor dmax,
                            torch::Tensor config, torch::Tensor coeff,
                            torch::Tensor hi, torch::Tensor lo,
                            torch::Tensor basis, torch::Tensor scales,
                            int64_t n_cb) {
    int M = X.size(0), K = X.size(1);
    int N = dmin.size(0) / n_cb;
    auto Xc = X.contiguous();
    auto Y = torch::empty({M, N}, X.options().dtype(torch::kFloat32));
    int cols_per_block = SWARPS * 16;
    int rows_per_block = TM * 16;
    dim3 grid((N + cols_per_block - 1) / cols_per_block, (M + rows_per_block - 1) / rows_per_block);
    sx8_wmma_kernel<<<grid, 256>>>(
        (const __half*)Xc.data_ptr(),
        (const __half*)dmin.data_ptr(), (const __half*)dmax.data_ptr(),
        (const unsigned char*)config.data_ptr(), (const unsigned char*)coeff.data_ptr(),
        (const unsigned char*)hi.data_ptr(), (const unsigned char*)lo.data_ptr(),
        (const float*)basis.data_ptr(), (const float*)scales.data_ptr(),
        (float*)Y.data_ptr(), M, K, N, (int)n_cb);
    return Y;
}