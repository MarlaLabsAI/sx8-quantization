// sx8_decode1_v3.cu — decode M=1 (GEMV) para layout v4.4 — KERNEL v2 COALESCENTE
//
// Layout v4.4 (kb-major, 30 B/bloque EXACTO):
//   hdr (n_blk, HDR_BYTES):  dmin(fp16) dmax(fp16) config(u8) coeff(u8)
//                            HDR_BYTES=6 (exacto, 4.373 GB) | 8 (pad 2, alineado 8 B)
//   levels (n_blk, 24): por sub-bloque sb: hi[4] (bytes sb*4) + lo[2] (bytes 16+sb*2)
//
// Reformulación PCA (validate_pca_reform.py):
//   Z0[kb], Z1[kb] precomputados por capa (kernel compute_z44):
//     Z0[kb] = s0[kb] * sum_t X[kb*32+t] * b0[kb][t]
//     Z1[kb] = s1[kb] * sum_t X[kb*32+t] * b1[kb][t]
//   => Y[n] = sum_kb [ core(kb,n) + c0(kb,n)*Z0[kb] + c1(kb,n)*Z1[kb] ]
//   el PCA pasa de 2 FMA + 2 loads fp32 POR PESO a 2 FMA POR BLOQUE.
//
// DISEÑO v2 (coalescente — FASE A3.3):
//   En kb-major (bid = kb*N + n), los bloques de columnas CONSECUTIVAS de la
//   misma kb son CONTIGUOS en DRAM. Por eso el warp mapea:
//     col_local = t >> 2   (0..7)  -> 8 columnas consecutivas n = n0 + col_local
//     sb        = t & 3    (0..3)  -> 4 sub-bloques de la columna
//   y el bucle recorre kb fijo por iteración. Cada iteración el warp lee:
//     levels: 8 bloques x 24 B = 192 B CONTIGUOS (coalescencia perfecta)
//     hdr:    8 bloques x HDR_BYTES B contiguos (broadcast por columna)
//     X:      X[kb*32 .. kb*32+31] = 64 B, COMPARTIDO por las 8 columnas (L1)
//   Reducción final: shfl_xor entre los 4 sub-bloques de cada columna.
//
// Correcto vs kernel numba v43: test de igualdad logits (maxdiff < 1e-3).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float __half_raw_to_float(unsigned short s) {
    return __half2float(__ushort_as_half(s));
}

// ---------------------------------------------------------------------------
// compute_z44: precomputa Z0/Z1 por capa (1 bloque = 1 warp, grid = n_cb)
//   Z0[kb] = s0[kb] * sum_t X[kb*32+t] * b0[kb][t]
//   Z1[kb] = s1[kb] * sum_t X[kb*32+t] * b1[kb][t]
// bases (n_cb, 64) fp32: b0 = bases[kb, 0:32], b1 = bases[kb, 32:64]
// scales (n_cb, 2) fp32
// ---------------------------------------------------------------------------
__global__ void compute_z44_kernel(
        const __half* __restrict__ X,     // (K,)
        const float* __restrict__ basis,  // (n_cb, 64)
        const float* __restrict__ scales, // (n_cb, 2)
        float* __restrict__ Z,            // (n_cb, 2) -> Z0[kb], Z1[kb]
        int n_cb) {
    int kb = blockIdx.x;
    if (kb >= n_cb) return;
    int t = threadIdx.x;          // 32 threads
    float xv = __half2float(X[kb * 32 + t]);
    float acc0 = xv * basis[kb * 64 + t];
    float acc1 = xv * basis[kb * 64 + 32 + t];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_xor_sync(0xFFFFFFFF, acc0, off);
        acc1 += __shfl_xor_sync(0xFFFFFFFF, acc1, off);
    }
    if (t == 0) {
        Z[kb * 2 + 0] = scales[kb * 2 + 0] * acc0;
        Z[kb * 2 + 1] = scales[kb * 2 + 1] * acc1;
    }
}

// ---------------------------------------------------------------------------
// decode1_v44_kernel (v3: v2 COALESCENTE + SPLIT-K)
//   thread: col_local = t>>2, sb = t&3 ; columna n = n0 + col_local
//   grid: (N/64) x split_k bloques, block = 256 threads (8 warps -> 64 cols)
//   Cada bloque (split_id = blockIdx.y) procesa un rango de kb; la columna
//   completa se suma con atomicAdd (Y pre-inicializado a 0).
//   split_k=1 -> sin atomics (escritura directa).
// ---------------------------------------------------------------------------
template <int HDR_BYTES>
__global__ void decode1_v44_kernel(
        const __half* __restrict__ X,      // (K,)
        const unsigned char* __restrict__ hdr,    // (n_blk, HDR_BYTES) kb-major
        const unsigned char* __restrict__ levels, // (n_blk, 24) kb-major
        const float* __restrict__ Z,        // (n_cb, 2)
        float* __restrict__ Y,              // (N,)
        int K, int N, int n_cb, int split_k) {
    int t = threadIdx.x & 31;      // lane
    int wid = threadIdx.x >> 5;    // warp id (0..7)
    int col_local = t >> 2;        // 0..7
    int sb = t & 3;                // sub-bloque 0..3
    int n = blockIdx.x * 64 + wid * 8 + col_local;  // columna
    if (n >= N) return;

    int split_id = blockIdx.y;
    int kb_per = (n_cb + split_k - 1) / split_k;
    int kb_start = split_id * kb_per;
    int kb_end = min(kb_start + kb_per, n_cb);

    float acc = 0.0f;

    #pragma unroll 4
    for (int kb = kb_start; kb < kb_end; kb++) {
        int bid = kb * N + n;      // kb-major; columnas consecutivas -> contiguos

        // ---- hdr (HDR_BYTES): dmin fp16, dmax fp16, config, coeff ----
        const unsigned char* h = hdr + bid * HDR_BYTES;
        unsigned short h0, h1;
        memcpy(&h0, h + 0, 2);
        memcpy(&h1, h + 2, 2);
        unsigned char cfg = h[4];
        unsigned char coeff = h[5];
        float lo_f = __half_raw_to_float(h0);
        float hi_f = __half_raw_to_float(h1);

        // ---- rlo/step en vuelo para MI sub-bloque ----
        float q = (hi_f - lo_f) * 0.25f;
        int s = (cfg >> (sb * 2)) & 3;
        float rlo = lo_f + q * (3 * (s == 2) + (s == 3));
        float rhi = hi_f - q * (3 * (s == 1) + (s == 3));
        float step = (rhi - rlo) * 0.015873f;
        if (step < 1e-10f) step = 1e-10f;

        // ---- coeff PCA (signed 4-bit x2) ----
        int c0 = coeff & 0xF;  if (c0 >= 8) c0 -= 16;
        int c1 = (coeff >> 4) & 0xF;  if (c1 >= 8) c1 -= 16;

        // ---- levels de mi sub-bloque: hi[4] (sb*4) + lo[2] (16+sb*2) ----
        const unsigned char* lv = levels + bid * 24;
        unsigned int hi4;
        unsigned short lo2;
        memcpy(&hi4, lv + sb * 4, 4);          // alineado a 4
        memcpy(&lo2, lv + 16 + sb * 2, 2);     // alineado a 2

        // X: 8 halfs contiguos de mi sub-bloque (carga vectorizada half2 x4);
        //    las 8 columnas comparten el mismo X[kb*32..+31] (broadcast L1)
        const __half2* X2 = (const __half2*)(X + kb * 32 + sb * 8);
        __half2 xa = X2[0];
        __half2 xb = X2[1];
        __half2 xc = X2[2];
        __half2 xd = X2[3];

        // 8 pesos: nibble hi en bits i*4 ; quad lo en bits (i>>2)*8 + (3-(i&3))*2
        float xv[8] = {
            __low2float(xa), __high2float(xa),
            __low2float(xb), __high2float(xb),
            __low2float(xc), __high2float(xc),
            __low2float(xd), __high2float(xd),
        };
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int nib = (int)((hi4 >> (i * 4)) & 0xF);
            int qua = (int)((lo2 >> ((i >> 2) * 8 + (3 - (i & 3)) * 2)) & 0x3);
            int lv_i = (nib << 2) | qua;
            float w = rlo + step * (float)lv_i;
            acc += xv[i] * w;
        }

        // ---- PCA por bloque: 1 vez (thread del sub-bloque 0) ----
        if (sb == 0)
            acc += (float)c0 * Z[kb * 2 + 0] + (float)c1 * Z[kb * 2 + 1];
    }

    // ---- reducir los 4 sub-bloques de la misma columna -> lane col_local*4 ----
    #pragma unroll
    for (int off = 2; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    // Determinista SIN atomicAdd: cada split escribe su fila en Y (split_k, N);
    // una segunda pasada (reduce_split_kernel) suma las filas en orden fijo.
    if (sb == 0)
        Y[(long long)split_id * N + n] = acc;
}

// reduce_split_kernel: Y (split_k, N) -> Yout (N,) — suma en orden fijo 0..split_k-1
__global__ void reduce_split_kernel(
        const float* __restrict__ Y,   // (split_k, N)
        float* __restrict__ Yout,      // (N,)
        int N, int split_k) {
    int n = blockIdx.x * 256 + threadIdx.x;
    if (n >= N) return;
    float s = 0.0f;
    for (int i = 0; i < split_k; i++)
        s += Y[(long long)i * N + n];
    Yout[n] = s;
}

// ---------------------------------------------------------------------------
// wrappers (variantes HDR_BYTES 6 y 8)
// ---------------------------------------------------------------------------
torch::Tensor compute_z44(torch::Tensor X, torch::Tensor basis, torch::Tensor scales,
                          int64_t n_cb) {
    int K = X.size(0);
    auto Xc = X.contiguous();
    auto Z = torch::empty({n_cb, 2}, X.options().dtype(torch::kFloat32));
    compute_z44_kernel<<<n_cb, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)Xc.data_ptr(),
        (const float*)basis.data_ptr(),
        (const float*)scales.data_ptr(),
        (float*)Z.data_ptr(), (int)n_cb);
    return Z;
}

torch::Tensor decode1_v44(torch::Tensor X, torch::Tensor hdr, torch::Tensor levels,
                          torch::Tensor Z, int64_t n_cb, int64_t hdr_bytes,
                          int64_t split_k) {
    int K = X.size(0);
    int N = hdr.size(0) / n_cb;   // hdr en kb-major: n_blk = n_cb * N
    auto Xc = X.contiguous();
    auto Y = torch::empty({(int)split_k, N}, X.options().dtype(torch::kFloat32));
    auto Yout = torch::empty({N}, X.options().dtype(torch::kFloat32));
    int blocks = (N + 63) / 64;
    dim3 grid(blocks, (int)split_k);
    if (hdr_bytes == 8) {
        decode1_v44_kernel<8><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            (const __half*)Xc.data_ptr(),
            (const unsigned char*)hdr.data_ptr(),
            (const unsigned char*)levels.data_ptr(),
            (const float*)Z.data_ptr(),
            (float*)Y.data_ptr(), K, N, (int)n_cb, (int)split_k);
    } else {
        decode1_v44_kernel<6><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            (const __half*)Xc.data_ptr(),
            (const unsigned char*)hdr.data_ptr(),
            (const unsigned char*)levels.data_ptr(),
            (const float*)Z.data_ptr(),
            (float*)Y.data_ptr(), K, N, (int)n_cb, (int)split_k);
    }
    reduce_split_kernel<<<(N + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const float*)Y.data_ptr(), (float*)Yout.data_ptr(), N, (int)split_k);
    return Yout;
}
