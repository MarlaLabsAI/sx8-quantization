#include <torch/extension.h>
#include <cuda_fp16.h>

__global__ void sx8_decode1_fast_kernel(
        const __half* __restrict__ X,          // (K,)
        const __half* __restrict__ rlo,        // (n_blk, 4) fp16 — precomputado
        const __half* __restrict__ step,       // (n_blk, 4) fp16 — precomputado
        const unsigned char* __restrict__ coeff,
        const unsigned char* __restrict__ hi_arr,
        const unsigned char* __restrict__ lo_arr,
        const float* __restrict__ basis,
        const float* __restrict__ scales,
        float* __restrict__ Y,                 // (N,)
        int K, int N, int n_cb) {
    int ltid = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int n = blockIdx.x * 8 + warp_id;
    if (n >= N) return;

    float acc = 0.0f;
    for (int kb = 0; kb < n_cb; kb += 2) {
        #pragma unroll
        for (int hb = 0; hb < 2; hb++) {
            int kb2 = kb + hb;
            if (kb2 >= n_cb) break;
            int bid2 = kb2 * N + n;
            int co = coeff[bid2];
            int c0r = co & 0xF; if (c0r >= 8) c0r -= 16;
            int c1r = (co >> 4) & 0xF; if (c1r >= 8) c1r -= 16;
            float s0 = scales[kb2 * 2];
            float s1v = scales[kb2 * 2 + 1];
            float p0 = c0r * s0;
            float p1 = c1r * s1v;
            int t = ltid;
            int k = kb2 * 32 + t;
            if (k >= K) continue;
            int hi = (hi_arr[bid2 * 16 + (t >> 1)] >> ((t & 1) * 4)) & 0xF;
            int lo = (lo_arr[bid2 * 8 + (t >> 2)] >> ((3 - (t & 3)) * 2)) & 0x3;
            int lv = (hi << 2) | lo;
            int sb = t >> 3;
            float r = __half2float(rlo[bid2 * 4 + sb]);
            float st = __half2float(step[bid2 * 4 + sb]);
            float w = r + st * (float)lv
                    + p0 * basis[kb2 * 64 + t] + p1 * basis[kb2 * 64 + 32 + t];
            acc += __half2float(X[k]) * w;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    if (ltid == 0)
        Y[n] = acc;
}

torch::Tensor decode1_forward(torch::Tensor X, torch::Tensor rlo, torch::Tensor step,
                              torch::Tensor coeff, torch::Tensor hi, torch::Tensor lo,
                              torch::Tensor basis, torch::Tensor scales,
                              int64_t n_cb) {
    int K = X.size(0);
    int N = coeff.size(0) / n_cb;   // coeff en kb-major: n_blk = n_cb * N
    auto Xc = X.contiguous();
    auto Y = torch::empty({N}, X.options().dtype(torch::kFloat32));
    int blocks = (N + 7) / 8;
    sx8_decode1_fast_kernel<<<blocks, 256>>>(
        (const __half*)Xc.data_ptr(),
        (const __half*)rlo.data_ptr(), (const __half*)step.data_ptr(),
        (const unsigned char*)coeff.data_ptr(),
        (const unsigned char*)hi.data_ptr(), (const unsigned char*)lo.data_ptr(),
        (const float*)basis.data_ptr(), (const float*)scales.data_ptr(),
        (float*)Y.data_ptr(), K, N, (int)n_cb);
    return Y;
}