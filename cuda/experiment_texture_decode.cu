// experiment_texture_decode.cu — ¿puede el HARDWARE DE TEXTURA acelerar el decode SX8?
//
// Hipótesis del usuario: los samplers de textura hacen interpolación lineal en
// hardware (TMUs). El decode V6 es w = rlo + step*lv == lerp(rlo, rhi, lv/63).
// Si (rlo,rhi) de cada sub-bloque se guardan como 2 texels contiguos en una
// textura 1D half con cudaFilterModeLinear, muestrear en
//   u = (bid*4 + sb)*2 + lv/63
// hace el lerp EN EL HARDWARE DE TEXTURA (0 ALU para el decode V6).
//
// E1: tex1Dfetch sobre levels (uchar, sin filtro) — ¿el texture path supera
//     al load global en el patrón v4.4? (mismo kernel que v3, niveles vía tex)
// E2: sampler con cudaFilterModeLinear sobre (rlo,rhi) — lerp hw + fetch del
//     nivel por tex1Dfetch. Mide BW y EXACTITUD vs decode exacto.
//
// Baseline: kernel v3 (load global) ya mide ~317 GB/s (80 tok/s puro).
// Veredicto: E1/E2 >= baseline -> TMUs ayudan; si no, se documenta.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float __half_raw_to_float(unsigned short s) {
    return __half2float(__ushort_as_half(s));
}

// ---------------------------------------------------------------------------
// E1: niveles vía textura uchar (cudaTextureObject_t, point) — resto igual que v3
// ---------------------------------------------------------------------------
__global__ void tex_fetch_levels_kernel(
        const __half* __restrict__ X,
        cudaTextureObject_t tex_hi,   // levels (n_blk,24) uchar
        cudaTextureObject_t tex_lo,
        const unsigned char* __restrict__ hdr,
        const float* __restrict__ Z,
        float* __restrict__ Y,
        int K, int N, int n_cb) {
    int t = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int col_local = t >> 2;
    int sb = t & 3;
    int n = blockIdx.x * 64 + wid * 8 + col_local;
    if (n >= N) return;
    float acc = 0.0f;
    #pragma unroll 4
    for (int kb = 0; kb < n_cb; kb++) {
        int bid = kb * N + n;
        const unsigned char* h = hdr + bid * 6;
        unsigned short h0, h1;
        memcpy(&h0, h + 0, 2); memcpy(&h1, h + 2, 2);
        unsigned char cfg = h[4];
        unsigned char coeff = h[5];
        float lo_f = __half_raw_to_float(h0);
        float hi_f = __half_raw_to_float(h1);
        float q = (hi_f - lo_f) * 0.25f;
        int s = (cfg >> (sb * 2)) & 3;
        float rlo = lo_f + q * (3 * (s == 2) + (s == 3));
        float rhi = hi_f - q * (3 * (s == 1) + (s == 3));
        float step = (rhi - rlo) * 0.015873f;
        if (step < 1e-10f) step = 0.015873f;
        int c0 = coeff & 0xF;  if (c0 >= 8) c0 -= 16;
        int c1 = (coeff >> 4) & 0xF;  if (c1 >= 8) c1 -= 16;
        // niveles vía TEXTURA (en vez de load global)
        int base = bid * 24 + sb * 4;
        unsigned int hi4 = 0, lo2 = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) hi4 |= (unsigned int)tex1Dfetch<unsigned char>(tex_hi, base + j) << (j * 8);
        #pragma unroll
        for (int j = 0; j < 2; j++) lo2 |= (unsigned int)tex1Dfetch<unsigned char>(tex_lo, 16 + sb * 2 + j) << (j * 8);
        const __half2* X2 = (const __half2*)(X + kb * 32 + sb * 8);
        __half2 xa = X2[0], xb = X2[1], xc = X2[2], xd = X2[3];
        float xv[8] = { __low2float(xa), __high2float(xa), __low2float(xb), __high2float(xb),
                        __low2float(xc), __high2float(xc), __low2float(xd), __high2float(xd) };
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int nib = (int)((hi4 >> (i * 4)) & 0xF);
            int qua = (int)((lo2 >> ((i >> 2) * 8 + (3 - (i & 3)) * 2)) & 0x3);
            int lv = (nib << 2) | qua;
            float w = rlo + step * (float)lv;
            acc += xv[i] * w;
        }
        if (sb == 0)
            acc += (float)c0 * Z[kb * 2 + 0] + (float)c1 * Z[kb * 2 + 1];
    }
    #pragma unroll
    for (int off = 2; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    if (sb == 0) Y[n] = acc;
}

// ---------------------------------------------------------------------------
// E2: sampler LINEAR — w = lerp_hw(rlo, rhi, lv/63) + PCA
//     textura rr: half x (n_blk*4*2); texels [bid*8 + sb*2] = rlo, [+1] = rhi
//     muestreo en u = (bid*4 + sb)*2 + lv/63 (filtro lineal -> lerp hw)
// ---------------------------------------------------------------------------
__global__ void sampler_lerp_kernel(
        const __half* __restrict__ X,
        cudaTextureObject_t tex_rr,   // half, filtro LINEAR
        cudaTextureObject_t tex_lv,   // uchar point (levels)
        const unsigned char* __restrict__ hdr,
        const float* __restrict__ Z,
        float* __restrict__ Y,
        int K, int N, int n_cb) {
    int t = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int col_local = t >> 2;
    int sb = t & 3;
    int n = blockIdx.x * 64 + wid * 8 + col_local;
    if (n >= N) return;
    float acc = 0.0f;
    #pragma unroll 4
    for (int kb = 0; kb < n_cb; kb++) {
        int bid = kb * N + n;
        const unsigned char* h = hdr + bid * 6;
        unsigned char coeff = h[5];
        int c0 = coeff & 0xF;  if (c0 >= 8) c0 -= 16;
        int c1 = (coeff >> 4) & 0xF;  if (c1 >= 8) c1 -= 16;
        // niveles vía textura
        int base = bid * 24 + sb * 4;
        unsigned int hi4 = 0, lo2 = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) hi4 |= (unsigned int)tex1Dfetch<unsigned char>(tex_lv, base + j) << (j * 8);
        #pragma unroll
        for (int j = 0; j < 2; j++) lo2 |= (unsigned int)tex1Dfetch<unsigned char>(tex_lv, 16 + sb * 2 + j) << (j * 8);
        const __half2* X2 = (const __half2*)(X + kb * 32 + sb * 8);
        __half2 xa = X2[0], xb = X2[1], xc = X2[2], xd = X2[3];
        float xv[8] = { __low2float(xa), __high2float(xa), __low2float(xb), __high2float(xb),
                        __low2float(xc), __high2float(xc), __low2float(xd), __high2float(xd) };
        float u_base = (float)(bid * 4 + sb) * 2.0f;   // texel (rlo,rhi) del sub-bloque
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int nib = (int)((hi4 >> (i * 4)) & 0xF);
            int qua = (int)((lo2 >> ((i >> 2) * 8 + (3 - (i & 3)) * 2)) & 0x3);
            int lv = (nib << 2) | qua;
            // LERP EN HARDWARE: tex1D con filtro lineal en u -> w = rlo + (rhi-rlo)*fract
            // texel i centrado en u=i+0.5: rlo en u_base+0.5, rhi en u_base+1.5
            float u = u_base + 0.5f + (float)lv * 0.015873f;  // lerp(rlo,rhi,lv/63)
            float w = tex1D<float>(tex_rr, u);
            acc += xv[i] * w;
        }
        if (sb == 0)
            acc += (float)c0 * Z[kb * 2 + 0] + (float)c1 * Z[kb * 2 + 1];
    }
    #pragma unroll
    for (int off = 2; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    if (sb == 0) Y[n] = acc;
}

// ---------------------------------------------------------------------------
// wrappers con cudaTextureObject_t
// ---------------------------------------------------------------------------
static cudaTextureObject_t make_tex_u8(const unsigned char* p, long n) {
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = (void*)p;
    rd.res.linear.desc = cudaCreateChannelDesc<unsigned char>();
    rd.res.linear.sizeInBytes = n;
    cudaTextureDesc td = {};
    td.addressMode[0] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t to;
    cudaCreateTextureObject(&to, &rd, &td, nullptr);
    return to;
}

static cudaTextureObject_t make_tex_float_linear(const float* p, long n) {
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeLinear;
    rd.res.linear.devPtr = (void*)p;
    rd.res.linear.desc = cudaCreateChannelDesc<float>();
    rd.res.linear.sizeInBytes = n * 4;
    cudaTextureDesc td = {};
    td.addressMode[0] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;   // LERP HARDWARE
    td.readMode = cudaReadModeElementType;
    cudaTextureObject_t to;
    cudaCreateTextureObject(&to, &rd, &td, nullptr);
    return to;
}

// texturas pre-creadas por llamada (se destruyen al final del wrapper)
torch::Tensor tex_fetch_levels(torch::Tensor X, torch::Tensor levels,
                               torch::Tensor hdr, torch::Tensor Z,
                               int64_t n_cb) {
    int N = hdr.size(0) / n_cb;
    int K = X.size(0);
    auto Xc = X.contiguous();
    auto Y = torch::zeros({N}, X.options().dtype(torch::kFloat32));
    auto t_hi = make_tex_u8((const unsigned char*)levels.data_ptr(), levels.numel());
    auto t_lo = make_tex_u8((const unsigned char*)levels.data_ptr(), levels.numel());
    tex_fetch_levels_kernel<<<(N + 63) / 64, 256>>>(
        (const __half*)Xc.data_ptr(), t_hi, t_lo,
        (const unsigned char*)hdr.data_ptr(), (const float*)Z.data_ptr(),
        (float*)Y.data_ptr(), K, N, (int)n_cb);
    cudaDestroyTextureObject(t_hi);
    cudaDestroyTextureObject(t_lo);
    return Y;
}

torch::Tensor sampler_lerp(torch::Tensor X, torch::Tensor levels,
                           torch::Tensor rr, torch::Tensor hdr, torch::Tensor Z,
                           int64_t n_cb) {
    int N = hdr.size(0) / n_cb;
    int K = X.size(0);
    auto Xc = X.contiguous();
    auto Y = torch::zeros({N}, X.options().dtype(torch::kFloat32));
    auto t_rr = make_tex_float_linear((const float*)rr.data_ptr(), rr.numel());
    auto t_lv = make_tex_u8((const unsigned char*)levels.data_ptr(), levels.numel());
    sampler_lerp_kernel<<<(N + 63) / 64, 256>>>(
        (const __half*)Xc.data_ptr(), t_rr, t_lv,
        (const unsigned char*)hdr.data_ptr(), (const float*)Z.data_ptr(),
        (float*)Y.data_ptr(), K, N, (int)n_cb);
    cudaDestroyTextureObject(t_rr);
    cudaDestroyTextureObject(t_lv);
    return Y;
}
