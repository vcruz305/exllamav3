// Inkling2Loop single-token (T == 1, bsz == 1) decode kernels. Each one replaces a chain of 10-30 small aten ops of
// the eager path in inkling2.py with one launch. fp32 math in the same order of operations as the eager code (the
// reductions differ only in summation order). Shapes are the model's: H = hidden, D = head dim (128), R = d_rel.
#include <cuda_fp16.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include "inkling2.cuh"

#define INK_MAXS 16     // attention-residual sources (block states + partial)
#define INK_MAXQ 4      // attention-residual queries
#define INK_MAXK 8      // short-conv taps + 1
#define INK_MAXG 64     // d_rel (gate rank) upper bound

__device__ __forceinline__ float ink_warp_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float ink_warp_max(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide sum / max (blockDim.x multiple of 32, <= 1024). sm: >= 32 floats. Every thread gets the result.
__device__ __forceinline__ float ink_block_sum(float v, float* sm)
{
    v = ink_warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) sm[w] = v;
    __syncthreads();
    float s = ink_warp_sum(l < nw ? sm[l] : 0.f);
    __syncthreads();
    return s;
}

__device__ __forceinline__ float ink_block_max(float v, float* sm)
{
    v = ink_warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) sm[w] = v;
    __syncthreads();
    float s = ink_warp_max(l < nw ? sm[l] : -INFINITY);
    __syncthreads();
    return s;
}

__device__ __forceinline__ void ink_load4h(const __half* p, float* f)
{
    uint2 u = *reinterpret_cast<const uint2*>(p);
    float2 a = __half22float2(*reinterpret_cast<__half2*>(&u.x));
    float2 b = __half22float2(*reinterpret_cast<__half2*>(&u.y));
    f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

__device__ __forceinline__ float ink_in(const float* x, long long i) { return x[i]; }
__device__ __forceinline__ float ink_in(const __half* x, long long i) { return __half2float(x[i]); }

// ---------------------------------------------------------------------------------------------------------------------
// Attention residual for one token: out[q, :] = sum_j softmax_j(<rms(x_j), qT[:, q]>) * rms(x_j), j over the sources
// (block states, then the partial state). qT: (H, NQ) pre-scaled queries, row-major.
// ---------------------------------------------------------------------------------------------------------------------
struct InkSrcs { const float* p[INK_MAXS]; };

__global__ void __launch_bounds__(256) ink_attn_res_kernel(
    InkSrcs srcs, int S, int NQ, int H, const float* __restrict__ qT, float eps, float* __restrict__ out)
{
    constexpr int NACC = INK_MAXS * INK_MAXQ + INK_MAXS;     // dot products [j * NQ + q], sums of squares [SQ + j]
    __shared__ float wsum[8][NACC];
    __shared__ float red[NACC];
    __shared__ float sc[INK_MAXS * INK_MAXQ];
    __shared__ float rs[INK_MAXS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int SQ = S * NQ, NA = SQ + S;
    const float* p[INK_MAXS];
    #pragma unroll
    for (int j = 0; j < INK_MAXS; ++j) p[j] = j < S ? srcs.p[j] : srcs.p[0];

    float acc[NACC];
    #pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
    for (int d = tid; d < H; d += blockDim.x) {
        float qv[INK_MAXQ];
        #pragma unroll
        for (int q = 0; q < INK_MAXQ; ++q) qv[q] = q < NQ ? qT[d * NQ + q] : 0.f;
        #pragma unroll
        for (int j = 0; j < INK_MAXS; ++j) {
            if (j < S) {
                float x = p[j][d];
                acc[SQ + j] += x * x;
                #pragma unroll
                for (int q = 0; q < INK_MAXQ; ++q) if (q < NQ) acc[j * NQ + q] += x * qv[q];
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < NACC; ++i) {
        float v = ink_warp_sum(acc[i]);
        if (lane == 0) wsum[warp][i] = v;
    }
    __syncthreads();
    if (tid < NA) {
        float s = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += wsum[w][tid];
        red[tid] = s;
    }
    __syncthreads();
    if (tid == 0) {
        for (int j = 0; j < S; ++j) rs[j] = rsqrtf(red[SQ + j] / (float)H + eps);
        for (int q = 0; q < NQ; ++q) {
            float m = -INFINITY;
            for (int j = 0; j < S; ++j) {
                float s = rs[j] * red[j * NQ + q];
                sc[j * NQ + q] = s;
                m = fmaxf(m, s);
            }
            float den = 0.f;
            for (int j = 0; j < S; ++j) {
                float e = expf(sc[j * NQ + q] - m);
                sc[j * NQ + q] = e;
                den += e;
            }
            for (int j = 0; j < S; ++j) sc[j * NQ + q] /= den;
        }
    }
    __syncthreads();
    for (int d = tid; d < H; d += blockDim.x) {
        for (int q = 0; q < NQ; ++q) {
            float o = 0.f;
            for (int j = 0; j < S; ++j) o += sc[j * NQ + q] * rs[j] * p[j][d];
            out[q * H + d] = o;
        }
    }
}

// Two launches, spread over H / 512 blocks of 128 threads (one thread per 4 channels). Every source's float4 is loaded
// up front (one memory round trip per phase, not one per source). Same math as ink_attn_res_kernel (sums in another
// order). Needs H % 4 == 0 and 16-byte aligned sources; other shapes take the generic kernel.
#define INK_ARBS 128                       // threads per block of the attention-residual kernels
#define INK_ACC (INK_MAXS * INK_MAXQ + INK_MAXS)   // 80: dot products at [j * 4 + q], sums of squares at [64 + j]

// partial[b][i] = sum over the block's channels of the NACC products (block b covers channels b * 512 .. + 512)
__global__ void __launch_bounds__(INK_ARBS) ink_attn_res_part_kernel(
    InkSrcs srcs, int S, int NQ, int H4, const float* __restrict__ qT, float* __restrict__ partial, long long rst,
    int nblk)
{
    __shared__ float wsum[INK_ARBS / 32][INK_ACC];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int d4 = blockIdx.x * INK_ARBS + tid;
    const bool ok = d4 < H4;
    const int dd = ok ? d4 : 0;
    const float valid = ok ? 1.f : 0.f;

    float4 xv[INK_MAXS];
    #pragma unroll
    for (int j = 0; j < INK_MAXS; ++j)
        xv[j] = j < S ? reinterpret_cast<const float4*>(srcs.p[j] + blockIdx.y * rst)[dd] : make_float4(0.f, 0.f, 0.f, 0.f);
    float qv[INK_MAXQ][4];
    #pragma unroll
    for (int q = 0; q < INK_MAXQ; ++q)
        #pragma unroll
        for (int c = 0; c < 4; ++c) qv[q][c] = (q < NQ) ? qT[(4 * dd + c) * NQ + q] * valid : 0.f;

    float acc[INK_ACC];
    #pragma unroll
    for (int i = 0; i < INK_ACC; ++i) acc[i] = 0.f;
    #pragma unroll
    for (int j = 0; j < INK_MAXS; ++j) {
        if (j < S) {
            const float4 x = xv[j];
            const float4 xm = make_float4(x.x * valid, x.y * valid, x.z * valid, x.w * valid);
            acc[INK_MAXS * INK_MAXQ + j] = xm.x * xm.x + xm.y * xm.y + xm.z * xm.z + xm.w * xm.w;
            #pragma unroll
            for (int q = 0; q < INK_MAXQ; ++q)
                if (q < NQ) acc[j * INK_MAXQ + q] = xm.x * qv[q][0] + xm.y * qv[q][1] + xm.z * qv[q][2] + xm.w * qv[q][3];
        }
    }
    // step-major butterfly: the reductions are independent, so each round issues them back to back
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        #pragma unroll
        for (int i = 0; i < INK_ACC; ++i) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], o);
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < INK_ACC; ++i) wsum[warp][i] = acc[i];
    }
    __syncthreads();
    if (tid < INK_ACC) {
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < INK_ARBS / 32; ++w) s += wsum[w][tid];
        partial[((size_t)blockIdx.y * nblk + blockIdx.x) * INK_ACC + tid] = s;
    }
}

// Finish: every block reduces the partials, computes the softmax weights (warp q = query q, lane j = source j) and
// writes its 128 channel groups: out[q][d] = sum_j w_jq rs_j x_j[d].
__global__ void __launch_bounds__(INK_ARBS) ink_attn_res_fin_kernel(
    InkSrcs srcs, int S, int NQ, int H, int H4, int nblk, const float* __restrict__ partial, float eps,
    float* __restrict__ out, long long rst)
{
    __shared__ float red[INK_ACC];
    __shared__ float rs[INK_MAXS];
    __shared__ float coef[INK_MAXQ][INK_MAXS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int d4 = blockIdx.x * INK_ARBS + tid;
    float4 xv[INK_MAXS];
    #pragma unroll
    for (int j = 0; j < INK_MAXS; ++j)
        xv[j] = (j < S && d4 < H4) ? reinterpret_cast<const float4*>(srcs.p[j] + blockIdx.y * rst)[d4] : make_float4(0.f, 0.f, 0.f, 0.f);
    if (tid < INK_ACC) {
        float s = 0.f;
        for (int b = 0; b < nblk; ++b) s += partial[((size_t)blockIdx.y * nblk + b) * INK_ACC + tid];
        red[tid] = s;
    }
    __syncthreads();
    if (tid < S) rs[tid] = rsqrtf(red[INK_MAXS * INK_MAXQ + tid] / (float)H + eps);
    __syncthreads();
    if (warp < NQ) {
        // softmax over the sources of query `warp`: logits rs_j * dot_jq
        const bool in = lane < S;
        const float s = in ? rs[lane] * red[lane * INK_MAXQ + warp] : -INFINITY;
        float m = s;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        const float e = in ? expf(s - m) : 0.f;
        float den = e;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
        if (in) coef[warp][lane] = e / den * rs[lane];
    }
    __syncthreads();
    if (d4 < H4) {
        #pragma unroll
        for (int q = 0; q < INK_MAXQ; ++q) {
            if (q < NQ) {
                float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
                #pragma unroll
                for (int j = 0; j < INK_MAXS; ++j) {
                    if (j < S) {
                        const float cf = coef[q][j];
                        o.x += cf * xv[j].x; o.y += cf * xv[j].y; o.z += cf * xv[j].z; o.w += cf * xv[j].w;
                    }
                }
                reinterpret_cast<float4*>(out + ((size_t)blockIdx.y * NQ + q) * H)[d4] = o;
            }
        }
    }
}

at::Tensor ink_attn_res(std::vector<at::Tensor> srcs, at::Tensor qT, double eps)
{
    const int S = (int)srcs.size();
    TORCH_CHECK(S >= 1 && S <= INK_MAXS, "ink_attn_res: 1..16 sources");
    TORCH_CHECK(qT.dim() == 2 && qT.size(1) >= 1 && qT.size(1) <= INK_MAXQ, "ink_attn_res: qT (H, nq<=4)");
    const int H = (int)qT.size(0), NQ = (int)qT.size(1);
    const at::cuda::OptionalCUDAGuard guard(qT.device());
    InkSrcs a;
    bool aligned = true;
    for (int j = 0; j < S; ++j) {
        TORCH_CHECK(srcs[j].scalar_type() == at::kFloat && srcs[j].is_contiguous() && srcs[j].numel() == H,
                    "ink_attn_res: sources must be contiguous fp32 vectors of H");
        a.p[j] = srcs[j].data_ptr<float>();
        aligned = aligned && (((uintptr_t)a.p[j]) % 16 == 0);
    }
    TORCH_CHECK(qT.scalar_type() == at::kFloat && qT.is_contiguous(), "ink_attn_res: qT must be contiguous fp32");
    auto out = at::empty({NQ, H}, qT.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (H % 4 == 0 && aligned) {
        const int H4 = H / 4;
        const int nblk = (H4 + INK_ARBS - 1) / INK_ARBS;
        auto partial = at::empty({nblk, INK_ACC}, qT.options());
        ink_attn_res_part_kernel<<<nblk, INK_ARBS, 0, stream>>>(
            a, S, NQ, H4, qT.data_ptr<float>(), partial.data_ptr<float>(), 0LL, nblk);
        ink_attn_res_fin_kernel<<<nblk, INK_ARBS, 0, stream>>>(
            a, S, NQ, H, H4, nblk, partial.data_ptr<float>(), (float)eps, out.data_ptr<float>(), 0LL);
    } else {
        ink_attn_res_kernel<<<1, 256, 0, stream>>>(
            a, S, NQ, H, qT.data_ptr<float>(), (float)eps, out.data_ptr<float>());
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Message gates for G consecutive gates (indices first .. first + G) on rows X[g], M[g] (strides in elements):
//   z = [rms(x), rms(m)] (2H), g = z @ down_g (R), t = tanh(g @ up_g) (2H), out = (1 - t[:H]) * x + t[H:] * m
// gd: (4, 2H, R) row-major, gu: (4, R, 2H) row-major.
// ---------------------------------------------------------------------------------------------------------------------
// Two launches, both spread over the GPU (the old one-block-per-gate kernel was latency-bound: serial strided loads):
//   ink_gate_down_kernel: grid (2H / 128 row chunks, G). Normalizes z = [rms(x), rms(m)] (block reductions over H), then
//                         each block contracts its 128 rows of gd into a partial g (R values) -> part[G][NSPL][R].
//   ink_gate_out_kernel:  grid (H / 64, G). Sums the partials into g (R values), computes t = tanh(g @ gu) for the 64
//                         forget and the 64 gain columns of its h range, and writes out[h] = (1 - t_f) x + t_g m.
#define INK_GRC 128        // rows of the 2H gate-down matrix per partial block
#define INK_GHW 64         // output h columns per block of the second gate kernel

__global__ void __launch_bounds__(256) ink_gate_down_kernel(
    const float* __restrict__ X, long long xs, const float* __restrict__ M, long long ms,
    const float* __restrict__ gd, int first, int H, int R, int NSPL, float eps, float* __restrict__ part,
    int Gm, long long xrs, long long mrs)
{
    __shared__ float red[2][32];
    __shared__ float zs[INK_GRC];
    __shared__ float sw[8][INK_MAXG];
    const int c = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const float* x = X + (g / Gm) * xrs + (g % Gm) * xs;
    const float* m = M + (g / Gm) * mrs + (g % Gm) * ms;
    float sx = 0.f, sm = 0.f;
    if (H <= 9 * 256) {
        // loads first (one memory round trip), then the sums
        float xa[9], ma[9];
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            const int i = tid + t * 256;
            xa[t] = i < H ? x[i] : 0.f;
            ma[t] = i < H ? m[i] : 0.f;
        }
        #pragma unroll
        for (int t = 0; t < 9; ++t) { sx += xa[t] * xa[t]; sm += ma[t] * ma[t]; }
    } else {
        for (int i = tid; i < H; i += blockDim.x) { const float a = x[i], b = m[i]; sx += a * a; sm += b * b; }
    }
    sx = ink_block_sum(sx, red[0]);
    sm = ink_block_sum(sm, red[1]);
    const float rx = rsqrtf(sx / (float)H + eps), rm = rsqrtf(sm / (float)H + eps);
    const int k0 = c * INK_GRC;
    if (tid < INK_GRC) {
        const int k = k0 + tid;
        zs[tid] = k >= 2 * H ? 0.f : (k < H ? x[k] * rx : m[k - H] * rm);
    }
    __syncthreads();

    // thread (r4, kk): float4 of ranks 4 r4 .. 4 r4 + 3 for rows kk, kk + per, ...
    const int R4 = R / 4, per = blockDim.x / R4;
    const int r4 = tid % R4, kk = tid / R4;
    const float* W = gd + (size_t)(first + g % Gm) * 2 * H * R;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int kl = kk; kl < INK_GRC; kl += per) {
        const int k = k0 + kl;
        if (k < 2 * H) {
            const float z = zs[kl];
            const float4 w = reinterpret_cast<const float4*>(W + (size_t)k * R)[r4];
            a.x += z * w.x; a.y += z * w.y; a.z += z * w.z; a.w += z * w.w;
        }
    }
    const int lane = tid & 31, warp = tid >> 5;
    for (int o = R4; o < 32; o <<= 1) {
        a.x += __shfl_xor_sync(0xffffffffu, a.x, o);
        a.y += __shfl_xor_sync(0xffffffffu, a.y, o);
        a.z += __shfl_xor_sync(0xffffffffu, a.z, o);
        a.w += __shfl_xor_sync(0xffffffffu, a.w, o);
    }
    if (lane < R4) {
        sw[warp][4 * lane + 0] = a.x;
        sw[warp][4 * lane + 1] = a.y;
        sw[warp][4 * lane + 2] = a.z;
        sw[warp][4 * lane + 3] = a.w;
    }
    __syncthreads();
    if (tid < R) {
        float s = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) s += sw[w][tid];
        part[((size_t)g * NSPL + c) * R + tid] = s;
    }
}

__global__ void __launch_bounds__(256) ink_gate_out_kernel(
    const float* __restrict__ X, long long xs, const float* __restrict__ M, long long ms,
    const float* __restrict__ gu, const float* __restrict__ part, int first, int H, int R, int NSPL,
    float* __restrict__ out, int Gm, long long xrs, long long mrs)
{
    __shared__ float pz[16][INK_MAXG];
    __shared__ float gv[INK_MAXG];
    __shared__ float sm2[2][2 * INK_GHW];
    __shared__ float tv[2 * INK_GHW];
    const int g = blockIdx.y, a0 = blockIdx.x * INK_GHW, tid = threadIdx.x;

    // g_r = sum over the NSPL partials (16 groups of partials, then the groups)
    for (int idx = tid; idx < 16 * R; idx += blockDim.x) {
        const int r = idx % R, grp = idx / R;
        float s = 0.f;
        #pragma unroll 4
        for (int c = grp; c < NSPL; c += 16) s += part[((size_t)g * NSPL + c) * R + r];
        pz[grp][r] = s;
    }
    __syncthreads();
    if (tid < R) {
        float s = 0.f;
        for (int q = 0; q < 16; ++q) s += pz[q][tid];
        gv[tid] = s;
    }
    __syncthreads();

    // t_n = tanh(sum_r g_r gu[gi, r, n]); thread (j, rh) sums the rows of half rh; j < 64 forget at h = a0 + j,
    // j >= 64 gain at h = a0 + j - 64
    const int j = tid & (2 * INK_GHW - 1), rh = tid / (2 * INK_GHW);
    const int hh = a0 + (j & (INK_GHW - 1));
    const int n = (j < INK_GHW ? 0 : H) + (hh < H ? hh : 0);
    const float* W2 = gu + (size_t)(first + g % Gm) * R * 2 * H;
    const int RH = R / 2;
    float s = 0.f;
    #pragma unroll 8
    for (int r = rh * RH; r < rh * RH + RH; ++r) s += gv[r] * W2[(size_t)r * 2 * H + n];
    sm2[rh][j] = s;
    __syncthreads();
    if (tid < 2 * INK_GHW) tv[tid] = tanhf(sm2[0][tid] + sm2[1][tid]);
    __syncthreads();
    if (tid < INK_GHW) {
        const int h = a0 + tid;
        if (h < H) {
            const float x = X[(g / Gm) * xrs + (g % Gm) * xs + h], m = M[(g / Gm) * mrs + (g % Gm) * ms + h];
            out[g * H + h] = (1.f - tv[tid]) * x + tv[INK_GHW + tid] * m;
        }
    }
}

at::Tensor ink_gates(at::Tensor X, at::Tensor M, at::Tensor gd, at::Tensor gu, int64_t first, int64_t G, double eps)
{
    const int H = (int)X.size(1), R = (int)gd.size(2);
    TORCH_CHECK(X.dim() == 2 && M.dim() == 2 && X.size(0) == G && M.size(0) == G, "ink_gates: X, M (G, H)");
    TORCH_CHECK(X.stride(1) == 1 && M.stride(1) == 1, "ink_gates: rows must be unit-stride");
    TORCH_CHECK(R >= 4 && R <= INK_MAXG && R % 4 == 0 && (R & (R - 1)) == 0, "ink_gates: rank a power of two in [4, 64]");
    TORCH_CHECK(gd.size(1) == 2 * H && gu.size(1) == R && gu.size(2) == 2 * H, "ink_gates: weight shapes");
    TORCH_CHECK(X.scalar_type() == at::kFloat && M.scalar_type() == at::kFloat && gd.is_contiguous() && gu.is_contiguous()
                && gd.scalar_type() == at::kFloat && gu.scalar_type() == at::kFloat, "ink_gates: fp32 contiguous");
    TORCH_CHECK(first >= 0 && first + G <= gd.size(0), "ink_gates: gate range");
    const at::cuda::OptionalCUDAGuard guard(X.device());
    const int NSPL = (2 * H + INK_GRC - 1) / INK_GRC;
    auto part = at::empty({G, NSPL, R}, X.options());
    auto out = at::empty({G, H}, X.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    ink_gate_down_kernel<<<dim3(NSPL, (unsigned)G), 256, 0, stream>>>(
        X.data_ptr<float>(), X.stride(0), M.data_ptr<float>(), M.stride(0), gd.data_ptr<float>(),
        (int)first, H, R, NSPL, (float)eps, part.data_ptr<float>(), (int)G, 0LL, 0LL);
    ink_gate_out_kernel<<<dim3((H + INK_GHW - 1) / INK_GHW, (unsigned)G), 256, 0, stream>>>(
        X.data_ptr<float>(), X.stride(0), M.data_ptr<float>(), M.stride(0), gu.data_ptr<float>(),
        part.data_ptr<float>(), (int)first, H, R, NSPL, out.data_ptr<float>(), (int)G, 0LL, 0LL);
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Short causal conv of one token over one channel group of a ring slot (C channels, taps = k - 1):
//   full = [st[c, :taps] | x[c]],  out[c] = sum_t full[t] * w[c, t]  (+ resid[c] when given)
//   history update: st[c, :taps] <- full[1:] (or the record tail st[c, -n:] <- full[-n:]), as Inkling2Ring._store.
// ---------------------------------------------------------------------------------------------------------------------
template <typename T>
__global__ void ink_sconv_kernel(float* __restrict__ st, long long ld, int C, const T* __restrict__ x,
                                 const float* __restrict__ w, const float* __restrict__ resid,
                                 float* __restrict__ out, int taps, int record)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int K1 = taps + 1;
    float* h = st + (long long)c * ld;
    float full[INK_MAXK];
    for (int t = 0; t < taps; ++t) full[t] = h[t];
    full[taps] = ink_in(x, c);
    float s = 0.f;
    for (int t = 0; t < K1; ++t) s += full[t] * w[c * K1 + t];
    out[c] = resid ? resid[c] + s : s;
    if (!record) {
        for (int t = 0; t < taps; ++t) h[t] = full[t + 1];
    } else {
        const int n = (int)min(ld, (long long)K1);
        for (int i = 0; i < n; ++i) h[ld - n + i] = full[K1 - n + i];
    }
}

at::Tensor ink_sconv(at::Tensor st, at::Tensor x, at::Tensor w, c10::optional<at::Tensor> resid, int64_t taps, bool record)
{
    const int C = (int)st.size(0);
    TORCH_CHECK(st.dim() == 2 && st.stride(1) == 1 && st.scalar_type() == at::kFloat, "ink_sconv: st (C, ld) fp32 view");
    TORCH_CHECK(taps >= 1 && taps + 1 <= INK_MAXK, "ink_sconv: taps");
    TORCH_CHECK(x.numel() == C && x.is_contiguous() && w.numel() == C * (taps + 1) && w.is_contiguous()
                && w.scalar_type() == at::kFloat, "ink_sconv: x (C), w (C, taps+1) fp32");
    const at::cuda::OptionalCUDAGuard guard(st.device());
    auto out = at::empty({C}, st.options());
    const float* rp = nullptr;
    if (resid.has_value()) {
        TORCH_CHECK(resid->numel() == C && resid->is_contiguous() && resid->scalar_type() == at::kFloat,
                    "ink_sconv: resid (C) fp32");
        rp = resid->data_ptr<float>();
    }
    const int threads = 128, blocks = (C + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kHalf) {
        ink_sconv_kernel<__half><<<blocks, threads, 0, stream>>>(
            st.data_ptr<float>(), st.stride(0), C, reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            w.data_ptr<float>(), rp, out.data_ptr<float>(), (int)taps, record ? 1 : 0);
    } else {
        TORCH_CHECK(x.scalar_type() == at::kFloat, "ink_sconv: x must be fp32 or fp16");
        ink_sconv_kernel<float><<<blocks, threads, 0, stream>>>(
            st.data_ptr<float>(), st.stride(0), C, x.data_ptr<float>(), w.data_ptr<float>(), rp, out.data_ptr<float>(),
            (int)taps, record ? 1 : 0);
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Query side of one single-token attention sublayer, one block per head:
//   q = l2clip(q_f * q_scale) * tau, r = l2clip(r_f * r_scale) * tau, rel = r @ rel_proj, sink = q_in @ sink_w^T,
//   gate = sigmoid(g_f). l2clip(x) = x * min(1, thr / (|x| + 1e-6)) over each head's last dim.
// ---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) ink_q_glue_kernel(
    const float* __restrict__ q_f, const float* __restrict__ r_f, const float* __restrict__ g_f,
    const float* __restrict__ q_in, const float* __restrict__ q_scale, const float* __restrict__ r_scale,
    const float* __restrict__ rel_proj, const float* __restrict__ sink_w,
    float* __restrict__ q_out, float* __restrict__ rel_out, float* __restrict__ sink_out,
    float* __restrict__ gate_out, int H, int D, int R, int E, float qk_clip, float rel_clip, float tau_h, const double* __restrict__ tau_p,
    long long tst)
{
    const long long rw = blockIdx.y, HQ = gridDim.x;
    const float tau = tau_p ? (float)tau_p[rw * tst] : tau_h;
    q_f += rw * HQ * D; r_f += rw * HQ * R; g_f += rw * HQ * D; q_in += rw * H;
    q_out += rw * HQ * D; rel_out += rw * HQ * E; sink_out += rw * HQ; gate_out += rw * HQ * D;
    __shared__ float red[32];
    __shared__ float rsm[INK_MAXG];
    const int h = blockIdx.x, tid = threadIdx.x;
    const int idx = h * D + tid;
    float v = 0.f;
    if (tid < D) v = q_f[idx] * q_scale[idx];
    const float ss = ink_block_sum(v * v, red);
    const float sc = fminf((1.f / (sqrtf(ss) + 1e-6f)) * qk_clip, 1.f);
    if (tid < D) {
        q_out[idx] = v * sc * tau;
        gate_out[idx] = 1.f / (1.f + expf(-g_f[idx]));
    }
    float s = 0.f;
    if (H <= 9 * 256) {
        float qa[9], wa[9];
        #pragma unroll
        for (int t = 0; t < 9; ++t) {
            const int k = tid + t * 256;
            qa[t] = k < H ? q_in[k] : 0.f;
            wa[t] = k < H ? sink_w[h * H + k] : 0.f;
        }
        #pragma unroll
        for (int t = 0; t < 9; ++t) s += qa[t] * wa[t];
    } else {
        for (int k = tid; k < H; k += blockDim.x) s += q_in[k] * sink_w[h * H + k];
    }
    s = ink_block_sum(s, red);
    if (tid == 0) sink_out[h] = s;

    float rv = 0.f;
    if (tid < R) rv = r_f[h * R + tid] * r_scale[h * R + tid];
    const float rss = ink_block_sum(rv * rv, red);
    const float rsc = fminf((1.f / (sqrtf(rss) + 1e-6f)) * rel_clip, 1.f);
    if (tid < R) rsm[tid] = rv * rsc * tau;
    __syncthreads();
    for (int e = tid; e < E; e += blockDim.x) {
        float a = 0.f;
        #pragma unroll 8
        for (int d = 0; d < R; ++d) a += rsm[d] * rel_proj[d * E + e];
        rel_out[h * E + e] = a;
    }
}

std::vector<at::Tensor> ink_q_glue(at::Tensor q_f, at::Tensor r_f, at::Tensor g_f, at::Tensor q_in, at::Tensor q_scale,
                                   at::Tensor r_scale, at::Tensor rel_proj, at::Tensor sink_w, double qk_clip,
                                   double rel_clip, double tau)
{
    const int Hq = (int)sink_w.size(0), H = (int)sink_w.size(1), R = (int)rel_proj.size(0), E = (int)rel_proj.size(1);
    const int D = (int)(q_f.numel() / Hq);
    TORCH_CHECK(D * Hq == q_f.numel() && D <= 256 && R <= INK_MAXG && r_f.numel() == Hq * R, "ink_q_glue: shapes");
    TORCH_CHECK(q_in.numel() == H && g_f.numel() == Hq * D && q_scale.numel() == Hq * D && r_scale.numel() == Hq * R,
                "ink_q_glue: shapes");
    for (const at::Tensor* t : {&q_f, &r_f, &g_f, &q_in, &q_scale, &r_scale, &rel_proj, &sink_w})
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous(), "ink_q_glue: fp32 contiguous inputs");
    const at::cuda::OptionalCUDAGuard guard(q_f.device());
    auto opt = q_f.options();
    auto q = at::empty({Hq * D}, opt), rel = at::empty({Hq * E}, opt), sink = at::empty({Hq}, opt),
         gate = at::empty({Hq * D}, opt);
    ink_q_glue_kernel<<<Hq, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_f.data_ptr<float>(), r_f.data_ptr<float>(), g_f.data_ptr<float>(), q_in.data_ptr<float>(),
        q_scale.data_ptr<float>(), r_scale.data_ptr<float>(), rel_proj.data_ptr<float>(), sink_w.data_ptr<float>(),
        q.data_ptr<float>(), rel.data_ptr<float>(), sink.data_ptr<float>(), gate.data_ptr<float>(), H, D, R, E,
        (float)qk_clip, (float)rel_clip, (float)tau, nullptr, 0LL);
    return {q, rel, sink, gate};
}

std::vector<at::Tensor> ink_q_glue_dev(at::Tensor q_f, at::Tensor r_f, at::Tensor g_f, at::Tensor q_in, at::Tensor q_scale,
                                   at::Tensor r_scale, at::Tensor rel_proj, at::Tensor sink_w, double qk_clip,
                                   double rel_clip, at::Tensor tau_dev)
{
    const int Hq = (int)sink_w.size(0), H = (int)sink_w.size(1), R = (int)rel_proj.size(0), E = (int)rel_proj.size(1);
    const int D = (int)(q_f.numel() / Hq);
    TORCH_CHECK(D * Hq == q_f.numel() && D <= 256 && R <= INK_MAXG && r_f.numel() == Hq * R, "ink_q_glue: shapes");
    TORCH_CHECK(q_in.numel() == H && g_f.numel() == Hq * D && q_scale.numel() == Hq * D && r_scale.numel() == Hq * R,
                "ink_q_glue: shapes");
    for (const at::Tensor* t : {&q_f, &r_f, &g_f, &q_in, &q_scale, &r_scale, &rel_proj, &sink_w})
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous(), "ink_q_glue: fp32 contiguous inputs");
    const at::cuda::OptionalCUDAGuard guard(q_f.device());
    auto opt = q_f.options();
    auto q = at::empty({Hq * D}, opt), rel = at::empty({Hq * E}, opt), sink = at::empty({Hq}, opt),
         gate = at::empty({Hq * D}, opt);
    ink_q_glue_kernel<<<Hq, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_f.data_ptr<float>(), r_f.data_ptr<float>(), g_f.data_ptr<float>(), q_in.data_ptr<float>(),
        q_scale.data_ptr<float>(), r_scale.data_ptr<float>(), rel_proj.data_ptr<float>(), sink_w.data_ptr<float>(),
        q.data_ptr<float>(), rel.data_ptr<float>(), sink.data_ptr<float>(), gate.data_ptr<float>(), H, D, R, E,
        (float)qk_clip, (float)rel_clip, 0.f, tau_dev.data_ptr<double>(), 0LL);
    return {q, rel, sink, gate};
}

// ---------------------------------------------------------------------------------------------------------------------
// K and V side of one single-token attention sublayer, one block per (kv head, K or V), one thread per channel:
//   short conv over [history | raw] (full layers add the per-token taps dyn = proj @ in), history update, then
//   K: l2clip per head (qk_clip), V: rms per head; written as fp16 into cache row `row` (both caches).
// st: (2C, ld) the K channels then the V channels of the ring slot.
// ---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(128) ink_kv_glue_kernel(
    const float* __restrict__ k_raw, const float* __restrict__ v_raw, float* __restrict__ st, long long ld,
    const float* __restrict__ k_in, const float* __restrict__ v_in,
    const float* __restrict__ wk, const float* __restrict__ wv,
    const float* __restrict__ pk, const float* __restrict__ pv,
    __half* __restrict__ kc, __half* __restrict__ vc, long long pos_h, const long long* __restrict__ pos_p,
    const int* __restrict__ pt, int C, int H, int D, int taps, int record, float qk_clip, float eps)
{
    // the new token's cache row: logical position -> page table -> physical row
    const long long pos = pos_p ? pos_p[0] : pos_h;
    const long long row = (long long)pt[pos / INK_PAGE] * INK_PAGE + pos % INK_PAGE;
    __shared__ float red[32];
    __shared__ float dyn[INK_MAXK];
    const int h = blockIdx.x, which = blockIdx.y, d = threadIdx.x;
    const int c = h * D + d;
    const int K1 = taps + 1;
    const float* raw = which ? v_raw : k_raw;
    const float* in = which ? v_in : k_in;
    const float* w = which ? wv : wk;
    const float* proj = which ? pv : pk;
    float* base = st + (long long)(which * C + c) * ld;

    if (proj) {
        float a[INK_MAXK];
        #pragma unroll
        for (int t = 0; t < INK_MAXK; ++t) a[t] = 0.f;
        #pragma unroll 6
        for (int i = d; i < H; i += blockDim.x) {
            const float x = in[i];
            #pragma unroll
            for (int t = 0; t < INK_MAXK; ++t)
                if (t < K1) a[t] += proj[(long long)(h * K1 + t) * H + i] * x;
        }
        #pragma unroll
        for (int t = 0; t < INK_MAXK; ++t)
            if (t < K1) {
                const float s = ink_block_sum(a[t], red);
                if (d == 0) dyn[t] = s;
            }
        __syncthreads();
    }
    float full[INK_MAXK];
    for (int t = 0; t < taps; ++t) full[t] = base[t];
    full[taps] = raw[c];
    float y = 0.f;
    for (int t = 0; t < K1; ++t) y += full[t] * (w[c * K1 + t] + (proj ? dyn[t] : 0.f));
    if (!record) {
        for (int t = 0; t < taps; ++t) base[t] = full[t + 1];
    } else {
        const int n = (int)min(ld, (long long)K1);
        for (int i = 0; i < n; ++i) base[ld - n + i] = full[K1 - n + i];
    }

    const float ss = ink_block_sum(y * y, red);
    if (which == 0) {
        y *= fminf((1.f / (sqrtf(ss) + 1e-6f)) * qk_clip, 1.f);
        kc[row * C + c] = __float2half_rn(y);
    } else {
        y *= rsqrtf(ss / (float)D + eps);
        vc[row * C + c] = __float2half_rn(y);
    }
}

void ink_kv_glue(at::Tensor k_raw, at::Tensor v_raw, at::Tensor st, at::Tensor k_in, at::Tensor v_in, at::Tensor wk,
                 at::Tensor wv, c10::optional<at::Tensor> pk, c10::optional<at::Tensor> pv, at::Tensor kf,
                 at::Tensor vf, int64_t pos, at::Tensor pt, int64_t taps, bool record, double qk_clip, double eps)
{
    const int Hk = (int)kf.size(1), D = (int)kf.size(2), C = Hk * D, H = (int)k_in.numel();
    TORCH_CHECK(D == 128 && Hk * D == k_raw.numel() && v_raw.numel() == C, "ink_kv_glue: D must be 128, C = Hk * D");
    TORCH_CHECK(st.dim() == 2 && st.size(0) == 2 * C && st.stride(1) == 1 && st.scalar_type() == at::kFloat,
                "ink_kv_glue: st (2C, ld) fp32 view");
    TORCH_CHECK(taps + 1 <= INK_MAXK && taps >= 1, "ink_kv_glue: taps");
    TORCH_CHECK(kf.scalar_type() == at::kHalf && vf.scalar_type() == at::kHalf && kf.is_contiguous() && vf.is_contiguous(),
                "ink_kv_glue: fp16 contiguous cache");
    TORCH_CHECK(pt.scalar_type() == at::kInt && pt.is_cuda() && pt.is_contiguous(), "ink_kv_glue: pt is a device int32 table");
    TORCH_CHECK(pos >= 0 && pos / INK_PAGE < pt.numel(), "ink_kv_glue: position outside the page table");
    const at::cuda::OptionalCUDAGuard guard(st.device());
    const float* pkp = pk.has_value() ? pk->data_ptr<float>() : nullptr;
    const float* pvp = pv.has_value() ? pv->data_ptr<float>() : nullptr;
    dim3 grid(Hk, 2);
    ink_kv_glue_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        k_raw.data_ptr<float>(), v_raw.data_ptr<float>(), st.data_ptr<float>(), st.stride(0),
        k_in.data_ptr<float>(), v_in.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), pkp, pvp,
        reinterpret_cast<__half*>(kf.data_ptr<at::Half>()), reinterpret_cast<__half*>(vf.data_ptr<at::Half>()),
        (long long)pos, nullptr, pt.data_ptr<int>(), C, H, D, (int)taps, record ? 1 : 0, (float)qk_clip, (float)eps);
}

void ink_kv_glue_dev(at::Tensor k_raw, at::Tensor v_raw, at::Tensor st, at::Tensor k_in, at::Tensor v_in, at::Tensor wk,
                 at::Tensor wv, c10::optional<at::Tensor> pk, c10::optional<at::Tensor> pv, at::Tensor kf,
                 at::Tensor vf, at::Tensor pos_dev, at::Tensor pt, int64_t taps, bool record, double qk_clip, double eps)
{
    const int Hk = (int)kf.size(1), D = (int)kf.size(2), C = Hk * D, H = (int)k_in.numel();
    TORCH_CHECK(D == 128 && Hk * D == k_raw.numel() && v_raw.numel() == C, "ink_kv_glue: D must be 128, C = Hk * D");
    TORCH_CHECK(st.dim() == 2 && st.size(0) == 2 * C && st.stride(1) == 1 && st.scalar_type() == at::kFloat,
                "ink_kv_glue: st (2C, ld) fp32 view");
    TORCH_CHECK(taps + 1 <= INK_MAXK && taps >= 1, "ink_kv_glue: taps");
    TORCH_CHECK(kf.scalar_type() == at::kHalf && vf.scalar_type() == at::kHalf && kf.is_contiguous() && vf.is_contiguous(),
                "ink_kv_glue: fp16 contiguous cache");
    TORCH_CHECK(pos_dev.numel() == 1 && pos_dev.is_cuda() && pos_dev.scalar_type() == at::kLong,
                "ink_kv_glue_dev: pos must be a device int64 scalar");
    TORCH_CHECK(pt.scalar_type() == at::kInt && pt.is_cuda() && pt.is_contiguous(), "ink_kv_glue: pt is a device int32 table");
    const at::cuda::OptionalCUDAGuard guard(st.device());
    const float* pkp = pk.has_value() ? pk->data_ptr<float>() : nullptr;
    const float* pvp = pv.has_value() ? pv->data_ptr<float>() : nullptr;
    dim3 grid(Hk, 2);
    ink_kv_glue_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        k_raw.data_ptr<float>(), v_raw.data_ptr<float>(), st.data_ptr<float>(), st.stride(0),
        k_in.data_ptr<float>(), v_in.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), pkp, pvp,
        reinterpret_cast<__half*>(kf.data_ptr<at::Half>()), reinterpret_cast<__half*>(vf.data_ptr<at::Half>()),
        0LL, reinterpret_cast<const long long*>(pos_dev.data_ptr<int64_t>()), pt.data_ptr<int>(),
        C, H, D, (int)taps, record ? 1 : 0, (float)qk_clip, (float)eps);
}


// ---------------------------------------------------------------------------------------------------------------------
// Single-query attention over the logical keys [lo, lo + L) of one sequence (the query is the last of them), one
// block per query head group: logits = scale * <q, k> + rel[L - 1 - j] (distance < E), softmax with the sink logit
// (value 0), o = sum p v, output (o * gate) as fp16. The key at logical position p is cache row
// pt[p / INK_PAGE] * INK_PAGE + p % INK_PAGE (page table pt, device int32), so any page order works.
// ---------------------------------------------------------------------------------------------------------------------
// Split over key chunks so that every SM gets work: grid (INK_GRID, kv heads), block b handles chunks b, b + grid, ...
// The three or fewer query heads of a kv head share the K / V loads. Then a combine over the chunks with the sink:
//   part: logits = scale * <q, k> + rel[L - 1 - j] (distance < E), chunk max m_c, p = exp(logit - m_c), l_c = sum p,
//         o_c = sum p v (fp32), written per query head at chunk slot c (stride nstride).
//   comb: M = max(sink, m_c), den = exp(sink - M) + sum_c exp(m_c - M) l_c, o = sum_c exp(m_c - M) o_c / den * gate.
// A captured CUDA graph uses the same grid at any position: chunks past the window are skipped.
#define INK_ACH 128        // keys per chunk (8 warps x 16 keys)
#define INK_MAXGQ 4        // query heads per kv head
static_assert(INK_ACH == INK_ACH_H, "chunk size must match inkling2.cuh");

__global__ void __launch_bounds__(256) ink_attend_part_kernel(
    const float* __restrict__ q, const float* __restrict__ rel, const __half* __restrict__ kf,
    const __half* __restrict__ vf, const int* __restrict__ pt, long long lo, int L, float sm_scale, int G, int Hk,
    int D, int E, int nstride, float* __restrict__ po, float* __restrict__ pml, const long long* __restrict__ par_p,
    long long pst)
{
    {
        const long long rw = blockIdx.z, HQ = (long long)Hk * G;
        if (par_p) { par_p += rw * pst; lo = par_p[0]; L = (int)par_p[1]; }
        q += rw * HQ * D; rel += rw * HQ * E; po += rw * HQ * nstride * D; pml += rw * HQ * nstride * 2;
    }
    __shared__ float slog[INK_MAXGQ][INK_ACH];
    __shared__ float sm32[32];
    __shared__ float osm[16][INK_MAXGQ][128];
    __shared__ long long srow[INK_ACH];                        // cache row of each key of the chunk
    const int hk = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long long KS = (long long)Hk * D;
    const int nc = (L + INK_ACH - 1) / INK_ACH;

    for (int c = blockIdx.x; c < nc; c += gridDim.x) {
        const int j0 = c * INK_ACH;
        const int n = min(INK_ACH, L - j0);
        if (tid < n) {
            const long long p = lo + j0 + tid;                 // logical position of the key
            srow[tid] = (long long)pt[p / INK_PAGE] * INK_PAGE + p % INK_PAGE;
        }
        __syncthreads();

        // step 1: warp = 16 keys, lane = 4 dims. Logits of the G query heads of this kv head.
        float qv[INK_MAXGQ][4];
        #pragma unroll
        for (int gi = 0; gi < INK_MAXGQ; ++gi)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                qv[gi][i] = gi < G ? q[(hk * G + gi) * D + lane * 4 + i] : 0.f;
        float kk[16][4];
        #pragma unroll
        for (int t = 0; t < 16; ++t) {
            const int jl = warp * 16 + t;
            if (jl < n) ink_load4h(kf + srow[jl] * KS + hk * D + lane * 4, kk[t]);
            else { kk[t][0] = kk[t][1] = kk[t][2] = kk[t][3] = 0.f; }
        }
        float part[INK_MAXGQ][16];
        #pragma unroll
        for (int gi = 0; gi < INK_MAXGQ; ++gi)
            #pragma unroll
            for (int t = 0; t < 16; ++t)
                part[gi][t] = qv[gi][0] * kk[t][0] + qv[gi][1] * kk[t][1] + qv[gi][2] * kk[t][2] + qv[gi][3] * kk[t][3];
        // reduce-scatter over the 64 logits (gi * 16 + t) of the warp: 62 shuffles instead of 320. Round r keeps the half
        // of its values selected by lane bit r, so lane L ends with the sums of logits 2L and 2L + 1
        float pv[64];
        #pragma unroll
        for (int gi = 0; gi < INK_MAXGQ; ++gi)
            #pragma unroll
            for (int t = 0; t < 16; ++t) pv[gi * 16 + t] = part[gi][t];
        #pragma unroll
        for (int half = 32, off = 16; half >= 2; half >>= 1, off >>= 1) {
            const bool b = (lane & off) != 0;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                if (i < half) {
                    const float send = b ? pv[i] : pv[i + half];
                    const float keep = b ? pv[i + half] : pv[i];
                    pv[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = 2 * lane + j, gi = idx >> 4, t = idx & 15;
            const int jl = warp * 16 + t;
            if (gi < G && jl < n) {
                float v = pv[j] * sm_scale;
                const int dist = L - 1 - (j0 + jl);
                if (dist < E) v += rel[(long long)(hk * G + gi) * E + dist];
                slog[gi][jl] = v;
            }
        }
        __syncthreads();

        // step 2: per query head, chunk max and exp (in place), chunk sum
        for (int gi = 0; gi < G; ++gi) {
            const float v = tid < n ? slog[gi][tid] : -INFINITY;
            const float m = ink_block_max(v, sm32);
            const float e = tid < n ? expf(v - m) : 0.f;
            const float s = ink_block_sum(e, sm32);
            if (tid < n) slog[gi][tid] = e;
            if (tid == 0) {
                const int hq = hk * G + gi;
                pml[((long long)hq * nstride + c) * 2] = m;
                pml[((long long)hq * nstride + c) * 2 + 1] = s;
            }
        }
        __syncthreads();

        // step 3: o_c = sum_j p_j v_j. Thread: 8 consecutive dims (one 16-byte load per key), keys kg, kg + 16, ... of
        // the chunk; all 8 loads are issued before the first use. 16 key groups are reduced at the end.
        const int dg = tid & 15, kg = tid >> 4;
        uint4 vr[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int jl = kg + 16 * t;
            vr[t] = jl < n ? *reinterpret_cast<const uint4*>(vf + srow[jl] * KS + hk * D + dg * 8)
                           : make_uint4(0u, 0u, 0u, 0u);
        }
        float acc[INK_MAXGQ][8];
        #pragma unroll
        for (int gi = 0; gi < INK_MAXGQ; ++gi)
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[gi][e] = 0.f;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const int jl = kg + 16 * t;
            const __half2* h2 = reinterpret_cast<const __half2*>(&vr[t]);
            float v[8];
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float2 f = __half22float2(h2[p]);
                v[2 * p] = f.x; v[2 * p + 1] = f.y;
            }
            #pragma unroll
            for (int gi = 0; gi < INK_MAXGQ; ++gi) {
                if (gi < G && jl < n) {
                    const float pj = slog[gi][jl];
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[gi][e] += pj * v[e];
                }
            }
        }
        #pragma unroll
        for (int gi = 0; gi < INK_MAXGQ; ++gi)
            if (gi < G) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) osm[kg][gi][dg * 8 + e] = acc[gi][e];
            }
        __syncthreads();
        for (int idx = tid; idx < G * 128; idx += blockDim.x) {      // G * 128 outputs
            const int gi = idx >> 7, dd = idx & 127;
            const int hq = hk * G + gi;
            float s = 0.f;
            #pragma unroll
            for (int g2 = 0; g2 < 16; ++g2) s += osm[g2][gi][dd];
            po[((long long)hq * nstride + c) * D + dd] = s;
        }
        __syncthreads();                                             // shared buffers free for the next chunk
    }
}

__global__ void __launch_bounds__(128) ink_attend_comb_kernel(
    const float* __restrict__ sink, const float* __restrict__ gate, const float* __restrict__ po,
    const float* __restrict__ pml, int nstride, int nc, const long long* __restrict__ par_p, int D,
    __half* __restrict__ o_out, long long pst)
{
    const long long rw = blockIdx.y, HQ = gridDim.x;
    const int ncn = par_p ? (int)par_p[rw * pst + 2] : nc;
    sink += rw * HQ; gate += rw * HQ * D; po += rw * HQ * nstride * D; pml += rw * HQ * nstride * 2; o_out += rw * HQ * D;
    __shared__ float w[INK_NSTORE];
    __shared__ float inv;
    const int hq = blockIdx.x, tid = threadIdx.x;
    if (tid == 0) {
        float M = sink[hq];
        for (int c = 0; c < ncn; ++c) M = fmaxf(M, pml[((long long)hq * nstride + c) * 2]);
        float den = expf(sink[hq] - M);
        for (int c = 0; c < ncn; ++c) {
            const float e = expf(pml[((long long)hq * nstride + c) * 2] - M);
            w[c] = e;
            den += e * pml[((long long)hq * nstride + c) * 2 + 1];
        }
        inv = 1.f / den;
    }
    __syncthreads();
    if (tid < D) {
        float o = 0.f;
        for (int c = 0; c < ncn; ++c) o += w[c] * po[((long long)hq * nstride + c) * D + tid];
        o_out[hq * D + tid] = __float2half_rn(o * inv * gate[hq * D + tid]);
    }
}

static void ink_attend_check(const at::Tensor& q, const at::Tensor& rel, const at::Tensor& sink, const at::Tensor& gate,
                             const at::Tensor& kf, const at::Tensor& vf, const at::Tensor& pt, int64_t Hk, int64_t E)
{
    const int Hq = (int)sink.numel(), D = (int)kf.size(2);
    TORCH_CHECK(D == 128 && Hq % Hk == 0 && Hq <= 256, "ink_attend: D must be 128, Hq a multiple of Hk");
    TORCH_CHECK(Hq / Hk <= INK_MAXGQ, "ink_attend: at most 4 query heads per kv head");
    TORCH_CHECK(q.numel() == Hq * D && gate.numel() == Hq * D && rel.numel() == Hq * E, "ink_attend: shapes");
    TORCH_CHECK(q.scalar_type() == at::kFloat && rel.scalar_type() == at::kFloat && sink.scalar_type() == at::kFloat
                && gate.scalar_type() == at::kFloat && q.is_contiguous() && rel.is_contiguous()
                && sink.is_contiguous() && gate.is_contiguous(), "ink_attend: fp32 contiguous q/rel/sink/gate");
    TORCH_CHECK(kf.scalar_type() == at::kHalf && vf.scalar_type() == at::kHalf, "ink_attend: fp16 cache");
    TORCH_CHECK(pt.scalar_type() == at::kInt && pt.is_cuda() && pt.is_contiguous(), "ink_attend: pt is a device int32 table");
}

at::Tensor ink_attend(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                      at::Tensor pt, int64_t lo, int64_t L, double sm_scale, int64_t Hk, int64_t E)
{
    ink_attend_check(q, rel, sink, gate, kf, vf, pt, Hk, E);
    TORCH_CHECK(L >= 1 && L <= (int64_t)INK_NSTORE * INK_ACH && lo >= 0, "ink_attend: key range");
    TORCH_CHECK((lo + L - 1) / INK_PAGE < pt.numel(), "ink_attend: keys outside the page table");
    const int Hq = (int)sink.numel(), D = (int)kf.size(2);
    const at::cuda::OptionalCUDAGuard guard(q.device());
    const int G = Hq / (int)Hk;
    const int nc = (int)((L + INK_ACH - 1) / INK_ACH);
    auto po = at::empty({Hq, nc, D}, q.options());
    auto pml = at::empty({Hq, nc, 2}, q.options());
    auto o = at::empty({Hq * D}, q.options().dtype(at::kHalf));
    auto stream = at::cuda::getCurrentCUDAStream();
    ink_attend_part_kernel<<<dim3(nc < INK_GRID ? nc : INK_GRID, (unsigned)Hk), 256, 0, stream>>>(
        q.data_ptr<float>(), rel.data_ptr<float>(),
        reinterpret_cast<const __half*>(kf.data_ptr<at::Half>()), reinterpret_cast<const __half*>(vf.data_ptr<at::Half>()),
        pt.data_ptr<int>(), (long long)lo, (int)L, (float)sm_scale, G, (int)Hk, D, (int)E, nc,
        po.data_ptr<float>(), pml.data_ptr<float>(), nullptr, 0LL);
    ink_attend_comb_kernel<<<Hq, 128, 0, stream>>>(
        sink.data_ptr<float>(), gate.data_ptr<float>(), po.data_ptr<float>(), pml.data_ptr<float>(), nc, nc, nullptr, D,
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()), 0LL);
    return o;
}

// Same as ink_attend with the window in a device tensor par = [lo, L, nc] (int64, nc = ceil(L / INK_ACH)); the grid
// is fixed (INK_GRID x kv heads) and the split buffers hold INK_NSTORE chunks, so a captured CUDA graph replays at
// any position up to INK_NSTORE * INK_ACH keys.
at::Tensor ink_attend_dev(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                          at::Tensor pt, at::Tensor par, double sm_scale, int64_t Hk, int64_t E)
{
    ink_attend_check(q, rel, sink, gate, kf, vf, pt, Hk, E);
    TORCH_CHECK(par.numel() == 3 && par.scalar_type() == at::kLong && par.is_cuda(), "ink_attend_dev: par");
    const int Hq = (int)sink.numel(), D = (int)kf.size(2);
    const int G = Hq / (int)Hk;
    auto po = at::empty({Hq, INK_NSTORE, D}, q.options());
    auto pml = at::empty({Hq, INK_NSTORE, 2}, q.options());
    auto o = at::empty({Hq * D}, q.options().dtype(at::kHalf));
    auto stream = at::cuda::getCurrentCUDAStream();
    ink_attend_part_kernel<<<dim3(INK_GRID, (unsigned)Hk), 256, 0, stream>>>(
        q.data_ptr<float>(), rel.data_ptr<float>(),
        reinterpret_cast<const __half*>(kf.data_ptr<at::Half>()), reinterpret_cast<const __half*>(vf.data_ptr<at::Half>()),
        pt.data_ptr<int>(), 0LL, 1, (float)sm_scale, G, (int)Hk, D, (int)E, INK_NSTORE,
        po.data_ptr<float>(), pml.data_ptr<float>(), reinterpret_cast<const long long*>(par.data_ptr<int64_t>()), 0LL);
    ink_attend_comb_kernel<<<Hq, 128, 0, stream>>>(
        sink.data_ptr<float>(), gate.data_ptr<float>(), po.data_ptr<float>(), pml.data_ptr<float>(), INK_NSTORE,
        INK_NSTORE, reinterpret_cast<const long long*>(par.data_ptr<int64_t>()), D, reinterpret_cast<__half*>(o.data_ptr<at::Half>()), 0LL);
    return o;
}


// =====================================================================================================================
// Row-batched variants (InkToken::step_rows2): T rows of one sequence (positions pos0 .. pos0 + T - 1) per launch, the
// row is a grid dimension. Each row runs the arithmetic of the single-token kernel above (same operations in the same
// order), so a row's result has the bits of a single-token step at that row.
// =====================================================================================================================
#define INK_MAXT 16        // rows per forward upper bound (history window buffers)

// srcs: S tensors (T, H) fp32 contiguous -> (T, NQ, H)
at::Tensor ink_attn_res_rows(std::vector<at::Tensor> srcs, at::Tensor qT, double eps)
{
    const int S = (int)srcs.size();
    TORCH_CHECK(S >= 1 && S <= INK_MAXS, "ink_attn_res_rows: 1..16 sources");
    TORCH_CHECK(qT.dim() == 2 && qT.size(1) >= 1 && qT.size(1) <= INK_MAXQ, "ink_attn_res_rows: qT (H, nq<=4)");
    const int H = (int)qT.size(0), NQ = (int)qT.size(1);
    const int64_t T = srcs[0].size(0);
    const at::cuda::OptionalCUDAGuard guard(qT.device());
    InkSrcs a;
    bool aligned = true;
    for (int j = 0; j < S; ++j) {
        TORCH_CHECK(srcs[j].scalar_type() == at::kFloat && srcs[j].is_contiguous() && srcs[j].dim() == 2
                    && srcs[j].size(0) == T && srcs[j].size(1) == H, "ink_attn_res_rows: sources (T, H) fp32 contiguous");
        a.p[j] = srcs[j].data_ptr<float>();
        aligned = aligned && (((uintptr_t)a.p[j]) % 16 == 0);
    }
    TORCH_CHECK(qT.scalar_type() == at::kFloat && qT.is_contiguous(), "ink_attn_res_rows: qT must be contiguous fp32");
    TORCH_CHECK(H % 4 == 0 && aligned, "ink_attn_res_rows: H % 4 == 0 and 16-byte aligned sources");
    auto out = at::empty({T, NQ, H}, qT.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int H4 = H / 4;
    const int nblk = (H4 + INK_ARBS - 1) / INK_ARBS;
    auto partial = at::empty({T, nblk, INK_ACC}, qT.options());
    ink_attn_res_part_kernel<<<dim3(nblk, (unsigned)T), INK_ARBS, 0, stream>>>(
        a, S, NQ, H4, qT.data_ptr<float>(), partial.data_ptr<float>(), (long long)H, nblk);
    ink_attn_res_fin_kernel<<<dim3(nblk, (unsigned)T), INK_ARBS, 0, stream>>>(
        a, S, NQ, H, H4, nblk, partial.data_ptr<float>(), (float)eps, out.data_ptr<float>(), (long long)H);
    return out;
}

// X: (T, G, H) fp32 contiguous; msgs: (T, NM, H) fp32 contiguous, row (r, first + g) is the message of gate first + g
// -> (T, G, H)
at::Tensor ink_gates_rows(at::Tensor X, at::Tensor msgs, at::Tensor gd, at::Tensor gu, int64_t first, int64_t G, double eps)
{
    TORCH_CHECK(X.dim() == 3 && msgs.dim() == 3 && X.size(1) == G && msgs.size(0) == X.size(0)
                && msgs.size(2) == X.size(2) && first >= 0 && first + G <= msgs.size(1), "ink_gates_rows: X (T, G, H), msgs (T, NM, H)");
    const int64_t T = X.size(0), NM = msgs.size(1);
    const int H = (int)X.size(2), R = (int)gd.size(2);
    TORCH_CHECK(X.is_contiguous() && msgs.is_contiguous() && X.scalar_type() == at::kFloat && msgs.scalar_type() == at::kFloat,
                "ink_gates_rows: fp32 contiguous rows");
    TORCH_CHECK(R >= 4 && R <= INK_MAXG && R % 4 == 0 && (R & (R - 1)) == 0, "ink_gates_rows: rank a power of two in [4, 64]");
    TORCH_CHECK(gd.size(1) == 2 * H && gu.size(1) == R && gu.size(2) == 2 * H && gd.is_contiguous() && gu.is_contiguous()
                && gd.scalar_type() == at::kFloat && gu.scalar_type() == at::kFloat && first + G <= gd.size(0),
                "ink_gates_rows: weights");
    const at::cuda::OptionalCUDAGuard guard(X.device());
    const int NSPL = (2 * H + INK_GRC - 1) / INK_GRC;
    auto part = at::empty({T * G, NSPL, R}, X.options());
    auto out = at::empty({T, G, H}, X.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* Mp = msgs.data_ptr<float>() + first * H;
    ink_gate_down_kernel<<<dim3(NSPL, (unsigned)(T * G)), 256, 0, stream>>>(
        X.data_ptr<float>(), (long long)H, Mp, (long long)H, gd.data_ptr<float>(),
        (int)first, H, R, NSPL, (float)eps, part.data_ptr<float>(), (int)G, (long long)G * H, (long long)NM * H);
    ink_gate_out_kernel<<<dim3((H + INK_GHW - 1) / INK_GHW, (unsigned)(T * G)), 256, 0, stream>>>(
        X.data_ptr<float>(), (long long)H, Mp, (long long)H, gu.data_ptr<float>(),
        part.data_ptr<float>(), (int)first, H, R, NSPL, out.data_ptr<float>(), (int)G, (long long)G * H, (long long)NM * H);
    return out;
}

// Short conv of T rows over one channel group (C channels): row r sees the history after rows < r. x: (T, C) fp32,
// resid: (T, C) fp32 or none -> (T, C). History after the rows as Inkling2Ring._store: the live taps advance by T
// rows, or (record) the live taps stay and the tail takes the last columns of [history | rows].
__global__ void ink_sconv_rows_kernel(float* __restrict__ st, long long ld, int C, const float* __restrict__ x,
                                      const float* __restrict__ w, const float* __restrict__ resid,
                                      float* __restrict__ out, int taps, int T, int record)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int K1 = taps + 1;
    float* h = st + (long long)c * ld;
    float buf[INK_MAXK + INK_MAXT];
    for (int t = 0; t < taps; ++t) buf[t] = h[t];
    for (int r = 0; r < T; ++r) buf[taps + r] = x[(long long)r * C + c];
    for (int r = 0; r < T; ++r) {
        float full[INK_MAXK];
        for (int t = 0; t < K1; ++t) full[t] = buf[r + t];
        float s = 0.f;
        for (int t = 0; t < K1; ++t) s += full[t] * w[c * K1 + t];
        out[(long long)r * C + c] = resid ? resid[(long long)r * C + c] + s : s;
    }
    if (!record) {
        for (int t = 0; t < taps; ++t) h[t] = buf[T + t];
    } else {
        const int n = (int)min(ld, (long long)(taps + T));
        for (int i = 0; i < n; ++i) h[ld - n + i] = buf[taps + T - n + i];
    }
}

at::Tensor ink_sconv_rows(at::Tensor st, at::Tensor x, at::Tensor w, c10::optional<at::Tensor> resid, int64_t taps, bool record)
{
    const int C = (int)st.size(0);
    TORCH_CHECK(st.dim() == 2 && st.stride(1) == 1 && st.scalar_type() == at::kFloat, "ink_sconv_rows: st (C, ld) fp32 view");
    TORCH_CHECK(taps >= 1 && taps + 1 <= INK_MAXK, "ink_sconv_rows: taps");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == C && x.is_contiguous() && x.scalar_type() == at::kFloat && x.size(0) <= INK_MAXT
                && w.numel() == C * (taps + 1) && w.is_contiguous() && w.scalar_type() == at::kFloat,
                "ink_sconv_rows: x (T, C) fp32, w (C, taps+1) fp32");
    const int T = (int)x.size(0);
    const at::cuda::OptionalCUDAGuard guard(st.device());
    auto out = at::empty({T, C}, st.options());
    const float* rp = nullptr;
    if (resid.has_value()) {
        TORCH_CHECK(resid->numel() == (int64_t)T * C && resid->is_contiguous() && resid->scalar_type() == at::kFloat,
                    "ink_sconv_rows: resid (T, C) fp32");
        rp = resid->data_ptr<float>();
    }
    const int threads = 128, blocks = (C + threads - 1) / threads;
    ink_sconv_rows_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        st.data_ptr<float>(), st.stride(0), C, x.data_ptr<float>(), w.data_ptr<float>(), rp, out.data_ptr<float>(),
        (int)taps, T, record ? 1 : 0);
    return out;
}

// History of one channel group after T rows with the conv inputs x (T, C) (see ink_sconv_rows); no outputs.
__global__ void ink_hist_rows_kernel(float* __restrict__ st, long long ld, int C, const float* __restrict__ x, int taps,
                                     int T, int record)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* h = st + (long long)c * ld;
    float buf[INK_MAXK + INK_MAXT];
    for (int t = 0; t < taps; ++t) buf[t] = h[t];
    for (int r = 0; r < T; ++r) buf[taps + r] = x[(long long)r * C + c];
    if (!record) {
        for (int t = 0; t < taps; ++t) h[t] = buf[T + t];
    } else {
        const int n = (int)min(ld, (long long)(taps + T));
        for (int i = 0; i < n; ++i) h[ld - n + i] = buf[taps + T - n + i];
    }
}

void ink_hist_rows(at::Tensor st, at::Tensor x, int64_t taps, bool record)
{
    const int C = (int)st.size(0);
    TORCH_CHECK(st.dim() == 2 && st.stride(1) == 1 && st.scalar_type() == at::kFloat, "ink_hist_rows: st (C, ld) fp32 view");
    TORCH_CHECK(taps >= 1 && taps + 1 <= INK_MAXK && x.dim() == 2 && x.size(1) == C && x.size(0) <= INK_MAXT
                && x.is_contiguous() && x.scalar_type() == at::kFloat, "ink_hist_rows: x (T, C) fp32");
    const at::cuda::OptionalCUDAGuard guard(st.device());
    const int threads = 128, blocks = (C + threads - 1) / threads;
    ink_hist_rows_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        st.data_ptr<float>(), st.stride(0), C, x.data_ptr<float>(), (int)taps, (int)x.size(0), record ? 1 : 0);
}

// Rewind of recorded conv histories: state (B, C, ld) of every layer, slot `slot`: the taps before the last
// num_tokens recorded columns become the current taps (columns 0..taps-1)
__global__ void ink_rewind_kernel(float* __restrict__ st, long long ld, int C, int taps, int nt)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* h = st + (long long)c * ld;
    float buf[INK_MAXK];
    const long long p = ld - nt - taps;
    for (int t = 0; t < taps; ++t) buf[t] = h[p + t];
    for (int t = 0; t < taps; ++t) h[t] = buf[t];
}

void ink_rewind_all(std::vector<at::Tensor> states, int64_t slot, int64_t taps, int64_t num_tokens)
{
    if (states.empty()) return;
    const at::cuda::OptionalCUDAGuard guard(states[0].device());
    auto stream = at::cuda::getCurrentCUDAStream();
    for (const at::Tensor& st : states) {
        TORCH_CHECK(st.dim() == 3 && st.is_contiguous() && st.scalar_type() == at::kFloat && st.is_cuda()
                    && slot >= 0 && slot < st.size(0), "ink_rewind_all: state (B, C, ld) fp32");
        const int64_t ld = st.size(2);
        TORCH_CHECK(taps >= 1 && taps < INK_MAXK && num_tokens >= 0 && ld - num_tokens - taps >= 0, "ink_rewind_all: range");
        const int C = (int)st.size(1);
        const int threads = 128, blocks = (C + threads - 1) / threads;
        ink_rewind_kernel<<<blocks, threads, 0, stream>>>(st.data_ptr<float>() + slot * st.stride(0), ld, C, (int)taps,
                                                           (int)num_tokens);
    }
}

// Query glue of T rows: q_f (T, Hq D), r_f (T, Hq R), g_f (T, Hq D), q_in (T, H); tau of row r at tau_dev[r * tst]
// -> {q (T, Hq D), rel (T, Hq E), sink (T, Hq), gate (T, Hq D)}
std::vector<at::Tensor> ink_q_glue_rows(at::Tensor q_f, at::Tensor r_f, at::Tensor g_f, at::Tensor q_in, at::Tensor q_scale,
                                        at::Tensor r_scale, at::Tensor rel_proj, at::Tensor sink_w, double qk_clip,
                                        double rel_clip, at::Tensor tau_dev, int64_t tst)
{
    const int Hq = (int)sink_w.size(0), H = (int)sink_w.size(1), R = (int)rel_proj.size(0), E = (int)rel_proj.size(1);
    TORCH_CHECK(q_f.dim() == 2 && r_f.dim() == 2 && g_f.dim() == 2 && q_in.dim() == 2, "ink_q_glue_rows: (T, n) inputs");
    const int64_t T = q_f.size(0);
    const int D = (int)(q_f.size(1) / Hq);
    TORCH_CHECK(D * Hq == q_f.size(1) && D <= 256 && R <= INK_MAXG && r_f.size(1) == Hq * R && r_f.size(0) == T
                && g_f.size(0) == T && g_f.size(1) == Hq * D && q_in.size(0) == T && q_in.size(1) == H
                && q_scale.numel() == Hq * D && r_scale.numel() == Hq * R, "ink_q_glue_rows: shapes");
    for (const at::Tensor* t : {&q_f, &r_f, &g_f, &q_in, &q_scale, &r_scale, &rel_proj, &sink_w})
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous(), "ink_q_glue_rows: fp32 contiguous inputs");
    TORCH_CHECK(tau_dev.is_cuda() && tau_dev.scalar_type() == at::kDouble, "ink_q_glue_rows: tau");
    const at::cuda::OptionalCUDAGuard guard(q_f.device());
    auto opt = q_f.options();
    auto q = at::empty({T, Hq * D}, opt), rel = at::empty({T, Hq * E}, opt), sink = at::empty({T, Hq}, opt),
         gate = at::empty({T, Hq * D}, opt);
    ink_q_glue_kernel<<<dim3(Hq, (unsigned)T), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        q_f.data_ptr<float>(), r_f.data_ptr<float>(), g_f.data_ptr<float>(), q_in.data_ptr<float>(),
        q_scale.data_ptr<float>(), r_scale.data_ptr<float>(), rel_proj.data_ptr<float>(), sink_w.data_ptr<float>(),
        q.data_ptr<float>(), rel.data_ptr<float>(), sink.data_ptr<float>(), gate.data_ptr<float>(), H, D, R, E,
        (float)qk_clip, (float)rel_clip, 0.f, tau_dev.data_ptr<double>(), (long long)tst);
    return {q, rel, sink, gate};
}

// K / V glue of T rows, grid (kv head, K or V, row). Row r convolves over the history after rows < r, read from the
// unmodified live history and the raw rows before it: window = columns r .. r + taps of [history | raw rows]. The
// history itself is not written here (ink_hist_rows after this kernel). Cache row of row r: position pos_p[r * pst].
__global__ void __launch_bounds__(128) ink_kv_glue_rows_kernel(
    const float* __restrict__ k_raw, const float* __restrict__ v_raw, const float* __restrict__ st, long long ld,
    const float* __restrict__ k_in, const float* __restrict__ v_in,
    const float* __restrict__ wk, const float* __restrict__ wv,
    const float* __restrict__ pk, const float* __restrict__ pv,
    __half* __restrict__ kc, __half* __restrict__ vc, const long long* __restrict__ pos_p, long long pst,
    const int* __restrict__ pt, int C, int H, int D, int taps, float qk_clip, float eps)
{
    const int rw = blockIdx.z;
    const long long pos = pos_p[rw * pst];
    const long long row = (long long)pt[pos / INK_PAGE] * INK_PAGE + pos % INK_PAGE;
    __shared__ float red[32];
    __shared__ float dyn[INK_MAXK];
    const int h = blockIdx.x, which = blockIdx.y, d = threadIdx.x;
    const int c = h * D + d;
    const int K1 = taps + 1;
    const float* raw = which ? v_raw : k_raw;
    const float* in = (which ? v_in : k_in) + (long long)rw * H;
    const float* w = which ? wv : wk;
    const float* proj = which ? pv : pk;
    const float* base = st + (long long)(which * C + c) * ld;

    if (proj) {
        float a[INK_MAXK];
        #pragma unroll
        for (int t = 0; t < INK_MAXK; ++t) a[t] = 0.f;
        #pragma unroll 6
        for (int i = d; i < H; i += blockDim.x) {
            const float x = in[i];
            #pragma unroll
            for (int t = 0; t < INK_MAXK; ++t)
                if (t < K1) a[t] += proj[(long long)(h * K1 + t) * H + i] * x;
        }
        #pragma unroll
        for (int t = 0; t < INK_MAXK; ++t)
            if (t < K1) {
                const float s = ink_block_sum(a[t], red);
                if (d == 0) dyn[t] = s;
            }
        __syncthreads();
    }
    float full[INK_MAXK];
    for (int t = 0; t < taps; ++t) {
        const int i = rw + t;
        full[t] = i < taps ? base[i] : raw[(long long)(i - taps) * C + c];
    }
    full[taps] = raw[(long long)rw * C + c];
    float y = 0.f;
    for (int t = 0; t < K1; ++t) y += full[t] * (w[c * K1 + t] + (proj ? dyn[t] : 0.f));

    const float ss = ink_block_sum(y * y, red);
    if (which == 0) {
        y *= fminf((1.f / (sqrtf(ss) + 1e-6f)) * qk_clip, 1.f);
        kc[row * C + c] = __float2half_rn(y);
    } else {
        y *= rsqrtf(ss / (float)D + eps);
        vc[row * C + c] = __float2half_rn(y);
    }
}

void ink_kv_glue_rows(at::Tensor k_raw, at::Tensor v_raw, at::Tensor st, at::Tensor k_in, at::Tensor v_in, at::Tensor wk,
                      at::Tensor wv, c10::optional<at::Tensor> pk, c10::optional<at::Tensor> pv, at::Tensor kf,
                      at::Tensor vf, at::Tensor pos_dev, int64_t pst, at::Tensor pt, int64_t taps, double qk_clip, double eps)
{
    const int Hk = (int)kf.size(1), D = (int)kf.size(2), C = Hk * D;
    TORCH_CHECK(k_raw.dim() == 2 && v_raw.dim() == 2 && k_in.dim() == 2 && v_in.dim() == 2, "ink_kv_glue_rows: (T, n) inputs");
    const int64_t T = k_raw.size(0);
    const int H = (int)k_in.size(1);
    TORCH_CHECK(D == 128 && k_raw.size(1) == C && v_raw.size(0) == T && v_raw.size(1) == C && k_in.size(0) == T
                && v_in.size(0) == T && v_in.size(1) == H && T <= INK_MAXT, "ink_kv_glue_rows: D must be 128, C = Hk * D");
    for (const at::Tensor* t : {&k_raw, &v_raw, &k_in, &v_in, &wk, &wv})
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous(), "ink_kv_glue_rows: fp32 contiguous inputs");
    TORCH_CHECK(st.dim() == 2 && st.size(0) == 2 * C && st.stride(1) == 1 && st.scalar_type() == at::kFloat,
                "ink_kv_glue_rows: st (2C, ld) fp32 view");
    TORCH_CHECK(taps + 1 <= INK_MAXK && taps >= 1, "ink_kv_glue_rows: taps");
    TORCH_CHECK(kf.scalar_type() == at::kHalf && vf.scalar_type() == at::kHalf && kf.is_contiguous() && vf.is_contiguous(),
                "ink_kv_glue_rows: fp16 contiguous cache");
    TORCH_CHECK(pos_dev.is_cuda() && pos_dev.scalar_type() == at::kLong, "ink_kv_glue_rows: pos must be device int64");
    TORCH_CHECK(pt.scalar_type() == at::kInt && pt.is_cuda() && pt.is_contiguous(), "ink_kv_glue_rows: pt is a device int32 table");
    const at::cuda::OptionalCUDAGuard guard(st.device());
    const float* pkp = pk.has_value() ? pk->data_ptr<float>() : nullptr;
    const float* pvp = pv.has_value() ? pv->data_ptr<float>() : nullptr;
    ink_kv_glue_rows_kernel<<<dim3(Hk, 2, (unsigned)T), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        k_raw.data_ptr<float>(), v_raw.data_ptr<float>(), st.data_ptr<float>(), st.stride(0),
        k_in.data_ptr<float>(), v_in.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), pkp, pvp,
        reinterpret_cast<__half*>(kf.data_ptr<at::Half>()), reinterpret_cast<__half*>(vf.data_ptr<at::Half>()),
        reinterpret_cast<const long long*>(pos_dev.data_ptr<int64_t>()), (long long)pst, pt.data_ptr<int>(),
        C, H, D, (int)taps, (float)qk_clip, (float)eps);
}

// Attention of T rows: q (T, Hq D), rel (T, Hq E), sink (T, Hq), gate (T, Hq D); window of row r at par[r * pst ..]
// = [lo, L, nc] -> (T, Hq D) fp16
at::Tensor ink_attend_rows_dev(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                               at::Tensor pt, at::Tensor par, int64_t pst, double sm_scale, int64_t Hk, int64_t E)
{
    TORCH_CHECK(q.dim() == 2 && rel.dim() == 2 && sink.dim() == 2 && gate.dim() == 2, "ink_attend_rows_dev: (T, n) inputs");
    const int64_t T = q.size(0);
    const int Hq = (int)sink.size(1), D = (int)kf.size(2);
    TORCH_CHECK(D == 128 && Hq % Hk == 0 && Hq <= 256 && Hq / Hk <= INK_MAXGQ, "ink_attend_rows_dev: heads");
    TORCH_CHECK(q.size(1) == Hq * D && gate.size(1) == Hq * D && rel.size(1) == Hq * E && rel.size(0) == T
                && sink.size(0) == T && gate.size(0) == T, "ink_attend_rows_dev: shapes");
    for (const at::Tensor* t : {&q, &rel, &sink, &gate})
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->is_contiguous(), "ink_attend_rows_dev: fp32 contiguous inputs");
    TORCH_CHECK(kf.scalar_type() == at::kHalf && vf.scalar_type() == at::kHalf, "ink_attend_rows_dev: fp16 cache");
    TORCH_CHECK(pt.scalar_type() == at::kInt && pt.is_cuda() && pt.is_contiguous(), "ink_attend_rows_dev: pt");
    TORCH_CHECK(par.scalar_type() == at::kLong && par.is_cuda(), "ink_attend_rows_dev: par");
    const int G = Hq / (int)Hk;
    auto po = at::empty({T, Hq, INK_NSTORE, D}, q.options());
    auto pml = at::empty({T, Hq, INK_NSTORE, 2}, q.options());
    auto o = at::empty({T, Hq * D}, q.options().dtype(at::kHalf));
    auto stream = at::cuda::getCurrentCUDAStream();
    const long long* pp = reinterpret_cast<const long long*>(par.data_ptr<int64_t>());
    ink_attend_part_kernel<<<dim3(INK_GRID, (unsigned)Hk, (unsigned)T), 256, 0, stream>>>(
        q.data_ptr<float>(), rel.data_ptr<float>(),
        reinterpret_cast<const __half*>(kf.data_ptr<at::Half>()), reinterpret_cast<const __half*>(vf.data_ptr<at::Half>()),
        pt.data_ptr<int>(), 0LL, 1, (float)sm_scale, G, (int)Hk, D, (int)E, INK_NSTORE,
        po.data_ptr<float>(), pml.data_ptr<float>(), pp, (long long)pst);
    ink_attend_comb_kernel<<<dim3(Hq, (unsigned)T), 128, 0, stream>>>(
        sink.data_ptr<float>(), gate.data_ptr<float>(), po.data_ptr<float>(), pml.data_ptr<float>(), INK_NSTORE,
        INK_NSTORE, pp, D, reinterpret_cast<__half*>(o.data_ptr<at::Half>()), (long long)pst);
    return o;
}
