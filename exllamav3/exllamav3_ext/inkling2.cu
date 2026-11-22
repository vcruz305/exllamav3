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

at::Tensor ink_attn_res(std::vector<at::Tensor> srcs, at::Tensor qT, double eps)
{
    const int S = (int)srcs.size();
    TORCH_CHECK(S >= 1 && S <= INK_MAXS, "ink_attn_res: 1..16 sources");
    TORCH_CHECK(qT.dim() == 2 && qT.size(1) >= 1 && qT.size(1) <= INK_MAXQ, "ink_attn_res: qT (H, nq<=4)");
    const int H = (int)qT.size(0), NQ = (int)qT.size(1);
    const at::cuda::OptionalCUDAGuard guard(qT.device());
    InkSrcs a;
    for (int j = 0; j < S; ++j) {
        TORCH_CHECK(srcs[j].scalar_type() == at::kFloat && srcs[j].is_contiguous() && srcs[j].numel() == H,
                    "ink_attn_res: sources must be contiguous fp32 vectors of H");
        a.p[j] = srcs[j].data_ptr<float>();
    }
    TORCH_CHECK(qT.scalar_type() == at::kFloat && qT.is_contiguous(), "ink_attn_res: qT must be contiguous fp32");
    auto out = at::empty({NQ, H}, qT.options());
    ink_attn_res_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        a, S, NQ, H, qT.data_ptr<float>(), (float)eps, out.data_ptr<float>());
    return out;
}

// ---------------------------------------------------------------------------------------------------------------------
// Message gates for G consecutive gates (indices first .. first + G) on rows X[g], M[g] (strides in elements):
//   z = [rms(x), rms(m)] (2H), g = z @ down_g (R), t = tanh(g @ up_g) (2H), out = (1 - t[:H]) * x + t[H:] * m
// gd: (4, 2H, R) row-major, gu: (4, R, 2H) row-major.
// ---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(1024) ink_gates_kernel(
    const float* __restrict__ X, long long xs, const float* __restrict__ M, long long ms,
    const float* __restrict__ gd, const float* __restrict__ gu, int first, int H, int R, float eps,
    float* __restrict__ out)
{
    extern __shared__ float z[];                    // 2H: normalized [x | m], then t = tanh(...)
    __shared__ float red[32];
    __shared__ float gp[1024];
    __shared__ float gv[INK_MAXG];
    const int g = blockIdx.x, gi = first + g, tid = threadIdx.x;
    const float* x = X + g * xs;
    const float* m = M + g * ms;
    float sx = 0.f, sm = 0.f;
    for (int i = tid; i < H; i += blockDim.x) { sx += x[i] * x[i]; sm += m[i] * m[i]; }
    sx = ink_block_sum(sx, red);
    sm = ink_block_sum(sm, red);
    const float rx = rsqrtf(sx / (float)H + eps), rm = rsqrtf(sm / (float)H + eps);
    for (int i = tid; i < H; i += blockDim.x) { z[i] = x[i] * rx; z[H + i] = m[i] * rm; }
    __syncthreads();

    // g_r = sum_k z[k] * gd[gi, k, r]: nparts threads per rank, each over a strided slice of k
    const float* W1 = gd + (size_t)gi * 2 * H * R;
    const int nparts = blockDim.x / R;
    if (tid < nparts * R) {
        const int r = tid % R, part = tid / R;
        float a = 0.f;
        for (int k = part; k < 2 * H; k += nparts) a += z[k] * W1[(size_t)k * R + r];
        gp[part * R + r] = a;
    }
    __syncthreads();
    if (tid < R) {
        float s = 0.f;
        for (int p = 0; p < nparts; ++p) s += gp[p * R + tid];
        gv[tid] = s;
    }
    __syncthreads();
    // t_n = tanh(sum_r g_r * gu[gi, r, n]) for n < 2H; z is no longer read
    const float* W2 = gu + (size_t)gi * R * 2 * H;
    for (int n = tid; n < 2 * H; n += blockDim.x) {
        float s = 0.f;
        for (int r = 0; r < R; ++r) s += gv[r] * W2[(size_t)r * 2 * H + n];
        z[n] = tanhf(s);
    }
    __syncthreads();
    for (int h = tid; h < H; h += blockDim.x)
        out[g * H + h] = (1.f - z[h]) * x[h] + z[H + h] * m[h];
}

at::Tensor ink_gates(at::Tensor X, at::Tensor M, at::Tensor gd, at::Tensor gu, int64_t first, int64_t G, double eps)
{
    const int H = (int)X.size(1), R = (int)gd.size(2);
    TORCH_CHECK(X.dim() == 2 && M.dim() == 2 && X.size(0) == G && M.size(0) == G, "ink_gates: X, M (G, H)");
    TORCH_CHECK(X.stride(1) == 1 && M.stride(1) == 1, "ink_gates: rows must be unit-stride");
    TORCH_CHECK(R >= 1 && R <= INK_MAXG && 1024 % R == 0, "ink_gates: rank must divide 1024 and be <= 64");
    TORCH_CHECK(gd.size(1) == 2 * H && gu.size(1) == R && gu.size(2) == 2 * H, "ink_gates: weight shapes");
    TORCH_CHECK(X.scalar_type() == at::kFloat && M.scalar_type() == at::kFloat && gd.is_contiguous() && gu.is_contiguous()
                && gd.scalar_type() == at::kFloat && gu.scalar_type() == at::kFloat, "ink_gates: fp32 contiguous");
    const at::cuda::OptionalCUDAGuard guard(X.device());
    auto out = at::empty({G, H}, X.options());
    ink_gates_kernel<<<(unsigned)G, 1024, 2 * H * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        X.data_ptr<float>(), X.stride(0), M.data_ptr<float>(), M.stride(0), gd.data_ptr<float>(), gu.data_ptr<float>(),
        (int)first, H, R, (float)eps, out.data_ptr<float>());
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
    float* __restrict__ gate_out, int H, int D, int R, int E, float qk_clip, float rel_clip, float tau)
{
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
    for (int k = tid; k < H; k += blockDim.x) s += q_in[k] * sink_w[h * H + k];
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
        (float)qk_clip, (float)rel_clip, (float)tau);
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
    __half* __restrict__ kc, __half* __restrict__ vc, long long row, int C, int H, int D, int taps, int record,
    float qk_clip, float eps)
{
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
                 at::Tensor vf, int64_t row, int64_t taps, bool record, double qk_clip, double eps)
{
    const int Hk = (int)kf.size(1), D = (int)kf.size(2), C = Hk * D, H = (int)k_in.numel();
    TORCH_CHECK(D == 128 && Hk * D == k_raw.numel() && v_raw.numel() == C, "ink_kv_glue: D must be 128, C = Hk * D");
    TORCH_CHECK(st.dim() == 2 && st.size(0) == 2 * C && st.stride(1) == 1 && st.scalar_type() == at::kFloat,
                "ink_kv_glue: st (2C, ld) fp32 view");
    TORCH_CHECK(taps + 1 <= INK_MAXK && taps >= 1, "ink_kv_glue: taps");
    TORCH_CHECK(kf.scalar_type() == at::kHalf && vf.scalar_type() == at::kHalf && kf.is_contiguous() && vf.is_contiguous(),
                "ink_kv_glue: fp16 contiguous cache");
    TORCH_CHECK(row >= 0 && (row + 1) * C <= kf.numel(), "ink_kv_glue: row outside the cache");
    const at::cuda::OptionalCUDAGuard guard(st.device());
    const float* pkp = pk.has_value() ? pk->data_ptr<float>() : nullptr;
    const float* pvp = pv.has_value() ? pv->data_ptr<float>() : nullptr;
    dim3 grid(Hk, 2);
    ink_kv_glue_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        k_raw.data_ptr<float>(), v_raw.data_ptr<float>(), st.data_ptr<float>(), st.stride(0),
        k_in.data_ptr<float>(), v_in.data_ptr<float>(), wk.data_ptr<float>(), wv.data_ptr<float>(), pkp, pvp,
        reinterpret_cast<__half*>(kf.data_ptr<at::Half>()), reinterpret_cast<__half*>(vf.data_ptr<at::Half>()),
        row, C, H, D, (int)taps, record ? 1 : 0, (float)qk_clip, (float)eps);
}

// ---------------------------------------------------------------------------------------------------------------------
// Single-query attention over cache rows [row0, row0 + L) (the query sits at the last of them), one block per query
// head: logits = scale * <q, k> + rel[L - 1 - j] (distance < E), softmax with the sink logit (value 0), o = sum p v,
// output (o * gate) as fp16.
// ---------------------------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256) ink_attend_kernel(
    const float* __restrict__ q, const float* __restrict__ rel, const float* __restrict__ sink,
    const float* __restrict__ gate, const __half* __restrict__ kf, const __half* __restrict__ vf,
    long long row0, int L, float sm_scale, int Hq, int Hk, int D, int E, float* __restrict__ logits,
    __half* __restrict__ o_out)
{
    __shared__ float red[32];
    __shared__ float osum[8][128];
    __shared__ float inv_s;
    const int hq = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int G = Hq / Hk, hk = hq / G;
    const long long KS = (long long)Hk * D;
    float qv[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) qv[i] = q[hq * D + lane * 4 + i];
    const float* rel_h = rel + (long long)hq * E;
    float* lg = logits + (long long)hq * L;

    // pass 1: logits
    for (int j0 = warp * 8; j0 < L; j0 += 64) {
        float kk[8][4];
        #pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            const int j = j0 + kb;
            if (j < L) ink_load4h(kf + (row0 + j) * KS + hk * D + lane * 4, kk[kb]);
            else { kk[kb][0] = kk[kb][1] = kk[kb][2] = kk[kb][3] = 0.f; }
        }
        float part[8];
        #pragma unroll
        for (int kb = 0; kb < 8; ++kb)
            part[kb] = qv[0] * kk[kb][0] + qv[1] * kk[kb][1] + qv[2] * kk[kb][2] + qv[3] * kk[kb][3];
        #pragma unroll
        for (int kb = 0; kb < 8; ++kb) part[kb] = ink_warp_sum(part[kb]);
        if (lane == 0) {
            #pragma unroll
            for (int kb = 0; kb < 8; ++kb) {
                const int j = j0 + kb;
                if (j < L) {
                    float v = part[kb] * sm_scale;
                    const int dist = L - 1 - j;
                    if (dist < E) v += rel_h[dist];
                    lg[j] = v;
                }
            }
        }
    }
    __syncthreads();

    // pass 2: softmax over the keys plus the sink
    float m = -INFINITY;
    for (int j = tid; j < L; j += blockDim.x) m = fmaxf(m, lg[j]);
    m = ink_block_max(m, red);
    m = fmaxf(m, sink[hq]);
    float s = 0.f;
    for (int j = tid; j < L; j += blockDim.x) {
        const float e = expf(lg[j] - m);
        lg[j] = e;
        s += e;
    }
    s = ink_block_sum(s, red);
    if (tid == 0) inv_s = 1.f / (s + expf(sink[hq] - m));
    __syncthreads();

    // pass 3: o = sum_j e_j v_j, reduced over the warps
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j0 = warp * 8; j0 < L; j0 += 64) {
        float vv[8][4];
        #pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            const int j = j0 + kb;
            if (j < L) ink_load4h(vf + (row0 + j) * KS + hk * D + lane * 4, vv[kb]);
            else { vv[kb][0] = vv[kb][1] = vv[kb][2] = vv[kb][3] = 0.f; }
        }
        #pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            const int j = j0 + kb;
            const float e = j < L ? lg[j] : 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] += e * vv[kb][i];
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) osum[warp][lane * 4 + i] = acc[i];
    __syncthreads();
    if (tid < D) {
        float o = 0.f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) o += osum[w][tid];
        o = o * inv_s * gate[hq * D + tid];
        o_out[hq * D + tid] = __float2half_rn(o);
    }
}

at::Tensor ink_attend(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                      int64_t row0, int64_t L, double sm_scale, int64_t Hk, int64_t E)
{
    const int Hq = (int)sink.numel(), D = (int)kf.size(2);
    TORCH_CHECK(D == 128 && Hq % Hk == 0 && Hq <= 256, "ink_attend: D must be 128, Hq a multiple of Hk");
    TORCH_CHECK(L >= 1 && row0 >= 0 && (row0 + L) * (int64_t)(Hk * D) <= kf.numel(), "ink_attend: key range");
    TORCH_CHECK(q.numel() == Hq * D && gate.numel() == Hq * D && rel.numel() == Hq * E, "ink_attend: shapes");
    TORCH_CHECK(q.scalar_type() == at::kFloat && rel.scalar_type() == at::kFloat && sink.scalar_type() == at::kFloat
                && gate.scalar_type() == at::kFloat && q.is_contiguous() && rel.is_contiguous()
                && sink.is_contiguous() && gate.is_contiguous(), "ink_attend: fp32 contiguous q/rel/sink/gate");
    TORCH_CHECK(kf.scalar_type() == at::kHalf && vf.scalar_type() == at::kHalf, "ink_attend: fp16 cache");
    const at::cuda::OptionalCUDAGuard guard(q.device());
    auto logits = at::empty({Hq, L}, q.options());
    auto o = at::empty({Hq * D}, q.options().dtype(at::kHalf));
    ink_attend_kernel<<<Hq, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        q.data_ptr<float>(), rel.data_ptr<float>(), sink.data_ptr<float>(), gate.data_ptr<float>(),
        reinterpret_cast<const __half*>(kf.data_ptr<at::Half>()), reinterpret_cast<const __half*>(vf.data_ptr<at::Half>()),
        row0, (int)L, (float)sm_scale, Hq, (int)Hk, D, (int)E, logits.data_ptr<float>(),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()));
    return o;
}
