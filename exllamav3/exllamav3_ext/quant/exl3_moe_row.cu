// Device-only one-row MoE (silu_softcap EXL3 experts). See exl3_moe_row.cuh for the math and the entry points.
// Kernel structure follows exl3_mlin.cu (one block = 32 output columns of one linear, 8 warps split k, row 0 of the
// mma A fragment is the input row, trellis decode via dq_dispatch, cp.async ring), with the per-block bitrate read
// from a device table (switch on K inside the block; block-uniform, no divergence).

#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "exl3_moe_row.cuh"
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"
#include "../routing.cuh"

namespace exl3_moe_row_ns
{

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int STAGES = 4;                   // cp.async ring depth per warp (stage = two 16x16 tiles)
constexpr int MAX_SLOTS = 16;
constexpr float RSQRT128 = 0.088388347648f;
constexpr int ROUTING_ACT_SIGMOID = 0;      // matches routing.cu ROUTING_ACT_SIGMOID

struct RowParams
{
    const int64_t* ptrs;        // (3 linears, 3 parts, E)
    const int32_t* kt;          // (3, E)
    const half* x;              // (H)
    const int64_t* sel;         // (top_k) expert ids
    const half* wts;            // (top_k) routing weights
    float* raw_a;               // (top_k, 2, I) gate / up raw sums, then transformed
    half* act;                  // (top_k, I) activations
    float* raw_b;               // (top_k, H) down raw sums, then weighted and transformed
    float* out;                 // (H)
    int* ctr_a;                 // (top_k, I/128)
    int* ctr_b;                 // (top_k, H/128)
    int* ctr_c;                 // (H/128)
    float cap_g, cap_u;
    int H, I, E, top_k, kmax;
};

// Raw GEMV sums are rounded to fp16 once (the current path's interm_dtype = half contract: gate/up/down outputs are
// stored in half, partials summed in fp32). Enabled by the Python gate for interm_dtype = half layers only.
__device__ __forceinline__ float4 round_h4(float4 v)
{
    v.x = __half2float(__float2half(v.x)); v.y = __half2float(__float2half(v.y));
    v.z = __half2float(__float2half(v.z)); v.w = __half2float(__float2half(v.w));
    return v;
}

// tanh(g / cg) * cg * sigmoid(g) * tanh(u / cu) * cu (the softcap formula of exl3_moe_coop_kernel.cuh act_softcap)
__device__ __forceinline__ float act_softcap_f(float g, float u, float cg, float cu)
{
    return tanhf(g / cg) * cg * (1.0f / (1.0f + __expf(-g))) * (tanhf(u / cu) * cu);
}

// One block: out32[0..31] = raw sums of columns [32 cb_blk, 32 cb_blk + 32) of x(1 x k) @ W(k x n), W decoded from the
// EXL3 trellis with K bits per weight. All 8 warps run; xh is the transformed input in shared memory.
template <int K, int CB>
__device__ __forceinline__ void linear32
(
    const uint16_t* __restrict__ trellis,
    const int k,
    const int n,
    const half* __restrict__ xh,
    uint4* __restrict__ ring,
    const int kmax,
    float* __restrict__ red,
    float* __restrict__ out32,
    const int cb_blk
)
{
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n16 = n >> 4;
    const int ni0 = cb_blk * 2;                 // first 16-column tile of this block
    const int per_warp = (k >> 4) / WARPS;      // k-slices (16 rows) per warp
    const int ks0 = warp * per_warp;
    constexpr int TILE_U16 = 16 * K;
    constexpr int CHUNKS = 4 * K;               // 16-byte chunks per stage (two tiles)
    uint4* wring = ring + (size_t) warp * STAGES * 4 * kmax;

    auto issue = [&] (int s_idx, int stage)
    {
        if (s_idx < per_warp)
        {
            const uint16_t* src = trellis + ((size_t) (ks0 + s_idx) * n16 + ni0) * TILE_U16;
            for (int c = lane; c < CHUNKS; c += 32)
                cp_async(&wring[stage * 4 * kmax + c], src + c * 8);
        }
        cp_async_fence();
    };

    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) issue(st, st);

    FragC acc[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;

    for (int i = 0; i < per_warp; ++i)
    {
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        issue(i + STAGES - 1, (i + STAGES - 1) % STAGES);
        const int stage = i % STAGES;
        const int ks = ks0 + i;

        FragA a;
        a[0] = __half2half2(__ushort_as_half(0));
        a[1] = a[0];
        a[2] = a[0];
        a[3] = a[0];
        if (lane < 4)
        {
            a[0] = *reinterpret_cast<const half2*>(&xh[ks * 16 + 2 * lane]);
            a[2] = *reinterpret_cast<const half2*>(&xh[ks * 16 + 8 + 2 * lane]);
        }

        #pragma unroll
        for (int t = 0; t < 2; ++t)
        {
            FragB f0, f1;
            dq_dispatch<K, CB, false>(reinterpret_cast<const uint32_t*>(wring + stage * 4 * kmax + t * 2 * K), lane << 3, f0, f1);
            ptx_mma_m16n8k16(a, f0, acc[2 * t]);
            ptx_mma_m16n8k16(a, f1, acc[2 * t + 1]);
        }
    }
    cp_async_wait<0>();

    // Row 0 sits in lanes 0..3: column of acc[j][e] is 8 j + 2 lane + e
    if (lane < 4)
    {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            red[warp * 32 + 8 * j + 2 * lane]     = acc[j][0];
            red[warp * 32 + 8 * j + 2 * lane + 1] = acc[j][1];
        }
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) v += red[w * 32 + threadIdx.x];
        out32[threadIdx.x] = v;
    }
}

template <int CB>
__device__ __forceinline__ void linear32_dispatch
(
    int kb,
    const uint16_t* __restrict__ trellis,
    const int k,
    const int n,
    const half* __restrict__ xh,
    uint4* __restrict__ ring,
    const int kmax,
    float* __restrict__ red,
    float* __restrict__ out32,
    const int cb_blk
)
{
    switch (kb)
    {
        case 1: linear32<1, CB>(trellis, k, n, xh, ring, kmax, red, out32, cb_blk); break;
        case 2: linear32<2, CB>(trellis, k, n, xh, ring, kmax, red, out32, cb_blk); break;
        case 3: linear32<3, CB>(trellis, k, n, xh, ring, kmax, red, out32, cb_blk); break;
        case 4: linear32<4, CB>(trellis, k, n, xh, ring, kmax, red, out32, cb_blk); break;
        case 5: linear32<5, CB>(trellis, k, n, xh, ring, kmax, red, out32, cb_blk); break;
        case 6: linear32<6, CB>(trellis, k, n, xh, ring, kmax, red, out32, cb_blk); break;
        case 7: linear32<7, CB>(trellis, k, n, xh, ring, kmax, red, out32, cb_blk); break;
        case 8: linear32<8, CB>(trellis, k, n, xh, ring, kmax, red, out32, cb_blk); break;
        default: break;
    }
}

// Shared memory layout (bytes): xh [max(H, I) half] | red [WARPS*32 float] | sh_in [256 float] | sh_out [256 float] |
// ring [WARPS][STAGES][4*kmax] uint4
__device__ __forceinline__ void smem_split
(
    unsigned char* dsm, int H, int I, int kmax, half*& xh, float*& red, float*& sh_in, float*& sh_out, uint4*& ring
)
{
    const int xhb = (H > I ? H : I) * 2;
    xh = reinterpret_cast<half*>(dsm);
    red = reinterpret_cast<float*>(dsm + xhb);
    sh_in = red + WARPS * 32;
    sh_out = sh_in + 256;
    ring = reinterpret_cast<uint4*>(dsm + xhb + 4096);
}

// Stage A: block = (slot, linear in {gate, up}, 32-column block). Input transform, gate/up linear, and for the last of
// the 8 blocks of a 128-column chunk (4 gate + 4 up): output Hadamard of both, softcapped activation into act[slot].
template <int CB>
__global__ void __launch_bounds__(THREADS) row_stage_a(const RowParams p)
{
    extern __shared__ __align__(16) unsigned char dsm[];
    half* xh; float* red; float* sh_in; float* sh_out; uint4* ring;
    smem_split(dsm, p.H, p.I, p.kmax, xh, red, sh_in, sh_out, ring);
    __shared__ int sh_flag;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = p.I / 32;
    const int slot = blockIdx.x / (2 * nb);
    const int rem = blockIdx.x % (2 * nb);
    const int lin = rem / nb;                   // 0 gate, 1 up
    const int cb = rem % nb;
    const int e = (int) p.sel[slot];
    const int kb = p.kt[lin * p.E + e];
    const uint16_t* trellis = reinterpret_cast<const uint16_t*>((uintptr_t) p.ptrs[(lin * 3 + 0) * p.E + e]);
    const half* suh = reinterpret_cast<const half*>((uintptr_t) p.ptrs[(lin * 3 + 1) * p.E + e]);

    // Input row: xh = H128(fp16(x * suh)) per 128-chunk
    for (int c = warp; c < (p.H >> 7); c += WARPS)
        had_hf_r_128_inner<true, false>(p.x + c * 128, xh + c * 128, suh + c * 128, RSQRT128);
    __syncthreads();

    float* raw = p.raw_a + ((size_t) slot * 2 + lin) * p.I;
    linear32_dispatch<CB>(kb, trellis, p.H, p.I, xh, ring, p.kmax, red, raw + cb * 32, cb);

    // Completion of the 128-column chunk: 8 arrivals (4 gate blocks, 4 up blocks)
    __threadfence();
    __syncthreads();
    const int nch = p.I >> 7;
    const int chunk = cb >> 2;
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd(p.ctr_a + slot * nch + chunk, 1);
        sh_flag = (old == 7);
    }
    __syncthreads();
    if (!sh_flag) return;

    __threadfence();
    if (warp < 2)
    {
        const int l = warp;                     // 0 gate, 1 up
        const float* src = p.raw_a + ((size_t) slot * 2 + l) * p.I + chunk * 128;
        reinterpret_cast<float4*>(sh_in + l * 128)[lane] = round_h4(__ldcg(reinterpret_cast<const float4*>(src) + lane));
        __syncwarp();
        const half* svh = reinterpret_cast<const half*>((uintptr_t) p.ptrs[(l * 3 + 2) * p.E + e]) + chunk * 128;
        had_ff_r_128_inner<false, true>(sh_in + l * 128, sh_out + l * 128, svh, RSQRT128);
    }
    __syncthreads();
    if (threadIdx.x < 128)
    {
        const float g = sh_out[threadIdx.x];
        const float u = sh_out[128 + threadIdx.x];
        p.act[(size_t) slot * p.I + chunk * 128 + threadIdx.x] = __float2half(act_softcap_f(g, u, p.cap_g, p.cap_u));
    }
    if (threadIdx.x == 0) p.ctr_a[slot * nch + chunk] = 0;
}

// Stage B: block = (slot, 32-column block of H). Input transform of act[slot]; down linear; for the last of the 4 blocks
// of a 128-column chunk: output Hadamard, routing weight; for the last of the top_k slots: fixed-order slot sum.
template <int CB>
__global__ void __launch_bounds__(THREADS) row_stage_b(const RowParams p)
{
    extern __shared__ __align__(16) unsigned char dsm[];
    half* xh; float* red; float* sh_in; float* sh_out; uint4* ring;
    smem_split(dsm, p.H, p.I, p.kmax, xh, red, sh_in, sh_out, ring);
    __shared__ int sh_flag;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nb = p.H / 32;
    const int slot = blockIdx.x / nb;
    const int cb = blockIdx.x % nb;
    const int e = (int) p.sel[slot];
    const int kb = p.kt[2 * p.E + e];
    const uint16_t* trellis = reinterpret_cast<const uint16_t*>((uintptr_t) p.ptrs[(2 * 3 + 0) * p.E + e]);
    const half* suh = reinterpret_cast<const half*>((uintptr_t) p.ptrs[(2 * 3 + 1) * p.E + e]);

    const half* a_in = p.act + (size_t) slot * p.I;
    for (int c = warp; c < (p.I >> 7); c += WARPS)
        had_hf_r_128_inner<true, false>(a_in + c * 128, xh + c * 128, suh + c * 128, RSQRT128);
    __syncthreads();

    float* raw = p.raw_b + (size_t) slot * p.H;
    linear32_dispatch<CB>(kb, trellis, p.I, p.H, xh, ring, p.kmax, red, raw + cb * 32, cb);

    __threadfence();
    __syncthreads();
    const int nch = p.H >> 7;
    const int chunk = cb >> 2;
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd(p.ctr_b + slot * nch + chunk, 1);
        sh_flag = (old == 3);
    }
    __syncthreads();
    if (!sh_flag) return;

    __threadfence();
    if (warp == 0)
    {
        const float* src = raw + chunk * 128;
        reinterpret_cast<float4*>(sh_in)[lane] = round_h4(__ldcg(reinterpret_cast<const float4*>(src) + lane));
        __syncwarp();
        const half* svh = reinterpret_cast<const half*>((uintptr_t) p.ptrs[(2 * 3 + 2) * p.E + e]) + chunk * 128;
        had_ff_r_128_inner<false, true>(sh_in, sh_out, svh, RSQRT128);
        __syncwarp();
        const float wgt = __half2float(p.wts[slot]);
        float4 o = reinterpret_cast<float4*>(sh_out)[lane];
        o.x *= wgt; o.y *= wgt; o.z *= wgt; o.w *= wgt;
        reinterpret_cast<float4*>(raw + chunk * 128)[lane] = o;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        p.ctr_b[slot * nch + chunk] = 0;
        const int old = atomicAdd(p.ctr_c + chunk, 1);
        sh_flag = (old == p.top_k - 1);
    }
    __syncthreads();
    if (!sh_flag) return;

    __threadfence();
    if (threadIdx.x < 128)
    {
        float acc = 0.f;
        for (int s = 0; s < p.top_k; ++s)
            acc += __ldcg(p.raw_b + (size_t) s * p.H + chunk * 128 + threadIdx.x);
        p.out[chunk * 128 + threadIdx.x] = acc;
    }
    if (threadIdx.x == 0) p.ctr_c[chunk] = 0;
}

template <int CB>
void launch_stages(const RowParams& p, cudaStream_t stream)
{
    const int smem = (p.H > p.I ? p.H : p.I) * 2 + 4096 + p.kmax * 2048;
    row_stage_a<CB><<<p.top_k * 2 * (p.I / 32), THREADS, smem, stream>>>(p);
    row_stage_b<CB><<<p.top_k * (p.H / 32), THREADS, smem, stream>>>(p);
}

} // namespace exl3_moe_row_ns

using namespace exl3_moe_row_ns;

ExlMoeRow::ExlMoeRow
(
    int H_,
    int I_,
    int E_,
    int top_k_,
    at::Tensor ptrs_,
    at::Tensor kt_,
    int cb_,
    float cap_g_,
    float cap_u_,
    at::Tensor gate_t_,
    c10::optional<at::Tensor> bias_f32_,
    float route_scale_
)
{
    H = H_; I = I_; E = E_; top_k = top_k_; cb = cb_;
    cap_g = cap_g_; cap_u = cap_u_; route_scale = route_scale_;
    TORCH_CHECK(H % 128 == 0 && I % 128 == 0 && (H / 16) % WARPS == 0 && (I / 16) % WARPS == 0,
                "exl3_moe_row: H and I must be multiples of 128");
    TORCH_CHECK(H <= 4096 && I <= 4096, "exl3_moe_row: H, I <= 4096");
    TORCH_CHECK(top_k >= 1 && top_k <= MAX_SLOTS && top_k <= E, "exl3_moe_row: top_k out of range");
    TORCH_CHECK(cb >= 0 && cb <= 2, "exl3_moe_row: bad codebook");
    TORCH_CHECK(cap_g > 0.f && cap_u > 0.f, "exl3_moe_row: caps must be positive");
    TORCH_CHECK(ptrs_.scalar_type() == at::kLong && ptrs_.numel() == 9 * E && ptrs_.is_contiguous(),
                "exl3_moe_row: ptrs must be int64 (9, E)");
    TORCH_CHECK(kt_.scalar_type() == at::kInt && kt_.numel() == 3 * E && kt_.is_contiguous(),
                "exl3_moe_row: kt must be int32 (3, E)");
    TORCH_CHECK(gate_t_.scalar_type() == at::kHalf && gate_t_.dim() == 2 && gate_t_.size(0) == E && gate_t_.size(1) == H
                && gate_t_.is_contiguous(), "exl3_moe_row: gate_t must be half (E, H) contiguous");
    const at::Tensor kt_cpu = kt_.to(at::kCPU);
    int km = 1;
    for (int64_t i = 0; i < kt_cpu.numel(); ++i)
    {
        const int k = kt_cpu.data_ptr<int32_t>()[i];
        TORCH_CHECK(k >= 1 && k <= 8, "exl3_moe_row: bitrates must be integers 1..8");
        if (k > km) km = k;
    }
    kmax = km;
    ptrs = ptrs_; kt = kt_; gate_t = gate_t_;
    bias = bias_f32_.has_value() ? bias_f32_.value() : at::Tensor();
    if (bias.defined()) TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.numel() == E, "exl3_moe_row: bias must be fp32 (E)");

    const auto dev = gate_t_.device();
    const auto f32 = at::TensorOptions().dtype(at::kFloat).device(dev);
    const auto f16 = at::TensorOptions().dtype(at::kHalf).device(dev);
    const auto i32 = at::TensorOptions().dtype(at::kInt).device(dev);
    const auto i64 = at::TensorOptions().dtype(at::kLong).device(dev);
    logits = at::zeros({1, E}, f32);
    sel = at::zeros({1, top_k}, i64);
    w = at::zeros({1, top_k}, f16);
    raw_a = at::zeros({top_k, 2, I}, f32);
    act = at::zeros({top_k, I}, f16);
    raw_b = at::zeros({top_k, H}, f32);
    ctr_a = at::zeros({top_k * (I / 128)}, i32);
    ctr_b = at::zeros({top_k * (H / 128)}, i32);
    ctr_c = at::zeros({H / 128}, i32);
}

void ExlMoeRow::run_experts(const at::Tensor& y, const at::Tensor& sel_, const at::Tensor& w_, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard device_guard(gate_t.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(y.numel() == H && y.scalar_type() == at::kHalf && y.is_contiguous(), "exl3_moe_row: y must be half (H)");
    TORCH_CHECK(sel_.numel() == top_k && sel_.scalar_type() == at::kLong && sel_.is_contiguous(), "exl3_moe_row: sel must be int64 (top_k)");
    TORCH_CHECK(w_.numel() == top_k && w_.scalar_type() == at::kHalf && w_.is_contiguous(), "exl3_moe_row: w must be half (top_k)");
    TORCH_CHECK(out.numel() == H && out.scalar_type() == at::kFloat && out.is_contiguous(), "exl3_moe_row: out must be fp32 (H)");

    RowParams p{};
    p.ptrs = ptrs.data_ptr<int64_t>();
    p.kt = kt.data_ptr<int32_t>();
    p.x = reinterpret_cast<const half*>(y.data_ptr());
    p.sel = sel_.data_ptr<int64_t>();
    p.wts = reinterpret_cast<const half*>(w_.data_ptr());
    p.raw_a = raw_a.data_ptr<float>();
    p.act = reinterpret_cast<half*>(act.data_ptr());
    p.raw_b = raw_b.data_ptr<float>();
    p.out = out.data_ptr<float>();
    p.ctr_a = ctr_a.data_ptr<int32_t>();
    p.ctr_b = ctr_b.data_ptr<int32_t>();
    p.ctr_c = ctr_c.data_ptr<int32_t>();
    p.cap_g = cap_g;
    p.cap_u = cap_u;
    p.H = H; p.I = I; p.E = E; p.top_k = top_k; p.kmax = kmax;

    switch (cb)
    {
        case 0: launch_stages<0>(p, stream); break;
        case 1: launch_stages<1>(p, stream); break;
        case 2: launch_stages<2>(p, stream); break;
        default: TORCH_CHECK(false, "exl3_moe_row: bad codebook");
    }
    cuda_check(cudaPeekAtLastError());
}

void ExlMoeRow::forward(const at::Tensor& y, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard device_guard(gate_t.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(y.numel() == H && y.scalar_type() == at::kHalf && y.is_contiguous(), "exl3_moe_row: y must be half (H)");
    // Router: fp32 logits (fixed-order GEMV) and top-k with sigmoid scores, selection bias and normalised weights:
    // the same two kernels the bsz = 1 routing_ds3 path runs (n_group = 1, topk_group = 1)
    routing_gemv_f32(y.view({H}), gate_t, logits.view({E}));
    routing_ds3_group(logits, bias.defined() ? c10::optional<at::Tensor>(bias) : c10::nullopt,
                      sel, w, route_scale, ROUTING_ACT_SIGMOID, 1, 1);
    run_experts(y, sel.view({top_k}), w.view({top_k}), out);
}
