#include <cuda_fp16.h>
#include "exl3_mlin.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"

namespace exl3_mlin_ns
{

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int STAGES = 4;               // cp.async ring depth per warp (stage = two 16x16 tiles)

// One block: 32 output columns of one linear, all k. Shared memory: transformed input, per-warp copy ring, reduction.
template <int K, int CB>
__global__ void __launch_bounds__(THREADS) mlin_kernel(const ExlMLinBatch batch)
{
    __shared__ __align__(16) half xh[EXL3_MLIN_MAX_K];
    __shared__ __align__(16) uint4 ring[WARPS][STAGES][4 * K];     // stage = 2 tiles = 2 * 16 * K uint16 = 4 * K uint4
    __shared__ float red[WARPS][32];
    __shared__ __align__(16) float sh_out[128];
    __shared__ int last_flag;

    const int b = blockIdx.x;
    int li = 0;
    while (li + 1 < batch.count && b >= batch.s[li + 1].blk_begin) ++li;
    const ExlMLinSpec& s = batch.s[li];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int cb_blk = b - s.blk_begin;             // 32-column block within this linear
    const int n16 = s.n >> 4;
    const int ni0 = cb_blk * 2;                     // first 16-column tile of this block
    const int per_warp = (s.k >> 4) / WARPS;        // k-slices (16 rows) per warp
    const int ks0 = warp * per_warp;
    constexpr int TILE_U16 = 16 * K;
    constexpr int CHUNKS = 4 * K;                   // 16-byte chunks per stage (two tiles)

    // Stage the weight slice `s_idx` (warp-local) into ring[warp][stage]; always commits a group
    auto issue = [&] (int s_idx, int stage)
    {
        if (s_idx < per_warp)
        {
            const uint16_t* src = s.trellis + ((size_t) (ks0 + s_idx) * n16 + ni0) * TILE_U16;
            for (int c = lane; c < CHUNKS; c += 32)
                cp_async(&ring[warp][stage][c], src + c * 8);
        }
        cp_async_fence();
    };

    #pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) issue(st, st);

    // Input: x * suh, 128-point Hadamard, scale (same arithmetic as the single-linear path), into shared memory
    for (int c = warp; c < (s.k >> 7); c += WARPS)
        had_hf_r_128_inner<true, false>(s.in + c * 128, xh + c * 128, s.suh + c * 128, 0.088388347648f);
    __syncthreads();

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

        // A fragment: row 0 is the input row (lanes 0..3 hold k = 2q, 2q+1 and 2q+8, 2q+9), other rows zero
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
            dq_dispatch<K, CB, false>(reinterpret_cast<const uint32_t*>(&ring[warp][stage][t * 2 * K]), lane << 3, f0, f1);
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
            red[warp][8 * j + 2 * lane]     = acc[j][0];
            red[warp][8 * j + 2 * lane + 1] = acc[j][1];
        }
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float v = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) v += red[w][threadIdx.x];
        s.out[cb_blk * 32 + threadIdx.x] = v;      // raw sums; output Hadamard below
    }

    // The 128-column chunk is complete when its 4 blocks have written: the last one transforms it in place
    const int chunk = cb_blk >> 2;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd(&s.counters[chunk], 1);
        last_flag = (old == 3);
    }
    __syncthreads();
    if (last_flag)
    {
        __threadfence();
        if (threadIdx.x < 32)
        {
            float4 v = __ldcg(reinterpret_cast<const float4*>(s.out + chunk * 128) + threadIdx.x);
            reinterpret_cast<float4*>(sh_out)[threadIdx.x] = v;
            __syncwarp();
            had_ff_r_128_inner<false, true>(sh_out, s.out + chunk * 128, s.svh + chunk * 128, 0.088388347648f);
        }
        if (threadIdx.x == 0) s.counters[chunk] = 0;
    }
}

template <int K, int CB>
void launch_k(const ExlMLinBatch& b, cudaStream_t stream)
{
    mlin_kernel<K, CB><<<b.total_blocks, THREADS, 0, stream>>>(b);
}

template <int CB>
void launch_cb(const ExlMLinBatch& b, cudaStream_t stream)
{
    switch (b.K)
    {
        case 1: launch_k<1, CB>(b, stream); break;
        case 2: launch_k<2, CB>(b, stream); break;
        case 3: launch_k<3, CB>(b, stream); break;
        case 4: launch_k<4, CB>(b, stream); break;
        case 5: launch_k<5, CB>(b, stream); break;
        case 6: launch_k<6, CB>(b, stream); break;
        case 7: launch_k<7, CB>(b, stream); break;
        case 8: launch_k<8, CB>(b, stream); break;
        default: TORCH_CHECK(false, "exl3_mlin: K must be 1..8");
    }
}

} // namespace exl3_mlin_ns

void exl3_mlin_launch(const ExlMLinBatch& b, cudaStream_t stream)
{
    switch (b.cb)
    {
        case 0: exl3_mlin_ns::launch_cb<0>(b, stream); break;
        case 1: exl3_mlin_ns::launch_cb<1>(b, stream); break;
        case 2: exl3_mlin_ns::launch_cb<2>(b, stream); break;
        default: TORCH_CHECK(false, "exl3_mlin: bad codebook");
    }
    cuda_check(cudaPeekAtLastError());
}

int exl3_mlin_tensors
(
    const std::vector<at::Tensor>& in,
    const std::vector<at::Tensor>& trellis,
    const std::vector<at::Tensor>& suh,
    const std::vector<at::Tensor>& svh,
    const std::vector<at::Tensor>& out,
    const at::Tensor& counters,
    bool mcg,
    bool mul1
)
{
    const int count = (int) in.size();
    TORCH_CHECK(count >= 1 && count <= EXL3_MLIN_MAX, "exl3_mlin: 1..6 linears");
    TORCH_CHECK(trellis.size() == in.size() && suh.size() == in.size() && svh.size() == in.size() && out.size() == in.size(),
                "exl3_mlin: list sizes differ");
    TORCH_CHECK(!(mcg && mul1), "exl3_mlin: both mcg and mul1");
    const at::cuda::OptionalCUDAGuard device_guard(in[0].device());

    ExlMLinBatch b{};
    b.count = count;
    b.K = (int) (trellis[0].size(2) / 16);
    b.cb = mcg ? 1 : (mul1 ? 2 : 0);
    int blocks = 0;
    int ctr = 0;
    for (int i = 0; i < count; ++i)
    {
        const at::Tensor& t = trellis[i];
        TORCH_CHECK(t.dim() == 3 && t.scalar_type() == at::kShort, "exl3_mlin: trellis must be int16 (k/16, n/16, 16K)");
        TORCH_CHECK(t.size(2) == 16 * b.K, "exl3_mlin: all linears must share the bitrate");
        const int k = (int) t.size(0) * 16;
        const int n = (int) t.size(1) * 16;
        TORCH_CHECK(k % 128 == 0 && k <= EXL3_MLIN_MAX_K && (k / 16) % exl3_mlin_ns::WARPS == 0, "exl3_mlin: k must be a multiple of 128, <= 4096");
        TORCH_CHECK(n % 128 == 0, "exl3_mlin: n must be a multiple of 128");
        TORCH_CHECK(in[i].numel() == k && in[i].scalar_type() == at::kHalf && in[i].is_contiguous(), "exl3_mlin: bad input");
        TORCH_CHECK(suh[i].numel() == k && svh[i].numel() == n && suh[i].scalar_type() == at::kHalf && svh[i].scalar_type() == at::kHalf, "exl3_mlin: bad scales");
        TORCH_CHECK(out[i].numel() == n && out[i].scalar_type() == at::kFloat && out[i].is_contiguous(), "exl3_mlin: out must be fp32 (n)");
        ExlMLinSpec& s = b.s[i];
        s.trellis = reinterpret_cast<const uint16_t*>(t.data_ptr());
        s.suh = reinterpret_cast<const half*>(suh[i].data_ptr());
        s.svh = reinterpret_cast<const half*>(svh[i].data_ptr());
        s.in = reinterpret_cast<const half*>(in[i].data_ptr());
        s.out = reinterpret_cast<float*>(out[i].data_ptr());
        s.counters = reinterpret_cast<int*>(counters.data_ptr()) + ctr;
        s.k = k;
        s.n = n;
        s.blk_begin = blocks;
        blocks += n / 32;
        ctr += n / 128;
    }
    TORCH_CHECK(counters.numel() >= ctr && counters.scalar_type() == at::kInt, "exl3_mlin: counters must hold sum(n/128) int32");
    b.total_blocks = blocks;
    exl3_mlin_launch(b, at::cuda::getCurrentCUDAStream().stream());
    return blocks;
}
