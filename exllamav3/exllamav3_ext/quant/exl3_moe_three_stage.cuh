#pragma once

#include "exl3_moe_kernel.cuh"

// Three ordinary stream-ordered stages, not a cooperative launch/last-arrival scheme.
// Each GEMM CTA owns one complete-K 256-column tile. Its two 128-column Hadamard
// groups never cross CTA boundaries. grid.y MUST be 1 (blockIdx.y == 0): the
// unchanged Hadamard helpers add blockIdx.y to scale indices internally.
// Preserve all incumbent half stores, FP32 MMA and warp/CTA reduction arithmetic.
// Stage 0: gather/input Had. Stage 1: gate + up + GUAD. Stage 2: down + output Had.
template<int STAGE>
__global__ __launch_bounds__(512)
void exl3_moe_three_stage_kernel(EXL3_MOE_MIXEDK_KERNEL_ARGS)
{
    __shared__ int span[2];
    if (threadIdx.x == 0)
    {
        int start = 0, active = 0;
        span[0] = -1;
        for (int e = 0; e < num_experts; ++e)
        {
            const int count = expert_count[e];
            if (count > 0 && count <= max_tokens_per_expert && count >= count_lo && count <= count_hi)
            {
                if (active++ == (int) blockIdx.z)
                {
                    span[0] = e;
                    span[1] = start;
                    break;
                }
            }
            start += count;
        }
    }
    __syncthreads();
    const int expert = span[0];
    if (expert < 0) return;
    const int rows = expert_count[expert];
    if (rows == 0 || rows > max_tokens_per_expert || rows < count_lo || rows > count_hi) return;
    const int start = span[1];
    half* g = temp_state_g + (size_t) start * hidden_dim;
    half* u = temp_state_u + (size_t) start * hidden_dim;
    half* ig = temp_intermediate_g + (size_t) start * intermediate_dim;
    half* iu = temp_intermediate_u + (size_t) start * intermediate_dim;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    constexpr float scale = 0.088388347648f;

    if constexpr (STAGE == 0)
    {
        const int chunks = hidden_dim / 128;
        for (int w = warp; w < rows * chunks; w += warps)
        {
            const int row = w / chunks;
            const int col = (w % chunks) * 128;
            const half* x = hidden_state + token_sorted[start + row] * hidden_dim + col;
            had_hf_r_128_inner<true, false>(x, g + w * 128, gate_suh[expert] + col, scale);
            had_hf_r_128_inner<true, false>(x, u + w * 128, up_suh[expert] + col, scale);
        }
    }
    if constexpr (STAGE == 1 || STAGE == 2)
    {
        // Duplicate synthetic routes can exceed eight rows; do not truncate them.
        for (int row = 0; row < rows; row += 16)
        {
            if constexpr (STAGE == 1)
            {
                moe_gemm_tile<0, 2, 16, 256, 3, 3, true>
                    (g + (size_t) row * hidden_dim, gate_trellis[expert],
                     ig + (size_t) row * intermediate_dim, rows - row,
                     hidden_dim, intermediate_dim, nullptr, K_gate_arr[expert]);
                // Full-K helper returns after a cp.async drain + CTA barrier.
                // Up has independent pointers/K; never assume gate K == up K.
                moe_gemm_tile<0, 2, 16, 256, 3, 3, true>
                    (u + (size_t) row * hidden_dim, up_trellis[expert],
                     iu + (size_t) row * intermediate_dim, rows - row,
                     hidden_dim, intermediate_dim, nullptr, K_up_arr[expert]);
            }
            else
            {
                moe_gemm_tile<0, 2, 16, 256, 3, 3, true>
                    (ig + (size_t) row * intermediate_dim, down_trellis[expert],
                     g + (size_t) row * hidden_dim, rows - row,
                     intermediate_dim, hidden_dim, nullptr, K_down_arr[expert]);
            }
            // Full-K's write_sum_gl + CTA barrier publishes HALF output to these
            // same-CTA readers. No other CTA's raw GEMM output is consumed here.
            for (int w = warp; w < MIN(16, rows - row) * 2; w += warps)
            {
                const int r = row + w / 2;
                const int col = blockIdx.x * 256 + (w % 2) * 128;
                if constexpr (STAGE == 1)
                {
                    const size_t off = (size_t) r * intermediate_dim + col;
                    had_hf_r_128_guad_inner(ig + off, iu + off, ig + off,
                        gate_svh[expert] + col, up_svh[expert] + col, down_suh[expert] + col,
                        scale, act_limit, act_function);
                }
                else
                {
                    const size_t off = (size_t) r * hidden_dim + col;
                    float* dst = output_scratch + (fused_base[expert] + r) * hidden_dim + col;
                    had_hf_r_128_d_inner<false>(g + off, dst, down_svh[expert] + col,
                        scale * __half2float(weight_sorted[start + r]));
                    // The output helper uses 128 shared floats per physical warp.
                    // Complete its reads before reusing the same warp's shared span.
                    __syncwarp();
                }
            }
            // Uniform CTA barrier: every warp finishes the epilogue before the
            // next row chunk reuses GEMM/shared storage (including inactive warps).
            __syncthreads();
        }
    }
}
