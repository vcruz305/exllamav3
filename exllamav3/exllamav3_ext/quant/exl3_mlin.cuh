#pragma once

// Fused one-row EXL3 linears: up to EXL3_MLIN_MAX linear layers (each with its own input row, weights,
// scales, output width) computed by ONE kernel launch, spread over all SMs.
//
// Math per linear i (m = 1), identical to the single-linear path of exl3_gemm_kernel:
//   xh      = fp16( H128( fp32( fp16(x * suh) ) ) * 1/sqrt(128) )          (input Hadamard, per 128 inputs)
//   acc[n]  = sum_k xh[k] * W[k][n]                                        (W decoded from the EXL3 trellis, fp32 acc)
//   out[n]  = H128( acc[128-chunk] ) * 1/sqrt(128) * svh[n]                (output Hadamard, per 128 outputs, fp32)
//
// Work: one block per 32 output columns of each linear (grid = sum of n_i / 32), 256 threads (8 warps; warp w
// owns k-slices [w * k/128, (w + 1) * k/128)). Each block computes its own copy of the transformed input in shared
// memory, so no grid-wide barrier is needed. The output Hadamard of a 128-column chunk runs in the last of its 4
// blocks (threadfence reduction on a per-chunk counter, which the last block resets to 0), so one launch is complete.
//
// Graph capturable: no host sync, no allocation, fixed grid for a given batch; all pointers are fixed device
// addresses (per layer). The counters must be zero on the first launch and are left zero by every launch.
//
// C++ entry points (all in exl3_mlin.cu):
//   void exl3_mlin_launch(const ExlMLinBatch& b, cudaStream_t stream);
//   int  exl3_mlin_tensors(in, trellis, suh, svh, out, counters, mcg, mul1);   // tensor front end (for Python)

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#define EXL3_MLIN_MAX 6
#define EXL3_MLIN_MAX_K 4096

struct ExlMLinSpec
{
    const uint16_t* trellis;    // (k/16, n/16, 16*K) uint16, EXL3 layout
    const half* suh;            // (k)
    const half* svh;            // (n)
    const half* in;             // (k) fp16 input row, untransformed
    float* out;                 // (n) fp32 output
    int* counters;              // (n/128) int32, zero on entry
    int k;
    int n;
    int blk_begin;              // first block of this linear: sum of n_j / 32 over earlier linears
};

struct ExlMLinBatch
{
    ExlMLinSpec s[EXL3_MLIN_MAX];
    int count;                  // linears in use
    int total_blocks;           // sum of n_i / 32
    int K;                      // bitrate (integer 1..8)
    int cb;                     // codebook: 0 = 3inst, 1 = mcg, 2 = mul1
};

void exl3_mlin_launch(const ExlMLinBatch& b, cudaStream_t stream);

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
);
