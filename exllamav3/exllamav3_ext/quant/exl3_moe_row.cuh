#pragma once

// Device-only one-row MoE for silu_softcap EXL3 experts (Inkling-2): router + top-k + softcapped SwiGLU experts +
// weighted routed sum, for ONE row (bsz = 1), with per-expert bitrates read on the device (mixed-K and uniform-K
// layers share the same code). Launch-only: no host synchronization, no allocation, no host-side expert ids, so the
// whole call is CUDA-graph capturable. All buffers are allocated once in the constructor.
//
// Math (per row y, H = hidden, I = intermediate, slot s = 0..top_k-1, expert e = sel[s]):
//   logits   = routing_gemv_f32(y, gate_t)                         (fp32, existing kernel, bit-identical routing)
//   sel, w   = routing_ds3_group(logits, bias_f32, n_group=1, topk_group=1, scale = route_scale)   (existing kernel)
//   gate/up  (per linear, same as exl3_mlin): xh = H128(fp16(y * suh)) ; raw = xh @ W ; g = H128(raw) * svh
//   a[s]     = tanh(g/cg)*cg * sigmoid(g) * tanh(u/cu)*cu           (fp16 intermediate, as the coop path)
//   down     : xa = H128(fp16(a * suh_d)) ; raw_d = xa @ W_d ; y_s = w_s * H128(raw_d) * svh_d
//   out[c]   = sum_s y_s[c]  (fixed slot order: deterministic)
//
// Launches (per call): routing gemv, routing top-k, stage A (gate and up, 2 * I/32 blocks per slot, all experts),
// stage B (down, H/32 blocks per slot, then per-chunk tails and the fixed-order slot sum). The stage boundaries
// are kernel boundaries; within stage A the 128-column chunk tail (H128 + activation) and within stage B the
// per-chunk tail (H128, weight, slot sum) run in the last block to arrive (threadfence reduction; counters reset
// to zero by that block, so the next call starts clean).
//
// Bitrate per expert and linear: kt[L][e] (L: 0 gate, 1 up, 2 down; integer 1..8). Pointer table ptrs[(L*3+p)*E+e]
// (int64 device addresses; p: 0 trellis (k/16, n/16, 16K) int16, 1 suh (k) fp16, 2 svh (n) fp16).
// Codebook cb: 0 = 3inst, 1 = mcg, 2 = mul1 (one per layer, as the other EXL3 MoE kernels).
//
// C++ entry points (exl3_moe_row.cu), Python class ext.MoeRow (exl3_moe_row_py.cpp):
//   ExlMoeRow(H, I, E, top_k, ptrs, kt, cb, cap_g, cap_u, gate_t, bias_f32|None, route_scale)
//   forward(y /*half (H)*/, out /*fp32 (H)*/)                          routing + experts, one call
//   run_experts(y, sel /*int64 (top_k)*/, w /*half (top_k)*/, out)     experts only, device-resident selection

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <cstdint>

namespace pybind11 { class module_; }

class ExlMoeRow
{
public:
    ExlMoeRow
    (
        int H,
        int I,
        int E,
        int top_k,
        at::Tensor ptrs,            // int64 (9, E), device addresses as above
        at::Tensor kt,              // int32 (3, E), bitrates
        int cb,
        float cap_g,
        float cap_u,
        at::Tensor gate_t,          // half (E, H): router weight, transposed (as RoutingCFG.gate_tensor_t)
        c10::optional<at::Tensor> bias_f32,   // float (E): selection bias (e_score_bias_f32)
        float route_scale
    );

    // y: half (H) contiguous row; out: fp32 (H), written (not accumulated)
    void forward(const at::Tensor& y, at::Tensor out);

    // y: half (H); sel: int64 (top_k); w: half (top_k); out: fp32 (H)
    void run_experts(const at::Tensor& y, const at::Tensor& sel, const at::Tensor& w, at::Tensor out);

    int H, I, E, top_k, kmax, cb;
    float cap_g, cap_u, route_scale;
    at::Tensor ptrs, kt, gate_t, bias;
    at::Tensor logits, sel, w;                  // routing outputs (device), used by forward()
    at::Tensor raw_a, act, raw_b;               // scratch: stage A raw sums, activations, stage B raw / weighted
    at::Tensor ctr_a, ctr_b, ctr_c;            // int32 completion counters, zero between calls
};

// Registers class ext.MoeRow (called from bindings.cpp)
void exl3_moe_row_bind(pybind11::module_& m);
