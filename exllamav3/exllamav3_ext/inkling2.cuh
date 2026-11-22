#pragma once

#include <vector>
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

// Inkling2Loop single-token decode kernels (see inkling2.cu)
at::Tensor ink_attn_res(std::vector<at::Tensor> srcs, at::Tensor qT, double eps);
at::Tensor ink_gates(at::Tensor X, at::Tensor M, at::Tensor gd, at::Tensor gu, int64_t first, int64_t G, double eps);
at::Tensor ink_sconv(at::Tensor st, at::Tensor x, at::Tensor w, c10::optional<at::Tensor> resid, int64_t taps,
                     bool record);
std::vector<at::Tensor> ink_q_glue(at::Tensor q_f, at::Tensor r_f, at::Tensor g_f, at::Tensor q_in, at::Tensor q_scale,
                                   at::Tensor r_scale, at::Tensor rel_proj, at::Tensor sink_w, double qk_clip,
                                   double rel_clip, double tau);
void ink_kv_glue(at::Tensor k_raw, at::Tensor v_raw, at::Tensor st, at::Tensor k_in, at::Tensor v_in, at::Tensor wk,
                 at::Tensor wv, c10::optional<at::Tensor> pk, c10::optional<at::Tensor> pv, at::Tensor kf,
                 at::Tensor vf, int64_t row, int64_t taps, bool record, double qk_clip, double eps);
// --- device-scalar variants for CUDA graph replay (inkling2_token.cu) ---
constexpr int INK_ACH_H = 128;   // must equal INK_ACH in inkling2.cu
std::vector<at::Tensor> ink_q_glue_dev(at::Tensor q_f, at::Tensor r_f, at::Tensor g_f, at::Tensor q_in,
                                       at::Tensor q_scale, at::Tensor r_scale, at::Tensor rel_proj, at::Tensor sink_w,
                                       double qk_clip, double rel_clip, at::Tensor tau_dev);
void ink_kv_glue_dev(at::Tensor k_raw, at::Tensor v_raw, at::Tensor st, at::Tensor k_in, at::Tensor v_in, at::Tensor wk,
                     at::Tensor wv, c10::optional<at::Tensor> pk, c10::optional<at::Tensor> pv, at::Tensor kf,
                     at::Tensor vf, at::Tensor row_dev, int64_t taps, bool record, double qk_clip, double eps);
at::Tensor ink_attend_dev(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                          at::Tensor par, double sm_scale, int64_t Hk, int64_t E);
at::Tensor ink_attend(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                      int64_t row0, int64_t L, double sm_scale, int64_t Hk, int64_t E);
