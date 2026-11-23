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
// Paged single-token cache addressing. A logical position p of the sequence lives at cache row
// pt[p / INK_PAGE] * INK_PAGE + p % INK_PAGE, where pt is the device int32 page table of the sequence (any page
// order). kf / vf: (rows, Hk, D) fp16 cache views.
constexpr int INK_PAGE = 256;    // must equal PAGE_SIZE (exllamav3/constants.py)
constexpr int INK_ACH_H = 128;   // keys per attention chunk; must equal INK_ACH in inkling2.cu
constexpr int INK_GRID = 64;     // attention blocks per kv head (grid-stride over the chunks)
constexpr int INK_NSTORE = 256;  // chunk slots of the device-side split: at most 32768 keys
void ink_kv_glue(at::Tensor k_raw, at::Tensor v_raw, at::Tensor st, at::Tensor k_in, at::Tensor v_in, at::Tensor wk,
                 at::Tensor wv, c10::optional<at::Tensor> pk, c10::optional<at::Tensor> pv, at::Tensor kf,
                 at::Tensor vf, int64_t pos, at::Tensor pt, int64_t taps, bool record, double qk_clip, double eps);
// --- device-scalar variants for CUDA graph replay (inkling2_token.cu) ---
std::vector<at::Tensor> ink_q_glue_dev(at::Tensor q_f, at::Tensor r_f, at::Tensor g_f, at::Tensor q_in,
                                       at::Tensor q_scale, at::Tensor r_scale, at::Tensor rel_proj, at::Tensor sink_w,
                                       double qk_clip, double rel_clip, at::Tensor tau_dev);
void ink_kv_glue_dev(at::Tensor k_raw, at::Tensor v_raw, at::Tensor st, at::Tensor k_in, at::Tensor v_in, at::Tensor wk,
                     at::Tensor wv, c10::optional<at::Tensor> pk, c10::optional<at::Tensor> pv, at::Tensor kf,
                     at::Tensor vf, at::Tensor pos_dev, at::Tensor pt, int64_t taps, bool record, double qk_clip,
                     double eps);
// Attention over logical keys [lo, lo + L) of the sequence (the query is the last of them); par = [lo, L, nc] device
at::Tensor ink_attend_dev(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                          at::Tensor pt, at::Tensor par, double sm_scale, int64_t Hk, int64_t E);
at::Tensor ink_attend(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                      at::Tensor pt, int64_t lo, int64_t L, double sm_scale, int64_t Hk, int64_t E);

// --- row-batched variants (T rows of one sequence per launch; InkToken::step_rows2) ---
at::Tensor ink_attn_res_rows(std::vector<at::Tensor> srcs, at::Tensor qT, double eps);
at::Tensor ink_gates_rows(at::Tensor X, at::Tensor msgs, at::Tensor gd, at::Tensor gu, int64_t first, int64_t G, double eps);
at::Tensor ink_sconv_rows(at::Tensor st, at::Tensor x, at::Tensor w, c10::optional<at::Tensor> resid, int64_t taps,
                          bool record);
void ink_hist_rows(at::Tensor st, at::Tensor x, int64_t taps, bool record);
void ink_rewind_all(std::vector<at::Tensor> states, int64_t slot, int64_t taps, int64_t num_tokens);
std::vector<at::Tensor> ink_q_glue_rows(at::Tensor q_f, at::Tensor r_f, at::Tensor g_f, at::Tensor q_in, at::Tensor q_scale,
                                        at::Tensor r_scale, at::Tensor rel_proj, at::Tensor sink_w, double qk_clip,
                                        double rel_clip, at::Tensor tau_dev, int64_t tst);
void ink_kv_glue_rows(at::Tensor k_raw, at::Tensor v_raw, at::Tensor st, at::Tensor k_in, at::Tensor v_in, at::Tensor wk,
                      at::Tensor wv, c10::optional<at::Tensor> pk, c10::optional<at::Tensor> pv, at::Tensor kf,
                      at::Tensor vf, at::Tensor pos_dev, int64_t pst, at::Tensor pt, int64_t taps, double qk_clip,
                      double eps);
at::Tensor ink_attend_rows_dev(at::Tensor q, at::Tensor rel, at::Tensor sink, at::Tensor gate, at::Tensor kf, at::Tensor vf,
                               at::Tensor pt, at::Tensor par, int64_t pst, double sm_scale, int64_t Hk, int64_t E);
