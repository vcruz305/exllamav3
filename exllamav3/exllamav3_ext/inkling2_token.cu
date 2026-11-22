// Inkling2Loop whole-token decode (T == 1, one sequence, fp16 paged cache): one C++ call per token.
//
// InkToken::step() issues, on the current stream, the same kernels in the same order as the Python single-token
// path (Inkling2Embedding -> Inkling2Block._forward_t1 x 48 -> Inkling2Final): attention residuals, message gates,
// the EXL3 linears (BC_LinearEXL3::run_alloc), the query/K/V glue, the ring-conv and attention kernels, the
// BlockSparseMLP routing (routing_gemv_f32 + routing_ds3_group) and the expert kernels (CoopMK::run or
// BC_BlockSparseMLP::run_bszN). Python only builds the spec (inkling2_loop.py) and makes one call per token.
//
// Stage 1: scalars (position, cache row, window, tau) are host arguments. Stage 2 replaces them with a device
// parameter buffer so the step can be captured.
#include <cmath>
#include <algorithm>
#include <memory>
#include <vector>

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAEvent.h>

#include "inkling2.cuh"
#include "inkling2_token.cuh"
#include "libtorch/linear.h"
#include "libtorch/blocksparse_mlp.h"
#include "quant/exl3_moe_coopmk.cuh"
#include "routing.cuh"
#include "softcap.cuh"
#include "quant/exl3_mlin.cuh"

struct InkToken;

// attn_res as composed torch ops (inkling2.attn_res): used where the Python path calls the torch version
// (the in-loop messages and the final residual), not the fused ext kernel the block residuals use.
static at::Tensor attn_res_torch(const std::vector<at::Tensor>& srcs, const at::Tensor& qT)
{
    std::vector<at::Tensor> flat;
    for (auto& s : srcs) flat.push_back(s.reshape({-1}));
    at::Tensor stacked = at::stack(flat, 0);                                   // (S, H)
    const int64_t Hd = stacked.size(-1);
    stacked = at::rms_norm(stacked, {Hd}, c10::nullopt, 1e-6);                 // _rms(stacked, 1e-6)
    at::Tensor w = at::matmul(stacked, qT);                                    // (S, nq), pre-scaled query
    w = at::softmax(w, 0);                                                     // over the sources
    return at::matmul(w.transpose(0, 1), stacked);                             // (nq, H)
}

struct InkBlock {
    int64_t layer = 0, instance = 0, Hq = 0, Hk = 0, D = 0, R = 0, E = 0, kvc = 0, W = -1;
    bool l0 = false, full = false, boundary = false, in_loop = false, loop_start = false;
    bool log_scaling = false;
    double alpha = 0.0, nfloor = 1.0, sm_scale = 1.0, qk_clip = 0.0, rel_clip = 0.0, eps = 1e-6;
    std::shared_ptr<BC_LinearEXL3> q, k, v, r, g, o;
    at::Tensor qkv_norm, norm_mlp, q_attn_t, q_mlp_t, gd, gu, msg_t;
    at::Tensor kw, vw, conv_attn, conv_mlp, q_scale, rel_scale, rel_proj, sink_proj;
    c10::optional<at::Tensor> kp, vp;
    at::Tensor kf, vf, ring;                       // cache rows (rows, Hk, D); conv ring (max_batch, ch, ld)
    bool coop = false;
    std::shared_ptr<CoopMK> coopmk;
    std::shared_ptr<BC_BlockSparseMLP> bc_moe;
    // exl3_mlin for q, r, g, k, v (one launch): trellis / suh / svh per linear, fp32 outs, counters
    bool mlin = false;
    std::vector<at::Tensor> mlin_t, mlin_su, mlin_sv, mlin_out;
    at::Tensor mlin_cnt;
    at::Tensor moe_out, gate_t, logits, bias_f32, sel, w;
    double rsf = 1.0;
    int64_t n_group = 1, topk_group = 1;
};

static void mlin5(InkBlock& b, const at::Tensor& qh, const at::Tensor& kh, const at::Tensor& vh)
{
    ExlMLinBatch bt{};
    bt.count = 5;
    bt.K = (int)(b.mlin_t[0].size(2) / 16);
    bt.cb = 2;                                                  // mul1
    const at::Tensor* ins[5] = {&qh, &qh, &qh, &kh, &vh};
    int* cnt = reinterpret_cast<int*>(b.mlin_cnt.data_ptr());
    int blocks = 0, ctr = 0;
    for (int i = 0; i < 5; ++i) {
        const at::Tensor& t = b.mlin_t[i];
        ExlMLinSpec& s = bt.s[i];
        s.trellis = reinterpret_cast<const uint16_t*>(t.data_ptr());
        s.suh = reinterpret_cast<const half*>(b.mlin_su[i].data_ptr());
        s.svh = reinterpret_cast<const half*>(b.mlin_sv[i].data_ptr());
        s.in = reinterpret_cast<const half*>(ins[i]->data_ptr());
        s.out = reinterpret_cast<float*>(b.mlin_out[i].data_ptr());
        s.counters = cnt + ctr;
        s.k = (int)t.size(0) * 16;
        s.n = (int)t.size(1) * 16;
        s.blk_begin = blocks;
        blocks += s.n / 32;
        ctr += s.n / 128;
    }
    bt.total_blocks = blocks;
    exl3_mlin_launch(bt, at::cuda::getCurrentCUDAStream().stream());
}


template <class T>
T get(const py::dict& d, const char* k)
{
    return d[k].cast<T>();
}

template <class T>
c10::optional<T> get_opt(const py::dict& d, const char* k)
{
    py::object o = d[k];
    if (o.is_none()) return c10::nullopt;
    return o.cast<T>();
}

InkBlock parse_block(const py::dict& d)
{
    InkBlock b;
    b.layer = get<int64_t>(d, "layer");
    b.instance = get<int64_t>(d, "instance");
    b.Hq = get<int64_t>(d, "Hq");
    b.Hk = get<int64_t>(d, "Hk");
    b.D = get<int64_t>(d, "D");
    b.R = get<int64_t>(d, "R");
    b.E = get<int64_t>(d, "E");
    b.kvc = get<int64_t>(d, "kvc");
    b.W = get<int64_t>(d, "W");
    b.l0 = get<bool>(d, "l0");
    b.full = get<bool>(d, "full");
    b.boundary = get<bool>(d, "boundary");
    b.in_loop = get<bool>(d, "in_loop");
    b.loop_start = get<bool>(d, "loop_start");
    b.log_scaling = get<bool>(d, "log_scaling");
    b.alpha = get<double>(d, "alpha");
    b.nfloor = get<double>(d, "nfloor");
    b.sm_scale = get<double>(d, "sm_scale");
    b.qk_clip = get<double>(d, "qk_clip");
    b.rel_clip = get<double>(d, "rel_clip");
    b.eps = get<double>(d, "eps");
    b.q = get<std::shared_ptr<BC_LinearEXL3>>(d, "q");
    b.k = get<std::shared_ptr<BC_LinearEXL3>>(d, "k");
    b.v = get<std::shared_ptr<BC_LinearEXL3>>(d, "v");
    b.r = get<std::shared_ptr<BC_LinearEXL3>>(d, "r");
    b.g = get<std::shared_ptr<BC_LinearEXL3>>(d, "g");
    b.o = get<std::shared_ptr<BC_LinearEXL3>>(d, "o");
    b.qkv_norm = get<at::Tensor>(d, "qkv_norm");
    b.norm_mlp = get<at::Tensor>(d, "norm_mlp");
    if (!b.l0) b.q_attn_t = get<at::Tensor>(d, "q_attn_t");
    b.q_mlp_t = get<at::Tensor>(d, "q_mlp_t");
    if (b.in_loop) {
        b.gd = get<at::Tensor>(d, "gd");
        b.gu = get<at::Tensor>(d, "gu");
        b.msg_t = get<at::Tensor>(d, "msg_t");
    }
    b.kw = get<at::Tensor>(d, "kw");
    b.vw = get<at::Tensor>(d, "vw");
    b.kp = get_opt<at::Tensor>(d, "kp");
    b.vp = get_opt<at::Tensor>(d, "vp");
    b.conv_attn = get<at::Tensor>(d, "conv_attn");
    b.conv_mlp = get<at::Tensor>(d, "conv_mlp");
    b.q_scale = get<at::Tensor>(d, "q_scale");
    b.rel_scale = get<at::Tensor>(d, "rel_scale");
    b.rel_proj = get<at::Tensor>(d, "rel_proj");
    b.sink_proj = get<at::Tensor>(d, "sink_proj");
    b.kf = get<at::Tensor>(d, "kf");
    b.vf = get<at::Tensor>(d, "vf");
    b.ring = get<at::Tensor>(d, "ring");
    b.coop = get<bool>(d, "coop");
    b.mlin = get<bool>(d, "mlin");
    if (b.mlin) {
        b.mlin_t = get<std::vector<at::Tensor>>(d, "mlin_t");
        b.mlin_su = get<std::vector<at::Tensor>>(d, "mlin_su");
        b.mlin_sv = get<std::vector<at::Tensor>>(d, "mlin_sv");
        b.mlin_out = get<std::vector<at::Tensor>>(d, "mlin_out");
        b.mlin_cnt = get<at::Tensor>(d, "mlin_cnt");
    }
    if (b.coop) {
        b.coopmk = get<std::shared_ptr<CoopMK>>(d, "coopmk");
    } else {
        b.bc_moe = get<std::shared_ptr<BC_BlockSparseMLP>>(d, "bc_moe");
    }
    b.moe_out = get<at::Tensor>(d, "moe_out");
    b.gate_t = get<at::Tensor>(d, "gate_t");
    b.logits = get<at::Tensor>(d, "logits");
    b.bias_f32 = get<at::Tensor>(d, "bias_f32");
    b.sel = get<at::Tensor>(d, "sel");
    b.w = get<at::Tensor>(d, "w");
    b.rsf = get<double>(d, "rsf");
    b.n_group = get<int64_t>(d, "n_group");
    b.topk_group = get<int64_t>(d, "topk_group");
    return b;
}

struct InkToken {
    int64_t H = 0, V = 0, Vpad = 0, taps = 0, page = 256;
    double eps = 1e-6, eps_emb = 1e-6, eps_fin = 1e-6, final_cap = 0.0;
    at::Tensor emb, emb_norm, fin_q_t, fin_norm, ids;
    std::shared_ptr<BC_LinearEXL3> head;
    std::vector<InkBlock> blocks;
    // device-side per-token state (graph-replayable): dI = [pos, (lo, L, nc) per block, token], dD = tau per block.
    // Written by update() from pinned host slots; a slot is reused only after its copy event has completed.
    int64_t slot_ = 0, NB = 0;
    at::Tensor dI, dD, hI[4], hD[4];
    at::cuda::CUDAEvent evs_[4];
    int hs_ = 0;
    // page table of the sequence: logical page -> cache page, device int32 with capacity cap_ (one entry per cache
    // page). Uploaded by set_table() when it changes; the buffer keeps its address, so a captured step reads it.
    at::Tensor pt_;
    int64_t cap_ = 0;
    std::vector<int64_t> pt_host_;

    explicit InkToken(const py::dict& s)
    {
        H = get<int64_t>(s, "H");
        V = get<int64_t>(s, "V");
        Vpad = get<int64_t>(s, "Vpad");
        taps = get<int64_t>(s, "taps");
        page = get<int64_t>(s, "page");
        eps = get<double>(s, "eps");
        eps_emb = get<double>(s, "eps_emb");
        eps_fin = get<double>(s, "eps_fin");
        final_cap = get<double>(s, "final_cap");
        emb = get<at::Tensor>(s, "emb");
        emb_norm = get<at::Tensor>(s, "emb_norm");
        fin_q_t = get<at::Tensor>(s, "fin_q_t");
        fin_norm = get<at::Tensor>(s, "fin_norm");
        head = get<std::shared_ptr<BC_LinearEXL3>>(s, "head");
        ids = at::zeros({1}, emb.options().dtype(at::kLong));
        for (auto item : get<py::list>(s, "blocks")) {
            blocks.push_back(parse_block(item.cast<py::dict>()));
        }
        NB = (int64_t)blocks.size();
        TORCH_CHECK(page == INK_PAGE, "InkToken: page size must be INK_PAGE");
        cap_ = (int64_t)1 << 40;
        for (auto& b : blocks) cap_ = std::min<int64_t>(cap_, b.kf.size(0) / page);
        pt_ = at::zeros({cap_}, emb.options().dtype(at::kInt));
        dI = at::zeros({1 + 3 * NB + 1}, emb.options().dtype(at::kLong));
        dD = at::zeros({NB}, emb.options().dtype(at::kDouble));
        for (int k = 0; k < 4; ++k) {
            hI[k] = at::empty({1 + 3 * NB + 1}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
            hD[k] = at::empty({NB}, at::TensorOptions().dtype(at::kDouble).pinned_memory(true));
        }
    }

    // Page table of the sequence (logical page i -> cache page pages[i]). Only the pages that hold positions up to the
    // current one are read; the copy happens when the table changes (not per token). Stream-ordered, outside any
    // captured region: a graph replay reads the buffer at its address.
    void set_table(const std::vector<int64_t>& pages)
    {
        TORCH_CHECK((int64_t)pages.size() <= cap_, "InkToken: page table longer than the cache");
        for (int64_t p : pages) TORCH_CHECK(p >= 0 && p < cap_, "InkToken: page index outside the cache");
        if (pages == pt_host_) return;
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        std::vector<int32_t> h(pages.begin(), pages.end());
        pt_.narrow(0, 0, (int64_t)h.size()).copy_(at::tensor(h));
        pt_host_ = pages;
    }

    // Host side of one token: fills the next pinned slot and copies it to the device buffers (async, stream-ordered).
    // Call before step_dev(); no kernel launches here, so it is outside any CUDA graph.
    void update(int64_t token, int64_t pos, int64_t slot)
    {
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        slot_ = slot;
        const int k = hs_;
        hs_ = (hs_ + 1) % 4;
        if (evs_[k].isCreated()) evs_[k].synchronize();
        int64_t* I = hI[k].data_ptr<int64_t>();
        double* Dd = hD[k].data_ptr<double>();
        I[0] = pos;
        for (int64_t i = 0; i < NB; ++i) {
            const InkBlock& b = blocks[i];
            const int64_t lo = (b.W < 0) ? 0 : std::max<int64_t>(0, pos - (b.W - 1));
            const int64_t L = pos + 1 - lo;
            I[1 + 3 * i] = lo;
            I[2 + 3 * i] = L;
            I[3 + 3 * i] = (L + INK_ACH_H - 1) / INK_ACH_H;
            Dd[i] = b.log_scaling ? 1.0 + b.alpha * std::log(std::max(((double)(pos + 1)) / b.nfloor, 1.0)) : 1.0;
        }
        I[1 + 3 * NB] = token;
        dI.copy_(hI[k], /*non_blocking=*/true);
        dD.copy_(hD[k], /*non_blocking=*/true);
        evs_[k].record(at::cuda::getCurrentCUDAStream());
    }

    at::Tensor step(int64_t token, int64_t pos, int64_t slot, bool record)
    {
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        const int64_t Hc = H;

        // Embedding: F.embedding(...).float(), then rms * norm weight
        ids.fill_(token);
        at::Tensor e = at::embedding(emb, ids).to(at::kFloat);
        at::Tensor        x = at::mul(at::rms_norm(e, {Hc}, c10::nullopt, eps_emb), emb_norm);   // (1, H) fp32

        std::vector<at::Tensor> bs;                     // block states, (1, H) fp32 each
        at::Tensor pre_x;
        std::vector<at::Tensor> pre_bs;
        at::Tensor prev_b, prev_o;

        for (auto& b : blocks) {
            at::Tensor msgs;
            bool has_msg = false;
            if (b.in_loop) {
                if (b.loop_start) {
                    if (b.instance == 0) {
                        pre_x = x;
                        pre_bs = bs;
                    } else {
                        prev_b = bs.back();
                        prev_o = x;
                        x = pre_x;
                        bs = pre_bs;
                    }
                }
                if (b.instance > 0) {
                    msgs = attn_res_torch(std::vector<at::Tensor>{prev_b, prev_o}, b.msg_t);   // (4, H)
                    has_msg = true;
                }
            }

            // Attention residual -> sublayer inputs
            at::Tensor hs;
            if (b.l0) {
                hs = x;
            } else {
                std::vector<at::Tensor> srcs = bs;
                srcs.push_back(x);
                hs = ink_attn_res(srcs, b.q_attn_t, b.eps);                                     // (1, H): ext, as _forward_t1
            }
            at::Tensor q_in, k_in, v_in;
            bool same = false;
            if (!has_msg && !b.full) {
                at::Tensor y = at::rms_norm(hs[0], {Hc}, b.qkv_norm[0], b.eps).view({1, Hc});
                q_in = y;
                k_in = y;
                v_in = y;
                same = true;
            } else {
                at::Tensor X = hs.size(0) == 3 ? hs : hs.expand({3, Hc});
                if (has_msg) X = ink_gates(X, msgs.narrow(0, 0, 3), b.gd, b.gu, 0, 3, b.eps);
                at::Tensor y = at::mul(at::rms_norm(X, {Hc}, c10::nullopt, b.eps), b.qkv_norm);   // (3, H)
                q_in = y.narrow(0, 0, 1);
                k_in = y.narrow(0, 1, 1);
                v_in = y.narrow(0, 2, 1);
            }

            // project_t1: linears, then the query and K/V glue kernels
            at::Tensor qh = q_in.to(at::kHalf).contiguous();
            at::Tensor kh = same ? qh : k_in.to(at::kHalf).contiguous();
            at::Tensor vh = same ? qh : v_in.to(at::kHalf).contiguous();
            at::Tensor q_f, r_f, g_f, k_raw, v_raw;
            if (b.mlin) {
                mlin5(b, qh, kh, vh);
                q_f = b.mlin_out[0].reshape(-1);
                r_f = b.mlin_out[1].reshape(-1);
                g_f = b.mlin_out[2].reshape(-1);
                k_raw = b.mlin_out[3].reshape(-1);
                v_raw = b.mlin_out[4].reshape(-1);
            } else {
                q_f = b.q->run_alloc(qh, b.Hq * b.D, true);
                r_f = b.r->run_alloc(qh, b.Hq * b.R, true);
                g_f = b.g->run_alloc(qh, b.Hq * b.D, true);
                k_raw = b.k->run_alloc(kh, b.Hk * b.D, true);
                v_raw = b.v->run_alloc(vh, b.Hk * b.D, true);
            }

            double tau = 1.0;
            if (b.log_scaling) {
                tau = 1.0 + b.alpha * std::log(std::max(((double)(pos + 1)) / b.nfloor, 1.0));
            }
            std::vector<at::Tensor> qg = ink_q_glue(
                q_f.reshape(-1), r_f.reshape(-1), g_f.reshape(-1), q_in.reshape(-1), b.q_scale, b.rel_scale,
                b.rel_proj, b.sink_proj, b.qk_clip, b.rel_clip, tau);
            at::Tensor rs = b.ring.select(0, slot);                                  // (ch, ld)
            ink_kv_glue(k_raw, v_raw, rs.narrow(0, 0, 2 * b.kvc), k_in.reshape(-1), v_in.reshape(-1), b.kw, b.vw,
                        b.kp, b.vp, b.kf, b.vf, pos, pt_, taps, record, b.qk_clip, b.eps);

            // Attention over the logical keys of the window, output projection
            const int64_t lo = (b.W < 0) ? 0 : std::max<int64_t>(0, pos - (b.W - 1));
            const int64_t L = pos + 1 - lo;
            at::Tensor o = ink_attend(qg[0], qg[1], qg[2], qg[3], b.kf, b.vf, pt_, lo, L, b.sm_scale, b.Hk, b.E);
            at::Tensor a = b.o->run_alloc(o.view({1, -1}), H, true);                   // (1, H) fp32

            // Attention output short conv; a block boundary also appends the block state
            at::Tensor residual;
            at::Tensor ra = rs.narrow(0, 2 * b.kvc, H);
            if (b.boundary) {
                bs.push_back(x);
                residual = ink_sconv(ra, a.reshape(-1), b.conv_attn, c10::nullopt, taps, record);
            } else {
                residual = ink_sconv(ra, a.reshape(-1), b.conv_attn, x.reshape(-1), taps, record);
            }

            // MLP input: attention residual, message gate, norm, fp16 rows
            std::vector<at::Tensor> srcs2 = bs;
            srcs2.push_back(residual.reshape(-1));
            at::Tensor m = ink_attn_res(srcs2, b.q_mlp_t, b.eps);                       // (1, H): ext, as _forward_t1
            if (has_msg) m = ink_gates(m, msgs.narrow(0, 3, 1), b.gd, b.gu, 3, 1, b.eps);
            at::Tensor mn = at::rms_norm(m, {Hc}, b.norm_mlp, b.eps).to(at::kHalf).contiguous();   // (1, H)

            // BlockSparseMLP at one row: router, then the expert kernels
            routing_gemv_f32(mn.reshape(-1), b.gate_t, b.logits.view(-1));
            routing_ds3_group(b.logits, c10::optional<at::Tensor>(b.bias_f32), b.sel, b.w, (float)b.rsf, 0,
                              (int)b.n_group, (int)b.topk_group);
            if (b.coop) {
                b.coopmk->run(mn, b.sel, b.w, c10::nullopt, -1, -1, -1);
            } else {
                b.bc_moe->run_bszN(mn, b.sel, b.w);
            }
            at::Tensor mo = b.moe_out.narrow(0, 0, 1);                                  // (1, H) fp32

            // MLP output short conv with the residual
            at::Tensor rm = rs.narrow(0, 2 * b.kvc + H, H);
            x = ink_sconv(rm, mo.reshape(-1), b.conv_mlp, residual.reshape(-1), taps, record).view({1, Hc});
        }

        // Final: attention residual over the block states, norm, head, softcap
        std::vector<at::Tensor> fsrc = bs;
        fsrc.push_back(x);
        at::Tensor fx = attn_res_torch(fsrc, fin_q_t);                                 // (1, H): Inkling2Final's torch attn_res
        at::Tensor xn = at::mul(at::rms_norm(fx, {Hc}, c10::nullopt, eps_fin), fin_norm).to(at::kHalf).contiguous();
        at::Tensor logits = head->run_alloc(xn, Vpad, false);                           // LinearEXL3.forward at one row
        if (final_cap != 0.0) softcap(logits, logits, (float)final_cap);
        return logits.view({1, 1, Vpad}).narrow(-1, 0, V);
    }

    at::Tensor step_dev(bool record)
    {
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        const int64_t Hc = H;

        // Embedding: F.embedding(...).float(), then rms * norm weight
        at::Tensor e = at::embedding(emb, dI.narrow(0, 1 + 3 * NB, 1)).to(at::kFloat);
        at::Tensor        x = at::mul(at::rms_norm(e, {Hc}, c10::nullopt, eps_emb), emb_norm);   // (1, H) fp32

        std::vector<at::Tensor> bs;                     // block states, (1, H) fp32 each
        at::Tensor pre_x;
        std::vector<at::Tensor> pre_bs;
        at::Tensor prev_b, prev_o;

        int64_t bi = -1;
        for (auto& b : blocks) {
            ++bi;
            at::Tensor msgs;
            bool has_msg = false;
            if (b.in_loop) {
                if (b.loop_start) {
                    if (b.instance == 0) {
                        pre_x = x;
                        pre_bs = bs;
                    } else {
                        prev_b = bs.back();
                        prev_o = x;
                        x = pre_x;
                        bs = pre_bs;
                    }
                }
                if (b.instance > 0) {
                    msgs = attn_res_torch(std::vector<at::Tensor>{prev_b, prev_o}, b.msg_t);   // (4, H)
                    has_msg = true;
                }
            }

            // Attention residual -> sublayer inputs
            at::Tensor hs;
            if (b.l0) {
                hs = x;
            } else {
                std::vector<at::Tensor> srcs = bs;
                srcs.push_back(x);
                hs = ink_attn_res(srcs, b.q_attn_t, b.eps);                                     // (1, H): ext, as _forward_t1
            }
            at::Tensor q_in, k_in, v_in;
            bool same = false;
            if (!has_msg && !b.full) {
                at::Tensor y = at::rms_norm(hs[0], {Hc}, b.qkv_norm[0], b.eps).view({1, Hc});
                q_in = y;
                k_in = y;
                v_in = y;
                same = true;
            } else {
                at::Tensor X = hs.size(0) == 3 ? hs : hs.expand({3, Hc});
                if (has_msg) X = ink_gates(X, msgs.narrow(0, 0, 3), b.gd, b.gu, 0, 3, b.eps);
                at::Tensor y = at::mul(at::rms_norm(X, {Hc}, c10::nullopt, b.eps), b.qkv_norm);   // (3, H)
                q_in = y.narrow(0, 0, 1);
                k_in = y.narrow(0, 1, 1);
                v_in = y.narrow(0, 2, 1);
            }

            // project_t1: linears, then the query and K/V glue kernels
            at::Tensor qh = q_in.to(at::kHalf).contiguous();
            at::Tensor kh = same ? qh : k_in.to(at::kHalf).contiguous();
            at::Tensor vh = same ? qh : v_in.to(at::kHalf).contiguous();
            at::Tensor q_f, r_f, g_f, k_raw, v_raw;
            if (b.mlin) {
                mlin5(b, qh, kh, vh);
                q_f = b.mlin_out[0].reshape(-1);
                r_f = b.mlin_out[1].reshape(-1);
                g_f = b.mlin_out[2].reshape(-1);
                k_raw = b.mlin_out[3].reshape(-1);
                v_raw = b.mlin_out[4].reshape(-1);
            } else {
                q_f = b.q->run_alloc(qh, b.Hq * b.D, true);
                r_f = b.r->run_alloc(qh, b.Hq * b.R, true);
                g_f = b.g->run_alloc(qh, b.Hq * b.D, true);
                k_raw = b.k->run_alloc(kh, b.Hk * b.D, true);
                v_raw = b.v->run_alloc(vh, b.Hk * b.D, true);
            }
            std::vector<at::Tensor> qg = ink_q_glue_dev(
                q_f.reshape(-1), r_f.reshape(-1), g_f.reshape(-1), q_in.reshape(-1), b.q_scale, b.rel_scale,
                b.rel_proj, b.sink_proj, b.qk_clip, b.rel_clip, dD.narrow(0, bi, 1));
            at::Tensor rs = b.ring.select(0, slot_);                                  // (ch, ld)
            ink_kv_glue_dev(k_raw, v_raw, rs.narrow(0, 0, 2 * b.kvc), k_in.reshape(-1), v_in.reshape(-1), b.kw, b.vw,
                        b.kp, b.vp, b.kf, b.vf, dI.narrow(0, 0, 1), pt_, taps, record, b.qk_clip, b.eps);

            // Attention over the logical keys of the window, output projection
            
            
            
            at::Tensor o = ink_attend_dev(qg[0], qg[1], qg[2], qg[3], b.kf, b.vf, pt_, dI.narrow(0, 1 + 3 * bi, 3),
                                          b.sm_scale, b.Hk, b.E);
            at::Tensor a = b.o->run_alloc(o.view({1, -1}), H, true);                   // (1, H) fp32

            // Attention output short conv; a block boundary also appends the block state
            at::Tensor residual;
            at::Tensor ra = rs.narrow(0, 2 * b.kvc, H);
            if (b.boundary) {
                bs.push_back(x);
                residual = ink_sconv(ra, a.reshape(-1), b.conv_attn, c10::nullopt, taps, record);
            } else {
                residual = ink_sconv(ra, a.reshape(-1), b.conv_attn, x.reshape(-1), taps, record);
            }

            // MLP input: attention residual, message gate, norm, fp16 rows
            std::vector<at::Tensor> srcs2 = bs;
            srcs2.push_back(residual.reshape(-1));
            at::Tensor m = ink_attn_res(srcs2, b.q_mlp_t, b.eps);                       // (1, H): ext, as _forward_t1
            if (has_msg) m = ink_gates(m, msgs.narrow(0, 3, 1), b.gd, b.gu, 3, 1, b.eps);
            at::Tensor mn = at::rms_norm(m, {Hc}, b.norm_mlp, b.eps).to(at::kHalf).contiguous();   // (1, H)

            // BlockSparseMLP at one row: router, then the expert kernels
            routing_gemv_f32(mn.reshape(-1), b.gate_t, b.logits.view(-1));
            routing_ds3_group(b.logits, c10::optional<at::Tensor>(b.bias_f32), b.sel, b.w, (float)b.rsf, 0,
                              (int)b.n_group, (int)b.topk_group);
            if (b.coop) {
                b.coopmk->run(mn, b.sel, b.w, c10::nullopt, -1, -1, -1);
            } else {
                b.bc_moe->run_bszN(mn, b.sel, b.w);
            }
            at::Tensor mo = b.moe_out.narrow(0, 0, 1);                                  // (1, H) fp32

            // MLP output short conv with the residual
            at::Tensor rm = rs.narrow(0, 2 * b.kvc + H, H);
            x = ink_sconv(rm, mo.reshape(-1), b.conv_mlp, residual.reshape(-1), taps, record).view({1, Hc});
        }

        // Final: attention residual over the block states, norm, head, softcap
        std::vector<at::Tensor> fsrc = bs;
        fsrc.push_back(x);
        at::Tensor fx = attn_res_torch(fsrc, fin_q_t);                                 // (1, H): Inkling2Final's torch attn_res
        at::Tensor xn = at::mul(at::rms_norm(fx, {Hc}, c10::nullopt, eps_fin), fin_norm).to(at::kHalf).contiguous();
        at::Tensor logits = head->run_alloc(xn, Vpad, false);                           // LinearEXL3.forward at one row
        if (final_cap != 0.0) softcap(logits, logits, (float)final_cap);
        return logits.view({1, 1, Vpad}).narrow(-1, 0, V);
    }
};

void ink_token_bind(py::module_& m)
{
    py::class_<InkToken, std::shared_ptr<InkToken>>(m, "InkToken")
        .def(py::init<const py::dict&>())
        .def("set_table", &InkToken::set_table, py::arg("pages"))
        .def("step", &InkToken::step, py::arg("token"), py::arg("pos"), py::arg("slot"), py::arg("record"))
        .def("update", &InkToken::update, py::arg("token"), py::arg("pos"), py::arg("slot"))
        .def("step_dev", &InkToken::step_dev, py::arg("record"));
}
