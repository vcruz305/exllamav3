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
#include <cstdlib>
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

// attn_res_torch over T rows: srcs (T, H) each -> (T, nq, H)
static at::Tensor attn_res_torch_rows(const std::vector<at::Tensor>& srcs, const at::Tensor& qT)
{
    at::Tensor stacked = at::stack(srcs, 1);                                   // (T, S, H)
    const int64_t Hd = stacked.size(-1);
    stacked = at::rms_norm(stacked, {Hd}, c10::nullopt, 1e-6);
    at::Tensor w = at::softmax(at::matmul(stacked, qT), 1);                    // (T, S, nq), over the sources
    return at::matmul(w.transpose(1, 2), stacked).contiguous();                // (T, nq, H)
}

// debugging: block inputs of the token or row step (InkToken::dump_on)
static std::vector<at::Tensor> g_dump;
static bool g_dump_on = false;

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
    std::vector<at::Tensor> mlin_xh, mlin_rout;                 // rows launch: (rows_max, k) fp16, (rows_max, n) fp32
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

static void mlin5_rows(InkBlock& b, const at::Tensor& qh, const at::Tensor& kh, const at::Tensor& vh, int64_t T)
{
    ExlMLinBatch bt{};
    bt.count = 5;
    bt.K = (int)(b.mlin_t[0].size(2) / 16);
    bt.cb = 2;                                                  // mul1
    bt.rows = (int)T;
    const at::Tensor* ins[5] = {&qh, &qh, &qh, &kh, &vh};
    int* cnt = reinterpret_cast<int*>(b.mlin_cnt.data_ptr());
    int blocks = 0, ctr = 0, rot = 0;
    for (int i = 0; i < 5; ++i) {
        const at::Tensor& t = b.mlin_t[i];
        ExlMLinSpec& s = bt.s[i];
        s.trellis = reinterpret_cast<const uint16_t*>(t.data_ptr());
        s.suh = reinterpret_cast<const half*>(b.mlin_su[i].data_ptr());
        s.svh = reinterpret_cast<const half*>(b.mlin_sv[i].data_ptr());
        s.k = (int)t.size(0) * 16;
        s.n = (int)t.size(1) * 16;
        TORCH_CHECK(ins[i]->dim() == 2 && ins[i]->size(0) == T && ins[i]->size(1) == s.k && ins[i]->is_contiguous()
                    && ins[i]->scalar_type() == at::kHalf, "InkToken: mlin rows input");
        TORCH_CHECK(b.mlin_xh[i].size(0) >= T && b.mlin_xh[i].size(1) == s.k && b.mlin_rout[i].size(0) >= T
                    && b.mlin_rout[i].size(1) == s.n, "InkToken: mlin rows scratch");
        s.in = reinterpret_cast<const half*>(ins[i]->data_ptr());
        s.xh = reinterpret_cast<half*>(b.mlin_xh[i].data_ptr());
        s.out = reinterpret_cast<float*>(b.mlin_rout[i].data_ptr());
        s.counters = cnt + ctr;
        s.blk_begin = blocks;
        s.rot_begin = rot;
        blocks += s.n / 32;
        ctr += s.n / 128;
        rot += (int)T * (s.k / 128);
    }
    bt.total_blocks = blocks;
    bt.rot_items = rot;
    exl3_mlin_rows_launch(bt, at::cuda::getCurrentCUDAStream().stream());
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
        b.mlin_xh = get<std::vector<at::Tensor>>(d, "mlin_xh");
        b.mlin_rout = get<std::vector<at::Tensor>>(d, "mlin_rout");
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

    // drafter taps (milestone 3): tapb_[bi] = tapbuf slot of block call bi (-1: none), tapf_ = slot of the normed
    // final state. tapbuf is fp16 (4, TMAX, H); row 0 holds the T = 1 tap. Set by set_taps() before capture.
    static constexpr int64_t TMAX = 9;
    at::Tensor tapbuf;
    std::vector<int64_t> tapb_;
    int64_t tapf_ = -1;
    // T-row path (milestone 4): rows of the device parameter buffers are S = 2 + 3 NB int64 (pos, windows, token)
    // and NB doubles (tau); row r of a forward starts at r * S / r * NB. Tcur_ = rows of the staged forward.
    int64_t Tcur_ = 1;
    // batched rows (step_rows2) or the sequential rows (step_row per row; no record support)
    bool batched_ = true;
    // independent sequences (step_seqs): the staged rows are one token of seq_slots_.size() sequences. Row r reads
    // page-table row r (pt_ holds TMAX tables of cap_ entries) through positions offset by r * cap_ pages.
    bool seqs_ = false;
    std::vector<int64_t> seq_slots_;
    std::vector<std::vector<int64_t>> tabs_host_;

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
        pt_ = at::zeros({cap_ * TMAX}, emb.options().dtype(at::kInt));
        dI = at::zeros({TMAX * (2 + 3 * NB)}, emb.options().dtype(at::kLong));
        dD = at::zeros({TMAX * NB}, emb.options().dtype(at::kDouble));
        for (int k = 0; k < 4; ++k) {
            hI[k] = at::empty({TMAX * (2 + 3 * NB)}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
            hD[k] = at::empty({TMAX * NB}, at::TensorOptions().dtype(at::kDouble).pinned_memory(true));
        }
        tapbuf = at::zeros({4, TMAX, H}, emb.options().dtype(at::kHalf));
        tapb_.assign(NB, -1);
    }

    // Host side of a forward over T = toks.size() rows (one sequence, positions pos0 .. pos0 + T - 1): fills a pinned
    // slot row by row and copies it to the device (stream-ordered, no kernel launch: outside any graph capture).
    void update_rows(const std::vector<int64_t>& toks, int64_t pos0, int64_t slot)
    {
        const int64_t T = (int64_t)toks.size();
        TORCH_CHECK(T >= 1 && T <= TMAX, "InkToken: 1..9 rows per forward");
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        slot_ = slot;
        Tcur_ = T;
        seqs_ = false;
        const int k = hs_;
        hs_ = (hs_ + 1) % 4;
        if (evs_[k].isCreated()) evs_[k].synchronize();
        const int64_t S = 2 + 3 * NB;
        int64_t* I = hI[k].data_ptr<int64_t>();
        double* Dd = hD[k].data_ptr<double>();
        for (int64_t r = 0; r < T; ++r) {
            int64_t* Ir = I + r * S;
            double* Dr = Dd + r * NB;
            const int64_t pos = pos0 + r;
            Ir[0] = pos;
            for (int64_t i = 0; i < NB; ++i) {
                const InkBlock& b = blocks[i];
                const int64_t lo = (b.W < 0) ? 0 : std::max<int64_t>(0, pos - (b.W - 1));
                const int64_t L = pos + 1 - lo;
                Ir[1 + 3 * i] = lo;
                Ir[2 + 3 * i] = L;
                Ir[3 + 3 * i] = (L + INK_ACH_H - 1) / INK_ACH_H;
                Dr[i] = b.log_scaling ? 1.0 + b.alpha * std::log(std::max(((double)(pos + 1)) / b.nfloor, 1.0)) : 1.0;
            }
            Ir[1 + 3 * NB] = toks[r];
        }
        dI.narrow(0, 0, T * S).copy_(hI[k].narrow(0, 0, T * S), /*non_blocking=*/true);
        dD.narrow(0, 0, T * NB).copy_(hD[k].narrow(0, 0, T * NB), /*non_blocking=*/true);
        evs_[k].record(at::cuda::getCurrentCUDAStream());
    }

    // Tap export configuration: tapb[bi] = slot (0..3) of block call bi or -1; tapf = slot of the normed final state
    // or -1. Changes the captured graph (the caller keys its graphs on it).
    void set_taps(const std::vector<int64_t>& tapb, int64_t tapf)
    {
        TORCH_CHECK((int64_t)tapb.size() == NB, "InkToken: one tap entry per block call");
        for (int64_t t : tapb) TORCH_CHECK(t >= -1 && t < 4, "InkToken: tap slot");
        TORCH_CHECK(tapf >= -1 && tapf < 4, "InkToken: tap slot");
        tapb_ = tapb;
        tapf_ = tapf;
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
        tabs_host_.clear();
    }

    // Page tables of the sequences of a step_seqs forward: row r of pt_ (cap_ entries each) = tabs[r]
    void set_tables(const std::vector<std::vector<int64_t>>& tabs)
    {
        TORCH_CHECK(tabs.size() >= 1 && (int64_t)tabs.size() <= TMAX, "InkToken: 1..9 sequences");
        if (tabs == tabs_host_) return;
        std::vector<int32_t> h(tabs.size() * cap_, 0);
        for (size_t r = 0; r < tabs.size(); ++r) {
            TORCH_CHECK((int64_t)tabs[r].size() <= cap_, "InkToken: page table longer than the cache");
            for (size_t i = 0; i < tabs[r].size(); ++i) {
                TORCH_CHECK(tabs[r][i] >= 0 && tabs[r][i] < cap_, "InkToken: page index outside the cache");
                h[r * cap_ + i] = (int32_t)tabs[r][i];
            }
        }
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        pt_.narrow(0, 0, (int64_t)h.size()).copy_(at::tensor(h));
        tabs_host_ = tabs;
        pt_host_.assign(1, -1);                 // the single-sequence table is gone
    }

    // Host side of a forward of one token for each of B sequences: row r = (toks[r], poss[r], ring slot slots[r])
    void update_seqs(const std::vector<int64_t>& toks, const std::vector<int64_t>& poss, const std::vector<int64_t>& slots)
    {
        const int64_t T = (int64_t)toks.size();
        TORCH_CHECK(T >= 1 && T <= TMAX && (int64_t)poss.size() == T && (int64_t)slots.size() == T,
                    "InkToken: 1..9 sequences, one position and slot each");
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        slot_ = slots[0];
        Tcur_ = T;
        seqs_ = true;
        seq_slots_ = slots;
        const int k = hs_;
        hs_ = (hs_ + 1) % 4;
        if (evs_[k].isCreated()) evs_[k].synchronize();
        const int64_t S = 2 + 3 * NB;
        int64_t* I = hI[k].data_ptr<int64_t>();
        double* Dd = hD[k].data_ptr<double>();
        for (int64_t r = 0; r < T; ++r) {
            int64_t* Ir = I + r * S;
            double* Dr = Dd + r * NB;
            const int64_t pos = poss[r], off = r * cap_ * page;
            Ir[0] = pos + off;
            for (int64_t i = 0; i < NB; ++i) {
                const InkBlock& b = blocks[i];
                const int64_t lo = (b.W < 0) ? 0 : std::max<int64_t>(0, pos - (b.W - 1));
                const int64_t L = pos + 1 - lo;
                Ir[1 + 3 * i] = lo + off;
                Ir[2 + 3 * i] = L;
                Ir[3 + 3 * i] = (L + INK_ACH_H - 1) / INK_ACH_H;
                Dr[i] = b.log_scaling ? 1.0 + b.alpha * std::log(std::max(((double)(pos + 1)) / b.nfloor, 1.0)) : 1.0;
            }
            Ir[1 + 3 * NB] = toks[r];
        }
        dI.narrow(0, 0, T * S).copy_(hI[k].narrow(0, 0, T * S), /*non_blocking=*/true);
        dD.narrow(0, 0, T * NB).copy_(hD[k].narrow(0, 0, T * NB), /*non_blocking=*/true);
        evs_[k].record(at::cuda::getCurrentCUDAStream());
    }

    // One token of each staged sequence (update_seqs): (B, 1, V)
    at::Tensor step_seqs()
    {
        TORCH_CHECK(seqs_, "InkToken: step_seqs follows update_seqs");
        return step_rows3(false).transpose(0, 1);
    }

    // Host side of one token: fills the next pinned slot and copies it to the device buffers (async, stream-ordered).
    // Call before step_dev(); no kernel launches here, so it is outside any CUDA graph.
    void update(int64_t token, int64_t pos, int64_t slot)
    {
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        slot_ = slot;
        seqs_ = false;
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

    // One token of the staged forward: row r of the device parameter buffers (r = 0 for the T = 1 step).
    at::Tensor step_row(int64_t r, bool record)
    {
        const int64_t S = 2 + 3 * NB;
        const int64_t base = r * S, dbase = r * NB;
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        const int64_t Hc = H;

        // Embedding: F.embedding(...).float(), then rms * norm weight
        at::Tensor e = at::embedding(emb, dI.narrow(0, base + 1 + 3 * NB, 1)).to(at::kFloat);
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

            if (g_dump_on) g_dump.push_back(x.detach().clone());
            // drafter tap: the block input after the loop bookkeeping (what Inkling2Block exports)
            if (tapb_[bi] >= 0) tapbuf.select(0, tapb_[bi]).narrow(0, r, 1).copy_(x);

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
                b.rel_proj, b.sink_proj, b.qk_clip, b.rel_clip, dD.narrow(0, dbase + bi, 1));
            at::Tensor rs = b.ring.select(0, slot_);                                  // (ch, ld)
            ink_kv_glue_dev(k_raw, v_raw, rs.narrow(0, 0, 2 * b.kvc), k_in.reshape(-1), v_in.reshape(-1), b.kw, b.vw,
                        b.kp, b.vp, b.kf, b.vf, dI.narrow(0, base, 1), pt_, taps, record, b.qk_clip, b.eps);

            // Attention over the logical keys of the window, output projection
            
            
            
            at::Tensor o = ink_attend_dev(qg[0], qg[1], qg[2], qg[3], b.kf, b.vf, pt_, dI.narrow(0, base + 1 + 3 * bi, 3),
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
        if (tapf_ >= 0) tapbuf.select(0, tapf_).narrow(0, r, 1).copy_(xn);
        at::Tensor logits = head->run_alloc(xn, Vpad, false);                           // LinearEXL3.forward at one row
        if (final_cap != 0.0) softcap(logits, logits, (float)final_cap);
        return logits.narrow(-1, 0, V);                                                      // (1, V)
    }

    at::Tensor step_dev(bool record) { return step_row(0, record).unsqueeze(0); }      // (1, 1, V)

    // Row-batched forward of the Tcur_ staged rows (one sequence, positions pos0 .. pos0 + T - 1). Batched over the
    // rows: embedding, norms, the attention linears, the head, the routing ds3 group and the expert kernels (chunks of
    // 8 rows). Per row, in order: the K/V cache write and the conv history step (row r sees rows < r), the attention
    // over the cache, the attention-residual sources and the router gemv. record: the live conv history is restored
    // after the rows and the rewind tail takes the window [history | rows] (what Inkling2Ring._store writes).
    at::Tensor step_rows2(bool record)
    {
        const int64_t T = Tcur_, S = 2 + 3 * NB;
        TORCH_CHECK(T >= 1 && T <= TMAX, "InkToken: 1..9 rows per forward");
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        const int64_t Hc = H;
        auto row = [](const at::Tensor& t, int64_t r) { return t.narrow(0, r, 1); };
        // debugging switch (bisection of the batched forward): 1 linears per row, 2 MoE per row, 4 router groups per row
        const char* dbg_env = std::getenv("INKLING2_ROWS_MASK");
        const int64_t dbgm = dbg_env ? std::atoi(dbg_env) : 0;
        auto lin = [&](const std::shared_ptr<BC_LinearEXL3>& L, const at::Tensor& inp, int64_t nout) -> at::Tensor {
            if (!(dbgm & 1)) return L->run_alloc(inp, nout, true).view({T, -1});
            std::vector<at::Tensor> v;
            for (int64_t r = 0; r < T; ++r) v.push_back(L->run_alloc(row(inp, r), nout, true).view({1, -1}));
            return at::cat(v, 0);
        };

        at::Tensor tk = dI.view({TMAX, S}).narrow(0, 0, T).select(1, 1 + 3 * NB).contiguous();
        at::Tensor e = at::embedding(emb, tk).to(at::kFloat);
        at::Tensor x = at::mul(at::rms_norm(e, {Hc}, c10::nullopt, eps_emb), emb_norm);       // (T, H) fp32

        std::vector<at::Tensor> bs;                     // block states, (T, H) fp32 each
        at::Tensor pre_x;
        std::vector<at::Tensor> pre_bs;
        at::Tensor prev_b, prev_o;

        int64_t bi = -1;
        for (auto& b : blocks) {
            ++bi;
            at::Tensor rs = b.ring.select(0, slot_);                                       // (ch, ld)
            const int64_t Ch = rs.size(0), ld = rs.size(1), kv2 = 2 * b.kvc;
            at::Tensor Hs, Xr;
            if (record) {
                Hs = rs.narrow(1, 0, taps).clone();                                       // live history before the rows
                Xr = at::empty({T, Ch}, rs.options());                                    // conv inputs, one row each
            }

            at::Tensor msgs;                                                               // (T, 4, H)
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
                    std::vector<at::Tensor> ms;
                    for (int64_t r = 0; r < T; ++r)
                        ms.push_back(([](at::Tensor h, int64_t nq) { TORCH_CHECK(h.size(0) == nq, "InkToken: attention residual NQ"); return h; })(attn_res_torch(std::vector<at::Tensor>{row(prev_b, r), row(prev_o, r)}, b.msg_t), 4));
                    msgs = at::stack(ms, 0);
                    has_msg = true;
                }
            }

            if (g_dump_on) g_dump.push_back(x.detach().clone());
            if (tapb_[bi] >= 0) tapbuf.select(0, tapb_[bi]).narrow(0, 0, T).copy_(x);

            // attention residual -> sublayer inputs, per row. ink_attn_res returns (NQ, H): NQ = 3 on full blocks
            // (q, k, v mixes), 1 otherwise. Each row keeps its own (NQ, H); (T, H) is only formed for the plain path.
            std::vector<at::Tensor> hv;
            for (int64_t r = 0; r < T; ++r) {
                if (b.l0) {
                    hv.push_back(row(x, r));
                    continue;
                }
                std::vector<at::Tensor> srcs;
                for (auto& s : bs) srcs.push_back(row(s, r));
                srcs.push_back(row(x, r));
                at::Tensor h = ink_attn_res(srcs, b.q_attn_t, b.eps);
                TORCH_CHECK(h.size(0) == (b.full ? 3 : 1), "InkToken: attention residual NQ");
                hv.push_back(h);
            }
            at::Tensor q_in, k_in, v_in;
            bool same = false;
            if (!has_msg && !b.full) {
                at::Tensor hs = b.l0 ? x : at::cat(hv, 0);                                 // (T, H), one query per row
                at::Tensor y = at::rms_norm(hs, {Hc}, b.qkv_norm[0], b.eps);
                q_in = y;
                k_in = y;
                v_in = y;
                same = true;
            } else {
                std::vector<at::Tensor> xs;
                for (int64_t r = 0; r < T; ++r) {
                    at::Tensor X = hv[r].size(0) == 3 ? hv[r] : hv[r].expand({3, Hc});
                    if (has_msg) X = ink_gates(X, msgs.select(0, r).narrow(0, 0, 3), b.gd, b.gu, 0, 3, b.eps);
                    xs.push_back(X);
                }
                at::Tensor XX = at::stack(xs, 0);                                          // (T, 3, H)
                at::Tensor y = at::mul(at::rms_norm(XX, {Hc}, c10::nullopt, b.eps), b.qkv_norm);   // (T, 3, H)
                q_in = y.select(1, 0);
                k_in = y.select(1, 1);
                v_in = y.select(1, 2);
            }

            // projections, batched over the rows
            at::Tensor qh = q_in.to(at::kHalf).contiguous();
            at::Tensor kh = same ? qh : k_in.to(at::kHalf).contiguous();
            at::Tensor vh = same ? qh : v_in.to(at::kHalf).contiguous();
            at::Tensor q_f = lin(b.q, qh, b.Hq * b.D);
            at::Tensor r_f = lin(b.r, qh, b.Hq * b.R);
            at::Tensor g_f = lin(b.g, qh, b.Hq * b.D);
            at::Tensor k_raw = lin(b.k, kh, b.Hk * b.D);
            at::Tensor v_raw = lin(b.v, vh, b.Hk * b.D);

            // per row: query glue, K/V glue (conv history step, cache write), attention over the cache
            at::Tensor ra = rs.narrow(0, kv2, H);                                          // attention conv history
            at::Tensor rm = rs.narrow(0, kv2 + H, H);                                      // MLP conv history
            if (b.boundary) bs.push_back(x);
            std::vector<at::Tensor> ov;
            for (int64_t r = 0; r < T; ++r) {
                const int64_t base = r * S, dbase = r * NB;
                std::vector<at::Tensor> qg = ink_q_glue_dev(
                    row(q_f, r).reshape(-1), row(r_f, r).reshape(-1), row(g_f, r).reshape(-1), row(q_in, r).reshape(-1),
                    b.q_scale, b.rel_scale, b.rel_proj, b.sink_proj, b.qk_clip, b.rel_clip,
                    dD.narrow(0, dbase + bi, 1));
                ink_kv_glue_dev(row(k_raw, r).reshape(-1), row(v_raw, r).reshape(-1), rs.narrow(0, 0, kv2),
                                row(k_in, r).reshape(-1), row(v_in, r).reshape(-1), b.kw, b.vw, b.kp, b.vp, b.kf, b.vf,
                                dI.narrow(0, base, 1), pt_, taps, false, b.qk_clip, b.eps);
                if (record) {
                    Xr.narrow(0, r, 1).narrow(1, 0, b.kvc).copy_(row(k_raw, r));
                    Xr.narrow(0, r, 1).narrow(1, b.kvc, b.kvc).copy_(row(v_raw, r));
                }
                at::Tensor o = ink_attend_dev(qg[0], qg[1], qg[2], qg[3], b.kf, b.vf, pt_,
                                              dI.narrow(0, base + 1 + 3 * bi, 3), b.sm_scale, b.Hk, b.E);
                ov.push_back(o.reshape({1, -1}));
            }
            at::Tensor a = lin(b.o, at::cat(ov, 0).contiguous(), H);   // (T, H) fp32

            // attention output short conv (per row: the history steps in row order)
            std::vector<at::Tensor> rv;
            for (int64_t r = 0; r < T; ++r) {
                at::Tensor ar = row(a, r).reshape(-1);
                at::Tensor res = b.boundary
                    ? ink_sconv(ra, ar, b.conv_attn, c10::nullopt, taps, false)
                    : ink_sconv(ra, ar, b.conv_attn, row(x, r).reshape(-1), taps, false);
                if (record) Xr.narrow(0, r, 1).narrow(1, kv2, H).copy_(ar.view({1, -1}));
                rv.push_back(res.view({1, -1}));
            }
            at::Tensor RES = at::cat(rv, 0);                                                // (T, H)

            // MLP input: attention residual over the block states, message gate, norm
            std::vector<at::Tensor> mv;
            for (int64_t r = 0; r < T; ++r) {
                std::vector<at::Tensor> srcs2;
                for (auto& s : bs) srcs2.push_back(row(s, r));
                srcs2.push_back(row(RES, r));
                mv.push_back(([](at::Tensor h, int64_t nq) { TORCH_CHECK(h.size(0) == nq, "InkToken: attention residual NQ"); return h; })(ink_attn_res(srcs2, b.q_mlp_t, b.eps), 1));                        // (1, H)
            }
            at::Tensor M = at::cat(mv, 0);
            if (has_msg) {
                std::vector<at::Tensor> gv;
                for (int64_t r = 0; r < T; ++r)
                    gv.push_back(ink_gates(row(M, r), msgs.select(0, r).narrow(0, 3, 1), b.gd, b.gu, 3, 1, b.eps));
                M = at::cat(gv, 0);
            }
            at::Tensor mn = at::rms_norm(M, {Hc}, b.norm_mlp, b.eps).to(at::kHalf).contiguous();   // (T, H)

            // router: gemv per row, the group selection over all rows; experts in chunks of 8 rows
            const int64_t Ne = b.logits.numel(), K = b.sel.numel();
            at::Tensor LG = at::empty({T, Ne}, b.logits.options());
            for (int64_t r = 0; r < T; ++r)
                routing_gemv_f32(row(mn, r).reshape(-1), b.gate_t, row(LG, r).reshape(-1));
            at::Tensor SEL = at::empty({T, K}, b.sel.options());
            at::Tensor WW = at::empty({T, K}, b.w.options());
            if (dbgm & 4) {
                for (int64_t r = 0; r < T; ++r)
                    routing_ds3_group(row(LG, r), c10::optional<at::Tensor>(b.bias_f32), row(SEL, r), row(WW, r),
                                      (float)b.rsf, 0, (int)b.n_group, (int)b.topk_group);
            } else {
                routing_ds3_group(LG, c10::optional<at::Tensor>(b.bias_f32), SEL, WW, (float)b.rsf, 0,
                                  (int)b.n_group, (int)b.topk_group);
            }
            at::Tensor MO = at::empty({T, Hc}, x.options());
            const int64_t CH = (dbgm & 2) ? 1 : 8;
            for (int64_t c0 = 0; c0 < T; c0 += CH) {
                const int64_t n = std::min<int64_t>(CH, T - c0);
                at::Tensor xc = mn.narrow(0, c0, n), sc = SEL.narrow(0, c0, n), wc = WW.narrow(0, c0, n);
                if (b.coop) b.coopmk->run(xc, sc, wc, c10::nullopt, -1, -1, -1);
                else b.bc_moe->run_bszN(xc, sc, wc);
                MO.narrow(0, c0, n).copy_(b.moe_out.narrow(0, 0, n));
            }

            // MLP output short conv with the residual (per row)
            std::vector<at::Tensor> xv;
            for (int64_t r = 0; r < T; ++r) {
                at::Tensor mo = row(MO, r).reshape(-1);
                xv.push_back(ink_sconv(rm, mo, b.conv_mlp, row(RES, r).reshape(-1), taps, false).view({1, Hc}));
                if (record) Xr.narrow(0, r, 1).narrow(1, kv2 + H, H).copy_(mo.view({1, -1}));
            }
            x = at::cat(xv, 0);

            if (record) {
                // Inkling2Ring._store of a recorded pass: live history unchanged, tail = last n columns of [history | rows]
                rs.narrow(1, 0, taps).copy_(Hs);
                at::Tensor full = at::cat({Hs, Xr.t()}, 1);                                 // (ch, taps + T)
                const int64_t n = std::min<int64_t>(ld, taps + T);
                rs.narrow(1, ld - n, n).copy_(full.narrow(1, taps + T - n, n));
            }
        }

        // Final: attention residual over the block states, norm, head (batched), softcap
        std::vector<at::Tensor> fv;
        for (int64_t r = 0; r < T; ++r) {
            std::vector<at::Tensor> fsrc;
            for (auto& s : bs) fsrc.push_back(row(s, r));
            fsrc.push_back(row(x, r));
            fv.push_back(([](at::Tensor h, int64_t nq) { TORCH_CHECK(h.size(0) == nq, "InkToken: attention residual NQ"); return h; })(attn_res_torch(fsrc, fin_q_t), 1));                                   // (1, H)
        }
        at::Tensor FX = at::cat(fv, 0);
        at::Tensor xn = at::mul(at::rms_norm(FX, {Hc}, c10::nullopt, eps_fin), fin_norm).to(at::kHalf).contiguous();
        if (tapf_ >= 0) tapbuf.select(0, tapf_).narrow(0, 0, T).copy_(xn);
        at::Tensor logits = head->run_alloc(xn, Vpad, false).view({T, -1});
        if (final_cap != 0.0) softcap(logits, logits, (float)final_cap);
        return logits.narrow(-1, 0, V).unsqueeze(0);                                       // (1, T, V)
    }

    // Row-batched forward, row kernels: as step_rows2 with the per-row stages (attention residuals, message gates,
    // query and K/V glue, attention, short convs, router projection) as one launch over the rows each. Each of those
    // kernels runs the single-token arithmetic per row. INKLING2_ROWS_MASK bit 16: message / final mixes per row.
    at::Tensor step_rows3(bool record)
    {
        const int64_t T = Tcur_, S = 2 + 3 * NB;
        TORCH_CHECK(T >= 1 && T <= TMAX, "InkToken: 1..9 rows per forward");
        TORCH_CHECK(!(seqs_ && record), "InkToken: independent sequences do not record history");
        const at::cuda::OptionalCUDAGuard guard(emb.device());
        const int64_t Hc = H;
        auto row = [](const at::Tensor& t, int64_t r) { return t.narrow(0, r, 1); };
        const char* dbg_env = std::getenv("INKLING2_ROWS_MASK");
        const int64_t dbgm = dbg_env ? std::atoi(dbg_env) : 0;
        auto lin = [&](const std::shared_ptr<BC_LinearEXL3>& L, const at::Tensor& inp, int64_t nout) -> at::Tensor {
            if (!(dbgm & 1)) return L->run_alloc(inp, nout, true).view({T, -1});
            std::vector<at::Tensor> v;
            for (int64_t r = 0; r < T; ++r) v.push_back(L->run_alloc(row(inp, r), nout, true).view({1, -1}));
            return at::cat(v, 0);
        };
        auto mix = [&](const std::vector<at::Tensor>& srcs, const at::Tensor& qT) -> at::Tensor {     // (T, nq, H)
            if (!(dbgm & 16)) return attn_res_torch_rows(srcs, qT);
            std::vector<at::Tensor> v;
            for (int64_t r = 0; r < T; ++r) {
                std::vector<at::Tensor> sr;
                for (auto& t : srcs) sr.push_back(row(t, r));
                v.push_back(attn_res_torch(sr, qT));
            }
            return at::stack(v, 0).contiguous();
        };

        at::Tensor tk = dI.view({TMAX, S}).narrow(0, 0, T).select(1, 1 + 3 * NB).contiguous();
        at::Tensor e = at::embedding(emb, tk).to(at::kFloat);
        at::Tensor x = at::mul(at::rms_norm(e, {Hc}, c10::nullopt, eps_emb), emb_norm).contiguous();   // (T, H) fp32

        std::vector<at::Tensor> bs;                     // block states, (T, H) fp32 each
        at::Tensor pre_x;
        std::vector<at::Tensor> pre_bs;
        at::Tensor prev_b, prev_o;

        int64_t bi = -1;
        for (auto& b : blocks) {
            ++bi;
            at::Tensor rs = b.ring.select(0, slot_);                                       // (ch, ld)
            const int64_t kv2 = 2 * b.kvc;

            at::Tensor msgs;                                                               // (T, 4, H)
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
                    msgs = mix(std::vector<at::Tensor>{prev_b, prev_o}, b.msg_t);
                    TORCH_CHECK(msgs.size(1) == 4, "InkToken: message mixes");
                    has_msg = true;
                }
            }

            if (g_dump_on) g_dump.push_back(x.detach().clone());
            if (tapb_[bi] >= 0) tapbuf.select(0, tapb_[bi]).narrow(0, 0, T).copy_(x);

            // attention residual -> sublayer inputs: (T, NQ, H), NQ = 3 on full blocks (q, k, v mixes), 1 otherwise
            at::Tensor hs3;
            if (b.l0) {
                hs3 = x.view({T, 1, Hc});
            } else {
                std::vector<at::Tensor> srcs = bs;
                srcs.push_back(x);
                hs3 = ink_attn_res_rows(srcs, b.q_attn_t, b.eps);
                TORCH_CHECK(hs3.size(1) == (b.full ? 3 : 1), "InkToken: attention residual NQ");
            }
            at::Tensor q_in, k_in, v_in;
            bool same = false;
            if (!has_msg && !b.full) {
                at::Tensor y = at::rms_norm(hs3.view({T, Hc}), {Hc}, b.qkv_norm[0], b.eps).contiguous();
                q_in = y;
                k_in = y;
                v_in = y;
                same = true;
            } else {
                at::Tensor XX = (hs3.size(1) == 3 ? hs3 : hs3.expand({T, 3, Hc})).contiguous();
                if (has_msg) XX = ink_gates_rows(XX, msgs, b.gd, b.gu, 0, 3, b.eps);
                at::Tensor y = at::mul(at::rms_norm(XX, {Hc}, c10::nullopt, b.eps), b.qkv_norm);   // (T, 3, H)
                q_in = y.select(1, 0).contiguous();
                k_in = y.select(1, 1).contiguous();
                v_in = y.select(1, 2).contiguous();
            }

            // projections, batched over the rows
            at::Tensor qh = q_in.to(at::kHalf).contiguous();
            at::Tensor kh = same ? qh : k_in.to(at::kHalf).contiguous();
            at::Tensor vh = same ? qh : v_in.to(at::kHalf).contiguous();
            at::Tensor q_f, r_f, g_f, k_raw, v_raw;
            if (b.mlin && !b.mlin_xh.empty() && !(dbgm & (1 | 32))) {
                // INKLING2_ROWS_MASK bit 32 off: one weight pass for all five projections and all rows
                mlin5_rows(b, qh, kh, vh, T);
                q_f = b.mlin_rout[0].narrow(0, 0, T);
                r_f = b.mlin_rout[1].narrow(0, 0, T);
                g_f = b.mlin_rout[2].narrow(0, 0, T);
                k_raw = b.mlin_rout[3].narrow(0, 0, T);
                v_raw = b.mlin_rout[4].narrow(0, 0, T);
            } else {
                q_f = lin(b.q, qh, b.Hq * b.D).contiguous();
                r_f = lin(b.r, qh, b.Hq * b.R).contiguous();
                g_f = lin(b.g, qh, b.Hq * b.D).contiguous();
                k_raw = lin(b.k, kh, b.Hk * b.D).contiguous();
                v_raw = lin(b.v, vh, b.Hk * b.D).contiguous();
            }

            // query glue, K/V glue (cache rows of all rows, then the K/V conv history), attention over the cache:
            // row r attends its own window, which ends at its own position
            std::vector<at::Tensor> qg = ink_q_glue_rows(q_f, r_f, g_f, q_in, b.q_scale, b.rel_scale, b.rel_proj,
                                                         b.sink_proj, b.qk_clip, b.rel_clip,
                                                         dD.narrow(0, bi, (T - 1) * NB + 1), NB);
            if (seqs_) {
                ink_kv_glue_seqs(k_raw, v_raw, b.ring, seq_slots_, k_in, v_in, b.kw, b.vw, b.kp, b.vp, b.kf, b.vf,
                                 dI.narrow(0, 0, (T - 1) * S + 1), S, pt_, taps, b.qk_clip, b.eps);
                ink_hist_seqs(b.ring, seq_slots_, 0, at::cat({k_raw, v_raw}, 1).contiguous(), taps);
            } else {
                at::Tensor rkv = rs.narrow(0, 0, kv2);
                ink_kv_glue_rows(k_raw, v_raw, rkv, k_in, v_in, b.kw, b.vw, b.kp, b.vp, b.kf, b.vf,
                                 dI.narrow(0, 0, (T - 1) * S + 1), S, pt_, taps, b.qk_clip, b.eps);
                ink_hist_rows(rkv, at::cat({k_raw, v_raw}, 1).contiguous(), taps, record);
            }
            at::Tensor o = ink_attend_rows_dev(qg[0], qg[1], qg[2], qg[3], b.kf, b.vf, pt_,
                                               dI.narrow(0, 1 + 3 * bi, (T - 1) * S + 3), S, b.sm_scale, b.Hk, b.E);
            at::Tensor a = lin(b.o, o, H).contiguous();                                    // (T, H) fp32

            // attention output short conv; a block boundary also appends the block state
            at::Tensor ra = rs.narrow(0, kv2, H);
            at::Tensor rm = rs.narrow(0, kv2 + H, H);
            auto sconv = [&](const at::Tensor& st, int64_t ch0, const at::Tensor& xx, const at::Tensor& w,
                             c10::optional<at::Tensor> res) -> at::Tensor {
                return seqs_ ? ink_sconv_seqs(b.ring, seq_slots_, ch0, xx, w, res, taps)
                             : ink_sconv_rows(st, xx, w, res, taps, record);
            };
            at::Tensor RES;
            if (b.boundary) {
                bs.push_back(x);
                RES = sconv(ra, kv2, a, b.conv_attn, c10::nullopt);
            } else {
                RES = sconv(ra, kv2, a, b.conv_attn, x);
            }

            // MLP input: attention residual over the block states, message gate, norm
            std::vector<at::Tensor> srcs2 = bs;
            srcs2.push_back(RES);
            at::Tensor M = ink_attn_res_rows(srcs2, b.q_mlp_t, b.eps);                      // (T, 1, H)
            TORCH_CHECK(M.size(1) == 1, "InkToken: attention residual NQ");
            if (has_msg) M = ink_gates_rows(M, msgs, b.gd, b.gu, 3, 1, b.eps);
            at::Tensor mn = at::rms_norm(M.view({T, Hc}), {Hc}, b.norm_mlp, b.eps).to(at::kHalf).contiguous();

            // router projection and group selection over all rows; experts in chunks of 8 rows
            const int64_t Ne = b.logits.numel(), K = b.sel.numel();
            at::Tensor LG = at::empty({T, Ne}, b.logits.options());
            routing_gemv_f32(mn, b.gate_t, LG);
            at::Tensor SEL = at::empty({T, K}, b.sel.options());
            at::Tensor WW = at::empty({T, K}, b.w.options());
            if (dbgm & 4) {
                for (int64_t r = 0; r < T; ++r)
                    routing_ds3_group(row(LG, r), c10::optional<at::Tensor>(b.bias_f32), row(SEL, r), row(WW, r),
                                      (float)b.rsf, 0, (int)b.n_group, (int)b.topk_group);
            } else {
                routing_ds3_group(LG, c10::optional<at::Tensor>(b.bias_f32), SEL, WW, (float)b.rsf, 0,
                                  (int)b.n_group, (int)b.topk_group);
            }
            at::Tensor MO = at::empty({T, Hc}, x.options());
            const int64_t CH = (dbgm & 2) ? 1 : 8;
            for (int64_t c0 = 0; c0 < T; c0 += CH) {
                const int64_t n = std::min<int64_t>(CH, T - c0);
                at::Tensor xc = mn.narrow(0, c0, n), sc = SEL.narrow(0, c0, n), wc = WW.narrow(0, c0, n);
                if (b.coop) b.coopmk->run(xc, sc, wc, c10::nullopt, -1, -1, -1);
                else b.bc_moe->run_bszN(xc, sc, wc);
                MO.narrow(0, c0, n).copy_(b.moe_out.narrow(0, 0, n));
            }

            // MLP output short conv with the residual
            x = sconv(rm, kv2 + H, MO, b.conv_mlp, RES);
        }

        // Final: attention residual over the block states, norm, head (batched), softcap
        std::vector<at::Tensor> fsrc = bs;
        fsrc.push_back(x);
        at::Tensor FX = mix(fsrc, fin_q_t);
        TORCH_CHECK(FX.size(1) == 1, "InkToken: final mix");
        at::Tensor xn = at::mul(at::rms_norm(FX.view({T, Hc}), {Hc}, c10::nullopt, eps_fin), fin_norm).to(at::kHalf).contiguous();
        if (tapf_ >= 0) tapbuf.select(0, tapf_).narrow(0, 0, T).copy_(xn);
        at::Tensor logits = head->run_alloc(xn, Vpad, false).view({T, -1});
        if (final_cap != 0.0) softcap(logits, logits, (float)final_cap);
        return logits.narrow(-1, 0, V).unsqueeze(0);                                       // (1, T, V)
    }

    // All Tcur_ rows of the staged forward, one token after another (each row sees the state the previous rows
    // wrote). Returns (1, Tcur_, V).
    at::Tensor step_rows(bool record)
    {
        if (batched_) {
            // INKLING2_ROWS_MASK bit 8: the per-row stage loop (step_rows2) instead of the row kernels
            const char* me = std::getenv("INKLING2_ROWS_MASK");
            return ((me ? std::atoi(me) : 0) & 8) ? step_rows2(record) : step_rows3(record);
        }
        TORCH_CHECK(!record, "InkToken: sequential rows do not record history (set_batched(true))");
        std::vector<at::Tensor> rows;
        for (int64_t r = 0; r < Tcur_; ++r) rows.push_back(step_row(r, record));
        return at::cat(rows, 0).unsqueeze(0);
    }
};

void ink_token_bind(py::module_& m)
{
    m.def("ink_rewind_all", &ink_rewind_all, py::arg("states"), py::arg("slot"), py::arg("taps"), py::arg("num_tokens"));
    py::class_<InkToken, std::shared_ptr<InkToken>>(m, "InkToken")
        .def(py::init<const py::dict&>())
        .def("set_table", &InkToken::set_table, py::arg("pages"))
        .def("step", &InkToken::step, py::arg("token"), py::arg("pos"), py::arg("slot"), py::arg("record"))
        .def("update", &InkToken::update, py::arg("token"), py::arg("pos"), py::arg("slot"))
        .def("step_dev", &InkToken::step_dev, py::arg("record"))
        .def("set_taps", &InkToken::set_taps, py::arg("tapb"), py::arg("tapf"))
        .def("update_rows", &InkToken::update_rows, py::arg("tokens"), py::arg("pos0"), py::arg("slot"))
        .def("step_rows", &InkToken::step_rows, py::arg("record"))
        .def("set_tables", &InkToken::set_tables, py::arg("tables"))
        .def("update_seqs", &InkToken::update_seqs, py::arg("tokens"), py::arg("positions"), py::arg("slots"))
        .def("step_seqs", &InkToken::step_seqs)
        .def("dump_on", [](InkToken&, bool v) { g_dump_on = v; g_dump.clear(); }, py::arg("on"))
        .def("dump_get", [](InkToken&) { return g_dump; })
        .def("set_batched", [](InkToken& t, bool v) { t.batched_ = v; }, py::arg("batched"))
        .def("tapbuf", [](InkToken& t) { return t.tapbuf; });
}
