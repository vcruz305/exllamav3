from __future__ import annotations
from typing_extensions import override
import math
import os
import torch
import torch.nn.functional as F
from ...model.config import Config
from ...util.tensor import get_for_device
from ...constants import PAGE_SIZE
from ...cache.recurrent import host_copy
from ...ext import exllamav3_ext as ext
from ..module import Module
from ..linear import Linear

"""
Inkling2Loop (thinkingmachines/Inkling-2-Mini) text decoder modules.

The decoder is a stack of blocks whose core range is applied several times ("passes"). On top of the usual
residual stream, three kinds of state cross block boundaries:

  - block states: the residual stream at the start of every n_layers_per_attn_res_block'th layer. Each sublayer
    input is an attention-pooled mix ("attention residual") of the block states and the running residual.
  - the pre-loop snapshot: every pass after the first restarts from the residual/block states at the loop entry.
  - messages: from the second pass on, each loop layer receives four messages (query, key, value, MLP input)
    pooled from the previous pass' final block state and output, mixed in through low-rank tanh gates.

These live in params["inkling2"] for the duration of one forward call. Positions are encoded only through a
state-conditioned relative position bias (no RoPE), every attention row has a per-token sink logit, K and V
pass through a short causal convolution before entering the cache, and the outputs of both sublayers pass
through another short causal convolution.

Reference semantics: transformers `Inkling2LoopTextModel` (models/inkling2_loop/modeling_inkling2_loop.py).
"""

STATE_KEY = "inkling2"

# Precision of the activations handed to the quantized linears (the EXL3 kernels take fp16 rows) and of
# the K/V rows (the cache's). Validation against an fp32 reference raises both to take the rounding out
ACT_DTYPE = torch.half
KV_DTYPE = torch.half

# Single-token (T == 1, one sequence, paged cache) decode runs the pure-tensor segments as one CUDA kernel each
# (exllamav3_ext/inkling2.cu, see Inkling2Block._forward_t1). INKLING2_T1=0 keeps the eager path for A/B and checks.
_T1 = os.environ.get("INKLING2_T1", "1") != "0"

# Device page tables of the single-token path, one entry (the last one) per process: the page list of the sequence
# as an int32 tensor on the device. Rebuilt only when the list changes (a new page or a new sequence).
_PT_CACHE = [None, None]


def _page_table(pages: list, device) -> torch.Tensor:
    key = (device, tuple(pages))
    if _PT_CACHE[0] != key:
        _PT_CACHE[0] = key
        _PT_CACHE[1] = torch.tensor(pages, dtype = torch.int32, device = device)
    return _PT_CACHE[1]


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    # One fused kernel on CUDA; same math as x * rsqrt(mean(x^2) + eps), which is 5 aten ops when written out
    return F.rms_norm(x, (x.shape[-1],), None, eps)


def _fold_sliding(w: torch.Tensor) -> torch.Tensor:
    """
    Depthwise causal conv weight (C, 1, k) with the residual term (conv(x) + x) / sqrt(k + 1) folded in: (C, k),
    taps in time-ascending order (column k - 1 multiplies the current token). Applied as (window * folded).sum(-1).
    """
    k = w.shape[-1]
    s = math.sqrt(k + 1)
    f = w[:, 0, :] / s
    f[:, k - 1] += 1.0 / s
    return f.contiguous()


def _with_history(ctx: torch.Tensor | None, x: torch.Tensor, taps: int) -> torch.Tensor:
    # [history | x] for a call without recurrent state: zeros at sequence start
    return F.pad(x, (0, 0, taps, 0)) if ctx is None else ctx


def _l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    # x * min(1, threshold / (norm + 1e-6)); reciprocal + in-place scale (not the Python `threshold / t` wrapper)
    norm = torch.linalg.vector_norm(x, dim = -1, keepdim = True)
    scale = torch.reciprocal(norm + 1e-6).mul_(threshold).clamp_(max = 1.0)
    return x * scale


def _side(config: Config, key: str, device, optional: bool = False) -> torch.Tensor | None:
    # Read inside the loader's deferred bracket, where plain get_tensor() hands out buffers that are only
    # filled when the bracket closes: these are converted right away, so they must load eagerly
    t = config.stc.get_tensor(key, device, optional = optional, allow_bf16 = True, no_defer = True)
    return None if t is None else t.float()


def _prep_query(q: torch.Tensor) -> torch.Tensor:
    """Learned queries (nq, dim) -> (dim, nq) fp32, pre-scaled by dim ** -0.5 (the attention-residual logit scale)"""
    return (q.float() * (q.shape[-1] ** -0.5)).T.contiguous()


def attn_res(block_states: list[torch.Tensor], partial: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
    """
    Attention residual: softmax-pool the (unweighted-RMS-normalized) block states and the partial state with
    learned queries. block_states: list of fp32 (bsz, seq, dim), partial: fp32 (bsz, seq, dim), query_t: (dim, nq)
    from _prep_query. Returns (bsz, seq, nq, dim).
    """
    stacked = torch.stack(block_states + [partial], dim = -2)
    stacked = _rms(stacked, 1e-6)
    w = torch.matmul(stacked, query_t)                                       # (bsz, seq, S, nq), pre-scaled
    w = torch.softmax(w, dim = -2)
    return torch.matmul(w.transpose(-1, -2), stacked)                       # (bsz, seq, nq, dim)


class Inkling2ConvLayerState:
    """
    Per-sequence short-convolution history of one layer instance: the last (kernel - 1) inputs of the four
    causal convolutions (K projection, V projection, attention output, MLP output), concatenated along the
    channel axis. Layout and rewind semantics follow ShortConvLayerState: [..., :taps] is the live history,
    a pass that records history (speculative decoding) writes its inputs to the tail instead and rewind()
    settles the live history at the accepted length.
    """

    def __init__(self, module: Inkling2Block, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.taps = module.ksize - 1
        self.conv_state = torch.empty(
            (max_batch_size, module.ring_channels, self.taps + max_history), dtype = torch.float, device = "meta",
        )
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id

    def get_checkpoint_size(self):
        return self.module.ring_channels * self.taps * 4

    def storage_size(self):
        return self.conv_state.numel() * self.conv_state.element_size()

    def alloc(self, device):
        self.conv_state = torch.zeros_like(self.conv_state, device = device)
        self.device = device

    def free(self):
        self.conv_state = torch.empty_like(self.conv_state, device = "meta")
        self.device = None

    def clear(self, idx: int):
        if self.device is not None:
            self.conv_state[idx].zero_()

    def get_state_tensors(self):
        return (self.conv_state,)

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        assert num_tokens >= 0 and (last_history == 0 or num_tokens <= last_history), \
            f"inkling2 conv rewind: {num_tokens} tokens outside the recorded history of {last_history}"
        if last_history > 0 and self.max_history == 0:
            raise RuntimeError(
                "inkling2 conv rewind: this cache has max_history == 0, so no speculative history was kept "
                "for rewind; create the Cache with max_history > 0 to use recurrent history / rewind"
            )
        if last_history > 0:
            p = self.conv_state.shape[-1] - num_tokens
            self.conv_state[slot, :, :self.taps].copy_(self.conv_state[slot, :, p - self.taps : p].clone())

    @staticmethod
    def rewind_all(layers, slot: int, last_history: int, num_tokens: int):
        """rewind() of every layer of one cache in one extension call (one small kernel per layer)."""
        l0 = layers[0]
        if (last_history <= 0 or l0.device is None or torch.device(l0.device).type != "cuda"
                or any(l.taps != l0.taps or l.device != l0.device or not l.conv_state.is_contiguous() for l in layers)):
            for l in layers:
                l.rewind(slot, last_history, num_tokens)
            return
        assert 0 <= num_tokens <= last_history and l0.max_history > 0, \
            f"inkling2 conv rewind: {num_tokens} tokens outside the recorded history of {last_history}"
        ext.ink_rewind_all([l.conv_state for l in layers], slot, l0.taps, num_tokens)

    def stash(self, slot, position: int = 0):
        return host_copy(self.conv_state[slot, :, :self.taps])

    def unstash(self, slot, stashed, position: int = 0):
        self.conv_state[slot, :, :self.taps].copy_(stashed)

    def tp_export(self, plan):
        raise NotImplementedError("Inkling2Loop: tensor-parallel is not supported")


class Inkling2Ring:
    """One forward call's view of a layer instance's conv history for the batch (None-safe: no cache, no history)"""

    def __init__(self, state: torch.Tensor | None, slots: list | None, record: bool, taps: int):
        self.state = state
        self.slots = slots
        self.record = record
        self.taps = taps

    def exchange(self, c0: int, x: torch.Tensor) -> torch.Tensor | None:
        """
        Store x (bsz, seq, C) as the new history of channels [c0, c0 + C) and return [history | x] (bsz, taps + seq, C)
        for the same channels, or None without recurrent state
        """
        if self.state is None:
            return None
        C = x.shape[-1]
        if len(self.slots) == 1:
            # One sequence (the usual case): no per-slot list, no stack, and no cast when the dtypes already match
            st = self.state[self.slots[0], c0 : c0 + C]                  # (C, taps + max_history)
            xi = x[0] if x.dtype == st.dtype else x[0].to(st.dtype)
            full = torch.cat([st[:, :self.taps].T, xi], dim = 0)         # (taps + seq, C), a copy
            self._store(st, full)
            return full.unsqueeze(0)
        out = []
        for i, slot in enumerate(self.slots):
            st = self.state[slot, c0 : c0 + C]                          # (C, taps + max_history)
            full = torch.cat([st[:, :self.taps].T, x[i].to(st.dtype)], dim = 0)     # (taps + seq, C), a copy
            self._store(st, full)
            out.append(full)
        return torch.stack(out, dim = 0)

    def _store(self, st: torch.Tensor, full: torch.Tensor):
        if self.record:
            n = min(st.shape[-1], full.shape[0])
            st[:, -n:].copy_(full[-n:].T)
        else:
            st[:, :self.taps].copy_(full[-self.taps:].T)


def _export_tap(params: dict, t: torch.Tensor):
    """Append one DSpark drafter tap (fp16 copy of a (bsz, T, hidden) state) to params["export_states"]"""
    s = params.get("export_states")
    if s is None:
        s = params["export_states"] = []
    s.append(t.half())


class Inkling2State:
    """Cross-block state of one forward call (see module docstring)"""

    def __init__(self):
        self.block_states: list[torch.Tensor] = []
        self.pre_loop: tuple | None = None
        self.prev_pass: tuple | None = None

    @staticmethod
    def get(params: dict) -> Inkling2State:
        st = params.get(STATE_KEY)
        if st is None:
            st = params[STATE_KEY] = Inkling2State()
        return st


class Inkling2Embedding(Module):
    """Token embedding followed by a weighted RMSNorm"""

    def __init__(self, config: Config, key: str, key_norm: str, vocab_size: int, hidden_size: int, eps: float):
        super().__init__(config, key, None)
        self.module_name = "Inkling2Embedding"
        self.key_norm = key_norm
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.eps = eps
        self.embedding = None
        self.norm_weight = None
        self.caps.update({"prefer_cpu": False})
        self._numel = vocab_size * hidden_size

    @override
    def optimizer_targets(self):
        return []

    @override
    def can_defer_load(self):
        return False

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        w = self.config.stc.get_tensor(self.key + ".weight", device, allow_bf16 = True, no_defer = True)
        self.embedding = w
        self.norm_weight = _side(self.config, self.key_norm + ".weight", device)

    @override
    def unload(self):
        self.device = None
        self.embedding = None
        self.norm_weight = None

    @override
    def get_tensors(self):
        return {
            self.key + ".weight": self.embedding,
            self.key_norm + ".weight": self.norm_weight,
        }

    @override
    def weights_numel(self):
        return self._numel

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        # A fresh forward call starts a fresh cross-block state
        params[STATE_KEY] = Inkling2State()
        x = F.embedding(x.to(self.embedding.device), self.embedding).float()
        x = _rms(x, self.eps) * self.norm_weight
        return x


class Inkling2MessageGate:
    """(1 - tanh(forget)) * x + tanh(gain) * message, both gates low-rank functions of [norm(x), norm(message)]"""

    def __init__(self, config: Config, key: str, eps: float):
        self.config = config
        self.key = key
        self.eps = eps
        self.down = None
        self.forget = None
        self.gain = None

    def load(self, device):
        self.down = _side(self.config, self.key + ".down_weight", device)
        self.forget = _side(self.config, self.key + ".forget_weight", device)
        self.gain = _side(self.config, self.key + ".gain_weight", device)

    def unload(self):
        self.down = self.forget = self.gain = None

    def tensors(self):
        return {
            self.key + ".down_weight": self.down,
            self.key + ".forget_weight": self.forget,
            self.key + ".gain_weight": self.gain,
        }

    def __call__(self, x: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        x, message = x.float(), message.float()
        g = torch.cat([_rms(x, self.eps), _rms(message, self.eps)], dim = -1)
        g = F.linear(g, self.down)
        forget = torch.tanh(F.linear(g, self.forget))
        gain = torch.tanh(F.linear(g, self.gain))
        return (1 - forget) * x + gain * message


class Inkling2Attention(Module):
    """
    Sliding-window or full attention with a state-conditioned relative position bias, per-token sink logits,
    L2-clipped queries/keys, RMS-normalized values, short convolutions on K/V and a sigmoid output gate
    """

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        d_rel: int,
        rel_extent: int,
        sliding_window: int | None,
        conv_kernel_size: int,
        qk_clip: float,
        rel_clip: float,
        log_scaling: tuple | None,
        eps: float,
        qmap: str | None = "block.attn",
    ):
        super().__init__(config, key, None)
        self.module_name = "Inkling2Attention"
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.gqa = num_q_heads // num_kv_heads
        self.head_dim = head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.sliding_window = sliding_window
        self.is_full = sliding_window is None
        self.ksize = conv_kernel_size
        self.qk_clip = qk_clip
        self.rel_clip = rel_clip
        self.log_scaling = log_scaling      # (alpha, n_floor) on full-attention layers
        self.eps = eps
        self.sm_scale = head_dim ** -0.5
        self.q_chunk = 512

        def lin(name, n_out, qm):
            m = Linear(config, f"{key}.{name}", hidden_size, n_out, qmap = qm, out_dtype = torch.float)
            self.register_submodule(m)
            return m

        self.q_proj = lin("wq_du", num_q_heads * head_dim, qmap and qmap + ".q")
        self.k_proj = lin("wk_dv", num_kv_heads * head_dim, qmap and qmap + ".k")
        self.v_proj = lin("wv_dv", num_kv_heads * head_dim, qmap and qmap + ".v")
        self.r_proj = lin("wr_du", num_q_heads * d_rel, qmap and qmap + ".q")
        self.g_proj = lin("gate_attn", num_q_heads * head_dim, qmap and qmap + ".q")
        self.o_proj = Linear(
            config, f"{key}.wo_ud", num_q_heads * head_dim, hidden_size,
            qmap = qmap and qmap + ".o", out_dtype = torch.float,
        )
        self.register_submodule(self.o_proj)

        self.sink_proj = None
        self.q_scale = None
        self.rel_scale = None
        self.rel_proj = None
        self.k_conv = None
        self.v_conv = None
        self.k_fold = None
        self.v_fold = None
        self.caps.update({"kv_cache": True})
        self.cache_layers = []

    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.k_proj.optimizer_targets()
        v = self.v_proj.optimizer_targets()
        r = self.r_proj.optimizer_targets()
        g = self.g_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k, v, r, g, o]]

    def _conv_keys(self, name: str) -> dict:
        if self.is_full:
            return {"static": f"{self.key}.{name}.static_w", "proj": f"{self.key}.{name}.weight_proj"}
        return {"weight": f"{self.key}.{name}.weight"}

    def _fold_conv(self, conv: dict) -> dict:
        """
        Taps of one K/V short conv with the residual term and 1/sqrt(scale) folded in (time-ascending taps, see
        _fold_sliding). Full layers keep a static part (H, C/H, k) and a per-token projection (H * k, dim) of the
        per-head taps, with the projection rows reordered to the same tap order.
        """
        k = self.ksize
        if not self.is_full:
            return {"w": _fold_sliding(conv["weight"])}
        H = self.num_kv_heads
        C = H * self.head_dim
        s = math.sqrt(2 * k + 1)
        static = conv["static"].T.reshape(H, C // H, k).flip(-1) / s          # (H, C/H, k), tap j = lag k - 1 - j
        static[..., k - 1] += 1.0 / s
        proj = conv["proj"].reshape(H, k, -1).flip(1).reshape(H * k, -1) / s
        return {"static": static.contiguous(), "proj": proj.contiguous()}

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        c = self.config
        self.sink_proj = _side(c, f"{self.key}.attn_sink_proj.weight", device)
        self.q_scale = _side(c, f"{self.key}.q_l2_clip.scale", device)
        self.rel_scale = _side(c, f"{self.key}.rel_l2_clip.scale", device)
        self.rel_proj = _side(c, f"{self.key}.rel_logits_proj.proj", device)
        self.k_conv = {n: _side(c, k, device) for n, k in self._conv_keys("k_sconv").items()}
        self.v_conv = {n: _side(c, k, device) for n, k in self._conv_keys("v_sconv").items()}
        self.k_fold = self._fold_conv(self.k_conv)
        self.v_fold = self._fold_conv(self.v_conv)
        for cl in self.cache_layers:
            cl.alloc(device)

    @override
    def unload(self):
        self.sink_proj = self.q_scale = self.rel_scale = self.rel_proj = None
        self.k_conv = self.v_conv = self.k_fold = self.v_fold = None
        for cl in self.cache_layers:
            cl.free()
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for m in self.modules:
            t.update(m.get_tensors())
        t[f"{self.key}.attn_sink_proj.weight"] = self.sink_proj
        t[f"{self.key}.q_l2_clip.scale"] = self.q_scale
        t[f"{self.key}.rel_l2_clip.scale"] = self.rel_scale
        t[f"{self.key}.rel_logits_proj.proj"] = self.rel_proj
        for name, conv in (("k_sconv", self.k_conv), ("v_sconv", self.v_conv)):
            for n, k in self._conv_keys(name).items():
                t[k] = conv[n]
        return t

    def _log_scale(self, positions: torch.Tensor) -> torch.Tensor | None:
        # positions: (bsz, seq) absolute. tau = 1 + alpha * log(max((p + 1) / n_floor, 1))
        if self.log_scaling is None:
            return None
        alpha, n_floor = self.log_scaling
        return 1.0 + alpha * torch.log(((positions + 1).float() / n_floor).clamp(min = 1.0))

    def _tau_host(self, pos: int) -> float:
        # Same as _log_scale for one absolute position, in host math (exactly 1.0 below n_floor)
        alpha, n_floor = self.log_scaling
        return 1.0 + alpha * math.log(max((pos + 1) / n_floor, 1.0))

    def _tau(self, pos_host: list | None, positions: torch.Tensor | None, dev) -> torch.Tensor | None:
        """
        Log-scaling factor per row, (bsz, seq) or None when it is 1 everywhere. Single-token calls pass the host
        positions and skip the device work unless the factor differs from 1.
        """
        if self.log_scaling is None:
            return None
        if pos_host is not None:
            taus = [self._tau_host(p) for p in pos_host]
            if all(t == 1.0 for t in taus):
                return None
            return torch.tensor(taus, dtype = torch.float, device = dev).view(-1, 1)
        return self._log_scale(positions)

    def _kv_conv(self, ctx: torch.Tensor, dyn_in: torch.Tensor, fold: dict):
        """
        Short causal convolution over the K or V projection, scaled as (conv(x) + x) / sqrt(...). ctx: (bsz, taps + seq, C)
        fp32 = [history | current]. dyn_in: the projection's input, for the per-token taps of full layers.
        """
        k = self.ksize
        win = ctx.unfold(1, k, 1)                                               # (bsz, seq, C, k), time-ascending
        if not self.is_full:
            return (win * fold["w"]).sum(-1)
        bsz, seq = win.shape[0], win.shape[1]
        H, C = self.num_kv_heads, win.shape[2]
        dyn = F.linear(dyn_in.float(), fold["proj"]).view(bsz, seq, H, 1, k)    # per token, per head, per tap
        w = fold["static"] + dyn                                                # (bsz, seq, H, C/H, k)
        return (win.reshape(bsz, seq, H, C // H, k) * w).sum(-1).reshape(bsz, seq, C)

    def project_t1(self, q_in, k_in, v_in, tau: float, params: dict, ring: "Inkling2Ring", slot: int,
                   kf: torch.Tensor, vf: torch.Tensor, pos: int, pt: torch.Tensor):
        """
        project() for one token (T == 1, one sequence) at logical position pos: the linears run as usual, the query
        side and the K/V side (short conv, history, norms, cache write through the page table pt) run as one kernel
        each. Returns q (Hq * D), rel (Hq * E), sink (Hq), gate (Hq * D), all fp32 flat.
        """
        qh = q_in.to(ACT_DTYPE).contiguous()
        same = k_in is q_in and v_in is q_in
        kh = qh if same else k_in.to(ACT_DTYPE).contiguous()
        vh = qh if same else v_in.to(ACT_DTYPE).contiguous()
        q_f = self.q_proj.forward(qh, params).reshape(-1)
        r_f = self.r_proj.forward(qh, params).reshape(-1)
        g_f = self.g_proj.forward(qh, params).reshape(-1)
        k_raw = self.k_proj.forward(kh, params).reshape(-1)
        v_raw = self.v_proj.forward(vh, params).reshape(-1)
        q, rel, sink, gate = ext.ink_q_glue(
            q_f, r_f, g_f, q_in.reshape(-1), self.q_scale.reshape(-1), self.rel_scale.reshape(-1),
            self.rel_proj, self.sink_proj, float(self.qk_clip), float(self.rel_clip), float(tau),
        )
        C = self.num_kv_heads * self.head_dim
        fk, fv = self.k_fold, self.v_fold
        ext.ink_kv_glue(
            k_raw, v_raw, ring.state[slot, :2 * C], k_in.reshape(-1), v_in.reshape(-1),
            fk["w"] if "w" in fk else fk["static"], fv["w"] if "w" in fv else fv["static"],
            fk.get("proj"), fv.get("proj"), kf, vf, pos, pt, self.ksize - 1, ring.record,
            float(self.qk_clip), float(self.eps),
        )
        return q, rel, sink, gate

    def attend_t1(self, q, rel, sink, gate, kf: torch.Tensor, vf: torch.Tensor, pt: torch.Tensor, lo: int,
                  L: int) -> torch.Tensor:
        """Single query over the logical keys [lo, lo + L) (page table pt), output (o * gate) fp16 flat (Hq * D)"""
        return ext.ink_attend(q, rel, sink, gate, kf, vf, pt, lo, L, self.sm_scale, self.num_kv_heads, self.rel_extent)

    def project(self, q_in, k_in, v_in, tau: torch.Tensor | None, params: dict, ring: Inkling2Ring):
        """
        Returns q (bsz, seq, Hq, D), k / v (bsz, seq, Hkv, D), relative logits (bsz, seq, Hq, rel_extent),
        sink logits (bsz, seq, Hq) and output gate (bsz, seq, Hq * D). tau: log-scaling factor (bsz, seq) or None
        """
        bsz, seq, _ = q_in.shape
        qh = q_in.to(ACT_DTYPE).contiguous()
        same = k_in is q_in and v_in is q_in
        kh = qh if same else k_in.to(ACT_DTYPE).contiguous()
        vh = qh if same else v_in.to(ACT_DTYPE).contiguous()

        q_f = self.q_proj.forward(qh, params)
        r_f = self.r_proj.forward(qh, params)
        g_f = self.g_proj.forward(qh, params)
        q = q_f.view(bsz, seq, self.num_q_heads, self.head_dim)
        q = _l2_clip(q * self.q_scale, self.qk_clip)
        if tau is not None:
            q = q * tau[..., None, None]

        k_raw = self.k_proj.forward(kh, params)
        v_raw = self.v_proj.forward(vh, params)
        C = k_raw.shape[-1]
        taps = self.ksize - 1
        # K and V rows share one ring exchange: their history channels are adjacent in the ring
        kv_raw = torch.cat([k_raw, v_raw], dim = -1)
        kv_ctx = _with_history(ring.exchange(0, kv_raw), kv_raw, taps)
        k = self._kv_conv(kv_ctx[..., :C], k_in, self.k_fold).view(bsz, seq, self.num_kv_heads, self.head_dim)
        v = self._kv_conv(kv_ctx[..., C:], v_in, self.v_fold).view(bsz, seq, self.num_kv_heads, self.head_dim)
        # K and V take the cache's precision whether or not a cache is attached, so cached and uncached
        # passes attend over the same values
        k = _l2_clip(k, self.qk_clip).to(KV_DTYPE)
        v = _rms(v, self.eps).to(KV_DTYPE)

        r = r_f.view(bsz, seq, self.num_q_heads, self.d_rel)
        r = _l2_clip(r * self.rel_scale, self.rel_clip)
        if tau is not None:
            r = r * tau[..., None, None]
        rel = torch.matmul(r, self.rel_proj)                                    # (bsz, seq, Hq, rel_extent)

        sink = F.linear(q_in, self.sink_proj)                                   # (bsz, seq, Hq)
        gate = torch.sigmoid(g_f.view(bsz, seq, self.num_q_heads * self.head_dim))
        return q, k, v, rel, sink, gate

    def attend(self, q, rel, sink, k_all, v_all, q0: int, k0: int) -> torch.Tensor:
        """
        q: (bsz, Tq, Hq, D), rel: (bsz, Tq, Hq, E), sink: (bsz, Tq, Hq), k_all / v_all: (bsz, Tk, Hkv, D).
        Queries sit at absolute positions q0.., keys at k0.. (contiguous). Returns (bsz, Tq, Hq * D) fp32.
        """
        bsz, Tq, Hq, D = q.shape
        Hk, G = self.num_kv_heads, self.gqa
        E = self.rel_extent
        Tk = k_all.shape[1]
        W = self.sliding_window
        # Bound the (heads, queries, keys) logit block of one step to ~64M elements
        Tk_max = Tk if W is None else min(Tk, W + self.q_chunk)
        step = max(16, min(self.q_chunk, (1 << 26) // max(1, Hq * Tk_max * bsz)))
        out = []
        # Global layers: every chunk of queries reads the same keys, so they are converted once, and only the
        # newest keys (within rel_extent of the chunk, or after it) need a bias or a mask
        dense = W is None and Tq > 1
        if dense:
            hi_all = min(Tk, q0 + Tq - k0)
            kt = k_all[:, :hi_all].float().permute(0, 2, 3, 1).contiguous()
            qs = q * self.sm_scale
            # Cached values stay half: the probabilities are written as half too (times a power of two that keeps
            # small ones normal and the output in range) and the value product runs on half tensor cores. The key
            # product is fp32 on TF32 cores, exact for the fp16 keys
            fused = v_all.dtype == torch.half and q.dtype == torch.float32
            if fused:
                vt = v_all[:, :hi_all].transpose(1, 2).contiguous()
                lo_v, hi_v = vt.aminmax()
                p_scale = 2.0 ** max(0, min(12, math.floor(math.log2(30000.0 / max(1e-3, float(max(-lo_v, hi_v)))))))
            else:
                vt = v_all[:, :hi_all].float().transpose(1, 2).contiguous()
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
        for a in range(0, Tq, step):
            b = min(a + step, Tq)
            n = b - a
            if dense and n > 1:
                qc = qs[:, a:b].reshape(bsz, n, Hk, G, D).permute(0, 2, 3, 1, 4).reshape(bsz, Hk, G * n, D)
                logits = torch.matmul(qc, kt).view(bsz, Hq, n, hi_all)
                c0 = max(0, min(hi_all, q0 + a - (E - 1) - k0))
                dist = torch.arange(q0 + a, q0 + b, device = q.device)[:, None] \
                    - torch.arange(k0 + c0, k0 + hi_all, device = q.device)[None, :]
                tail = logits[..., c0:]
                bias = rel[:, a:b].transpose(1, 2).gather(-1, dist.clamp(0, E - 1).expand(bsz, Hq, n, hi_all - c0))
                tail += bias.masked_fill_(~((dist >= 0) & (dist < E)), 0.0)
                tail.masked_fill_(dist < 0, float("-inf"))
                del bias, tail
                if fused:
                    p = torch.empty(logits.shape, dtype = torch.half, device = logits.device)
                    ext.ink_softmax_sink(logits, sink[:, a:b].transpose(1, 2).float().contiguous(), p, p_scale)
                    del logits
                    o = torch.matmul(p.view(bsz, Hk, G * n, hi_all), vt).float().view(bsz, Hk, G, n, D)
                    del p
                    o *= 1.0 / p_scale
                    out.append(o.permute(0, 3, 1, 2, 4).reshape(bsz, n, Hq * D))
                    continue
                # softmax with the sink as one more logit, normalised on the small output instead of the block
                sk = sink[:, a:b].transpose(1, 2).unsqueeze(-1)
                m = torch.maximum(logits.amax(dim = -1, keepdim = True), sk)
                logits.sub_(m).exp_()
                den = logits.sum(dim = -1, keepdim = True) + (sk - m).exp()
                o = torch.matmul(logits.view(bsz, Hk, G * n, hi_all), vt).view(bsz, Hk, G, n, D)
                del logits
                o = o / den.view(bsz, Hk, G, n, 1)
                out.append(o.permute(0, 3, 1, 2, 4).reshape(bsz, n, Hq * D))
                continue
            # Keys this chunk can see: up to its last query, and for sliding layers from its first window on
            hi = min(Tk, q0 + b - k0)
            lo = 0 if W is None else max(0, q0 + a - (W - 1) - k0)
            kc = k_all[:, lo:hi].float()
            vc = v_all[:, lo:hi].float()
            # Query head h reads KV head h // G
            if n == 1:
                qc = q[:, a:b].reshape(bsz, Hk, G, D)                        # the permute/reshape below is the identity here
            else:
                qc = q[:, a:b].reshape(bsz, n, Hk, G, D).permute(0, 2, 3, 1, 4).reshape(bsz, Hk, G * n, D)
            logits = torch.matmul(qc, kc.permute(0, 2, 3, 1)).view(bsz, Hq, n, hi - lo)
            logits *= self.sm_scale
            if n == 1:
                # One query: keys [lo, hi) are all at or before it and, on sliding layers, inside its window, so
                # nothing is masked. The bias row is read back to front from the newest key; keys farther than
                # rel_extent get no bias
                L = hi - lo
                d0 = q0 + a - (k0 + hi - 1)                                      # distance to the newest key
                Lb = min(L, E - d0)
                if Lb > 0:
                    logits[..., L - Lb:] += rel[:, a:b, :, d0 : d0 + Lb].flip(-1).transpose(1, 2)
            else:
                qp = torch.arange(q0 + a, q0 + b, device = q.device)
                kp = torch.arange(k0 + lo, k0 + hi, device = q.device)
                dist = qp[:, None] - kp[None, :]                                 # (n, Tk)
                in_rel = (dist >= 0) & (dist < E)
                idx = dist.clamp(0, E - 1).expand(bsz, Hq, n, hi - lo)
                bias = rel[:, a:b].transpose(1, 2).gather(-1, idx)
                logits += bias.masked_fill_(~in_rel, 0.0)
                del bias
                masked = dist < 0
                if W is not None:
                    masked = masked | (dist >= W)
                logits.masked_fill_(masked, float("-inf"))
            # The sink is one extra logit per query with a zero value: it only absorbs probability mass
            logits = torch.cat([logits, sink[:, a:b].transpose(1, 2).unsqueeze(-1)], dim = -1)
            p = torch.softmax(logits, dim = -1)[..., :-1].reshape(bsz, Hk, G * n, hi - lo)
            del logits
            o = torch.matmul(p, vc.transpose(1, 2)).view(bsz, Hk, G, n, D)
            if n == 1:
                out.append(o.reshape(bsz, n, Hq * D))                         # same element order as the permute below
            else:
                out.append(o.permute(0, 3, 1, 2, 4).reshape(bsz, n, Hq * D))
        if dense:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return torch.cat(out, dim = 1) if len(out) > 1 else out[0]

    def finish(self, o: torch.Tensor, gate: torch.Tensor, params: dict) -> torch.Tensor:
        o = (o * gate).to(ACT_DTYPE).contiguous()
        return self.o_proj.forward(o, params, out_dtype = torch.float)

    @override
    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError("Inkling2Attention is driven by Inkling2Block")


class Inkling2MoE(Module):
    """
    Routed experts with softcapped SwiGLU: tanh(g / c_g) * c_g * sigmoid(g) * tanh(u / c_u) * c_u.
    Router: sigmoid scores, top-k chosen on scores + selection bias, weights are the selected scores
    normalized to sum 1 and scaled by route_scale. Routing and the weighted sum run in fp32.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        route_scale: float,
        gate_softcap: float,
        up_softcap: float,
        qmap: str | None = "block.mlp",
    ):
        super().__init__(config, key, None)
        self.module_name = "Inkling2MoE"
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.route_scale = route_scale
        self.gate_softcap = gate_softcap
        self.up_softcap = up_softcap
        self.gates, self.ups, self.downs = [], [], []
        for i in range(num_experts):
            g = Linear(config, f"{key}.experts.{i}.gate_proj", hidden_size, intermediate_size,
                       qmap = qmap and qmap + ".input", out_dtype = torch.float, qgroup = key + ".gate")
            u = Linear(config, f"{key}.experts.{i}.up_proj", hidden_size, intermediate_size,
                       qmap = qmap and qmap + ".input", out_dtype = torch.float, qgroup = key + ".up")
            d = Linear(config, f"{key}.experts.{i}.down_proj", intermediate_size, hidden_size,
                       qmap = qmap and qmap + f".e{i}.down", out_dtype = torch.float, qgroup = key + ".down")
            self.gates.append(g); self.ups.append(u); self.downs.append(d)
            self.register_submodule(g)
            self.register_submodule(u)
            self.register_submodule(d)
        self.router = None
        self.bias = None

    @override
    def optimizer_targets(self):
        g = [m.optimizer_targets() for m in self.gates]
        u = [m.optimizer_targets() for m in self.ups]
        d = [m.optimizer_targets() for m in self.downs]
        return [g + u, d]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.router = _side(self.config, f"{self.key}.gate.weight", device)
        self.bias = _side(self.config, f"{self.key}.gate.bias", device)

    @override
    def unload(self):
        self.router = self.bias = None
        super().unload()

    @override
    def get_tensors(self):
        t = super().get_tensors()
        for m in self.modules:
            t.update(m.get_tensors())
        t[f"{self.key}.gate.weight"] = self.router
        t[f"{self.key}.gate.bias"] = self.bias
        return t

    def route(self, x: torch.Tensor):
        scores = torch.sigmoid(F.linear(x.float(), self.router))
        idx = torch.topk(scores + self.bias, self.top_k, dim = -1, sorted = False).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(dim = -1, keepdim = True) + 1e-20) * self.route_scale
        return idx, w

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        idx, w = self.route(x)
        xh = x.to(ACT_DTYPE)
        y = torch.zeros_like(x, dtype = torch.float)
        cg, cu = self.gate_softcap, self.up_softcap
        if x.shape[0] <= 4:
            # Decode: one host sync for the routing, then each (row, expert) pair on its own
            for r, experts in enumerate(idx.tolist()):
                xe = xh[r:r + 1].unsqueeze(0)
                g = torch.cat([self.gates[e].forward(xe, params) for e in experts], dim = 1)     # (1, k, I)
                u = torch.cat([self.ups[e].forward(xe, params) for e in experts], dim = 1)
                a = (torch.tanh(g / cg) * cg * torch.sigmoid(g) * torch.tanh(u / cu) * cu).to(ACT_DTYPE)
                d = torch.cat([
                    self.downs[e].forward(a[:, i : i + 1].contiguous(), params, out_dtype = torch.float)
                    for i, e in enumerate(experts)
                ], dim = 1)
                y[r] = torch.matmul(w[r], d[0])
            return y.view(shape)
        # One host sync for the set of routed experts, ascending order
        for e in torch.unique(idx).tolist():
            rows, slot = torch.where(idx == e)
            xe = xh[rows].contiguous().unsqueeze(0)
            g = self.gates[e].forward(xe, params).float()
            u = self.ups[e].forward(xe, params).float()
            a = torch.tanh(g / cg) * cg * torch.sigmoid(g) * torch.tanh(u / cu) * cu
            d = self.downs[e].forward(a.to(ACT_DTYPE).contiguous(), params, out_dtype = torch.float)[0]
            y.index_add_(0, rows, d * w[rows, slot].unsqueeze(-1))
        return y.view(shape)


class Inkling2Block(Module):
    """One decoder layer. Applied once per pass; params["layer_instance"] is the pass for loop layers."""

    def __init__(
        self,
        config: Config,
        key: str,
        key_prefix: str,
        layer_idx: int,
        is_full: bool,
        loop_range: tuple,
        num_passes: int,
        block_len: int,
        hidden_size: int,
        conv_kernel_size: int,
        eps: float,
        attn: Inkling2Attention,
        mlp: Inkling2MoE,
    ):
        super().__init__(config, key, None)
        self.module_name = "Inkling2Block"
        self.layer_idx = layer_idx
        self.key_prefix = key_prefix
        self.is_full = is_full
        self.loop_start, self.loop_end = loop_range
        self.in_loop = self.loop_start <= layer_idx < self.loop_end
        self.num_passes = num_passes
        self.is_block_boundary = layer_idx % block_len == 0
        self.hidden_size = hidden_size
        self.ksize = conv_kernel_size
        self.eps = eps
        self.attn = attn
        self.mlp = mlp
        self.register_submodule(attn)
        self.register_submodule(mlp)
        self.gates = {
            n: Inkling2MessageGate(config, f"{key}.controllers.{n}", eps)
            for n in ("attn", "k", "v", "mlp")
        } if self.in_loop else {}
        self.norm_keys = (
            {n: f"{key}.{n}_input_norm.weight" for n in ("q", "k", "v")} if is_full
            else {"q": f"{key}.attn_norm.weight"}
        )
        self.norm_keys["mlp"] = f"{key}.mlp_norm.weight"
        self.conv_keys = {"attn": f"{key}.attn_sconv.weight", "mlp": f"{key}.mlp_sconv.weight"}
        # attn_queries.{i} belongs to layer i + 1 (layer 0 has no attention-input residual)
        self.query_keys = {"mlp": f"{key_prefix}.attn_res.mlp_queries.{layer_idx}"}
        if layer_idx > 0:
            self.query_keys["attn"] = f"{key_prefix}.attn_res.attn_queries.{layer_idx - 1}"
        self.msg_keys = (
            f"{key_prefix}.middle_cycle_cross_pass_queries",
            f"{key_prefix}.middle_cycle_cross_pass_kv_queries",
        )
        self.norms = None
        self.convs = None
        self.convs_f = None
        self.queries = None
        self.queries_t = None
        self.msg_query = None
        self.msg_query_t = None
        self.qkv_norm = self.gate_down = self.gate_up = None

        # Conv history channels: [K projection | V projection | attention output | MLP output]
        self.kv_channels = attn.num_kv_heads * attn.head_dim
        self.ring_channels = 2 * self.kv_channels + 2 * hidden_size
        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = Inkling2ConvLayerState
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}

    @override
    def optimizer_targets(self):
        return [self.attn.optimizer_targets(), self.mlp.optimizer_targets()]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        c = self.config
        self.norms = {n: _side(c, k, device) for n, k in self.norm_keys.items()}
        self.convs = {n: _side(c, k, device) for n, k in self.conv_keys.items()}
        self.convs_f = {n: _fold_sliding(w) for n, w in self.convs.items()}
        self.queries = {n: _side(c, k, device).view(-1, self.hidden_size) for n, k in self.query_keys.items()}
        self.queries_t = {n: _prep_query(q) for n, q in self.queries.items()}
        for g in self.gates.values():
            g.load(device)
        self.qkv_norm = torch.stack(
            [self.norms[n] for n in (("q", "k", "v") if self.is_full else ("q", "q", "q"))]
        ).contiguous()
        if self.in_loop:
            # The four message gates stacked for the batched gate computation (see _gates)
            gl = list(self.gates.values())
            self.gate_down = torch.stack([g.down.t() for g in gl]).contiguous()                     # (4, 2 dim, rank)
            self.gate_up = torch.stack([torch.cat([g.forget.t(), g.gain.t()], dim = 1) for g in gl]).contiguous()
        if self.in_loop:
            # Stored as two (2 * num_loop_layers, dim) tensors: [query, mlp] and [key, value] per loop layer
            i = self.layer_idx - self.loop_start
            qm = _side(c, self.msg_keys[0], device).view(-1, 2, self.hidden_size)[i]
            kv = _side(c, self.msg_keys[1], device).view(-1, 2, self.hidden_size)[i]
            self.msg_query = torch.stack([qm[0], kv[0], kv[1], qm[1]], dim = 0).contiguous()
            self.msg_query_t = _prep_query(self.msg_query)
        for rl in self.recurrent_layers:
            rl.alloc(device)

    @override
    def unload(self):
        self.norms = self.convs = self.convs_f = self.queries = self.msg_query = None
        self.queries_t = self.msg_query_t = None
        self.qkv_norm = self.gate_down = self.gate_up = None
        for g in self.gates.values():
            g.unload()
        for rl in self.recurrent_layers:
            rl.free()
        super().unload()

    def _ring(self, params: dict, instance: int) -> Inkling2Ring:
        rsg = params.get("recurrent_states")
        if not rsg:
            return Inkling2Ring(None, None, False, self.ksize - 1)
        (state,) = rsg[0].cache.get_recurrent_layer((self.layer_idx, instance)).get_state_tensors()
        slots = [r.slot for r in rsg]
        return Inkling2Ring(state, slots, bool(params.get("recurrent_history")), self.ksize - 1)

    @override
    def get_tensors(self):
        t = {}
        for m in self.modules:
            t.update(m.get_tensors())
        for n, k in self.norm_keys.items():
            t[k] = self.norms[n]
        for n, k in self.conv_keys.items():
            t[k] = self.convs[n]
        for n, k in self.query_keys.items():
            q = self.queries[n]
            t[k] = q[0] if q.shape[0] == 1 else q
        for g in self.gates.values():
            t.update(g.tensors())
        return t

    def _t1_rows(self, params: dict, ring: Inkling2Ring, bsz: int, seq: int):
        """Block-table rows when the single-token kernels apply (one sequence, pages up to the position), else None.
        The pages need not be contiguous or ordered: the kernels translate every logical position through the table"""
        if not _T1 or bsz != 1 or seq != 1 or ring.state is None or len(ring.slots) != 1:
            return None
        bt = params.get("block_table")
        if params.get("cache") is None or bt is None or bt.is_cuda:
            return None
        rows = bt.tolist()
        # Only the pages holding positions 0..pos have to be in the table: the generator hands over a table padded
        # with unused entries
        used = int(params["cache_seqlens"].tolist()[0]) // PAGE_SIZE + 1
        if used > len(rows[0]):
            return None
        return rows

    def _forward_t1(self, x: torch.Tensor, st: Inkling2State, messages, ring: Inkling2Ring, params: dict,
                    rows: list) -> torch.Tensor:
        """
        Single-token forward of one sequence on a paged cache: the same computation as the general path, with each
        pure-tensor segment as one kernel. x: (1, 1, dim) fp32, already loop-bookkept by forward().
        """
        attn = self.attn
        H, kvc, taps = self.hidden_size, self.kv_channels, self.ksize - 1
        instance = params.get("layer_instance", 0)
        slot = ring.slots[0]
        cache, bt = params["cache"], params["block_table"]
        pos = int(params["cache_seqlens"].tolist()[0])
        pt = _page_table(rows[0][:pos // PAGE_SIZE + 1], x.device)
        lo = 0 if attn.sliding_window is None else max(0, pos - (attn.sliding_window - 1))
        L = pos + 1 - lo
        kc, vc = cache.get_layer(self.layer_idx, params["cache_seqlens"], bt, instance = instance)
        kf = kc.view(-1, attn.num_kv_heads, attn.head_dim)
        vf = vc.view(-1, attn.num_kv_heads, attn.head_dim)
        tau = attn._tau_host(pos) if attn.log_scaling is not None else 1.0

        # attention residual over the block states, message gates, norms -> the sublayer inputs
        xf = x.reshape(H)
        if self.layer_idx == 0:
            hs = xf.view(1, H)
        else:
            hs = ext.ink_attn_res(st.block_states + [xf], self.queries_t["attn"], self.eps)      # (nq, H)
        if messages is None and not self.is_full:
            y = F.rms_norm(hs[0], (H,), self.qkv_norm[0], self.eps).view(1, 1, H)
            q_in = k_in = v_in = y
        else:
            X = hs if hs.shape[0] == 3 else hs.expand(3, H)
            if messages is not None:
                X = ext.ink_gates(X, messages.reshape(4, H)[0:3], self.gate_down, self.gate_up, 0, 3, self.eps)
            y = F.rms_norm(X, (H,), None, self.eps) * self.qkv_norm                             # (3, H)
            q_in, k_in, v_in = (y[i].view(1, 1, H) for i in range(3))

        q, rel, sink, gate = attn.project_t1(q_in, k_in, v_in, tau, params, ring, slot, kf, vf, pos, pt)
        o = attn.attend_t1(q, rel, sink, gate, kf, vf, pt, lo, L)
        cache.update_layer(self.layer_idx, params["cache_seqlens"], bt, kc, vc, 1, instance)
        a = attn.o_proj.forward(o.view(1, 1, -1), params, out_dtype = torch.float)              # (1, 1, H)

        # attention output conv; a block boundary also appends the block state (the residual is then the conv output)
        c_attn = 2 * kvc
        if self.is_block_boundary:
            st.block_states = st.block_states + [x]
            residual = ext.ink_sconv(ring.state[slot, c_attn:c_attn + H], a.reshape(H), self.convs_f["attn"], None,
                                     taps, ring.record)
        else:
            residual = ext.ink_sconv(ring.state[slot, c_attn:c_attn + H], a.reshape(H), self.convs_f["attn"], xf,
                                     taps, ring.record)
        residual = residual.view(1, 1, H)

        # MLP input: attention residual over the (extended) block states, message gate, norm
        m = ext.ink_attn_res(st.block_states + [residual.reshape(H)], self.queries_t["mlp"], self.eps)   # (1, H)
        if messages is not None:
            m = ext.ink_gates(m, messages.reshape(4, H)[3:4], self.gate_down, self.gate_up, 3, 1, self.eps)
        mn = F.rms_norm(m, (H,), self.norms["mlp"], self.eps).view(1, 1, H)
        mlp_in = mn if isinstance(self.mlp, Inkling2MoE) else mn.to(ACT_DTYPE).contiguous()
        m_out = self.mlp.forward(mlp_in, params)

        c_mlp = 2 * kvc + H
        out = ext.ink_sconv(ring.state[slot, c_mlp:c_mlp + H], m_out.reshape(H), self.convs_f["mlp"],
                            residual.reshape(H), taps, ring.record)
        return out.view(1, 1, H)

    def _norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.rms_norm(x.float(), (self.hidden_size,), self.norms[name], self.eps)

    def _gates(self, X: torch.Tensor, M: torch.Tensor, first: int) -> torch.Tensor:
        """
        Message gates first .. first + G on the rows X, M: (bsz, seq, G, dim) fp32, states and messages. Per row and gate
        this is Inkling2MessageGate: (1 - tanh(forget)) * X + tanh(gain) * M, the low-rank gates over [rms(X), rms(M)].
        All G gates run as two batched matmuls over the stacked weights (see load).
        """
        bsz, seq, G, H = X.shape
        N = bsz * seq
        xs = X.reshape(N, G, H).transpose(0, 1)                                 # (G, N, dim)
        ms = M.reshape(N, G, H).transpose(0, 1)
        z = F.rms_norm(torch.stack([xs, ms], dim = 2), (H,), None, self.eps).reshape(G, N, 2 * H)
        g = torch.bmm(z, self.gate_down[first : first + G])                     # (G, N, rank)
        t = torch.tanh(torch.bmm(g, self.gate_up[first : first + G]))           # (G, N, 2 * dim): forget | gain
        out = (1 - t[..., :H]) * xs + t[..., H:] * ms
        return out.transpose(0, 1).reshape(bsz, seq, G, H)

    def _attn_inputs(self, hs: torch.Tensor, messages: torch.Tensor | None):
        """
        Inputs (q, k, v) of the attention sublayer from its attention residual hs (bsz, seq, nq, dim). Full layers take
        the three inputs from three queries (one for layer 0), sliding layers take one. Message gates, when present, mix
        the messages into each input; every input is then normalized with its own norm (the q norm on sliding layers).
        """
        bsz, seq, nq, H = hs.shape
        if messages is None and not self.is_full:
            y = self._norm(hs[..., 0, :], "q")
            return y, y, y
        if nq != 3:
            hs = hs.expand(bsz, seq, 3, H)
        if messages is not None:
            hs = self._gates(hs, messages[..., :3, :], 0)
        y = F.rms_norm(hs.float(), (H,), None, self.eps) * self.qkv_norm
        return y[..., 0, :], y[..., 1, :], y[..., 2, :]

    def _mlp_input(self, x: torch.Tensor) -> torch.Tensor:
        # The Python MoE takes the fp32 normalized stream (its router runs in fp32); BlockSparseMLP's
        # expert kernels and router take the half-precision rows
        return x if isinstance(self.mlp, Inkling2MoE) else x.to(ACT_DTYPE).contiguous()

    def _sconv(self, x: torch.Tensor, name: str, ctx: torch.Tensor) -> torch.Tensor:
        # (conv(x) + x) / sqrt(k + 1), causal depthwise conv (scale and residual folded into convs_f);
        # ctx: (bsz, k - 1 + seq, dim) = [preceding inputs | x]
        return (ctx.unfold(1, self.ksize, 1) * self.convs_f[name]).sum(-1)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        st = Inkling2State.get(params)
        instance = params.get("layer_instance", 0)
        dev = self.device
        x = x.float()
        bsz, seq, _ = x.shape

        # Loop bookkeeping at the first loop layer: snapshot on the first pass, restart on later ones
        messages = None
        if self.in_loop:
            if self.layer_idx == self.loop_start:
                if instance == 0:
                    st.pre_loop = (x, list(st.block_states))
                else:
                    # The previous pass' last block state (its earlier ones restate the pre-loop residual)
                    # and its output are the message sources
                    st.prev_pass = (st.block_states[-1], x)
                    x, blocks = st.pre_loop
                    st.block_states = list(blocks)
            if instance > 0:
                prev_block, prev_out = st.prev_pass
                messages = attn_res([prev_block.to(dev)], prev_out.to(dev), self.msg_query_t)
        # DSpark drafter taps: the partial block entering this visit (vLLM loop.py aux_hidden_state_layers)
        exp = params.get("export_state_layers")
        if exp and self.visit_index(instance) in exp:
            _export_tap(params, x)
        st.block_states = [s if s.device == dev else s.to(dev) for s in st.block_states]
        ring = self._ring(params, instance)
        rows = self._t1_rows(params, ring, bsz, seq)
        if rows is not None:
            return self._forward_t1(x, st, messages, ring, params, rows)

        block_residual = x
        if self.layer_idx == 0:
            hs = x.unsqueeze(-2)
        else:
            hs = attn_res(st.block_states, x, self.queries_t["attn"])
        q_in, k_in, v_in = self._attn_inputs(hs, messages)

        a = self.attention(q_in, k_in, v_in, params, instance, ring)
        a = self._sconv(a, "attn", _with_history(ring.exchange(2 * self.kv_channels, a), a, self.ksize - 1))

        if self.is_block_boundary:
            st.block_states = st.block_states + [block_residual]
            residual = a
        else:
            residual = block_residual + a

        m = attn_res(st.block_states, residual, self.queries_t["mlp"])[..., 0, :]
        if messages is not None:
            m = self._gates(m.unsqueeze(-2), messages[..., 3:4, :], 3)[..., 0, :]
        m = self.mlp.forward(self._mlp_input(self._norm(m, "mlp")), params)
        m = self._sconv(m, "mlp", _with_history(
            ring.exchange(2 * self.kv_channels + self.hidden_size, m), m, self.ksize - 1))
        return residual + m

    def visit_index(self, instance: int) -> int:
        """Index of this block call in the forward's visit sequence (prelude, core x passes, coda): the
        numbering the drafter's target_layer_ids use"""
        ncore = self.loop_end - self.loop_start
        if self.in_loop:
            return self.layer_idx + instance * ncore
        if self.layer_idx >= self.loop_end:
            return self.layer_idx + (self.num_passes - 1) * ncore
        return self.layer_idx

    def attention(self, q_in, k_in, v_in, params: dict, instance: int, ring: Inkling2Ring) -> torch.Tensor:
        attn = self.attn
        bsz, seq, _ = q_in.shape
        dev = q_in.device
        cache = params.get("cache")

        if cache is None:
            pos0 = params.get("position", 0) or 0
            if seq == 1:
                tau = attn._tau([int(pos0)] * bsz, None, dev)
            else:
                pos = torch.arange(pos0, pos0 + seq, device = dev)
                tau = attn._tau(None, pos.unsqueeze(0).expand(bsz, -1), dev)
            q, k, v, rel, sink, gate = attn.project(q_in, k_in, v_in, tau, params, ring)
            o = attn.attend(q, rel, sink, k, v, pos0, pos0)
            return attn.finish(o, gate, params)

        # Paged cache: the new K/V rows go to positions cache_seqlens.. of each sequence's pages, then every
        # sequence attends over its own pages (all of them, or the ones its window reaches)
        block_table = params["block_table"]
        past = [int(p) for p in params["cache_seqlens"].tolist()]
        if seq == 1 and not block_table.is_cuda:
            rows = block_table.tolist()
            if all(row[j] == row[0] + j for row in rows for j in range(len(row))):
                return self._decode_contiguous(q_in, k_in, v_in, params, past, rows, instance, ring)

        # Device views of the staged tables (seeded by the generator's arena upload, so no per-call copy or sync):
        # positions come from cache_seqlens on the device, the host list gives the shapes
        block_table = get_for_device(params, "block_table", dev)
        cs_dev = get_for_device(params, "cache_seqlens", dev)
        ar = torch.arange(seq, device = dev)
        positions = cs_dev.long().unsqueeze(1) + ar.unsqueeze(0)
        q, k, v, rel, sink, gate = attn.project(q_in, k_in, v_in, attn._tau(None, positions, dev), params, ring)

        kc, vc = cache.get_layer(self.layer_idx, cs_dev, block_table, instance = instance)
        bt = block_table.to(kc.device).long()
        outs = []
        for i in range(bsz):
            pos = positions[i].to(kc.device)
            page, off = bt[i][pos // PAGE_SIZE], pos % PAGE_SIZE
            kc[page, off] = k[i].to(kc.device)
            vc[page, off] = v[i].to(kc.device)
            end = past[i] + seq
            lo = 0 if attn.sliding_window is None else max(0, past[i] - (attn.sliding_window - 1))
            p0, p1 = lo // PAGE_SIZE, (end - 1) // PAGE_SIZE + 1
            a, b = lo - p0 * PAGE_SIZE, end - p0 * PAGE_SIZE
            pages = bt[i, p0:p1]
            first = int(pages[0])
            if int(pages[-1]) - first == p1 - p0 - 1 and bool((pages[1:] - pages[:-1] == 1).all()):
                kp, vp = kc[first : first + p1 - p0], vc[first : first + p1 - p0]       # consecutive pages: no copy
            else:
                kp, vp = kc[pages], vc[pages]
            k_all = kp.view(-1, attn.num_kv_heads, attn.head_dim)[a:b].unsqueeze(0).to(dev)
            v_all = vp.view(-1, attn.num_kv_heads, attn.head_dim)[a:b].unsqueeze(0).to(dev)
            outs.append(attn.attend(q[i:i+1], rel[i:i+1], sink[i:i+1], k_all, v_all, past[i], lo))
        cache.update_layer(self.layer_idx, params["cache_seqlens"], block_table, kc, vc, seq, instance)
        o = torch.cat(outs, dim = 0) if bsz > 1 else outs[0]
        return attn.finish(o, gate, params)

    def _decode_contiguous(self, q_in, k_in, v_in, params: dict, past: list, rows: list, instance: int,
                           ring: Inkling2Ring) -> torch.Tensor:
        """
        Single-token paged step where every sequence's pages are consecutive: the cache is one flat run of rows, so
        the new row is a plain write and the attended keys are views (no gather). Host integers only.
        """
        attn = self.attn
        bsz = q_in.shape[0]
        dev = q_in.device
        cache = params["cache"]
        block_table = params["block_table"]
        P = PAGE_SIZE
        q, k, v, rel, sink, gate = attn.project(q_in, k_in, v_in, attn._tau(past, None, dev), params, ring)
        kc, vc = cache.get_layer(self.layer_idx, params["cache_seqlens"], block_table, instance = instance)
        kf = kc.view(-1, attn.num_kv_heads, attn.head_dim)
        vf = vc.view(-1, attn.num_kv_heads, attn.head_dim)
        outs = []
        for i in range(bsz):
            p = past[i]
            base = rows[i][0] * P
            flat = base + p
            kf[flat] = k[i, 0]
            vf[flat] = v[i, 0]
            lo = 0 if attn.sliding_window is None else max(0, p - (attn.sliding_window - 1))
            k_all = kf[base + lo : flat + 1].unsqueeze(0)
            v_all = vf[base + lo : flat + 1].unsqueeze(0)
            outs.append(attn.attend(q[i:i+1], rel[i:i+1], sink[i:i+1], k_all, v_all, p, lo))
        cache.update_layer(self.layer_idx, params["cache_seqlens"], block_table, kc, vc, 1, instance)
        o = torch.cat(outs, dim = 0) if bsz > 1 else outs[0]
        return attn.finish(o, gate, params)


class Inkling2Final(Module):
    """Final attention residual over the block states, weighted RMSNorm, unembedding and logit softcap"""

    def __init__(
        self,
        config: Config,
        key: str,
        key_query: str,
        key_norm: str,
        key_head: str,
        hidden_size: int,
        padded_vocab_size: int,
        vocab_size: int,
        eps: float,
        softcap: float,
    ):
        super().__init__(config, key, None)
        self.module_name = "Inkling2Final"
        self.key_query = key_query
        self.key_norm = key_norm
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.eps = eps
        self.head = Linear(
            config, key_head, hidden_size, padded_vocab_size,
            qmap = "block", qbits_key = "head_bits", softcap = softcap, out_dtype = torch.half,
        )
        self.register_submodule(self.head)
        self.caps.update({"logits_output": True})
        self.sparse_head = None
        self.query = None
        self.query_t = None
        self.norm_weight = None
        self.final_visit = None     # visit index past the last block (set by the model): tap 48

    @override
    def optimizer_targets(self):
        return [self.head.optimizer_targets()]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.query = _side(self.config, self.key_query, device).view(-1, self.hidden_size)
        self.query_t = _prep_query(self.query)
        self.norm_weight = _side(self.config, self.key_norm + ".weight", device)

    @override
    def unload(self):
        self.query = self.query_t = self.norm_weight = None
        super().unload()

    @override
    def get_tensors(self):
        t = self.head.get_tensors()
        t[self.key_query] = self.query[0]
        t[self.key_norm + ".weight"] = self.norm_weight
        return t

    @override
    def hidden(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Normed final state (the head's input). Also the DSpark drafter's last tap (visit n_visits)."""
        st = Inkling2State.get(params)
        n = x.shape[1]
        # The forward loop may have cropped x to the last tokens; the block states follow
        blocks = [s[:, -n:].to(self.device) for s in st.block_states]
        x = attn_res(blocks, x.float(), self.query_t)[..., 0, :]
        x = (_rms(x, self.eps) * self.norm_weight).to(ACT_DTYPE).contiguous()
        exp = params.get("export_state_layers")
        if exp and self.final_visit is not None and self.final_visit in exp:
            _export_tap(params, x)
        return x

    def draft_logits(self, x: torch.Tensor, params: dict, softcap: bool = True) -> torch.Tensor:
        """Head logits (padded width) of a drafter's normed state, with or without the head's softcap"""
        cap = self.head.softcap
        if not softcap:
            self.head.softcap = 0.0
        try:
            return self.head.forward(x, params)
        finally:
            self.head.softcap = cap

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        x = self.hidden(x, params)
        logits = self.head.forward(x, params, out_dtype)
        # Rows past the tokenizer's vocabulary are padding in the checkpoint
        return logits[..., :self.vocab_size]
