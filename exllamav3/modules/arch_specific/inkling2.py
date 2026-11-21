from __future__ import annotations
from typing_extensions import override
import math
import torch
import torch.nn.functional as F
from ...model.config import Config
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


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim = True) + eps)


def _l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim = -1, keepdim = True)
    return x * (threshold / (norm + 1e-6)).clamp(max = 1.0)


def _side(config: Config, key: str, device, optional: bool = False) -> torch.Tensor | None:
    # Read inside the loader's deferred bracket, where plain get_tensor() hands out buffers that are only
    # filled when the bracket closes: these are converted right away, so they must load eagerly
    t = config.stc.get_tensor(key, device, optional = optional, allow_bf16 = True, no_defer = True)
    return None if t is None else t.float()


def attn_res(block_states: list[torch.Tensor], partial: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    Attention residual: softmax-pool the (unweighted-RMS-normalized) block states and the partial state with
    learned queries. block_states: list of (bsz, seq, dim), partial: (bsz, seq, dim), query: (nq, dim).
    Returns (bsz, seq, nq, dim).
    """
    stacked = torch.stack([s.float() for s in block_states] + [partial.float()], dim = -2)
    stacked = _rms(stacked, 1e-6)
    dim = stacked.shape[-1]
    w = torch.matmul(stacked, query.float().T) * (dim ** -0.5)      # (bsz, seq, S, nq)
    w = torch.softmax(w, dim = -2)
    return torch.matmul(w.transpose(-1, -2), stacked)               # (bsz, seq, nq, dim)


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
            m = Linear(config, f"{key}.{name}", hidden_size, n_out, qmap = qm, out_dtype = torch.half)
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
        self.caps.update({"kv_cache": False})

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

    @override
    def unload(self):
        self.sink_proj = self.q_scale = self.rel_scale = self.rel_proj = None
        self.k_conv = self.v_conv = None
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

    def _kv_conv(self, x: torch.Tensor, dyn_in: torch.Tensor, conv: dict, history: torch.Tensor | None):
        """
        Short causal convolution over the K or V projection. x: (bsz, seq, C) fp32, history: (bsz, ksize - 1, C)
        of the preceding inputs (zeros at sequence start). Sliding layers: static depthwise taps; full layers:
        static per-channel taps plus per-token, per-head taps projected from the projection's own input.
        """
        bsz, seq, C = x.shape
        k = self.ksize
        if history is None:
            history = x.new_zeros((bsz, k - 1, C))
        ctx = torch.cat([history, x], dim = 1)                                  # (bsz, seq + k - 1, C)
        if not self.is_full:
            w = conv["weight"]                                                  # (C, 1, k), tap k - 1 = current
            y = F.conv1d(ctx.transpose(1, 2), w, None, padding = 0, groups = C).transpose(1, 2)
            return (y + x) / math.sqrt(k + 1)
        H = self.num_kv_heads
        dyn = F.linear(dyn_in.float(), conv["proj"]).view(bsz, seq, H, 1, k)    # per token, per head, per lag
        w = conv["static"].T.reshape(H, C // H, k) + dyn                        # (bsz, seq, H, C/H, k), lag 0 = current
        win = ctx.unfold(1, k, 1).flip(-1)                                      # (bsz, seq, C, k), [..., lag]
        win = win.reshape(bsz, seq, H, C // H, k)
        y = (win * w).sum(-1).reshape(bsz, seq, C)
        return (y + x) / math.sqrt(2 * k + 1)

    def project(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor, positions: torch.Tensor,
                params: dict, k_hist = None, v_hist = None):
        """
        Returns q (bsz, seq, Hq, D), k / v (bsz, seq, Hkv, D), relative logits (bsz, seq, Hq, rel_extent),
        sink logits (bsz, seq, Hq), output gate (bsz, seq, Hq * D), and the raw K/V projections (conv inputs)
        """
        bsz, seq, _ = q_in.shape
        qh = q_in.half().contiguous()
        same = k_in is q_in and v_in is q_in
        kh = qh if same else k_in.half().contiguous()
        vh = qh if same else v_in.half().contiguous()
        tau = self._log_scale(positions)

        q = self.q_proj.forward(qh, params).float().view(bsz, seq, self.num_q_heads, self.head_dim)
        q = _l2_clip(q * self.q_scale, self.qk_clip)
        if tau is not None:
            q = q * tau[..., None, None]

        k_raw = self.k_proj.forward(kh, params).float()
        v_raw = self.v_proj.forward(vh, params).float()
        k = self._kv_conv(k_raw, k_in, self.k_conv, k_hist).view(bsz, seq, self.num_kv_heads, self.head_dim)
        v = self._kv_conv(v_raw, v_in, self.v_conv, v_hist).view(bsz, seq, self.num_kv_heads, self.head_dim)
        k = _l2_clip(k, self.qk_clip)
        v = _rms(v, self.eps)

        r = self.r_proj.forward(qh, params).float().view(bsz, seq, self.num_q_heads, self.d_rel)
        r = _l2_clip(r * self.rel_scale, self.rel_clip)
        if tau is not None:
            r = r * tau[..., None, None]
        rel = torch.matmul(r, self.rel_proj)                                    # (bsz, seq, Hq, rel_extent)

        sink = F.linear(q_in.float(), self.sink_proj)                           # (bsz, seq, Hq)
        gate = torch.sigmoid(self.g_proj.forward(qh, params).float())
        return q, k, v, rel, sink, gate, k_raw, v_raw

    def attend(self, q, rel, sink, k_all, v_all, q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
        """
        q: (bsz, Tq, Hq, D), rel: (bsz, Tq, Hq, E), sink: (bsz, Tq, Hq), k_all / v_all: (bsz, Tk, Hkv, D),
        q_pos: (Tq,) and k_pos: (Tk,) absolute positions (ascending). Returns (bsz, Tq, Hq * D) fp32.
        """
        bsz, Tq, Hq, D = q.shape
        Hk, G = self.num_kv_heads, self.gqa
        E = self.rel_extent
        out = []
        for a in range(0, Tq, self.q_chunk):
            b = min(a + self.q_chunk, Tq)
            qp = q_pos[a:b]
            # Keys this chunk can see: up to its last query, and for sliding layers from its first window on
            hi = int(torch.searchsorted(k_pos, qp[-1], right = True))
            lo = 0
            if self.sliding_window is not None:
                lo = int(torch.searchsorted(k_pos, qp[0] - (self.sliding_window - 1), right = False))
            kp = k_pos[lo:hi]
            kc = k_all[:, lo:hi].float()
            vc = v_all[:, lo:hi].float()
            n = b - a
            qc = q[:, a:b].reshape(bsz, n, Hk, G, D)
            logits = torch.einsum("bqhgd,bkhd->bhgqk", qc, kc).reshape(bsz, Hq, n, hi - lo) * self.sm_scale
            dist = qp[:, None] - kp[None, :]                                     # (n, Tk)
            in_rel = (dist >= 0) & (dist < E)
            idx = dist.clamp(0, E - 1).expand(bsz, Hq, n, hi - lo)
            bias = rel[:, a:b].transpose(1, 2).gather(-1, idx)
            logits = logits + bias.masked_fill(~in_rel, 0.0)
            masked = dist < 0
            if self.sliding_window is not None:
                masked = masked | (dist >= self.sliding_window)
            logits = logits.masked_fill(masked, float("-inf"))
            # The sink is one extra logit per query with a zero value: it only absorbs probability mass
            logits = torch.cat([logits, sink[:, a:b].transpose(1, 2).unsqueeze(-1)], dim = -1)
            p = torch.softmax(logits, dim = -1)[..., :-1].reshape(bsz, Hk, G, n, hi - lo)
            o = torch.einsum("bhgqk,bkhd->bqhgd", p, vc)
            out.append(o.reshape(bsz, n, Hq * D))
        return torch.cat(out, dim = 1) if len(out) > 1 else out[0]

    def finish(self, o: torch.Tensor, gate: torch.Tensor, params: dict) -> torch.Tensor:
        o = (o * gate).half().contiguous()
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
                       qmap = qmap and qmap + ".input", out_dtype = torch.half, qgroup = key + ".gate")
            u = Linear(config, f"{key}.experts.{i}.up_proj", hidden_size, intermediate_size,
                       qmap = qmap and qmap + ".input", out_dtype = torch.half, qgroup = key + ".up")
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
        xh = x.half()
        y = torch.zeros_like(x, dtype = torch.float)
        cg, cu = self.gate_softcap, self.up_softcap
        # One host sync for the set of routed experts, ascending order
        for e in torch.unique(idx).tolist():
            rows, slot = torch.where(idx == e)
            xe = xh[rows].contiguous().unsqueeze(0)
            g = self.gates[e].forward(xe, params).float()
            u = self.ups[e].forward(xe, params).float()
            a = torch.tanh(g / cg) * cg * torch.sigmoid(g) * torch.tanh(u / cu) * cu
            d = self.downs[e].forward(a.half().contiguous(), params, out_dtype = torch.float)[0]
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
        self.queries = None
        self.msg_query = None

    @override
    def optimizer_targets(self):
        return [self.attn.optimizer_targets(), self.mlp.optimizer_targets()]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        c = self.config
        self.norms = {n: _side(c, k, device) for n, k in self.norm_keys.items()}
        self.convs = {n: _side(c, k, device) for n, k in self.conv_keys.items()}
        self.queries = {n: _side(c, k, device).view(-1, self.hidden_size) for n, k in self.query_keys.items()}
        for g in self.gates.values():
            g.load(device)
        if self.in_loop:
            # Stored as two (2 * num_loop_layers, dim) tensors: [query, mlp] and [key, value] per loop layer
            i = self.layer_idx - self.loop_start
            qm = _side(c, self.msg_keys[0], device).view(-1, 2, self.hidden_size)[i]
            kv = _side(c, self.msg_keys[1], device).view(-1, 2, self.hidden_size)[i]
            self.msg_query = torch.stack([qm[0], kv[0], kv[1], qm[1]], dim = 0).contiguous()

    @override
    def unload(self):
        self.norms = self.convs = self.queries = self.msg_query = None
        for g in self.gates.values():
            g.unload()
        super().unload()

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

    def _norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return _rms(x.float(), self.eps) * self.norms[name]

    def _sconv(self, x: torch.Tensor, name: str, history: torch.Tensor | None) -> torch.Tensor:
        # (conv(x) + x) / sqrt(k + 1), causal depthwise conv; history: (bsz, k - 1, dim) preceding inputs
        bsz, seq, C = x.shape
        if history is None:
            history = x.new_zeros((bsz, self.ksize - 1, C))
        ctx = torch.cat([history, x], dim = 1).transpose(1, 2)
        y = F.conv1d(ctx, self.convs[name], None, padding = 0, groups = C).transpose(1, 2)
        return (y + x) / math.sqrt(self.ksize + 1)

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
                messages = attn_res([prev_block.to(dev)], prev_out.to(dev), self.msg_query)
        st.block_states = [s.to(dev) for s in st.block_states]

        block_residual = x
        if self.layer_idx == 0:
            h = (x,)
        else:
            h = attn_res(st.block_states, x, self.queries["attn"]).unbind(-2)

        if self.is_full:
            q_in, k_in, v_in = h if len(h) == 3 else (h[0], h[0], h[0])
            if messages is not None:
                q_in = self.gates["attn"](q_in, messages[..., 0, :])
                k_in = self.gates["k"](k_in, messages[..., 1, :])
                v_in = self.gates["v"](v_in, messages[..., 2, :])
            q_in, k_in, v_in = self._norm(q_in, "q"), self._norm(k_in, "k"), self._norm(v_in, "v")
        elif messages is not None:
            q_in = self._norm(self.gates["attn"](h[0], messages[..., 0, :]), "q")
            k_in = self._norm(self.gates["k"](h[0], messages[..., 1, :]), "q")
            v_in = self._norm(self.gates["v"](h[0], messages[..., 2, :]), "q")
        else:
            q_in = k_in = v_in = self._norm(h[0], "q")

        a = self.attention(q_in, k_in, v_in, params, instance)
        a = self._sconv(a, "attn", None)

        if self.is_block_boundary:
            st.block_states = st.block_states + [block_residual]
            residual = a
        else:
            residual = block_residual + a

        m = attn_res(st.block_states, residual, self.queries["mlp"])[..., 0, :]
        if messages is not None:
            m = self.gates["mlp"](m, messages[..., 3, :])
        m = self.mlp.forward(self._norm(m, "mlp"), params)
        m = self._sconv(m, "mlp", None)
        return residual + m

    def attention(self, q_in, k_in, v_in, params: dict, instance: int) -> torch.Tensor:
        attn = self.attn
        bsz, seq, _ = q_in.shape
        mode = params.get("attn_mode", "flash_attn_nc")
        if mode != "flash_attn_nc":
            raise NotImplementedError("Inkling2Loop: cached attention is not implemented yet")
        pos0 = params.get("position", 0) or 0
        pos = torch.arange(pos0, pos0 + seq, device = q_in.device)
        positions = pos.unsqueeze(0).expand(bsz, -1)
        q, k, v, rel, sink, gate, _, _ = attn.project(q_in, k_in, v_in, positions, params)
        o = attn.attend(q, rel, sink, k, v, pos, pos)
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
        self.norm_weight = None

    @override
    def optimizer_targets(self):
        return [self.head.optimizer_targets()]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.query = _side(self.config, self.key_query, device).view(-1, self.hidden_size)
        self.norm_weight = _side(self.config, self.key_norm + ".weight", device)

    @override
    def unload(self):
        self.query = self.norm_weight = None
        super().unload()

    @override
    def get_tensors(self):
        t = self.head.get_tensors()
        t[self.key_query] = self.query[0]
        t[self.key_norm + ".weight"] = self.norm_weight
        return t

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        st = Inkling2State.get(params)
        n = x.shape[1]
        # The forward loop may have cropped x to the last tokens; the block states follow
        blocks = [s[:, -n:].to(self.device) for s in st.block_states]
        x = attn_res(blocks, x.float(), self.query)[..., 0, :]
        x = (_rms(x, self.eps) * self.norm_weight).half().contiguous()
        logits = self.head.forward(x, params, out_dtype)
        # Rows past the tokenizer's vocabulary are padding in the checkpoint
        return logits[..., :self.vocab_size]
