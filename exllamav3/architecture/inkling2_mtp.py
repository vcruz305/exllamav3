from __future__ import annotations

import math
import os
import weakref

import torch
import torch.nn.functional as F
from typing_extensions import override

from ..model.model import Model
from ..modules.module import Module
from ..modules.linear import Linear
from ..modules.arch_specific.inkling2 import _side

# Inkling2 MTP component: the DSpark block drafter stored in mtp.safetensors.
# Semantics follow the vLLM reference (ref nvidia/dspark.py, ops/dspark.py, speculator.py):
#   - L = processed length. The anchor (last sampled token) sits at position L and is not processed.
#   - Block row j (0..block-1) sits at position L+j: row 0 = normed embedding of the anchor,
#     rows 1.. = mask_embed. Each draft layer attends bidirectionally over [context K/V at 0..L-1 ; block].
#     Local layers see the last sliding_window context keys plus the block.
#   - Context K/V for layer i come from the target taps: mix_i = dflare_norm(sum softmax(fuse_weight[i])[t] * tap_t).
#   - Draft j (1..block-1) = greedy argmax of softcap(lm_head(h_j) + markov bias_j); the Markov head is a
#     rank-r recurrent gate over the previous token; its bias is folded through lm_head once.
# Batch 1 only (the context K/V buffers are per position).


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + eps)
    return x * w.float()


def _rope(x: torch.Tensor, pos: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Neox RoPE on x (T, H, D), fp32; pos (T,) long."""
    ang = pos.float()[:, None] * inv_freq[None, :]
    cos = ang.cos()[:, None, :]
    sin = ang.sin()[:, None, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim = -1)


class Inkling2MTPTensor(Module):
    """Plain tensor of the component file (norm weights, fuse weights, mask/markov embeddings)."""

    def __init__(self, config, key: str):
        super().__init__(config, key, None)
        self.module_name = "Inkling2MTPTensor"
        self.t = None

    def load(self, device: torch.device, **kwargs):
        self.device = device
        self.t = _side(self.config, self.key, device)

    def unload(self):
        self.device = None
        self.t = None

    def get_tensors(self):
        return {self.key: self.t}

    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError("Inkling2 MTP tensors are read by the drafter, not run as modules")

    def optimizer_targets(self):
        return []


class Inkling2MTPLayer(Module):
    """One draft layer: pre-norm attention (q/k norms, shared K/V ctx projections) and SwiGLU MLP."""

    def __init__(self, config, key: str, local: bool, hidden: int, heads: int, kv_heads: int, head_dim: int, eps: float):
        super().__init__(config, key, None)
        self.module_name = "Inkling2MTPLayer"
        self.local = local
        self.heads, self.kv_heads, self.head_dim, self.eps = heads, kv_heads, head_dim, eps

        def lin(name, n_in, n_out, bias = False):
            m = Linear(config, f"{key}.{name}", n_in, n_out, out_dtype = torch.half)
            self.register_submodule(m)
            return m

        def ten(name):
            m = Inkling2MTPTensor(config, f"{key}.{name}")
            self.register_submodule(m)
            return m

        self.q_proj = lin("self_attn.q_proj", hidden, heads * head_dim)
        self.k_proj = lin("self_attn.k_proj", hidden, kv_heads * head_dim)
        self.v_proj = lin("self_attn.v_proj", hidden, kv_heads * head_dim)
        self.o_proj = lin("self_attn.o_proj", heads * head_dim, hidden)
        self.k_proj_ctx = lin("self_attn.k_proj_ctx", hidden, kv_heads * head_dim)
        self.v_proj_ctx = lin("self_attn.v_proj_ctx", hidden, kv_heads * head_dim)
        self.q_norm = ten("self_attn.q_norm.weight")
        self.k_norm = ten("self_attn.k_norm.weight")
        self.k_norm_ctx = ten("self_attn.k_norm_ctx.weight")
        self.gate_proj = lin("mlp.gate_proj", hidden, config_intermediate(config))
        self.up_proj = lin("mlp.up_proj", hidden, config_intermediate(config))
        self.down_proj = lin("mlp.down_proj", config_intermediate(config), hidden)
        self.input_norm = ten("input_layernorm.weight")
        self.post_norm = ten("post_attention_layernorm.weight")
        self.dflare_norm = ten("dflare_norm.weight")

    def forward(self, x, params, out_dtype = None):
        raise NotImplementedError("the drafter runs its layers itself (Inkling2MTPModel._layer)")

    def optimizer_targets(self):
        return []


def config_intermediate(config) -> int:
    return config.mtp_config["intermediate_size"]


class Inkling2MTPModel(Model):
    config_class = None  # set by the loader (Inkling2LoopConfig)

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        m = config.mtp_config
        self.hidden = m["hidden_size"]
        self.n_layers = m["num_hidden_layers"]
        self.heads = m["num_attention_heads"]
        self.kv_heads = m["num_key_value_heads"]
        self.head_dim = m["head_dim"]
        self.eps = m["rms_norm_eps"]
        self.theta = m["rope_theta"]
        self.block = m["block_size"]
        # INKLING2_MTP_BLOCK: shorter draft block (anchor + block - 1 drafted tokens per round)
        self.block = max(2, min(self.block, int(os.environ.get("INKLING2_MTP_BLOCK", self.block))))
        config.block_size = self.block      # the generator reads the draft block size from the draft config
        self.rank = m["markov_rank"]
        self.window = m["sliding_window_size"]
        self.local_ids = set(m["local_layer_ids"])
        self.taps = list(m["target_layer_ids"])
        self.softcap = float(getattr(config, "final_logit_softcapping", 0.0) or 0.0)
        self.vocab_size = config.vocab_size

        self.layers = []
        for i in range(self.n_layers):
            layer = Inkling2MTPLayer(
                config, f"layers.{i}", i in self.local_ids, self.hidden,
                self.heads, self.kv_heads, self.head_dim, self.eps,
            )
            self.layers.append(layer)
            self.modules.append(layer)

        self.norm = Inkling2MTPTensor(config, "norm.weight")
        self.markov_w1 = Inkling2MTPTensor(config, "markov_head.markov_w1.weight")
        self.markov_dflare = Inkling2MTPTensor(config, "markov_head.dflare_norm.weight")
        self.fuse_weight = Inkling2MTPTensor(config, "fuse_weight")
        self.mask_embed = Inkling2MTPTensor(config, "mask_embed")
        self.rnn_proj = Linear(config, "markov_head.rnn_proj", 2 * self.rank + 2 * self.hidden, 3 * self.rank,
                               out_dtype = torch.float)
        self.hidden_out = Linear(config, "markov_head.hidden_out", self.rank, self.hidden, out_dtype = torch.float)
        for m_ in (self.norm, self.markov_w1, self.markov_dflare, self.fuse_weight, self.mask_embed,
                   self.rnn_proj, self.hidden_out):
            self.modules.append(m_)

        self.caps.update({
            "dflash_draft": True,
            "attach_target": True,
            "default_draft_size": self.block - 1,
            "supports_tp": False,
        })
        # Taps the target must export (4 taps: visit 5, 23, 41 partials and the final state at visit 48)
        self.draft_verifier_params = {"export_state_layers": set(self.taps)}

        self.lm = None           # the target's Inkling2Final (head)
        self.embed = None        # the target's Inkling2Embedding
        self.attached_model = None
        self._ready = False
        self._ctx_k = [None] * self.n_layers
        self._ctx_v = [None] * self.n_layers
        self._cap = 0
        self._ctx_len = 0
        self._f_rnn = None
        self.dev = None
        self.dense_context = True   # unpaged context: the generator must not reuse prefix pages (job.py)

    # ---- lifecycle ------------------------------------------------------------------------------------------

    def attach_to(self, target):
        self.attached_model = weakref.ref(target)
        self.embed = target.modules[0]
        self.lm = target.modules[target.logit_layer_idx]
        self.dev = self.lm.head.device
        self._ready = False

    @torch.inference_mode()
    def _prepare(self):
        """Dense fp32 copies of the Markov head, and the head folded through hidden_out (per vocab row)."""
        if self._ready:
            return
        r, H = self.rank, self.hidden
        rnn_w = self.rnn_proj.inner.get_weight_tensor().t().float()        # (3r, 2r+2H)
        self._rnn_w_state = rnn_w[:, : 2 * r].contiguous()                  # [state | prev]
        self._rnn_w_hid = rnn_w[:, 2 * r : 2 * r + H].contiguous()          # hidden
        self._rnn_w_frnn = rnn_w[:, 2 * r + H :].contiguous()               # f_rnn
        self._rnn_b = self.rnn_proj.inner.get_bias_tensor().float()
        ho_w = self.hidden_out.inner.get_weight_tensor().t().float()        # (H, r)
        ho_b = self.hidden_out.inner.get_bias_tensor().float()              # (H,)
        head = self.lm.head.inner.get_weight_tensor().t()                   # (V_pad, H) fp16
        V = head.shape[0]
        fold_w = torch.empty((V, r), dtype = torch.float, device = head.device)
        fold_b = torch.empty((V,), dtype = torch.float, device = head.device)
        for s in range(0, V, 8192):
            hc = head[s : s + 8192].float()
            fold_w[s : s + 8192] = hc @ ho_w
            fold_b[s : s + 8192] = hc @ ho_b
        self._fold_w, self._fold_b = fold_w, fold_b
        del head
        self._ready = True

    def _ensure_cap(self, n: int):
        if n <= self._cap:
            return
        new_cap = max(n, 2 * self._cap, 1024)
        for i in range(self.n_layers):
            for name in ("_ctx_k", "_ctx_v"):
                buf = getattr(self, name)
                nb = torch.zeros((new_cap, self.kv_heads, self.head_dim), dtype = torch.half, device = self.dev)
                if buf[i] is not None:
                    nb[: self._cap] = buf[i][: self._cap]
                buf[i] = nb
        self._cap = new_cap

    def _inv_freq(self):
        return 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype = torch.float, device = self.dev) / self.head_dim))

    def _alpha(self, row: int) -> torch.Tensor:
        return torch.softmax(self.fuse_weight.t.float()[row], dim = -1)

    def _mix(self, taps: torch.Tensor, row: int, norm_w: torch.Tensor) -> torch.Tensor:
        """taps (4, n, H) fp32 -> RMSNorm(sum_t alpha[t] * taps[t]) (fp32)."""
        alpha = self._alpha(row)
        mixed = torch.einsum("t,tnh->nh", alpha, taps)
        return _rms(mixed, norm_w, self.eps)

    # ---- drafter forward -----------------------------------------------------------------------------------

    def _write_ctx(self, taps: torch.Tensor, start: int):
        """Context K/V for positions [start, start + n) from taps (4, n, H) fp32."""
        n = taps.shape[1]
        self._ensure_cap(start + n)
        inv = self._inv_freq()
        pos = torch.arange(start, start + n, device = self.dev)
        for i, layer in enumerate(self.layers):
            m = self._mix(taps, i, layer.dflare_norm.t).half()
            k = layer.k_proj_ctx.forward(m, {}).float().view(n, self.kv_heads, self.head_dim)
            v = layer.v_proj_ctx.forward(m, {}).view(n, self.kv_heads, self.head_dim)
            k = _rope(_rms(k, layer.k_norm_ctx.t, self.eps), pos, inv)
            self._ctx_k[i][start : start + n] = k.half()
            self._ctx_v[i][start : start + n] = v.half()

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, params: dict, **kwargs) -> torch.Tensor:
        """Block forward for one anchor: returns the normed block states (1, block, hidden) fp16."""
        assert input_ids.shape[0] == 1, "Inkling2 MTP drafter: batch 1 only"
        self._prepare()
        L = int(params["cache_seqlens"][0])
        assert L == self._ctx_len, f"MTP drafter: context length {self._ctx_len} != position {L}"
        anchor = input_ids.reshape(-1)[:1].to(self.dev).long()
        params["mtp_anchor"] = anchor
        self._ensure_cap(L + self.block)

        emb = self.embed.embedding[anchor].float()
        x0 = _rms(emb, self.embed.norm_weight, self.embed.eps)          # (1, H)
        mask = self.mask_embed.t.float().expand(self.block - 1, self.hidden)
        x = torch.cat([x0, mask], dim = 0)                             # (block, H)
        pos = torch.arange(L, L + self.block, device = self.dev)
        inv = self._inv_freq()
        for i, layer in enumerate(self.layers):
            x = self._layer(i, layer, x, pos, L, inv)
        h = _rms(x, self.norm.t, self.eps)
        return h.half().unsqueeze(0)

    def _layer(self, i, layer, x, pos, L, inv):
        T = x.shape[0]
        h = _rms(x, layer.input_norm.t, self.eps).half()
        q = layer.q_proj.forward(h, {}).float().view(T, self.heads, self.head_dim)
        k = layer.k_proj.forward(h, {}).float().view(T, self.kv_heads, self.head_dim)
        v = layer.v_proj.forward(h, {}).float().view(T, self.kv_heads, self.head_dim)
        q = _rope(_rms(q, layer.q_norm.t, self.eps), pos, inv)
        k = _rope(_rms(k, layer.k_norm.t, self.eps), pos, inv)
        lo = max(0, L - self.window) if layer.local else 0
        K = torch.cat([self._ctx_k[i][lo:L].float(), k], dim = 0)     # (n, Hk, D)
        V = torch.cat([self._ctx_v[i][lo:L].float(), v], dim = 0)
        group = self.heads // self.kv_heads
        qg = q.view(T, self.kv_heads, group, self.head_dim)
        scores = torch.einsum("tkgd,nkd->tkgn", qg, K) * (self.head_dim ** -0.5)
        p = torch.softmax(scores, dim = -1)
        o = torch.einsum("tkgn,nkd->tkgd", p, V).reshape(T, self.heads * self.head_dim)
        x = x + layer.o_proj.forward(o.half(), {}).float()
        h2 = _rms(x, layer.post_norm.t, self.eps).half()
        g = layer.gate_proj.forward(h2, {}).float()
        u = layer.up_proj.forward(h2, {}).float()
        m = layer.down_proj.forward((F.silu(g) * u).half(), {}).float()
        return x + m

    @torch.inference_mode()
    def update_kv_from_target(self, target_hidden: list, cache, params: dict, lengths: list[int] = None):
        """Write context K/V (and the Markov context f_rnn) for the processed rows of this round.
        target_hidden: one (1, T, H) tensor per tap, in target_layer_ids order."""
        self._prepare()
        taps = [t.to(self.dev) for t in target_hidden]
        T = taps[0].shape[1]
        n = T if lengths is None else min(T, int(lengths[0]))
        if n <= 0:
            return
        start = int(params["cache_seqlens"][0])
        if start < self._ctx_len:
            self._ctx_len = start
        assert start == self._ctx_len, f"MTP drafter: taps at {start}, context ends at {self._ctx_len}"
        stacked = torch.stack([t[0, :n].float() for t in taps], dim = 0)  # (4, n, H)
        self._write_ctx(stacked, start)
        last = stacked[:, n - 1 : n]
        self._f_rnn = self._mix(last, self.n_layers, self.markov_dflare.t)[0]
        self._ctx_len = start + n

    @torch.inference_mode()
    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Greedy block draft: returns (1, block) ids [anchor, draft_1 .. draft_{block-1}]."""
        self._prepare()
        assert self._f_rnn is not None, "MTP drafter: no target context yet"
        anchor = params["mtp_anchor"]
        r, V = self.rank, self.vocab_size
        h = state[0, 1:].float()                                                # (block-1, H)
        base = self.lm.head.inner.forward(h.half(), {}, torch.float).float()    # raw lm_head (no softcap)
        ctx = (self._rnn_b[None, :] + h @ self._rnn_w_hid.t()
               + (self._f_rnn @ self._rnn_w_frnn.t())[None, :])                 # (block-1, 3r)
        sc = self.softcap
        state_r = torch.zeros(r, dtype = torch.float, device = self.dev)
        prev = anchor
        drafts = []
        for s in range(self.block - 1):
            emb = self.markov_w1.t[prev].float().reshape(-1)                    # (r,)
            raw = ctx[s] + self._rnn_w_state @ torch.cat([state_r, emb])
            gate = torch.sigmoid(raw[:r])
            cand = torch.tanh(raw[r : 2 * r])
            latent = torch.tanh(raw[2 * r :])
            state_r = gate * state_r + (1.0 - gate) * cand
            bias = self._fold_w @ latent + self._fold_b
            logits = base[s] + bias
            if sc:
                logits = sc * torch.tanh(logits / sc)
            d = torch.argmax(logits[:V]).reshape(1)
            drafts.append(d)
            prev = d
        return torch.cat([anchor] + drafts).view(1, self.block)

    @override
    def prepare_inputs(self, input_ids, params):
        return input_ids
