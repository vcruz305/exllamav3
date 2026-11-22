from __future__ import annotations
from typing_extensions import override
import os
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules.attn import prepare_for_attn
from ..modules.short_conv import ShortConvState
from ..modules.block_sparse_mlp import BlockSparseMLP, MIXEDK_MIN_ROWS
from ..modules.block_sparse_mlp_routing import routing_ds3, _gate_t
from ..modules.quant.exl3 import LinearEXL3
from ..modules.quant.fp16 import LinearFP16
from ..cache.recurrent_util import prepare_for_recurrence
from ..constants import PAGE_SIZE
from ..ext import exllamav3_ext as ext
from ..modules.arch_specific.inkling2 import (
    Inkling2Embedding,
    Inkling2Attention,
    Inkling2MoE,
    Inkling2Block,
    Inkling2Final,
)

"""
Inkling2Loop (thinkingmachines/Inkling-2-Mini), text decoder.

The checkpoint's vision tower and multi-token-prediction sidecar are not instantiated.
"""


def _build_moe(c: Config, key: str):
    """
    Routed experts of one layer. Default: the fork's BlockSparseMLP with the softcapped SwiGLU as activation
    "silu_softcap" and the DeepSeek-V3-style sigmoid router (selection bias). Mixed-K layers take the unified
    fused mixed-K kernel (EXL3_MOE_COOP_MIXEDK=1 adds the CoopMK decode kernels); uniform-K layers take the
    dense per-expert path, since the BC fused family has no softcap arm. INKLING2_PY_MOE=1 keeps the
    per-expert Python Inkling2MoE for A/B comparison.
    """
    if os.environ.get("INKLING2_PY_MOE", "0") == "1":
        return Inkling2MoE(
            config = c,
            key = f"{key}.mlp",
            hidden_size = c.hidden_size,
            intermediate_size = c.intermediate_size,
            num_experts = c.num_experts,
            num_experts_per_tok = c.num_experts_per_tok,
            route_scale = c.route_scale,
            gate_softcap = c.gate_softcap,
            up_softcap = c.up_softcap,
        )
    return BlockSparseMLP(
        config = c,
        key = f"{key}.mlp",
        hidden_size = c.hidden_size,
        intermediate_size = c.intermediate_size,
        num_experts = c.num_experts,
        num_experts_per_tok = c.num_experts_per_tok,
        key_gate = "experts.{expert_idx}.gate_proj",
        key_up = "experts.{expert_idx}.up_proj",
        key_down = "experts.{expert_idx}.down_proj",
        key_routing_gate = "gate",
        key_e_score_bias = "gate.bias",
        qmap = "block.mlp",
        out_dtype = torch.float,
        activation_fn = "silu_softcap",
        act_limit = c.gate_softcap,
        act_limit2 = c.up_softcap,
        router_type = "ds3",
        routed_scaling_factor = c.route_scale,
        n_group = 1,
        topk_group = 1,
    )


def _exl3_bc(lin, name: str):
    """BC_LinearEXL3 of a plain EXL3 linear (no padding, scale, trim or LoRA): the call Linear.forward_row makes"""
    if (type(lin.inner) is not LinearEXL3 or lin.in_features != lin.in_features_unpadded
            or lin.out_features != lin.out_features_unpadded or lin.pre_scale != 1.0 or lin.post_scale != 1.0
            or lin.trim_padded_out or lin.lora_a_tensors):
        raise ValueError(f"token path: {name} is not a plain EXL3 linear")
    return lin.inner.bc


# Whole-token decode (exllamav3_ext/inkling2_token.cu): one extension call per token instead of the Python block
# loop. INKLING2_TOKEN=0 keeps the Python single-token path (Inkling2Block._forward_t1) for A/B checks.
_TOKEN = os.environ.get("INKLING2_TOKEN", "1") != "0"
# exl3_mlin for the q, r, g, k, v projections of the token step (rounding-level change vs the per-linear BC path)
_MLIN_TOK = os.environ.get("INKLING2_TOK_MLIN", "1") != "0"
# CUDA-graph replay of the token step (per record flag); INKLING2_GRAPH=0 runs the same step eagerly
_GRAPH = os.environ.get("INKLING2_GRAPH", "1") != "0"
# Call arguments that select the general path (capture, overrides, reconstruct, all-expert or sensitivity passes)
_TOKEN_BLOCKERS = ("capture", "capture_qmaps", "capture_sens", "ovr", "reconstruct", "activate_all_experts",
                   "autosplit_measure", "tp_warmup")


def _build_token(model, params: dict):
    """Spec of the one-call token step over the fwd_modules sequence, for ext.InkToken. Raises when a layer is
    outside what the step implements; the caller then keeps the Python path"""
    cache = params["cache"]
    rcache = params["recurrent_states"][0].cache
    emb = model.modules[0]
    fin = [mm for mm, _, _ in model.fwd_modules if isinstance(mm, Inkling2Final)]
    if not isinstance(emb, Inkling2Embedding) or len(fin) != 1:
        raise ValueError("token path: unexpected module layout")
    fin = fin[0]
    blocks = []
    taps = None
    for module, instance, _ in model.fwd_modules:
        if not isinstance(module, Inkling2Block):
            continue
        taps = module.ksize - 1
        a, m = module.attn, module.mlp
        if not isinstance(m, BlockSparseMLP) or m.routing_fn is not routing_ds3:
            raise ValueError("token path: MoE is not the BlockSparseMLP ds3 router")
        for attr in ("router_pre_norm", "routed_pre_norm", "latent_in", "latent_out", "shared_experts",
                     "shared_gate", "cpu_split_first", "routed_post_norm", "routing_device"):
            if getattr(m, attr, None) is not None:
                raise ValueError(f"token path: MoE {attr} is not supported")
        if getattr(m, "alt_residual_channel", False) or getattr(m, "cpu_offload", False) or getattr(m, "tp_reduce", False) \
                or m.num_local_experts != m.num_experts or m.routing_gate is None:
            raise ValueError("token path: MoE layout is not supported")
        cfg = m.routing_cfg
        _gate_t(cfg)
        if cfg.e_score_bias_f32 is None and cfg.e_score_correction_bias is not None:
            cfg.e_score_bias_f32 = cfg.e_score_correction_bias.float().contiguous()
        if cfg.router_logits_f32_bsz1 is None:
            cfg.router_logits_f32_bsz1 = torch.empty((1, cfg.num_experts), dtype = torch.float, device = m.device)
        if m.coopmk is not None:
            coop, bc_moe, moe_out = True, None, m.coopmk_out
        else:
            # Same branch choice as BlockSparseMLP._forward_rows at one row: run_bszN only when the torch,
            # reconstruct and mixed-K tiers would not be taken first
            if (m.bc is None or m.experts_cfg.out_bszn is None or not m.is_quantized
                    or m.config.infer_params.no_reconstruct
                    or (getattr(m, "mixedk_unified", False) and m.num_experts_per_tok >= MIXEDK_MIN_ROWS)):
                raise ValueError("token path: MoE would take the torch/reconstruct tier at one row")
            coop, bc_moe, moe_out = False, m.bc, m.experts_cfg.out_bszn
        if type(cache.layers[module.layer_idx, instance]).__name__ != "CacheLayer_fp16":
            raise ValueError("token path: needs the fp16 paged cache")
        Hk, D = a.num_kv_heads, a.head_dim
        kc, vc = cache.get_layer(module.layer_idx, params["cache_seqlens"], params["block_table"], instance = instance)
        ring = rcache.get_recurrent_layer((module.layer_idx, instance)).get_state_tensors()[0]
        ln = module.layer_idx == module.loop_start and module.in_loop
        blocks.append(dict(
            layer = module.layer_idx, instance = instance, Hq = a.num_q_heads, Hk = Hk, D = D, R = a.d_rel,
            E = a.rel_extent, kvc = Hk * D, W = -1 if a.sliding_window is None else int(a.sliding_window),
            l0 = module.layer_idx == 0, full = bool(module.is_full), boundary = bool(module.is_block_boundary),
            in_loop = bool(module.in_loop), loop_start = bool(ln),
            log_scaling = a.log_scaling is not None,
            alpha = float(a.log_scaling[0]) if a.log_scaling is not None else 0.0,
            nfloor = float(a.log_scaling[1]) if a.log_scaling is not None else 1.0,
            sm_scale = float(a.sm_scale), qk_clip = float(a.qk_clip), rel_clip = float(a.rel_clip),
            eps = float(module.eps),
            q = _exl3_bc(a.q_proj, "q"), k = _exl3_bc(a.k_proj, "k"), v = _exl3_bc(a.v_proj, "v"),
            r = _exl3_bc(a.r_proj, "r"), g = _exl3_bc(a.g_proj, "g"), o = _exl3_bc(a.o_proj, "o"),
            qkv_norm = module.qkv_norm, norm_mlp = module.norms["mlp"],
            q_attn_t = module.queries_t.get("attn"), q_mlp_t = module.queries_t["mlp"],
            gd = module.gate_down if module.in_loop else None, gu = module.gate_up if module.in_loop else None,
            msg_t = module.msg_query_t if module.in_loop else None,
            kw = a.k_fold["w"] if "w" in a.k_fold else a.k_fold["static"], kp = a.k_fold.get("proj"),
            vw = a.v_fold["w"] if "w" in a.v_fold else a.v_fold["static"], vp = a.v_fold.get("proj"),
            conv_attn = module.convs_f["attn"], conv_mlp = module.convs_f["mlp"],
            q_scale = a.q_scale.reshape(-1), rel_scale = a.rel_scale.reshape(-1),
            rel_proj = a.rel_proj, sink_proj = a.sink_proj,
            kf = kc.view(-1, Hk, D), vf = vc.view(-1, Hk, D), ring = ring,
            coop = coop, coopmk = m.coopmk if coop else None, bc_moe = bc_moe, moe_out = moe_out,
            **_mlin_fields(a, m.device),
            gate_t = cfg.gate_tensor_t, logits = cfg.router_logits_f32_bsz1, bias_f32 = cfg.e_score_bias_f32,
            sel = cfg.selected_experts_bsz1, w = cfg.routing_weights_bsz1,
            rsf = float(cfg.routed_scaling_factor), n_group = int(cfg.n_group), topk_group = int(cfg.topk_group),
        ))
    head = fin.head
    if type(head.inner) is not LinearEXL3 or head.inner.bias is not None or head.in_features != head.in_features_unpadded \
            or head.pre_scale != 1.0 or head.post_scale != 1.0 or head.lora_a_tensors:
        raise ValueError(f"token path: head is not a plain EXL3 linear (inner={type(head.inner).__name__})")
    return dict(
        H = model.config.hidden_size, V = fin.vocab_size, Vpad = head.inner.out_features, taps = taps,
        page = PAGE_SIZE, eps = float(module.eps), eps_emb = float(emb.eps), eps_fin = float(fin.eps),
        final_cap = float(head.softcap or 0.0),
        emb = emb.embedding, emb_norm = emb.norm_weight, fin_q_t = fin.query_t, fin_norm = fin.norm_weight,
        head = head.inner.bc, blocks = blocks,
    )


def _mlin_fields(a, dev):
    """exl3_mlin spec of q, r, g, k, v (the one-launch projection path of the token step), or the disabled spec"""
    off = dict(mlin = False, mlin_t = [], mlin_su = [], mlin_sv = [], mlin_out = [],
               mlin_cnt = torch.zeros(1, dtype = torch.int32, device = dev))
    if not _MLIN_TOK:
        return off
    lins = [a.q_proj, a.r_proj, a.g_proj, a.k_proj, a.v_proj]

    def plain(l):
        i = l.inner
        return (type(i) is LinearEXL3 and getattr(i, "K", None) == 6 and getattr(i, "mul1", False)
                and not getattr(i, "mcg", False) and getattr(i, "bias", None) is None
                and getattr(i, "out_dtype", torch.float) == torch.float and l.pre_scale == 1.0 and l.post_scale == 1.0
                and l.softcap == 0.0 and not l.trim_padded_out and not l.lora_a_tensors
                and l.in_features == l.in_features_unpadded and l.out_features == l.out_features_unpadded
                and l.out_features % 128 == 0 and l.in_features % 128 == 0)
    if not all(plain(l) for l in lins):
        print(" -- Inkling2 token path: exl3_mlin not used for a block (linears are not plain K=6 mul1)")
        return off
    outs = [torch.empty((1, l.out_features), dtype = torch.float, device = dev) for l in lins]
    cnt = torch.zeros(sum(l.out_features // 128 for l in lins), dtype = torch.int32, device = dev)
    return dict(mlin = True, mlin_t = [l.inner.trellis for l in lins], mlin_su = [l.inner.suh for l in lins],
                mlin_sv = [l.inner.svh for l in lins], mlin_out = outs, mlin_cnt = cnt)


class Inkling2LoopConfig(Config):
    arch_string = "Inkling2LoopForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Inkling2LoopModel},
            **kwargs,
        )

        tc = self.config_dict["text_config"]

        def rd(key, default = no_default):
            if key in tc:
                return tc[key]
            if default is no_default:
                raise ValueError(f"text_config.{key} expected but not present")
            return default

        self.hidden_size = rd("hidden_size")
        self.num_hidden_layers = rd("num_hidden_layers")
        self.padded_vocab_size = rd("vocab_size")
        self.vocab_size = rd("unpadded_vocab_size", self.padded_vocab_size)
        self.rms_norm_eps = rd("rms_norm_eps", 1e-6)

        # Attention. Layers listed in local_layer_ids use the sliding-window geometry
        self.local_layer_ids = set(rd("local_layer_ids", []))
        self.num_q_heads = rd("num_attention_heads")
        self.num_kv_heads = rd("num_key_value_heads")
        self.head_dim = rd("head_dim")
        self.swa_num_q_heads = rd("swa_num_attention_heads", self.num_q_heads)
        self.swa_num_kv_heads = rd("swa_num_key_value_heads", self.num_kv_heads)
        self.swa_head_dim = rd("swa_head_dim", self.head_dim)
        self.sliding_window = rd("sliding_window_size")
        self.d_rel = rd("d_rel")
        self.rel_extent = rd("rel_extent")
        self.qk_clip = rd("qk_l2_clip_threshold")
        self.rel_clip = rd("rel_l2_clip_threshold")
        self.log_scaling_alpha = rd("log_scaling_alpha", None)
        self.log_scaling_n_floor = rd("log_scaling_n_floor", None)
        self.sconv_kernel_size = rd("sconv_kernel_size")
        self.block_len = rd("n_layers_per_attn_res_block")
        for key, expected in (
            ("q_bias", False), ("o_bias", False), ("use_value_norm_no_gamma", True),
            ("attn_res_normalize_values", True), ("n_shared_experts", 0), ("use_gate_bias", True),
        ):
            if rd(key, expected) != expected:
                raise ValueError(f"Inkling2Loop: text_config.{key} != {expected} is not supported")

        # MLP
        self.num_experts = rd("n_routed_experts")
        self.num_experts_per_tok = rd("num_experts_per_tok")
        self.intermediate_size = rd("intermediate_size")
        self.gate_softcap = rd("mlp_gate_softcap")
        self.up_softcap = rd("mlp_up_softcap")
        self.route_scale = rd("route_scale")
        self.final_logit_softcapping = rd("final_logit_softcapping", 0.0)

        # Loop
        lc = rd("loop_config")
        if lc.get("is_varloop"):
            raise ValueError("Inkling2Loop: variable-depth loops are not supported")
        self.loop_start = lc["core_start"]
        self.loop_end = lc["core_end"]
        self.num_passes = lc["num_passes"]

        self.max_seq_len = rd("model_max_length", 1 << 20)


class Inkling2LoopModel(Model):
    config_class = Inkling2LoopConfig

    def __init__(
        self,
        config: Inkling2LoopConfig,
        key_prefix: str = "model.llm",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self._tok = None
        self._tok_key = None
        self._tok_disabled = False
        self.tok_steps = 0          # decode tokens served by ext.InkToken (fast-path hit count)
        self.graph_captures = 0     # CUDA graphs captured for the token step
        self._gkey, self._graphs, self._gwarm = None, {}, {}
        c = config

        self.modules += [
            Inkling2Embedding(
                config = c,
                key = f"{key_prefix}.embed",
                key_norm = f"{key_prefix}.embed_norm",
                vocab_size = c.padded_vocab_size,
                hidden_size = c.hidden_size,
                eps = c.rms_norm_eps,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(c.num_hidden_layers):
            local = idx in c.local_layer_ids
            key = f"{key_prefix}.layers.{idx}"
            log_scaling = None
            if not local and c.log_scaling_alpha is not None and c.log_scaling_n_floor is not None:
                log_scaling = (c.log_scaling_alpha, c.log_scaling_n_floor)
            self.modules += [
                Inkling2Block(
                    config = c,
                    key = key,
                    key_prefix = key_prefix,
                    layer_idx = idx,
                    is_full = not local,
                    loop_range = (c.loop_start, c.loop_end),
                    num_passes = c.num_passes,
                    block_len = c.block_len,
                    hidden_size = c.hidden_size,
                    conv_kernel_size = c.sconv_kernel_size,
                    eps = c.rms_norm_eps,
                    attn = Inkling2Attention(
                        config = c,
                        key = f"{key}.attn",
                        layer_idx = idx,
                        hidden_size = c.hidden_size,
                        num_q_heads = c.swa_num_q_heads if local else c.num_q_heads,
                        num_kv_heads = c.swa_num_kv_heads if local else c.num_kv_heads,
                        head_dim = c.swa_head_dim if local else c.head_dim,
                        d_rel = c.d_rel,
                        rel_extent = c.rel_extent,
                        sliding_window = c.sliding_window if local else None,
                        conv_kernel_size = c.sconv_kernel_size,
                        qk_clip = c.qk_clip,
                        rel_clip = c.rel_clip,
                        log_scaling = log_scaling,
                        eps = c.rms_norm_eps,
                    ),
                    mlp = _build_moe(c, key),
                )
            ]
            self.modules[-1].layer_idx = idx

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            Inkling2Final(
                config = c,
                key = f"{key_prefix}.final",
                key_query = f"{key_prefix}.attn_res.output_query",
                key_norm = f"{key_prefix}.norm",
                key_head = f"{key_prefix}.unembed",
                hidden_size = c.hidden_size,
                padded_vocab_size = c.padded_vocab_size,
                vocab_size = c.vocab_size,
                eps = c.rms_norm_eps,
                softcap = c.final_logit_softcapping,
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        self.caps.update({
            "supports_tp": False,
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
        })
        # Short-convolution histories ride the generic conv-ring state (see Inkling2ConvLayerState)
        self.recurrent_state_cls = ShortConvState

        # The loop: layers [loop_start, loop_end) run num_passes times
        if not c.layer_map and not c.layer_map_str:
            core = list(range(c.loop_start, c.loop_end))
            c.layer_map = (
                list(range(c.loop_start)) + core * c.num_passes + list(range(c.loop_end, c.num_hidden_layers))
            )

    def _tok_ok(self, x: torch.Tensor, params: dict) -> bool:
        """One sequence, one token, fp16 paged cache with contiguous pages up to the position. Recorded conv history
        (params recurrent_history, speculative passes) is handled in the step by the record flag"""
        if self._tok_disabled or x.numel() != 1 or params.get("cache") is None:
            return False
        rs = params.get("recurrent_states")
        if not rs or len(rs) != 1:
            return False
        if any(k in params for k in _TOKEN_BLOCKERS):
            return False
        bt, cs = params.get("block_table"), params.get("cache_seqlens")
        if bt is None or cs is None or bt.is_cuda or cs.is_cuda:
            return False
        pos = int(cs.tolist()[0])
        row = bt.tolist()[0]
        used = pos // PAGE_SIZE + 1
        return used <= len(row) and all(row[j] == row[0] + j for j in range(used))

    def forward_ls(self, x: torch.Tensor, params: dict):
        if _TOKEN and self._tok_ok(x, params):
            key = (params["cache"], params["recurrent_states"][0].cache)
            if self._tok is None or self._tok_key[0] is not key[0] or self._tok_key[1] is not key[1]:
                try:
                    self._tok = ext.InkToken(_build_token(self, params))
                    self._tok_key = key
                except Exception as e:
                    self._tok_disabled = True
                    print(f" -- Inkling2 token path off ({e}); Python single-token path")
                    return super().forward_ls(x, params)
            pos = int(params["cache_seqlens"].tolist()[0])
            base_page = int(params["block_table"].tolist()[0][0])
            slot = params["recurrent_states"][0].slot
            record = bool(params.get("recurrent_history"))
            self.tok_steps += 1
            tok_id = int(x.reshape(-1)[0])
            if not _GRAPH:
                return self._tok.step(tok_id, pos, base_page, slot, record)
            # CUDA graph: per-token values go through the device buffers (update), the captured step reads them.
            gk = (id(self._tok), slot)
            if self._gkey != gk:
                self._gkey, self._graphs, self._gwarm = gk, {}, {}
            self._tok.update(tok_id, pos, base_page, slot)
            if record in self._graphs:
                g, out = self._graphs[record]
                g.replay()
                return out.clone()
            n = self._gwarm.get(record, 0)
            if n < 2:                       # eager steps first: cuBLAS handles and allocator state before capture
                self._gwarm[record] = n + 1
                return self._tok.step_dev(record).clone()
            g = torch.cuda.CUDAGraph()      # capture records only; the replay below executes this token once
            with torch.cuda.graph(g):
                out = self._tok.step_dev(record)
            g.replay()
            self._graphs[record] = (g, out)
            self.graph_captures += 1
            return out.clone()
        return super().forward_ls(x, params)

    @override
    def unload(self):
        self._tok = None
        self._tok_key = None
        super().unload()

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # Mirrors the checkpoint's chat template (system first, then the thinking-effort line)
        p = ""
        if system_prompt:
            p += f"<|message_system|><|content_text|>{system_prompt}<|end_message|>"
        p += "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
        p += f"<|message_user|><|content_text|>{prompt}<|end_message|>"
        return p
