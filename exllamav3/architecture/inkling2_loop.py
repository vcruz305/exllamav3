from __future__ import annotations
from typing_extensions import override
import os
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules.attn import prepare_for_attn
from ..modules.short_conv import ShortConvState
from ..modules.block_sparse_mlp import BlockSparseMLP
from ..cache.recurrent_util import prepare_for_recurrence
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
