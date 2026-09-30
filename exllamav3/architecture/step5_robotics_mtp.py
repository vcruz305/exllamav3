from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..model.model import Model
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    SlidingAttention,
    GatedMLP,
    Linear,
)
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .step5_robotics import Step5RoboticsConfig


"""
Step-5-Preview MTP (multi-token-prediction) depths -- the 3 nextn layers stored in the
checkpoint as ``model.layers.{92,93,94}.*``.

Shape follows NVIDIA NeMo ``components/models/step3p7/mtp.py`` (the only public reference
registered for ``model_type: step3p5v``), transposed onto exllamav3 modules:

  per depth (51 / 3 = 17 tensors each):
    fusion      : hnorm, enorm, eh_proj        (2H -> H, concat of normed hidden + normed embed)
    decoder blk : input_layernorm, self_attn.{q,k,v,o,g}_proj, self_attn.{q,k}_norm,
                  post_attention_layernorm, mlp.{gate,up,down}_proj   (DENSE -- no MoE)
    shared head : transformer.shared_head.norm, transformer.shared_head.output (H -> vocab)

NeMo's ``_make_mtp_block_config`` sets ``moe_layers_enum = ()`` and defaults the MTP layer
type to ``sliding_attention``, so each depth is a DENSE sliding-attention block. Our
checkpoint confirms this: layers 92..94 carry ``mlp.{gate,up,down}_proj`` and no
``moe.*`` tensors.

Unlike GLM-5.2 MTP (one shared lm_head borrowed from the target), Step-5 gives every depth
its OWN ``transformer.shared_head.output`` (H -> vocab). That is modelled here per depth.

MTP bits are budgeted through ``qbits_key = "mtp_bits"`` (NOT ``"bits"``), matching every
other ``*_mtp.py`` in this tree.
"""


class Step5RoboticsMTPModel(Model):

    def __init__(
        self,
        config: "Step5RoboticsConfig",
        key_prefix: str = "model",
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        first = config.mtp_base_layer_idx          # 92
        n_depths = config.mtp_num_layers           # 3
        assert n_depths > 0, "Step5RoboticsMTPModel built with num_nextn_predict_layers == 0"

        self.modules = []
        self.depth_head_idxs: list[int] = []

        for depth in range(n_depths):
            li = first + depth
            key = f"{key_prefix}.layers.{li}"

            # ---- fusion: enorm(embed) || hnorm(target_hidden) --2H -> H -------------
            self.modules.append(
                Qwen3_5MTPInputLayer(
                    config = config,
                    key = f"{key}.input",
                    key_pre_fc_norm_hidden = f"{key}.hnorm",
                    key_pre_fc_norm_embedding = f"{key}.enorm",
                    key_fc = f"{key}.eh_proj",
                    hidden_size = config.hidden_size,
                    rms_norm_eps = config.rms_norm_eps,
                    native_draft_len = 1,
                    out_dtype = torch.float,
                    qbits_key = "mtp_bits",
                    constant_bias = 0.0,
                )
            )

            # ---- dense decoder block (sliding attention + dense MLP) ----------------
            is_swa = config.layer_types[li] == "sliding_attention"
            act_limit = float(config.swiglu_limits[li])
            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = li,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    attn = (
                        SlidingAttention(
                            config = config,
                            key = f"{key}.self_attn",
                            layer_idx = li,
                            hidden_size = config.hidden_size,
                            head_dim = config.head_dim,
                            num_q_heads = config.num_q_heads if not is_swa else config.alt_num_q_heads,
                            num_kv_heads = config.num_kv_heads if not is_swa else config.alt_num_kv_heads,
                            rope_settings = config.rope_settings_list[li],
                            sm_scale = None,
                            sliding_window = config.sliding_window if is_swa else -1,
                            key_q = "q_proj",
                            key_k = "k_proj",
                            key_v = "v_proj",
                            key_o = "o_proj",
                            key_g = "g_proj",
                            qmap = "block.attn",
                            q_norm = RMSNorm(
                                config = config,
                                key = f"{key}.self_attn.q_norm",
                                rms_norm_eps = config.rms_norm_eps,
                                constant_bias = 1.0,
                            ) if config.use_qk_norm else None,
                            k_norm = RMSNorm(
                                config = config,
                                key = f"{key}.self_attn.k_norm",
                                rms_norm_eps = config.rms_norm_eps,
                                constant_bias = 1.0,
                            ) if config.use_qk_norm else None,
                            out_dtype = torch.float,
                            select_hq_bits = 2,
                            qbits_key = "mtp_bits",
                        )
                        if is_swa else
                        Attention(
                            config = config,
                            key = f"{key}.self_attn",
                            layer_idx = li,
                            hidden_size = config.hidden_size,
                            head_dim = config.head_dim,
                            num_q_heads = config.num_q_heads,
                            num_kv_heads = config.num_kv_heads,
                            rope_settings = config.rope_settings_list[li],
                            sm_scale = None,
                            sliding_window = -1,
                            key_q = "q_proj",
                            key_k = "k_proj",
                            key_v = "v_proj",
                            key_o = "o_proj",
                            key_g = "g_proj",
                            qmap = "block.attn",
                            q_norm = RMSNorm(
                                config = config,
                                key = f"{key}.self_attn.q_norm",
                                rms_norm_eps = config.rms_norm_eps,
                                constant_bias = 1.0,
                            ) if config.use_qk_norm else None,
                            k_norm = RMSNorm(
                                config = config,
                                key = f"{key}.self_attn.k_norm",
                                rms_norm_eps = config.rms_norm_eps,
                                constant_bias = 1.0,
                            ) if config.use_qk_norm else None,
                            out_dtype = torch.float,
                            tp_split_norm = False,
                            select_hq_bits = 2,
                            qbits_key = "mtp_bits",
                        )
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        act_limit = act_limit,
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        select_hq_bits = 1,
                        qbits_key = "mtp_bits",
                    ),
                )
            )

            # ---- per-depth shared head: RMSNorm -> Linear(H -> vocab) ---------------
            self.modules.append(
                RMSNorm(
                    config = config,
                    key = f"{key}.transformer.shared_head.norm",
                    rms_norm_eps = config.rms_norm_eps,
                    out_dtype = torch.half,
                    constant_bias = 1.0,
                )
            )
            self.depth_head_idxs.append(len(self.modules))
            self.modules.append(
                Linear(
                    config = config,
                    key = f"{key}.transformer.shared_head.output",
                    qbits_key = "head_bits",
                    in_features = config.hidden_size,
                    out_features = config.vocab_size,
                    qmap = "block",
                    caps = {"logits_output": True},
                )
            )

        self.last_kv_module_idx = self.depth_head_idxs[0] - 2

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": n_depths,
            "autosplit_load_fwd": False,
        })

        # Activate all experts during H capture pass in quantization (no experts here,
        # but kept for parity with the other *_mtp.py models).
        self.calibration_all_experts = True

        # Which trunk state hnorm consumes: post-final-norm (NeMo/HyV3/Qwen3.5) or the
        # pre-norm residual (DeepSeek-V3 paper). Settled empirically on GLM-5.2; kept as an
        # attribute so the comparison stays reproducible.
        self.pre_norm_tap = False

        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # MTP does not push input_ids through its own Embedding -- the fusion layer consumes
        # the target's embeddings. prepare_for_attn still wires flash-attn params.
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")

    def attach_to(self, target):
        """Bind to the target model: borrow embed_tokens and learn which hidden tap hnorm sees."""
        self.attached_model = weakref.ref(target)

        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        if self.pre_norm_tap:
            self.draft_verifier_params = {
                "export_state_layers": {self.config.num_hidden_layers - 1},
            }
        else:
            target_norm = target.modules[target.logit_layer_idx - 1]
            assert isinstance(target_norm, RMSNorm), \
                "Expected target final RMSNorm immediately before lm_head"
            self.draft_verifier_params = {
                "export_state_norm_keys": {target_norm.key},
            }

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict,
    ) -> torch.Tensor:
        # Per-depth heads are the draft logits; the target lm_head is only used when the
        # generator asks for a target-shaped sample (parity with the other MTP models).
        if not self.attached_model().loaded_tp:
            ll = self.attached_model().logit_layer_idx
            lm = self.attached_model().modules[ll]
            logits = lm.prepare_for_device(state, params)
            logits = lm.forward(logits, params)
            if params.get("export_draft_conf"):
                logits = logits[..., :self.attached_model().config.vocab_size]
                conf, ids = torch.max(logits, dim = -1)
                params["draft_conf"] = conf
                return ids
            return torch.argmax(logits, dim = -1)
        else:
            state = self.attached_model().tp_producer.send(state)
            return self.attached_model().tp_dispatch_lm_head_argmax((state, {}))
