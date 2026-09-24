from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import Config, no_default
from ..util.rope import RopeStyle
from .step3_5 import Step3_5Model
from .step3_7 import Step3_7Config
from .step5_robotics_mtp import Step5RoboticsMTPModel

"""
Step-5-Preview -- MMGPTStepRoboticsForCausalLM (model_type "step3p5v").

Body architecture is the Step-3.5/3.7 family: the pinned Step3_5Model matches this
checkpoint name-for-name on 1,195 body tensors (all 92 layers: q/k/v/o/g_proj, q/k norms,
input/post-attention layernorms, MoE router + stacked experts via the fkey/fidx split
path, shared expert, embed/lm_head/norm). Config contract mirrors Step3_7Config, which
already reads everything under "text_config".

SCOPE OF THIS PORT (explicit, fail-closed):
  * text body: layers 0..num_hidden_layers-1 (92) via Step3_5Model.
  * MTP depths 92..94 via Step5RoboticsMTPModel (this commit): per-depth fusion
    (hnorm/enorm/eh_proj), dense sliding-attention decoder block, and a per-depth
    transformer.shared_head. Budgeted on qbits_key "mtp_bits".
  * the sparse (CSA) indexer on the 23 "full_attention" layers is DESCRIBED here
    (attributes below) but its tensors are not yet loadable by this module set: until the
    indexer is implemented, those layers run DENSE full attention. That is an
    approximation on the attention path only; it must be stated in any report derived
    from a capture made with this port. NOTE: no public runtime (NeMo, vLLM, StepFun
    Step-3.7-Flash) implements this CSA indexer -- see work/plans/PORT_SPEC.md §3.
  * the vision tower is excluded.
"""


class Step5RoboticsConfig(Step3_7Config):

    arch_string = "MMGPTStepRoboticsForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(directory, **kwargs)

        # Text body + MTP depths. The vision model is deliberately dropped from the
        # class mapping (vision stays out of scope for these packs).
        self.model_classes = {"text": Step5RoboticsModel, "mtp": Step5RoboticsMTPModel}

        # ---- sparse (CSA) indexer descriptor -------------------------------------
        sc = self.read_cfg(dict, "text_config->sparse_config", None) or {}
        self.sparse_enabled = bool(sc.get("enabled", False))
        self.sparse_topk = sc.get("topk")
        self.sparse_region_block_size = sc.get("region_block_size")
        self.sparse_proxy_dim = sc.get("proxy_dim")
        self.sparse_num_heads = sc.get("sparse_indexer_num_heads")
        self.sparse_num_k_heads = sc.get("sparse_indexer_num_k_heads")
        self.sparse_rope_dim = sc.get("sparse_indexer_rope_dim")
        self.sparse_use_rope = sc.get("sparse_indexer_use_rope")
        self.sparse_q_norm_type = sc.get("sparse_indexer_q_norm_type")
        self.sparse_k_norm_type = sc.get("sparse_indexer_k_norm_type")
        self.sparse_csa_z_norm_type = sc.get("sparse_indexer_csa_z_norm_type")
        self.sparse_softmax_variant = sc.get("sparse_indexer_softmax_variant")
        self.sparse_ssmax_granularity = sc.get("sparse_indexer_ssmax_s_granularity")
        self.sparse_compression_method = sc.get("compression_method")
        self.sparse_attention_impl = sc.get("attention_impl")
        self.sparse_apply_to_layer_types = sc.get("apply_to_layer_types") or []

        # Full-attention layers that carry the indexer in the source checkpoint
        self.sparse_indexer_layers = [
            idx for idx in range(self.num_hidden_layers)
            if self.layer_types[idx] in self.sparse_apply_to_layer_types
        ]

        # MTP / nextn layers (model.layers.{92,93,94}). Per-layer rope_theta,
        # partial_rotary_factors, swiglu_limits, swiglu_limits_shared and layer_types are all
        # 95-long in this checkpoint, so the MTP depths already have config values at
        # indices 92..94 (sliding_attention + dense MLP, rope_theta 10000, prf 1.0).
        self.mtp_num_layers = self.read_cfg(int, "text_config->num_nextn_predict_layers", 0)
        self.mtp_base_layer_idx = self.num_hidden_layers

        # The component only exists when the checkpoint actually carries the tensors.
        mtp_key = f"model.layers.{self.mtp_base_layer_idx}.eh_proj"
        if self.mtp_num_layers == 0 or not any(
            self.stc.has_tensor(f"{mtp_key}.{t}") for t in ("weight", "trellis")
        ):
            self.model_classes.pop("mtp", None)


class Step5RoboticsModel(Step3_5Model):

    config_class = Step5RoboticsConfig
