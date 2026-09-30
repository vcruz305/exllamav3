from __future__ import annotations
from typing_extensions import override
import torch
from torch import nn

from ..model.config import Config
from . import Module


class Step5SSMaxScale(Module):
    """
    Scalable-Softmax per-head scale for the MAIN attention of a Step-5 full-attention layer.

    Checkpoint tensor: ``model.layers.N.self_attn.ssmax_s`` -- note there is **no** ``.weight``
    suffix, so ``tensor_key == self.key``.

    Shape [64] F32, where 64 == ``num_attention_heads`` of the MAIN attention (not the 16
    indexer heads). Measured on the real checkpoint as a FROZEN CONSTANT,
    0.08495759963989258, bit-identical across all 64 entries and all 23 layers that carry it
    (port-notes/03, byte-range read of the BF16 shards).

    Semantics (Scalable-Softmax, Nakanishi 2025, arXiv 2501.19399): the attention logit scale
    is ``s * log(n)`` instead of the usual ``1/sqrt(head_dim)``, where ``s`` is this value and
    ``n`` is the softmax length. See ``exllamav3/modules/ssmax.py`` for the scale helper.

    UNVERIFIED: whether the StepFun runtime applies ``s`` alone or ``s * log(n)``. The paper's
    definition includes ``log(n)``; that is the default. This module deliberately does NOT fold
    the scale into any projection -- it just carries the tensor so the pack is complete and the
    attention forward can read it.

    This lives here rather than on ``Attention`` because it is a bare parameter with no
    quantization and no compute of its own; attaching it to the layer's indexer module keeps
    core ``Attention`` untouched for every other architecture.
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Step5SSMaxScale"
        self.out_dtype = out_dtype
        # No ".weight" suffix: the tensor IS the key.
        self.tensor_key = self.key
        self.s = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        # Passthrough. This module only carries the ssmax_s tensor; the scale is applied
        # inside the main attention softmax (see exllamav3/modules/ssmax.py), not here.
        return x

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        t = self.config.stc.get_tensor(
            self.tensor_key, self.device, float2half = False, allow_bf16 = False,
        )
        self.s = nn.Parameter(t.float(), requires_grad = False)

    @override
    def unload(self):
        self.device = None
        self.s = None

    @override
    def get_tensors(self, **kwargs):
        return {self.tensor_key: self.s} if self.s is not None else {}

    def num_heads(self) -> int:
        return 0 if self.s is None else int(self.s.shape[0])
