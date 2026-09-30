"""
Scalable-Softmax (SSMax) attention logit scaling.

Implements the attention-logit scale of Scalable-Softmax (Nakanishi 2025,
arXiv:2501.19399): instead of the standard ``1/sqrt(head_dim)`` scale, logits are
scaled per query head by a learned parameter ``s``:

    SSMax(z)_i = n^(s * z_i) / sum_j n^(s * z_j)
               = softmax over (s * log(n) * z_i)

i.e. an effective logit scale of ``s * log(n)`` where ``n`` is the softmax length
(the number of keys the query attends to). Two candidate semantics are supported
and switchable via ``variant``:

  * ``"log_n"``    -> scale = s * log(n)     (full Scalable-Softmax; DEFAULT —
                     this is what the SSMax paper defines)
  * ``"constant"`` -> scale = s              (plain learned per-head temperature;
                     matches the "constant softmax scale 0.08496" description in
                     the Step-5 GGUF conversion notes)

Step-5-Preview checkpoint facts (measured, see port-notes/03 and 05):
  * ``model.layers.N.self_attn.ssmax_s`` is a plain parameter shaped [64] F32,
    64 = num_attention_heads of the MAIN attention (head_dim 192), present on
    exactly the 23 full_attention layers (3, 7, ..., 91).
  * The value is a measured constant 0.08495759963989258, bit-identical across
    all 64 entries and all layers -> per-head shape is structural, not learned
    variation.
  * 1/sqrt(192) = 0.07217; ssmax_s = 0.08496 is close but not equal — consistent
    with a learned ``s``, NOT with a re-derived inverse-sqrt.

==========================================================================
!!! UNVERIFIED — FLAGGED OPEN ITEM !!!
The runtime formula (``s`` vs ``s * log(n)``) is NOT verified against the
StepFun reference implementation. No public runtime implements this tensor
(llama.cpp drops it; vLLM/SGLang have no step3p5v sparse path), and the
checkpoint alone cannot distinguish the two readings: with ``s`` constant,
``s * log(n)`` is "constant" at fixed ``n``. The default here ("log_n") follows
the SSMax paper, where the log(n) factor is the entire point (it is what fixes
attention fading). Validate against the reference runtime before shipping.
If the reference turns out to use the constant form, pass
``variant="constant"`` — nothing else changes.
==========================================================================

This module is deliberately torch-free at import time (tensor math is duck-typed
against the input tensor type), so it can be imported and unit-tested anywhere.
"""

import math
from typing import Any

# Variant names
SSMAX_VARIANT_LOGN = "log_n"        # scale = s * log(n)   (full SSMax)
SSMAX_VARIANT_CONSTANT = "constant" # scale = s
SSMAX_VARIANTS = (SSMAX_VARIANT_LOGN, SSMAX_VARIANT_CONSTANT)
DEFAULT_SSMAX_VARIANT = SSMAX_VARIANT_LOGN

# Step-5-Preview measured constant (self_attn.ssmax_s, [64] F32, all entries equal).
# Reference value only; the runtime should load the tensor from the checkpoint.
SSMAX_S_STEP5 = 0.08495759963989258


def _check_variant(variant):
    if variant not in SSMAX_VARIANTS:
        raise ValueError(f"ssmax: unknown variant {variant!r}, expected one of {SSMAX_VARIANTS}")


def _check_softmax_len(softmax_len):
    # A softmax length is a key count: require an integral value >= 1. Integral
    # floats (4.0) are accepted; anything fractional is rejected loudly rather
    # than silently producing a wrong log(n) scale.
    if isinstance(softmax_len, bool) or not isinstance(softmax_len, (int, float)):
        raise ValueError(f"ssmax: softmax_len must be a scalar number, got {type(softmax_len).__name__}")
    if not float(softmax_len).is_integer():
        raise ValueError(f"ssmax: softmax_len must be an integer count, got {softmax_len}")
    if not (softmax_len >= 1):
        raise ValueError(f"ssmax: softmax_len must be >= 1, got {softmax_len}")


def ssmax_logit_scale(s, softmax_len, variant = DEFAULT_SSMAX_VARIANT):
    """
    Scalar SSMax logit scale for one head.

    variant "log_n":     s * log(n)   (n = softmax_len; natural log, per the paper)
    variant "constant":  s

    Returns a float. Note n = 1 gives scale 0 under "log_n" (log(1) = 0); that is
    mathematically harmless — with a single key the softmax output is [1] either
    way — but it is why callers must not pass 0 or negative lengths.
    """
    _check_variant(variant)
    _check_softmax_len(softmax_len)
    s = float(s)
    if variant == SSMAX_VARIANT_CONSTANT:
        return s
    return s * math.log(softmax_len)


def ssmax_per_head_scale(
    num_heads: int,
    head_dim: int,
    ssmax_s: Any,
    softmax_len,
    variant = DEFAULT_SSMAX_VARIANT,
):
    """
    Per-head SSMax logit scale.

    Args:
        num_heads:   number of MAIN attention query heads (Step-5: 64).
        head_dim:    attention head dim (Step-5: 192). Not used in the SSMax
                     formula — accepted for interface parity with
                     head_dim ** -0.5 call sites and for validation only. When
                     SSMax is active it REPLACES 1/sqrt(head_dim); it is never
                     multiplied into it.
        ssmax_s:     the checkpoint parameter ``self_attn.ssmax_s`` — a tensor
                     with 1 or num_heads elements (Step-5: [64] F32), or a plain
                     number. Must already be on the target device.
        softmax_len: scalar n — the softmax length (number of keys attended to).
                     Semantics of n (full context vs. selected keys vs. per-row
                     causal length) are CALLER-DEFINED and unverified; see the
                     module docstring.
        variant:     "log_n" (default) or "constant".

    Returns:
        Per-head scale as a tensor of shape [num_heads] when ``ssmax_s`` is a
        tensor (same tensor library/device as the input, float32), or a plain
        tuple of floats when ``ssmax_s`` is a Python number or sequence.
    """
    _check_variant(variant)
    _check_softmax_len(softmax_len)
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ValueError(f"ssmax: num_heads must be a positive int, got {num_heads!r}")
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim < 1:
        raise ValueError(f"ssmax: head_dim must be a positive int, got {head_dim!r}")

    factor = 1.0 if variant == SSMAX_VARIANT_CONSTANT else math.log(softmax_len)

    # Tensor path (torch or anything with numel/reshape/expand/__mul__)
    numel = getattr(ssmax_s, "numel", None)
    if callable(numel):
        n = int(numel())
        if n not in (1, num_heads):
            raise ValueError(
                f"ssmax: ssmax_s has {n} elements, expected 1 or num_heads={num_heads} "
                f"(granularity 'q_head')"
            )
        s = ssmax_s.float().reshape(-1)
        if n == 1:
            s = s.expand(num_heads)
        if factor == 1.0:
            return s.contiguous()
        return (s * factor).contiguous()

    # Sequence path (plain list/tuple of per-head values)
    if isinstance(ssmax_s, (list, tuple)):
        if len(ssmax_s) not in (1, num_heads):
            raise ValueError(
                f"ssmax: ssmax_s has {len(ssmax_s)} elements, expected 1 or "
                f"num_heads={num_heads} (granularity 'q_head')"
            )
        vals = [float(v) for v in ssmax_s]
        if len(vals) == 1:
            vals = vals * num_heads
        return tuple(v * factor for v in vals)

    # Plain-number path
    if isinstance(ssmax_s, bool) or not isinstance(ssmax_s, (int, float)):
        raise ValueError(
            f"ssmax: ssmax_s must be a tensor, a sequence, or a number, got "
            f"{type(ssmax_s).__name__}"
        )
    return tuple(float(ssmax_s) * factor for _ in range(num_heads))


def fold_ssmax_into_q(q: Any, per_head_scale: Any):
    """
    Apply a per-head logit scale by folding it into q.

    Identity used: scale_h * (q_h . k) = (scale_h * q_h) . k — so a per-head (or
    per-head-per-row) scale needs NO kernel changes when folded into q after
    RoPE/QK-norm and before the attention kernel. This is the recommended
    integration path (see port-notes/06-ssmax-patch.md).

    Args:
        q:               (..., num_heads, head_dim) query tensor, post-RoPE.
        per_head_scale:  [num_heads] (broadcast over rows) or [..., num_heads]
                         tensor broadcastable to q's head axis, last dim ==
                         num_heads.

    Returns a new tensor; ``q`` is not modified.
    """
    n_heads = q.shape[-2]
    scale_shape = tuple(per_head_scale.shape)
    if not scale_shape or scale_shape[-1] != n_heads:
        raise ValueError(
            f"ssmax: per_head_scale shape {scale_shape} does not end in num_heads={n_heads}"
        )
    # Broadcast: align scale's last dim to q's head axis (second-to-last), its
    # leading dims to q's leading dims.
    view_shape = (1,) * (q.dim() - len(scale_shape) - 1) + scale_shape + (1,)
    if len(view_shape) != q.dim():
        raise ValueError(
            f"ssmax: cannot broadcast scale shape {scale_shape} over q shape {tuple(q.shape)}"
        )
    for vs, qs in zip(view_shape, q.shape):
        if vs != 1 and vs != qs:
            raise ValueError(
                f"ssmax: scale shape {scale_shape} does not broadcast over q shape "
                f"{tuple(q.shape)}"
            )
    return q * per_head_scale.view(*view_shape)
