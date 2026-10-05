"""
Draft-only pruned LM head: logits over a leading EXL3 column slice of the target's lm_head.

Drafters (MTP heads, DFlash2) that share the target's head only need candidates, not the full
vocabulary: a draft whose true argmax lies past the slice is just a rejection at verify time,
and verification keeps the full head, so the verified (greedy) output is unchanged. The 8-bit
RED-SNOW head is 634 MB; a 65,536-column slice reads 268 MB.
"""
from __future__ import annotations
import torch
from ..ext import exllamav3_ext as ext

_cache = {}


def pruned_head(lm, n: int, device):
    """(trellis slice, svh slice, width) for the first n columns (rounded down to 128), or None
    when the head is not a bias-free EXL3 linear or the slice would not be narrower."""
    key = (id(lm), n, str(device))
    hit = _cache.get(key)
    if hit is not None:
        return hit or None
    inner = getattr(lm, "inner", None)
    tr = getattr(inner, "trellis", None)
    if tr is None or getattr(inner, "bias", None) is not None or getattr(inner, "svh", None) is None:
        _cache[key] = False
        return None
    n_full = tr.shape[1] * 16
    n2 = min(n, n_full) // 128 * 128
    if n2 <= 0 or n2 >= n_full:
        _cache[key] = False
        return None
    res = (tr[:, :n2 // 16, :].contiguous().to(device), inner.svh[:n2].contiguous().to(device), n2)
    _cache[key] = res
    print(f" -- pruned draft head: {n2}/{n_full} columns", flush = True)
    return res


def pruned_logits(lm, x: torch.Tensor, n: int):
    """x (..., k) -> half logits (..., n2) over the slice, or None (caller uses the full head)."""
    ph = pruned_head(lm, n, x.device)
    if ph is None:
        return None
    tr, svh, n2 = ph
    inner = lm.inner
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    if x2.dtype != torch.half:
        x2 = x2.half()
    x2 = x2.contiguous()
    xh = torch.empty_like(x2)
    y = torch.empty((x2.shape[0], n2), dtype = torch.half, device = x2.device)
    ext.exl3_gemm(x2, tr, y, inner.suh, xh, svh, -1, inner.mcg, inner.mul1, 0)
    return y.view(*lead, n2)
