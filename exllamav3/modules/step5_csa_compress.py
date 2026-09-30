from __future__ import annotations

import torch

from ..model.config import Config
from .module import Module
from .linear import Linear
from ..ext import exllamav3_ext as ext
from ..constants import PAGE_SIZE

"""
Step-5-Preview CSA block compressor (``compression_method: "csa_block_compress"``).

One per full-attention layer. Maps the checkpoint's SINGLE compression projection
``model.layers.N.self_attn.sparse_indexer_z.weight [proxy_dim, hidden]`` (BF16) onto
exllamav3's fused DSV4 window compressor (``exllamav3_ext/dsv4_compress.cu``). See
``port-notes/07-z-compressor-mapping.md`` for the full decision record.

MAPPING (hypothesis A -- degenerate gate, mean pooling):

  z rows        = sparse_indexer_z(x)              (seq, W) fp16, W = hd = proxy_dim (256)
  gate rows     = 0                               -> per-column softmax = 1/m -> MEAN pool
  window        = m = region_block_size (8) tokens, NON-overlapping (W == hd -> kernel
                  non-overlap branch, dsv4_compress.cu:413-414)
  entry w       = mean(z rows of tokens [w*8, w*8+8))   -- equals z(mean hidden of block)
  norm          = NONE (config sparse_indexer_csa_z_norm_type "none"): the kernel's
                  RMSNorm stage is bypassed with the ``no_norm`` flag
  rope          = GPT-J interleaved pairs on the trailing rope_dim (32) columns at the
                  window's first-token position w*8 (dsv4_compress.cu:190-202)
  ape           = zeros((m, W)) fp32 -- the checkpoint has no ape tensor (kernel still
                  requires the argument, dsv4_compress.cu:405)

The ONLY learned tensor on this path is the z projection; there is no gate projection and
no learned positional bias in the checkpoint (Step-5 ships 8 sparse tensors per layer,
none of them a gate/ape; see port-notes/05 §1).

NORM BYPASS: the fused kernel unconditionally RMS-normalizes its output
(dsv4_compress.cu:170-186 in the unpatched tree), so ``norm_type: "none"`` cannot be
expressed with ``norm_w = ones``. This module therefore calls the kernel with the
``no_norm`` flag introduced by the patch recorded in port-notes/07. Against an extension
build WITHOUT that flag (or without the bindings.cpp companion diff), ``forward_fused``
falls back to the pure-torch chunk path in this file -- never to the norm-applying kernel,
which would silently produce wrong (unit-RMS) entries.
"""


class Step5CSACompressor(Module):
    """
    Block compressor for one Step-5 full-attention layer's sparse indexer.

    Checkpoint key: pass ``key = "model.layers.N.self_attn.sparse_indexer_z"`` (the Linear
    loads ``{key}.weight``, i.e. exactly the checkpoint tensor). ``proxy_dim``, block size,
    rope config and z norm type are read from the Step-5 sparse config attributes on the
    Config (``sparse_proxy_dim``, ``sparse_region_block_size``, ``sparse_rope_dim``,
    ``sparse_use_rope``, ``sparse_csa_z_norm_type``); explicit ctor args override.

    Semantics (mapping A, norm bypassed):

    - ``forward``: torch reference over complete 8-token blocks of one chunk
      (``(bsz, nw, hd)`` fp16, roped at absolute window positions).
    - ``forward_fused``: cached-path entry -- projects z, then runs the fused
      ``ext.dsv4_compress`` with zero gate, zero ape and ``no_norm = True``, writing
      emitted entries straight into paged pools and appending the projected rows to the
      caller-supplied ring buffers (single job, matching ``DSV4Compressor.forward_fused``).
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        qmap: str | None = None,
        select_hq_bits: int = 0,
        proxy_dim: int | None = None,
        block_size: int | None = None,
        rope_dim: int | None = None,
        norm_type: str | None = None,
        wz: Linear | None = None,
    ):
        super().__init__(config, key, qmap)

        self.hd = proxy_dim if proxy_dim is not None else getattr(config, "sparse_proxy_dim", None)
        self.m = block_size if block_size is not None else getattr(config, "sparse_region_block_size", None)
        self.rope_dim = rope_dim if rope_dim is not None else getattr(config, "sparse_rope_dim", 0)
        self.norm_type = norm_type if norm_type is not None else getattr(config, "sparse_csa_z_norm_type", "none")
        use_rope = getattr(config, "sparse_use_rope", True)
        if self.hd is None or self.m is None:
            raise ValueError(f"{key}: proxy_dim / region_block_size not resolvable from config")
        if not use_rope:
            self.rope_dim = 0

        # Fail closed on anything this mapping does not implement
        if self.norm_type != "none":
            raise NotImplementedError(
                f"{key}: sparse_indexer_csa_z_norm_type {self.norm_type!r} not implemented; "
                f"this mapping supports \"none\" only (see port-notes/07)"
            )
        assert self.m and PAGE_SIZE % self.m == 0, \
            f"{key}: region_block_size {self.m} must divide PAGE_SIZE {PAGE_SIZE}"
        assert self.rope_dim % 2 == 0 and self.rope_dim <= self.hd, \
            f"{key}: bad sparse_indexer_rope_dim {self.rope_dim} for proxy_dim {self.hd}"

        # hd == W == proxy_dim -> kernel non-overlapping branch (W == hd)
        self.hd = int(self.hd)
        self.m = int(self.m)
        self.rope_dim = int(self.rope_dim)
        self.head_dim = self.hd
        self.proj_width = self.hd
        self.hidden_size = hidden_size

        self.wz = wz if wz is not None else Linear(
            config,
            key,
            hidden_size,
            self.hd,
            qmap = qmap,
            out_dtype = torch.half,
            trim_padded_out = True,
            select_hq_bits = select_hq_bits,
        )
        self.register_submodule(self.wz)

        self.fused_inv_freq = None     # (rope_dim / 2,) fp32, set by make_fused
        self._ape = None               # (m, W) fp32 zeros, allocated per device
        self._gate0 = None             # (seq, W) fp16 zero gate scratch, grown on demand
        self._norm_w1 = None           # (hd,) fp16 ones; required arg, unused with no_norm

    # ------------------------------------------------------------------ setup

    def make_fused(self, inv_freq: torch.Tensor):
        """Arm the fused cached path with the compress rope table (rope_dim / 2 entries)."""
        assert inv_freq.dtype == torch.float
        assert inv_freq.shape[0] * 2 == self.rope_dim, \
            f"{self.key}: inv_freq length {inv_freq.shape[0]} does not match rope_dim {self.rope_dim}"
        self.fused_inv_freq = inv_freq

    def unmake_fused(self):
        self.fused_inv_freq = None
        self._ape = self._gate0 = self._norm_w1 = None

    def _fused_buffers(self, device):
        if self._ape is None or self._ape.device != device:
            # Step-5 has no ape tensor: the positional-bias term is exactly zero
            self._ape = torch.zeros((self.m, self.proj_width), dtype = torch.float, device = device)
            self._norm_w1 = torch.ones((self.hd,), dtype = torch.half, device = device)
            self._gate0 = None
        if self._gate0 is None or self._gate0.device != device:
            self._gate0 = None    # allocated per call to the exact chunk width

    @staticmethod
    def _rope_gptj_(comp: torch.Tensor, wpos: torch.Tensor, inv_freq: torch.Tensor):
        """
        In-place GPT-J (interleaved-pair) rotation of the trailing rd columns of comp
        (N, hd) fp32 at positions wpos (N,), matching dsv4_compress.cu:190-202 and
        tests/test_dsv4_compress_kernel.py:43-49: (e, o) -> (e*cos - o*sin, o*cos + e*sin).
        """
        rd = inv_freq.shape[0] * 2
        c0 = comp.shape[-1] - rd
        theta = wpos.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        cos, sin = theta.cos(), theta.sin()
        e = comp[:, c0 + 0::2]
        o = comp[:, c0 + 1::2]
        comp[:, c0:] = torch.stack((e * cos - o * sin, o * cos + e * sin), dim = -1).flatten(-2)

    def _pool_blocks(self, kvw: torch.Tensor) -> torch.Tensor:
        """
        Per-column softmax-gated pooling with the degenerate (all-zero) gate over window
        entries: softmax(0) = 1/m for every entry, so the pool is exactly the MEAN of the
        m rows, computed in fp32 like the kernel's online accumulator. kvw: (nw, m, hd)
        fp32 -> (nw, hd) fp32. No norm is applied (z_norm_type "none").
        """
        return kvw.mean(dim = 1)

    # --------------------------------------------------------------- reference

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
        inv_freq: torch.Tensor | None = None,
        position: int = 0,
    ) -> torch.Tensor:
        """
        Torch reference path. x: (bsz, seq, hidden). Returns complete-window block entries
        (bsz, nw, hd) fp16, nw = seq // m (sub-window remainder dropped), roped at absolute
        window positions (position // m + w) * m. No norm (z_norm_type "none").
        """
        z = self.wz.forward(x, params)                       # (bsz, seq, hd) fp16
        bsz, seq, _ = z.shape
        usable = (seq // self.m) * self.m
        if usable == 0:
            return z.new_zeros((bsz, 0, self.hd))
        kvw = z[:, :usable].float().view(bsz, -1, self.m, self.hd)
        comp = self._pool_blocks(kvw)                        # (bsz, nw, hd) fp32
        inv = inv_freq if inv_freq is not None else self.fused_inv_freq
        if self.rope_dim and inv is not None:
            nw = comp.shape[1]
            wpos = (torch.arange(nw, device = x.device, dtype = torch.long) + position // self.m) * self.m
            self._rope_gptj_(comp.view(-1, self.hd), wpos, inv)
        return comp.half() if out_dtype is None else comp.to(out_dtype)

    def optimizer_targets(self):
        return []

    # ------------------------------------------------------------ cached path

    def forward_fused(
        self,
        x: torch.Tensor,
        params: dict,
        buf_kv: torch.Tensor,
        buf_gate: torch.Tensor,
        dest_a: torch.Tensor,
        dest_b: torch.Tensor | None,
        position: int,
        inv_freq: torch.Tensor | None = None,
        pool_bt: torch.Tensor | None = None,
        pool_epp: int = 0,
        stage_rel: bool = False,
    ):
        """
        Cached-path forward (single job): project z, pool complete windows of this chunk
        plus any rows carried in the ring, rope at the window positions, and write emitted
        entries into the pools exactly like dsv4_compress (entry ec0 + w, block-table
        remap when pool_bt is given, rows [0, nw) when stage_rel). This chunk's z rows are
        appended to buf_kv / buf_gate at ring rows (position + j) % buf_rows.

        buf_kv / buf_gate: (buf_rows, W) fp16 rings (buf_rows = PAGE_SIZE + m);
        dest_a / dest_b: fp16 pools (dest_b None -> all hd columns into dest_a, pool_idx
        layout). The zero gate ring (buf_gate) is written but always reads as zero.

        Uses the fused kernel with no_norm = True when the extension exposes the flag
        (see port-notes/07); falls back to the exact torch chunk path otherwise. It never
        calls the kernel without the flag: an unpatched kernel would RMS-normalize the
        entries, which is wrong for z_norm_type "none".
        """
        assert x.shape[0] == 1, f"{self.key}: forward_fused is single-job"
        assert self.fused_inv_freq is not None or self.rope_dim == 0, \
            f"{self.key}: call make_fused(inv_freq) before forward_fused"
        z = self.wz.forward(x, params)[0]                    # (seq, hd) fp16
        seq, W = z.shape
        assert W == self.proj_width
        inv = inv_freq if inv_freq is not None else self.fused_inv_freq
        if inv is None:
            # rope_dim == 0: the kernel still type/shape-checks inv_freq (dsv4_compress.cu:400,
            # :416 with rd = 0), so hand it an empty table
            inv = torch.zeros((0,), dtype = torch.float, device = z.device)
        self._fused_buffers(z.device)
        gate = torch.zeros((seq, W), dtype = torch.half, device = z.device)
        try:
            ext.dsv4_compress(
                z, gate, buf_kv, buf_gate, None,              # ovl: non-overlapping
                self._ape, self._norm_w1, 1e-6, inv, dest_a, dest_b,
                position, None, self.m, None, pool_bt, pool_epp, stage_rel,
                True,                                          # no_norm
            )
        except TypeError:
            # Extension predates the no_norm flag (or its bindings.cpp companion diff):
            # compute the identical chunk semantics in torch instead
            self._forward_fused_torch(z, gate, buf_kv, buf_gate, dest_a, dest_b, position,
                                      inv, pool_bt, pool_epp, stage_rel)

    def _forward_fused_torch(
        self,
        z: torch.Tensor,
        gate: torch.Tensor,
        buf_kv: torch.Tensor,
        buf_gate: torch.Tensor,
        dest_a: torch.Tensor,
        dest_b: torch.Tensor | None,
        position: int,
        inv_freq: torch.Tensor | None,
        pool_bt: torch.Tensor | None,
        pool_epp: int,
        stage_rel: bool,
    ):
        """
        Pure-torch equivalent of one dsv4_compress call for this mapping (zero gate, zero
        ape, no norm): window gather across the chunk/ring boundary, mean pooling, rope,
        pool scatter, ring store. Mirrors the addressing invariants of
        dsv4_compress.cu:20-24, 107-109, 204-210 and 270-275.
        """
        seq, W = z.shape
        m = self.m
        hd = self.hd
        buf_rows = buf_kv.shape[0]
        ec0 = position // m
        nw = (position + seq) // m - ec0

        if nw > 0:
            # Window source rows: absolute positions [(ec0 + w) * m, (ec0 + w + 1) * m),
            # from this chunk when abs >= position else from the ring (abs % buf_rows)
            w = torch.arange(nw, device = z.device, dtype = torch.long)
            e = torch.arange(m, device = z.device, dtype = torch.long)
            abs_pos = (ec0 + w).unsqueeze(1) * m + e.unsqueeze(0)          # (nw, m)
            from_new = abs_pos >= position
            src_new = z.float()[(abs_pos - position).clamp_(max = seq - 1)]
            src_ring = buf_kv.float()[abs_pos.remainder_(buf_rows)]
            kvw = torch.where(from_new.unsqueeze(-1), src_new, src_ring)   # (nw, m, hd)
            comp = self._pool_blocks(kvw)                                  # (nw, hd) fp32
            if self.rope_dim and inv_freq is not None:
                self._rope_gptj_(comp, (ec0 + w) * m, inv_freq)

            # Pool scatter: entry ec0 + w (staging rows [0, nw) when stage_rel)
            drow = w if stage_rel else ec0 + w
            if pool_bt is not None and not stage_rel:
                assert pool_epp > 0, f"{self.key}: paged mode requires pool_epp"
                drow = pool_bt[drow // pool_epp] * pool_epp + drow.remainder(pool_epp)
            out = comp.half()
            if dest_b is None:
                dest_a[drow] = out
            else:
                wa = dest_a.shape[-1]
                dest_a[drow] = out[:, :wa]
                dest_b[drow] = out[:, wa:]

        # Ring store: trailing min(seq, buf_rows) rows at (position + j) % buf_rows
        j0 = seq - buf_rows if seq > buf_rows else 0
        j = torch.arange(j0, seq, device = z.device, dtype = torch.long)
        ridx = (position + j).remainder(buf_rows)
        buf_kv[ridx] = z[j]
        buf_gate[ridx] = gate[j]
