from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device
from . import Module, Linear, RMSNorm
from .layernorm import LayerNorm

# Selection output width is padded to a multiple of this (matching dsa_topk.cu's -1 padded
# int32 output and the BC graph's slot stride)
_TOPK_PAD = 32


class Step5CSAIndexer(Module):
    """
    CSA lightning indexer for Step-5-Preview (DeepSeek-V3.2-DSA style, raw-token keys).

    Per-layer tensors (checkpoint `(out, in)`, transposed on load like every exllamav3 Linear):

        sparse_indexer_q.weight        [H_i*D_i, hidden]   BF16   query projection (from raw hidden)
        sparse_indexer_q_norm.weight   [D_i]               F32    RMSNorm over the proxy dims
        sparse_indexer_k.weight        [D_i, hidden]       BF16   key projection (one shared key head)
        sparse_indexer_k_norm.weight   [D_i]               F32    LayerNorm weight (biased)
        sparse_indexer_k_norm.bias     [D_i]               F32    LayerNorm bias
        sparse_indexer_w.weight        [H_i, hidden]       F32    per-indexer-head scoring weights
        sparse_indexer_z.weight        [D_i, hidden]       BF16   CSA block-compress content proj

    Geometry (verified from safetensors headers): hidden 4096, H_i = 16 indexer heads,
    D_i = 256 proxy dim, one shared key head, rope_dim = 32, topk = 512,
    region_block_size = 8.

    Pipeline per token t (hidden state x_t):

        q[t] = q_norm( W_q x_t ).view(H_i, D_i)      # RMSNorm over the D_i dims of each head
        k[s] = k_norm( W_k x_s )                     # biased LayerNorm over the D_i dims
        w[t] = W_w x_t                               # raw linear, no norm, no bias

    RoPE is applied to BOTH q and k on the FIRST rope_dim = 32 dims, GPT-J interleaved
    pairing, using the main attention's rope table (mla_attn._indexer_rope_). On the key
    side the order is wk GEMM -> LayerNorm -> RoPE (the norm is applied before the
    rotation), on the query side wq GEMM -> RMSNorm -> RoPE.

    Scoring formula (ported exactly from mla_attn.py:744-746 and dsa_triton.py:618):

        I(t, s) = (D_i * H_i) ** -0.5 * SUM_h [ w[t, h] * ReLU( D_i ** -0.5 * (q[t, h] . k[s]) ) ]

    - w[t, h] multiplies the POST-ReLU value and may be negative (no norm, no bias).
    - ReLU is inside the head reduction; the two 1/sqrt scales are D_i ** -0.5 (inside the
      ReLU, mathematically identical to folding it into the kernel's single
      D_i ** -0.5 * H_i ** -0.5 post-scale) and H_i ** -0.5 outside.
    - The reduction runs in fp32; scores are emitted fp16.
    - There is NO softmax over the s axis anywhere: raw scores go to top-k.

    Selection: per query token (shared across all main attention heads), top-`topk` by
    score over causal entries only (a query may only select keys at position <= its own),
    emitted as int32 key indices in ascending order, -1 padded to a multiple of 32. If
    fewer than `topk` causal keys exist, all of them are selected and the row is -1
    padded. This reference path resolves score ties in ascending key-index order (like
    dsa_topk.cu) via a stable descending sort.

    Two open items, both TODO hooks below:
      1. Region/block granularity: `region_block_size = 8` suggests selection at 8-token
         block granularity; this module implements token-granular top-k only
         (see `_select` and `select_blocks`).
      2. The CSA z/compressor: `sparse_indexer_z` (csa_block_compress) content projection
         is declared but the compressor itself is NOT implemented here; whether its block
         key is a mean-pool of member z vectors or a self-gated (softmax gate) pool is
         unresolved (see `csa_compressed_keys`).
    """

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int = 4096,
        num_heads: int = 16,
        proxy_dim: int = 256,
        rope_dim: int = 32,
        topk: int = 512,
        region_block_size: int = 8,
        rope_settings: RopeSettings | None = None,
        norm_eps: float = 1e-5,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        qbits_key: str = "bits",
        select_hq_bits: int = 0,
        key_q: str = "sparse_indexer_q",
        key_q_norm: str = "sparse_indexer_q_norm",
        key_k: str = "sparse_indexer_k",
        key_k_norm: str = "sparse_indexer_k_norm",
        key_w: str = "sparse_indexer_w",
        key_z: str = "sparse_indexer_z",
        submodules: dict | None = None,
    ):
        """
        :param config: Model config
        :param key: Tensor key prefix of the owning attention module, e.g.
                    "model.layers.3.self_attn"; every submodule key is f"{key}.{key_*}"
        :param layer_idx: Layer index (bookkeeping; selections are per-layer)
        :param hidden_size: Input width (4096 for Step-5)
        :param num_heads: Indexer heads H_i (16)
        :param proxy_dim: Indexer proxy dim D_i (256)
        :param rope_dim: Width of the partial rope, applied to the FIRST rope_dim dims (32)
        :param topk: Keys selected per query token (512)
        :param region_block_size: Selection granularity in tokens (8). Currently advisory
                                  only: selection is token-granular (see class docstring)
        :param rope_settings: Rope settings of the MAIN attention; the indexer shares its
                              table. Pairing must be GPT-J interleaved (RopeStyle.GPTJ),
                              and rotary_dim / inv_freq must cover the rope_dim-wide head
        :param norm_eps: Eps for both q_norm (RMSNorm) and k_norm (LayerNorm); the model
                         config carries rms_norm_eps = 1e-5
        :param qmap: Quantization state label for the hidden input of the quantizable
                     projections (q, z)
        :param out_dtype: Output dtype of the module's forward (unused; selection output
                          is int32), kept for Module signature parity
        :param qbits_key: Quantization bits key for q/z (EXL3)
        :param select_hq_bits: Half-integer bitrate selection for q/z
        :param key_q, key_q_norm, key_k, key_k_norm, key_w, key_z: Checkpoint key suffixes
        :param submodules: Prebuilt submodules (TP worker import), keyed by local name
        """
        super().__init__(config, key, qmap)

        assert rope_dim <= proxy_dim, "rope_dim must fit in the proxy dim"
        assert proxy_dim % 2 == 0, "proxy_dim must be even"
        assert region_block_size >= 1

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads          # H_i, indexer scoring heads
        self.proxy_dim = proxy_dim          # D_i, indexer proxy width
        self.rope_dim = rope_dim            # partial rope width (first dims)
        self.topk = topk
        self.region_block_size = region_block_size
        self.rope_settings = rope_settings
        self.rope: RoPE | None = None
        self.norm_eps = norm_eps
        self.out_dtype = out_dtype

        # Cache layers size the indexer-key plane off this (one D_i-wide key per token)
        self.idx_plane_dim = proxy_dim

        # In a TP worker the submodules arrive prebuilt (imported from the parent process)
        # instead of being constructed against the tensor collection
        def _sub(name, factory):
            m = submodules.get(name) if submodules is not None else factory()
            self.register_submodule(m)
            return m

        qmap_in = qmap + ".input" if qmap is not None else None

        # Query projection: quantizable, from the raw hidden state (no q_a latent on this
        # architecture; sparse_indexer_q is hidden -> H_i * D_i)
        self.idx_q = _sub("idx_q", lambda: Linear(
            config, f"{key}.{key_q}", hidden_size, num_heads * proxy_dim,
            qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits, qbits_key = qbits_key,
        ))
        self.idx_q_norm = _sub("idx_q_norm", lambda: RMSNorm(
            config, f"{key}.{key_q_norm}", norm_eps, out_dtype = torch.half,
        ))
        # Key head and per-head scoring weights are router-like: tiny, and selection noise
        # is coherent across every layer sharing them, so they stay unquantized
        self.idx_k = _sub("idx_k", lambda: Linear(
            config, f"{key}.{key_k}", hidden_size, proxy_dim,
            qmap = None, out_dtype = torch.half, pad_to = 1,
        ))
        # Biased LayerNorm over the D_i dims, applied before the rotation
        self.idx_k_norm = _sub("idx_k_norm", lambda: LayerNorm(
            config, f"{key}.{key_k_norm}", norm_eps, out_dtype = torch.half,
        ))
        self.idx_w = _sub("idx_w", lambda: Linear(
            config, f"{key}.{key_w}", hidden_size, num_heads,
            qmap = None, out_dtype = torch.half, pad_to = 1,
        ))
        # CSA block-compress content projection. Declared so its checkpoint tensors load,
        # but the compressor that pools z into block keys is NOT implemented here (TODO
        # hook: csa_compressed_keys). Whether the block key is a plain mean-pool of the
        # member z vectors or a self-gated (softmax gate) pool is an open item.
        self.idx_z = _sub("idx_z", lambda: Linear(
            config, f"{key}.{key_z}", hidden_size, proxy_dim,
            qmap = None, out_dtype = torch.half, pad_to = 1,
        ))

    @override
    def optimizer_targets(self):
        # The two full-size projections (q, z) sit in the quantization/optimizer group; the
        # router-like w/k projections and the norms do not
        q = self.idx_q.optimizer_targets() + self.idx_z.optimizer_targets()
        return [[q]]

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        if self.rope_settings and self.rope is None:
            self.rope = RoPE(device, self.rope_settings)

    @override
    def unload(self):
        super().unload()
        self.rope = None

    def _indexer_rope_(self, x4, position, positions, position_ids, inv_freq):
        """
        In-place partial rope on the leading rope_dim dims of a (bsz, seqlen, heads, D_i)
        tensor, via a strided trailing-slice view: the kernel reads through the head stride,
        so no slice copy or writeback is made (the eager mirror of mla_attn's
        _indexer_rope_ / the BC path's rope_gr narrow views). GPT-J interleaved pairing,
        same table as the main attention.
        """
        from ..ext import exllamav3_ext as ext
        if self.rope is None or self.rope_dim == 0:
            return
        rope = self.rope
        v = x4[..., : self.rope_dim]
        ext.rope(
            v, v, None, None,
            rope.inv_freq if inv_freq is None else inv_freq,
            position,
            positions.contiguous() if positions is not None else None,
            position_ids.contiguous() if position_ids is not None else None,
            int(rope.rope_settings.rope_style),
            rope.attn_factor,
            None, None, self.norm_eps, 0.0,
            rope.llama_4_scaling_beta, rope.llama_4_scaling_original,
            rope.rope_settings.rotate_dims, 0,
        )

    def project_q(
        self,
        x: torch.Tensor,
        params: dict,
        position: int | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inv_freq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Indexer queries, (bsz, seqlen, H_i, D_i), head-major, roped on the first rope_dim
        dims. q_norm is an RMSNorm over the D_i dims of each reshaped head.
        """
        bsz, seqlen, _ = x.shape
        q = self.idx_q.forward(x, params)
        q = q.view(bsz, seqlen, self.num_heads, self.proxy_dim)
        q = self.idx_q_norm.forward(q, params, out_dtype = torch.half)
        q = q.contiguous().view(bsz, seqlen, self.num_heads, self.proxy_dim)
        self._indexer_rope_(q, position, positions, position_ids, inv_freq)
        return q

    def project_k(
        self,
        x: torch.Tensor,
        params: dict,
        position: int | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inv_freq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Roped indexer keys for the current chunk, (bsz, seqlen, D_i), one shared key head.
        Order: wk GEMM -> biased LayerNorm over the D_i dims -> partial rope on the first
        rope_dim dims. Append these to the cache's indexer-key plane.
        """
        k = self.idx_k.forward(x, params)
        k = self.idx_k_norm.forward(k, params, out_dtype = torch.half).contiguous()
        bsz, seqlen, D_i = k.shape
        k = k.view(bsz, seqlen, 1, D_i)
        self._indexer_rope_(k, position, positions, position_ids, inv_freq)
        return k.view(bsz, seqlen, D_i)

    def project_w(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Per-head scoring weights w[t, h], (bsz, seqlen, H_i). Raw linear, no norm, no
        bias; may be negative."""
        return self.idx_w.forward(x, params)

    def indexer_scores(
        self,
        q_idx: torch.Tensor,
        w: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lightning-indexer scores over a key plane, fp16 out.

        :param q_idx: (bsz, seqlen, H_i, D_i) roped queries
        :param w: (bsz, seqlen, H_i) per-head weights
        :param keys: (bsz, kv_len, D_i) roped keys; keys[:, s] sits at absolute position s
        :return: (bsz, seqlen, kv_len) fp16 scores, no softmax

            I(t, s) = (D_i * H_i) ** -0.5 * SUM_h w[t, h] * ReLU(D_i ** -0.5 * q[t, h] . k[s])

        fp32 accumulation (the relu-weighted head reduction runs in fp32, matching the
        reference); the returned scores are fp16, matching dsa_triton.py:658.
        """
        D_i = self.proxy_dim
        qf = q_idx.float()
        kf = keys.float()
        sc = torch.einsum("bthd,bsd->bths", qf, kf)
        sc = F.relu(sc * (D_i ** -0.5))
        sc = torch.einsum("bth,bths->bts", w.float() * (self.num_heads ** -0.5), sc)
        return sc.to(torch.half)

    def _select(
        self,
        scores: torch.Tensor,
        host_seqlens: list[int],
    ) -> torch.Tensor:
        """
        Token-granular causal top-k selection over raw scores.

        :param scores: (bsz, seqlen, kv_len) unmasked fp16 scores
        :param host_seqlens: per-batch-row cache length BEFORE this chunk (position of the
                             row's first query token)
        :return: (bsz * seqlen, K_pad) int32 key indices, ascending per row, -1 padded.
                 K_pad = ceil(min(topk, max_visible) / 32) * 32.

        Causality lives in the selection: query row r of batch row b sits at absolute
        position host_seqlens[b] + r and may only select keys at position <= its own, so
        the downstream gathered attention needs no mask of its own. Rows with fewer than
        topk causal keys select all of them.

        NOTE (TODO): this is token granularity. The block-granular variant
        (region_block_size = 8) would score/aggregate per 8-token region and emit block
        indices expanded to token indices; see select_blocks.
        """
        bsz, seqlen, kv_len = scores.shape
        dev = scores.device
        t_max = max(host_seqlens) + seqlen
        k_pad = -(-min(self.topk, t_max) // _TOPK_PAD) * _TOPK_PAD

        # Absolute query positions, (bsz, seqlen); causal bound per row is position + 1
        qpos = torch.tensor(host_seqlens, dtype = torch.long, device = dev).unsqueeze(1)
        qpos = qpos + torch.arange(seqlen, dtype = torch.long, device = dev).unsqueeze(0)
        spos = torch.arange(kv_len, dtype = torch.long, device = dev).unsqueeze(0).unsqueeze(0)
        visible = spos <= qpos.unsqueeze(-1)
        sc = scores.float().masked_fill(~visible, -float("inf"))

        # Stable descending sort = exact top-k with score ties resolved in ascending key
        # index order (matching dsa_topk.cu's ordered compaction)
        k_cand = min(self.topk, kv_len)
        idx = torch.argsort(sc, dim = -1, stable = True, descending = True)[..., :k_cand]
        vals = sc.gather(-1, idx)
        keep = vals > -float("inf")
        idx = torch.where(keep, idx, torch.full_like(idx, kv_len))
        # Ascending key order, discarded entries collapse to the end
        idx = idx.sort(dim = -1).values
        idx = torch.where(idx < kv_len, idx, torch.full_like(idx, -1))
        idx = idx[..., :k_pad]
        out = torch.full((bsz, seqlen, k_pad), -1, dtype = torch.int32, device = dev)
        out[..., : idx.shape[-1]] = idx.to(torch.int32)
        return out.view(bsz * seqlen, k_pad)

    def select_blocks(self, *args, **kwargs):
        """
        TODO (open item 1): region/block-granular selection at region_block_size = 8.
        Step-5's sparse_config carries region_block_size = 8, which points at scoring and
        selecting whole 8-token regions (like DSV4's compressor-entry granularity) rather
        than single tokens. Only token-granular selection is implemented (_select); when
        the block variant lands it must still emit raw token indices so the gathered
        attention path is unchanged.
        """
        raise NotImplementedError(
            "Step5CSAIndexer: block-granular selection (region_block_size) is not "
            "implemented; use token-granular _select"
        )

    def csa_compressed_keys(self, *args, **kwargs):
        """
        TODO (open item 2): CSA block compressor over the z projection
        (compression_method = "csa_block_compress").

        The sparse_indexer_z content projection is loaded (self.idx_z: hidden -> D_i) but
        the compressor that turns per-token z into block keys is NOT implemented here.
        Two candidate semantics, unresolved:
          - mean-pool: block key = mean of the member z vectors (plus a learned APE?)
          - self-gated: block key = softmax-gated pool of member z vectors, DSV4-style
            (dsv4.py compressor: comp = (kv * gate.softmax(dim = 2)).sum(dim = 2))
        Until this is resolved, index keys are raw per-token k vectors (project_k).
        """
        raise NotImplementedError(
            "Step5CSAIndexer: CSA z-block compressor is not implemented; keys are "
            "raw per-token k (project_k)"
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        keys: torch.Tensor,
        host_seqlens: list[int] | None = None,
        position: int | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inv_freq: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Full indexer step: score the key plane against this chunk's hidden states and
        select the sparse context.

        :param x: (bsz, seqlen, hidden) raw hidden states
        :param keys: (bsz, kv_len, D_i) roped indexer keys in absolute order; keys[:, s]
                     is the key at absolute position s (per-row valid range is
                     0 .. host_seqlens[b] + seqlen - 1; anything outside the causal bound
                     is masked out of the selection)
        :param host_seqlens: per-row cache length before the chunk; defaults to
                             params["cache_seqlens"] (host list or tensor)
        :param position, positions, position_ids, inv_freq: rope inputs, mirroring
                             mla_attn: taken from params unless passed explicitly
        :return: (bsz * seqlen, K_pad) int32 selected key indices, ascending per query
                 token, -1 padded to a multiple of 32. Selection is per query token and
                 shared by all main attention heads.
        """
        if position is None:
            position = params.get("position", 0)
        if positions is None:
            positions = get_for_device(params, "positions", self.device, None)
        if position_ids is None:
            position_ids = get_for_device(params, "position_ids", self.device, None)
        if inv_freq is None:
            inv_freq = get_for_device(params, "inv_freq", self.device, None)
        if host_seqlens is None:
            host_seqlens = params.get("cache_seqlens")
            if isinstance(host_seqlens, torch.Tensor):
                host_seqlens = host_seqlens.cpu().tolist()
        assert host_seqlens is not None, "host_seqlens or params['cache_seqlens'] required"

        q_idx = self.project_q(x, params, position, positions, position_ids, inv_freq)
        w = self.project_w(x, params)
        scores = self.indexer_scores(q_idx, w, keys)
        return self._select(scores, host_seqlens)
