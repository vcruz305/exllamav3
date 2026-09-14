from __future__ import annotations
from typing_extensions import override
import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .module import Module
from .linear import Linear
from ..model.config import Config
from ..util.tensor import get_for_device

"""
DeepSeek-V4.1 Engram: hashed n-gram embeddings added to the mHC residual streams before
layers `engram_layer_ids`. Each position hashes the compressed ids of the n = 2..max_ngram
tokens ending there, once per head (`n_hash_cols = (max_ngram - 1) * n_heads` rows of
`head_dim` fp8 values with one e8m0 scale per 32); `wkv` turns the concatenated rows into one
key per residual copy plus one shared value, and a normalized stream-vs-key dot product,
passed through a signed sqrt and a sigmoid, gates the value into every copy:

    streams <- streams + gate[copy] * value

Reference: DeepSeek inference/engram.py (hashing) and inference/model.py Engram (gate).
The tables stay on disk (two DiskTensorHandles per layer); rows are gathered per forward
with a thread pool, so NVMe or a network mount both work. Recurrent state per sequence:
the compressed ids of the previous max_ngram - 1 tokens (DEAD-filled at sequence start).
"""

DEAD = -1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def build_compressed_token_map(tokenizer_json: str, cache_dir: str | None = None):
    """Compressed id per token id (tokens that normalize alike collapse), from tokenizer.json;
    cached as engram_token_map.<sha>.npz in cache_dir (the tokenizer's directory by default)."""
    from tokenizers import Regex, Tokenizer, normalizers
    cache_dir = cache_dir or os.path.dirname(os.path.abspath(tokenizer_json))
    with open(tokenizer_json, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    cache = os.path.join(cache_dir, f"engram_token_map.{digest}.npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        return z["token_map"], int(z["vocab_size"])
    backend = Tokenizer.from_file(tokenizer_json)
    sentinel = ""  # keeps a lone-space token alive through Strip()
    normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(), normalizers.Replace(sentinel, " "),
    ])
    n = backend.get_vocab_size(with_added_tokens = True)
    key_to_new = {}
    lookup = np.zeros(n, dtype = np.int64)
    for token_id in range(n):
        text = backend.decode([token_id], skip_special_tokens = False)
        if "�" in text:
            key = backend.id_to_token(token_id)  # partial UTF-8 byte token: raw form
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    try:
        os.makedirs(cache_dir, exist_ok = True)
        np.savez(cache, token_map = lookup, vocab_size = np.int64(len(key_to_new)))
    except OSError:
        pass
    return lookup, len(key_to_new)


class EngramHasher:
    """Hash ids of the n-grams ending at each position, shared by the engram layers of a model."""

    def __init__(self, config, tokenizer_json: str | None = None, cache_dir: str | None = None):
        self.layer_ids = tuple(config.engram_layer_ids)
        self.max_ngram = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.num_embeddings = tuple(config.engram_num_embeddings)
        self.context_len = self.max_ngram - 1
        tokenizer_json = tokenizer_json or os.path.join(config.directory, "tokenizer.json")
        token_map, vocab_size = build_compressed_token_map(tokenizer_json, cache_dir)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(f"engram: compressed vocab {vocab_size} != config {config.engram_compressed_vocab_size}")
        self.pad_id = int(token_map[config.engram_pad_token_id])
        primes, seen = [], set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram - 1):
                sizes, current = [], config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(sizes)
            primes.append(per_ngram)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        for layer, sizes, rows in zip(self.layer_ids, flat, self.num_embeddings):
            if sum(sizes) != rows:
                raise ValueError(f"engram layer {layer}: primes sum to {sum(sizes)}, table has {rows} rows")
        bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        mults = []
        for layer_id in self.layer_ids:
            g = np.random.default_rng(10007 * layer_id)
            v = g.integers(low = 0, high = bound, size = (self.max_ngram,), dtype = np.int64)
            mults.append(torch.tensor(v * 2 + 1))
        self.primes = torch.tensor(primes, dtype = torch.int64)                      # [layers, max_ngram-1, heads]
        self.offsets = torch.tensor(np.array([np.cumsum([0, *s[:-1]]) for s in flat]))  # [layers, n_cols]
        self.multipliers = torch.stack(mults)                                          # [layers, max_ngram]
        self.token_map = torch.tensor(token_map, dtype = torch.int64)
        self.n_cols = (self.max_ngram - 1) * self.n_heads

    def compress(self, ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        c = self.token_map[ids.to("cpu", torch.int64)]
        if token_mask is not None:
            c = torch.where(token_mask.cpu(), c, torch.full_like(c, DEAD))
        return c

    def hash_window(self, compressed: torch.Tensor, start_pos: int) -> torch.Tensor:
        """compressed: (bsz, context_len + seq) compressed ids (DEAD where no token exists);
        returns (bsz, seq, n_layers, n_cols) row ids."""
        h = self.context_len
        bsz, hl = compressed.shape
        seq = hl - h
        positions = torch.arange(start_pos, start_pos + seq).expand(bsz, seq)
        tokens, blocked = [], torch.zeros_like(positions, dtype = torch.bool)
        for shift in range(self.max_ngram):
            source = compressed[:, h - shift: h - shift + seq]
            blocked = blocked | (positions < shift) | (source == DEAD)
            tokens.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        tokens = torch.stack(tokens, dim = -1)
        products = tokens.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim = -1) + self.offsets


class EngramTable:
    """fp8 rows + e8m0 scales of one engram layer, gathered from disk on demand."""

    def __init__(self, stc, key: str, head_dim: int, threads: int = 32):
        self.weight = stc.get_tensor_handle(f"{key}.weight")
        self.scale = stc.get_tensor_handle(f"{key}.scale")
        assert self.weight.dtype == torch.float8_e4m3fn and list(self.weight.row_shape) == [head_dim], \
            f"{key}.weight: expected fp8 rows of {head_dim}, got {self.weight.dtype} {self.weight.row_shape}"
        assert self.scale.dtype == torch.uint8 and self.scale.num_rows == self.weight.num_rows
        self.block = head_dim // int(self.scale.row_shape[0])
        self.head_dim = head_dim
        self.threads = threads
        self.pool = ThreadPoolExecutor(max_workers = threads) if threads > 1 else None

    def _gather(self, handle, ids: list, nbytes: int) -> bytes:
        fd = handle._ensure_open()
        base = handle.abs_offset

        def one(r):
            b = os.pread(fd, nbytes, base + r * nbytes)
            assert len(b) == nbytes
            return b
        if self.pool is None or len(ids) < 4:
            return b"".join(one(r) for r in ids)
        return b"".join(self.pool.map(one, ids, chunksize = 8))

    def rows(self, hash_ids: torch.Tensor, device) -> torch.Tensor:
        """hash_ids: any shape of row ids -> (*shape, head_dim) fp32 dequantized rows on device."""
        flat = hash_ids.reshape(-1).cpu().to(torch.int64)
        uniq, inverse = torch.unique(flat, return_inverse = True)
        ids = uniq.tolist()
        wb = self._gather(self.weight, ids, self.weight.row_bytes)
        sb = self._gather(self.scale, ids, self.scale.row_bytes)
        w = torch.frombuffer(bytearray(wb), dtype = torch.float8_e4m3fn).view(len(ids), self.head_dim).to(device)
        s = torch.frombuffer(bytearray(sb), dtype = torch.uint8).view(len(ids), -1).to(device)
        vals = w.float().view(len(ids), -1, self.block) * torch.exp2(s.float() - 127.0).unsqueeze(-1)
        vals = vals.view(len(ids), self.head_dim)
        return vals[inverse.to(device)].view(*hash_ids.shape, self.head_dim)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait = False)


class EngramLayerState:
    """Per-slot compressed-id context for the hashing (max_ngram - 1 ids), CPU-resident, with
    right-aligned history columns for rewind like the other recurrent states."""

    def __init__(self, module, max_batch_size: int, max_history: int, cache_id: int):
        self.module = module
        self.ctx = module.hasher.context_len
        self.id_state = torch.empty((max_batch_size, self.ctx + max_history), dtype = torch.long, device = "meta")
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id

    def get_checkpoint_size(self):
        return self.ctx * 8

    def storage_size(self):
        return self.id_state.numel() * 8

    def alloc(self, device):
        self.id_state = torch.full_like(self.id_state, DEAD, device = "cpu")
        self.device = device

    def free(self):
        self.id_state = torch.empty_like(self.id_state, device = "meta")
        self.device = None

    def clear(self, idx: int):
        if self.device is not None:
            self.id_state[idx].fill_(DEAD)

    def get_state_tensors(self):
        return (self.id_state,)

    def rewind(self, slot: int, last_history: int, num_tokens: int):
        assert num_tokens <= last_history
        if last_history > 0:
            p = self.id_state.shape[-1] - num_tokens
            temp = self.id_state[slot, p - self.ctx: p].clone()
            self.id_state[slot, :self.ctx].copy_(temp)

    def stash(self, slot, position: int = 0):
        return (self.id_state[slot, :self.ctx].cpu(),)

    def unstash(self, slot, stashed, position: int = 0):
        self.id_state[slot, :self.ctx].copy_(stashed[0])

    def tp_export(self, plan):
        return {"cls": EngramLayerState, "args": {"cache_id": self.cache_id, "max_history": self.max_history,
                                                  "max_batch_size": self.max_batch_size}}


class EngramLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        table_index: int,
        hasher: EngramHasher,
        hidden_size: int,
        hc_mult: int,
        rms_norm_eps: float,
        qmap: str | None = None,
        gather_threads: int = 32,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config = config, key = key, qmap = None)
        assert layer_idx < 0  # keeps the recurrent-state key distinct from the block at the same index
        self.layer_idx = layer_idx
        self.table_index = table_index
        self.hasher = hasher
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = rms_norm_eps
        self.clamp_value = 1e-6
        self.gather_threads = gather_threads
        self.out_dtype = out_dtype
        self.n_hash_cols = hasher.n_cols
        self.wkv = Linear(
            config = config,
            key = f"{key}.wkv",
            in_features = self.n_hash_cols * hasher.head_dim,
            out_features = hidden_size * (hc_mult + 1),
            qmap = qmap,
            out_dtype = torch.half,
        )
        self.register_submodule(self.wkv)
        self.qk = None
        self.table = None
        self.caps.update({"recurrent_cache": True})
        self.layer_state_cls = EngramLayerState
        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        q = stc.get_tensor(f"{self.key}.q_weight", device, allow_bf16 = True, no_defer = True).float()
        k = stc.get_tensor(f"{self.key}.k_weight", device, allow_bf16 = True, no_defer = True).float()
        self.qk = (q * k).contiguous()                                   # (hc_mult, hidden), only ever used as a product
        self.table = EngramTable(stc, f"{self.key}.embed", self.hasher.head_dim, self.gather_threads)
        for rl in self.recurrent_layers:
            rl.alloc(device)

    @override
    def unload(self):
        for rl in self.recurrent_layers:
            rl.free()
        if self.table is not None:
            self.table.close()
        self.table = None
        self.qk = None
        super().unload()

    @override
    def get_tensors(self):
        return {}

    @override
    def weights_numel(self):
        return self.wkv.weights_numel() + 2 * self.hc_mult * self.hidden_size

    @override
    def optimizer_targets(self):
        return self.wkv.optimizer_targets()

    def _history(self, ids: torch.Tensor, params: dict):
        """(compressed (bsz, ctx + seq), state layer or None, slots or None)."""
        comp = self.hasher.compress(ids)
        rsg = params.get("recurrent_states")
        if rsg:
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            (id_state,) = rsl.get_state_tensors()
            slots = get_for_device(params, "recurrent_slots", "cpu").tolist()
            assert len(slots) == ids.shape[0]
            prev = torch.stack([id_state[s, :self.hasher.context_len] for s in slots])
            return torch.cat((prev, comp), dim = 1), rsl, slots
        assert params.get("position", 0) == 0, "EngramLayer needs recurrent states for forwards past position 0"
        pad = torch.full((ids.shape[0], self.hasher.context_len), DEAD, dtype = torch.long)
        return torch.cat((pad, comp), dim = 1), None, None

    def forward_streams(self, streams: torch.Tensor, hash_ids: torch.Tensor, params: dict) -> torch.Tensor:
        """streams: (bsz, seq, hc_mult, hidden) fp32; hash_ids: (bsz, seq, n_hash_cols). Returns the
        delta to add to the streams."""
        bsz, seq, H, D = streams.shape
        rows = self.table.rows(hash_ids, streams.device)                       # (bsz, seq, cols, head_dim) fp32
        kv = self.wkv.forward(rows.view(bsz, seq, -1).half(), params)          # (bsz, seq, (H+1) * D) fp16
        key = kv[..., :H * D].float().view(bsz, seq, H, D)
        value = kv[..., H * D:].float()
        h = streams.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * self.qk * key).sum(-1) * rstd * (D ** -0.5)               # (bsz, seq, H)
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        return gate.unsqueeze(-1) * value.unsqueeze(-2)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        bsz, seq = x.shape[:2]
        ids = params.get("input_ids")
        if ids is None:
            ids = torch.zeros((bsz, seq), dtype = torch.long)               # measuring forward
        ids = ids.to("cpu", torch.int64)
        pos = int(params.get("position", 0))
        window, rsl, slots = self._history(ids, params)
        hash_ids = self.hasher.hash_window(window, pos)[:, :, self.table_index, :]
        delta = self.forward_streams(x, hash_ids, params)
        if rsl is not None:
            (id_state,) = rsl.get_state_tensors()
            ctx = self.hasher.context_len
            for i, s in enumerate(slots):
                if params.get("recurrent_history", False):
                    w = min(id_state.shape[-1], window.shape[-1])
                    id_state[s, -w:].copy_(window[i, -w:])
                else:
                    id_state[s, :ctx].copy_(window[i, -ctx:])
        return x + delta
