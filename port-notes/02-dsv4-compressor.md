# 02 — CSA block compressor (`dsv4_compress`) and the Step-5 `sparse_indexer_z` mapping

Scope: how exllamav3's fused DSV4 compressor pools a run of `m` tokens into one compressed
KV entry, and how Step-5-Preview's single `sparse_indexer_z.weight` projection maps onto it.

All paths relative to the repo root unless absolute. Code quotes are `file:line`.
Everything under "Inference" is NOT confirmed by code and must be validated against the
reference (the Step-5 modeling code is gated — see PORT_SPEC §3).

---

## 1. Pooling math

Two per-token projections produce two row tensors of equal width `W` (exllamav3/modules/dsv4.py:120-133):

- **`kv`** (`{key}.wkv`: hidden → `W`) — the *values* being pooled.
- **`gate`** (`{key}.wgate`: hidden → `W`) — the *logits* of the pooling weights.

`W = 2 * head_dim` when `overlapping`, else `W = head_dim` (dsv4.py:120). `gate` is the
gating signal, `kv` the content; both are linear projections of the same hidden state.

The pool is a **per-column softmax-gated average over the window entries**, fp32. Torch
reference (tests/test_dsv4_compress_kernel.py:41):

```python
comp = (kv * gate.softmax(dim = 1)).sum(dim = 1)   # dim 1 = window-entry axis
```

with `kv`/`gate` viewed `(nw, m_or_2m, W)` (test:30-31) — the softmax runs **independently
for every one of the `W` columns** over the `m` (or `2m`) entries of the window. The same
math stateless in `DSV4Compressor.forward` (dsv4.py:292 `comp = (kv * gate.softmax(dim = 2)).sum(dim = 2)`),
and online-softmaxed per column in the CUDA kernel
(exllamav3/exllamav3_ext/dsv4_compress.cu:117-168; header: "Gate softmax runs per COLUMN
over the window entries, fp32, matching (kv * gate.softmax(dim = 2)).sum(dim = 2)" —
dsv4_compress.cu:31-32).

**Learned positional bias ('ape')**: yes — `ape` is a learned `(m, W)` fp32 tensor added to
the **gate rows only**, indexed by the token's within-window slot (`abs_pos % m`):

- test:31 `gate = gate_rows[:nw * m].float().view(nw, m, W) + ape.unsqueeze(0)`
- dsv4.py:270 `gate = gate[:, :usable].view(...) + self.ape` (self.ape shape `(compress_rate, proj_width)` fp32, dsv4.py:135, loaded at dsv4.py:629)
- kernel dsv4_compress.cu:159 `gv += ape[(abs_pos % m) * W + sc];`

The saved overlap snapshot stores the gate **with the ape bias already applied**
(dsv4.py:288-289 comment "the saved gate slice already carries the position bias (ape); it
is not re-added when restored"; kernel dsv4_compress.cu:234 `snap[(m + e) * hd + c] = gv + ape[...]`).

Window addressing (kernel header, dsv4_compress.cu:20-24): window `w` emits pool entry
`ec0 + w` (`ec0 = pos0 / m`) from source rows at absolute positions
`[(ec0 + w) * m, (ec0 + w + 1) * m)`; row lookup is `ring row = abs_pos % buf_rows`,
`buf_rows = PAGE_SIZE + m` (PAGE_SIZE = 256, exllamav3/constants.py:2).

## 2. Norm and rope

**Norm = weighted RMSNorm over the pooled `hd` columns** (no mean subtraction):

```
comp = comp * rsqrt(mean(comp^2) + eps) * norm_w          # test:42
```

Kernel equivalent (dsv4_compress.cu:184-185):

```c
float rmr = rsqrtf(sh_red[0] / (float) hd + eps);
float normed = comp * rmr * (active ? __half2float(norm_w[c]) : 0.0f);
```

`norm_w` is the RMSNorm weight `(hd,)` (dsv4.py:134 `RMSNorm(cfg, f"{key}.norm", attn.rms_norm_eps)`,
cast to fp16 for the fused path at dsv4.py:157-159). The torch path calls
`self.norm.forward(comp.half(), params)` (dsv4.py:293). Yes — this is plain RMSNorm with a
learnable per-column gain.

**Rope**: applied AFTER the norm, only to the **trailing `rd` columns** of the pooled entry
(`rd = 2 * len(inv_freq) = rope_head_dim`; test:43-49, dsv4.py:122,295):

- slice: `comp[..., -self.rope_dim:]` (dsv4.py:295), kernel `c0 = hd - rd` (dsv4_compress.cu:192)
- style: **GPT-J interleaved pairs** — `e, o = x[0::2], x[1::2]` → `(e·cos − o·sin, o·cos + e·sin)`
  (test:48-49; kernel dsv4_compress.cu:199-201)
- position: the **window's first-token position** `wpos = (entry_index + entry_count) * m`
  (test:44 `wpos = torch.arange(nw) * m`; stateful dsv4.py:260 `fwp = state.entry_count * m`,
  294 `wpos = ... * m + fwp`; kernel dsv4_compress.cu:196 `theta = inv_freq[p] * (float)((ec0 + w) * m)`)
- inv_freq comes from the **compress rope table** (`yarn_inv_freq(rope_head_dim, compress_rope_theta=160000, rope_scaling)`,
  dsv4.py:618-619, 630). Leading `hd - rd` columns are left unroped (the "nope" part).

## 3. `overlapping`, `m`, and the shape variants

**`m` = `compress_rate`** = tokens per compressed entry; window stride is exactly `m`, so
`nw = seq // m` windows per chunk (test:26, kernel dsv4_compress.cu:109 `nw = (pos0 + seq) / m - ec0`).

**`overlapping`** = the CSA/indexer "Ca/Cb" scheme. Projected rows carry **two halves of
`hd` columns**: columns `[0, hd)` are the "Ca" slice (used by the NEXT window's first half),
columns `[hd, 2hd)` the "Cb" slice (this window's second half). Each pooled entry then
mixes `2m` source entries (test:32-40, dsv4.py:274-291, kernel dsv4_compress.cu:126-146):

- entries `0..m-1`: previous window's rows at `[(w−1)·m, w·m)`, **Ca columns** (`sc = c`)
- entries `m..2m−1`: this window's rows at `[w·m, (w+1)·m)`, **Cb columns** (`sc = hd + c`)

So consecutive entries share `m` source tokens — receptive field `2m`, stride `m`. For the
first window of the first chunk, the Ca half is masked out (`ec0 == 0` → skipped,
dsv4_compress.cu:129). Cross-chunk, the last window's Ca slice is carried in a **snapshot
ring** `ovl` of shape `(depth, 2, m, hd)` fp32 (kv slice + gate slice with ape),
dsv4_compress.cu:215-236; dsv4.py:283-290. Non-overlapping mode needs no snapshot
(`ovl = None`; dsv4_compress.cu:417 "overlapping mode requires snapshot ring").

Variants (test matrix tests/test_dsv4_compress_kernel.py:118-130):

| variant | hd | W | m | overlapping | source of dims |
|---|---|---|---|---|---|
| CSA (main compressor) | 512 | 1024 = 2·hd | **4** | yes | `compress_rate_csa = 4` (architecture/deepseek_v4.py:60), `_RATIO_TO_TYPE = {0: "sliding", 4: "csa", 128: "hca"}` (deepseek_v4.py:19) |
| HCA | 512 | 512 = hd | **128** | no | `compress_rate_hca = 128` (deepseek_v4.py:61) |
| indexer-shaped ("idx") | 128 | 256 = 2·hd | 4 | yes | `index_head_dim = 128` (deepseek_v4.py:48); built as `DSV4Compressor(..., index_head_dim, compress_rate, overlapping = True)` (dsv4.py:515-523) |

The kernel infers the mode from the width alone: `bool overlap = W == 2 * hd` with
`TORCH_CHECK(overlap || W == hd)` (dsv4_compress.cu:413-414). Output entry is split-stored:
columns `[0, Wa)` → `dest_a` (pool_c, nope), `[Wa, hd)` → `dest_b` (pool_r, rope), or all to
`dest_a` when `dest_b == nullptr` (pool_idx) — dsv4_compress.cu:35-36, 204-210; pool layout
`pool_c/pool_r/pool_idx` in exllamav3/cache/dsa.py:42-54 (`epp = PAGE_SIZE // m`, dsa.py:77).

## 4. Mapping Step-5's `sparse_indexer_z` onto the pipeline

### What is CONFIRMED

- **Pipeline contract** (from the code above): per-token `kv_rows` + `gate_rows`, both
  `(seq, W)` fp16, `W ∈ {hd, 2·hd}`; learned `ape (m, W)` fp32 added to the gate; per-column
  softmax pooling over `m` (or `2m`) entries; weighted RMSNorm over `hd`; GPT-J rope on the
  trailing `rd` cols at `w·m`. Entry `w` summarizes source rows `[w·m, (w+1)·m)`.
- **Step-5 tensors/config** (checkpoint + `text_config.sparse_config`, mirrored in
  exllamav3/architecture/step5_robotics.py:50-73): per full-attention layer there is exactly
  ONE compression projection `sparse_indexer_z.weight [256, 4096]` (bf16, proxy_dim 256 ×
  hidden 4096), `compression_method: "csa_block_compress"`, `region_block_size: 8`,
  `sparse_indexer_csa_z_norm_type: "none"`, `sparse_indexer_rope_dim: 32`,
  `sparse_indexer_use_rope: true`, `topk: 512` (PORT_SPEC §3). Sibling tensors
  `sparse_indexer_{q,k,w}` + `q/k_norm` + `ssmax_s` are scoring-path, not compression:
  `q [4096, 4096]` = 16 heads × 256, `k [256, 4096]` = single 256-dim key head
  (`sparse_indexer_num_k_heads: 1`, step5_robotics.py:57).
- **There is no gate tensor and no `ape` tensor** in the Step-5 checkpoint (PORT_SPEC §3
  tensor table lists all `sparse_indexer_*` tensors). DSV4 needs both (dsv4.py:126-135,629).
- **"none" norm cannot be faked with `norm_w = ones`**: the kernel unconditionally applies
  `rsqrt(mean-square + eps)` (dsv4_compress.cu:184-185), which rescales the output no matter
  what `norm_w` is. A true `none` requires either a kernel bypass flag or the torch path
  with the norm stage skipped. This is a hard gap in the fused path.
- `m = 8` is legal for the paged pool: `assert m and PAGE_SIZE % m == 0` (dsa.py:75), 256 % 8
  == 0, `epp = 32`.

### What is INFERRED (proposals, strongest first)

**I1 — dims: `hd = W = proxy_dim = 256`, non-overlapping.** The pooled entry must be
score-compatible with the indexer query/key space, which is 256-wide (`q` per-head 256, `k`
256). So the compressed block key is 256-dim → `hd = 256`; z outputs 256 cols → `W = 256 =
hd` → the kernel's non-overlapping branch (dsv4_compress.cu:413-414). An overlapping mapping
would need z of width 512 (for hd 256) or would yield 128-dim entries (if hd = 128), neither
matching the proxy space. STRONG inference.

**I2 — `m = region_block_size = 8`, non-overlapping blocks.** "csa_block_compress" reads as
"compress each block of `region_block_size` tokens into one entry" — the block IS the
compression unit (DSV4's CSA instead uses overlapping `m = 4` windows). Entry `w` =
pool over tokens `[8w, 8w+8)`. Competing reading: `m = 4` with `region_block_size` used only
for selection granularity (PORT_SPEC §3's "candidate filtering (2,048 blocks × 8)"). MEDIUM
inference — resolve against the reference before wiring.

**I3 — gate: degenerate (uniform) → mean pooling.** With no gate tensor, the softmax gate
collapses: feed `gate_rows = 0` → per-column softmax = uniform `1/m` → `comp = mean(z rows of
the block)`. Alternative (I3b): self-gated `gate_rows = kv_rows = z(x)` (softmax-gated pooling
with logits from z itself). Both are expressible in the existing kernel without change
(both `kv_new` and `gate_new` are just input tensors). I3 is preferred: it makes z the ONLY
learned compressor tensor, consistent with `z_norm_type: "none"` (no learned gain anywhere on
the compression path) and with a deterministic block summary for selection. I3b is the main
competing hypothesis; pick by equivalence testing. INFERRED.

**I4 — norm: skip entirely.** Both placements of `sparse_indexer_csa_z_norm_type` (pre-pool
row norm on z, or post-pool output norm like DSV4's compressor norm) collapse to the same
implementation for `"none"`: apply no norm. Implementation: add an `apply_norm`/norm-type
switch to `dsv4_compress` (windows kernel lines 170-186), or pool in torch. REQUIRED CHANGE
to the fused path (see CONFIRMED gap above). If a future checkpoint says `"rmsnorm"`, the
existing stage is exactly right.

**I5 — rope: trailing 32 cols at block position.** `use_rope: true`, `rope_dim: 32` → pass
`inv_freq` of length 16 (rd = 32 ≤ hd, satisfies dsv4_compress.cu:416) and the kernel ropes
the trailing 32 columns of the pooled key at `w * m` out of the box (dsv4_compress.cu:190-202).
Whether Step-5 ropes the *compressed* key (vs. only per-token q/k) is unconfirmed. MEDIUM
inference.

**I6 — `ape = zeros((m, W))` fp32.** The kernel requires an `ape` tensor (`const at::Tensor&`
non-optional, dsv4_compress.cuh:36; shape check dsv4_compress.cu:405). Step-5 has no ape
tensor → pass zeros. CONFIRMED at the API level (a tensor must be passed), INFERRED that
zeros is semantically correct.

### Integration sketch (caller side)

```python
# per full-attention layer, indexer-shaped block compressor:
#   hd = 256 (proxy_dim), W = 256, m = 8 (region_block_size), overlapping = False
z_rows  = sparse_indexer_z(x)                    # (seq, 256) fp16
gate    = torch.zeros_like(z_rows)               # I3: uniform gate -> mean pool
ext.dsv4_compress(
    z_rows, gate,
    ring_kv, ring_gate,                          # (PAGE_SIZE + 8, 256) fp16 each
    None,                                        # ovl: none, non-overlapping
    torch.zeros((8, 256), dtype = torch.float),  # I6: no ape in checkpoint
    norm_w, eps,                                 # I4: norm BYPASS needed (kernel change)
    inv_freq16,                                  # I5: 32-dim rope table (len 16)
    dest, None,                                  # dest = pooled keys (cap, 256) fp16
    position, None, 8,
    None, pool_bt, pool_epp, False)
```

State/ring handling mirrors the existing CSA indexer path (`rsl.idx_buf_kv/idx_buf_gate`,
`idx_ovl = None`; cf. dsv4.py:1698-1702). For the stateful torch path, `DSV4Compressor` can
be constructed with `wkv = z Linear`, `wgate = zeros Linear`-equivalent, `norm = Identity`
substitute, `compress_rate = 8`, `overlapping = False` — but note the class hardwires
`RMSNorm` as default (dsv4.py:134) and the fused builder copies `self.norm.weight`
(dsv4.py:157), so the "none" path needs an explicit bypass there too.

## 5. Public API

### Fused kernel binding (what the cached path calls)

`ext.dsv4_compress(kv_new, gate_new, ring_kv, ring_gate, ovl, ape, norm_w, rms_norm_eps,
inv_freq, dest_a, dest_b, position, position_tensor, m, slot_ids, pool_bt, pool_epp,
stage_rel)` — signature exllamav3/exllamav3_ext/dsv4_compress.cuh:33-53; Python call sites
tests/test_dsv4_compress_kernel.py:73-75 and dsv4.py:241-245.

| arg | shape | dtype | notes |
|---|---|---|---|
| `kv_new`, `gate_new` | `(seq, W)` or batched `(B·seq, W)` | fp16 | projected rows of this chunk (dsv4_compress.cu:394-395) |
| `ring_kv`, `ring_gate` | `(buf_rows, W)` or `(slots, buf_rows, W)` | fp16 | row-addressed `abs % buf_rows`, `buf_rows = PAGE_SIZE + m` |
| `ovl` | `(depth, 2, m, hd)` or `(slots, depth, 2, m, hd)`, or None | fp32 | overlapping only (dsv4_compress.cu:417) |
| `ape` | `(m, W)` | fp32 | required (dsv4_compress.cu:398, 405) |
| `norm_w` | `(hd,)` | fp16 | RMSNorm weight (dsv4_compress.cu:399, 415) |
| `rms_norm_eps` | float | — | |
| `inv_freq` | `(rd/2,)` | fp32 | `rd` even, `rd ≤ hd ≤ 1024` (dsv4_compress.cu:416) |
| `dest_a`, `dest_b` | `(cap, Wa)`, `(cap, hd−Wa)` or `dest_b=None` | fp16 | `dest_b=None` → all `hd` cols to `dest_a` (dsv4_compress.cu:207-210) |
| `position` | int | — | absolute pos of row 0 of the chunk |
| `position_tensor` | `(B,)` int32 or None | int32 | device position override (graph replay); enables padded grid (dsv4_compress.cu:38-40) |
| `m` | int | — | tokens per entry |
| `slot_ids` | `(B,)` int32 or None | int32 | batched mode: state slot per job (requires `position_tensor`, dsv4_compress.cu:440) |
| `pool_bt` | `(B, npr)` int32 or None | int32 | paged pools: block-table row per job; entry row remapped `bt[row/epp]*epp + row%epp` (dsv4_compress.cu:205-206) |
| `pool_epp` | int | — | entries per page = `PAGE_SIZE // m` (required if `pool_bt`) |
| `stage_rel` | bool | — | `dest_a` is per-job staging rows `[0, nw)` for packed-pool quantization (dsv4_compress.cu:78-80) |

Mode is derived: `overlap = (W == 2·hd)` (dsv4_compress.cu:413). All tensors: one CUDA
device (`OptionalCUDAGuard`, dsv4_compress.cu:391), raw-pointer indexing → contiguous
required. Two launches: windows kernel (grid `(nw_or_padded, batch)`, one block per window,
one thread per column) + ring-store kernel (dsv4_compress.cu:475, 500).
Also exported: `dsv4_ring_append(kv, ring, pos, ring_beg, slot_ids)` (dsv4_compress.cuh:55-72).

### C++ bound companion (projection + pool in one transition)

`BC_DSV4Compressor` (exllamav3/exllamav3_ext/libtorch/dsv4_compressor.h:17-115; pybind in
dsv4_compressor_bc.h:1-38): holds `wkv`/`wgate` as `BC_LinearEXL3` or `BC_LinearFP16`
(exactly one of each pair), `ape`, `norm_w`, `rms_norm_eps`, `inv_freq`, `m`, fixed scratch,
and an optional 2-expert `exl3_mgemm` for the paired wkv+wgate projection
(dsv4_compressor.cpp:35-58). `.run(x, ring_kv, ring_gate, ovl, dest_a, dest_b, position,
position_tensor, mg_c, pool_bt, pool_epp, stage_rel)` with `x: (seq, hidden) fp16`,
`seq ≤ MAX_QLEN = 32` (dsv4_compressor.h:19; check dsv4_compressor.cpp:33). Graph-capturable
via `run_gr` + device position tensor.

### Python wrapper

- `DSV4Compressor.forward(x, params, inv_freq, state: DSV4CompressorState | None)` —
  dsv4.py:248-297. `x (bsz, seq, hidden)` fp16 → `(bsz, nw, head_dim)` fp16, roped at
  `(w + entry_count) * m`. Stateless: complete windows only, remainder discarded. Stateful:
  sub-window remainder buffered, Ca overlap carried, `entry_count` advanced
  (`DSV4CompressorState` contract: `entry_count / get_buffer / store_rows / advance_entries /
  get_overlap / set_overlap`, dsv4.py:42-90).
- `DSV4Compressor.forward_fused(x, params, buf_kv, buf_gate, ovl, dest_a, dest_b, position,
  pool_bt, pool_epp, stage_rel)` — dsv4.py:222-245; bsz 1 cached path; delegates to
  `BC_DSV4Compressor.run` when `seq ≤ BC_MAX_QLEN = 32` (dsv4.py:104, 233-237), else two
  `Linear.forward` + `ext.dsv4_compress` (dsv4.py:239-245).
- Device assumptions: CUDA throughout, fp16 projections (EXL3-quantized or fp16 weights both
  supported, dsv4.py:149-153, 182-188), `ape`/`inv_freq` fp32, outputs fp16. Padded
  projections disable the fused build (python fallback, dsv4.py:163-166).
