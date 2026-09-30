# 04 — Attention integration points for the CSA + lightning indexer

Scope: how exllamav3 wires a sparse-selection indexer into the attention forward pass and the
KV cache, and what a new architecture must provide. Repo state: `feat/step5-mtp-indexer`
(read-only survey). All line numbers from this tree.

**There are three indexer flavors in the tree.** They share the selection kernel stack
(`dsa_triton.py`: `dsa_indexer_scores`, `ext.dsa_topk`, `_dsa_pool_expand_kernel`) but differ in
module, cache plane and forward hooks:

| flavor | module attr | module class | cache planes | arch examples |
|---|---|---|---|---|
| QSA (Qwen block-topk) | `qsa_indexer` | `QSAIndexer` (`modules/qsa_indexer.py`) | `CacheLayer_qsa{,_quant}` `raw_k`/`pooled` (`cache/qsa.py`) | `architecture/qwen4_exp.py:173` |
| DSA lightning indexer on MLA (token topk) | `indexer_mode`, `index_*` | `MLAttention` (`modules/mla_attn.py`) | `CacheLayer_MLA_*` `k_idx`/`k_pool` (`cache/mla.py`) | `glm5_next.py`, `glm_moe_dsa.py` |
| CSA indexer on DSA pools (DeepSeek-V4) | `indexer` + `index_*` | `DSV4Attention` (`modules/dsv4.py`) | `CacheLayer_dsa` `pool_idx` (`cache/dsa.py`) | `deepseek_v4.py:163-165` |

Step-5 must pick one semantics: QSA = *block* top-k with a mask-shaped selection; MLA/DSA =
*token* top-k index lists gathered by the attention kernel ("lightning indexer"). The task title
(DSV4-style CSA + lightning indexer) matches the MLA/DSA columns.

---

## 1. Module attributes that arm the sparse path (exact gates)

### 1a. `Attention` + `qsa_indexer` (QSA)

- Constructor slot: `qsa_indexer: Module | None = None` (`modules/attn.py:192`), stored and
  registered at `attn.py:213-214` (`self.qsa_indexer = qsa_indexer; self.register_submodule(...)`).
- Eager (non-cached) gate, `attn.py:884-889`:
  ```python
  if self.qsa_indexer is not None:
      assert cu_seqlens is None, "QSA indexer: cu_seqlens batching not supported in nc mode"
      assert causal and position == 0 and positions is None
      if seqlen > self.qsa_indexer.sparse_threshold():
          qsa_q_idx, raw_k = self.qsa_indexer.project(x, self.rope, params)
          qsa_pooled = self.qsa_indexer.pool_keys(raw_k, self.rope, params)
  ```
- Cached gate, `attn.py:1079-1083`: `if self.qsa_indexer is not None:` resolve `qsa_layer` from
  the cache, take `qsa_seqlens_cpu = get_for_device(params, "cache_seqlens", "cpu")`, then
  `qsa_sparse = int(qsa_seqlens_cpu.max().item()) + seqlen > self.qsa_indexer.sparse_threshold()`.
- Cache-plane gate, `cache/qsa.py:22-26`:
  ```python
  idx = attention.qsa_indexer
  assert idx is not None
  self.index_head_dim = idx.head_dim
  self.compress_ratio = idx.compress_ratio
  assert PAGE_SIZE % self.compress_ratio == 0
  ```
- The indexer itself: `qsa_indexer.py:60` `assert kv_heads == 1, "QSAIndexer assumes a single
  raw key head"` (that is the *indexer's* raw key head, not the attention KV heads).
- BC-graph eligibility, `bc_attn.py:601-625` (`_qsa_module_eligible`): reads
  `getattr(m, "qsa_indexer", None)`; if present it must be fully armable — `index_qk_proj`
  exl3-quantized with `.inner.bc`, unpadded width `(idx.n_heads + 1) * idx.head_dim`, both
  layernorms weighted with `constant_bias == 1.0`, `_is_pow2(idx.head_dim)`,
  `PAGE_SIZE % idx.compress_ratio == 0`, and (the gathered sparse kernels support none of these)
  `(m.sliding_window is None or m.sliding_window < 0) and not m.logit_softcapping and
  getattr(m, "sinks", None) is None`. A module **with** an indexer that cannot be armed declines
  the BC path outright ("an unarmed graph would let the planes go stale under dense decode and
  poison later sparse selection", `bc_attn.py:597-600`).

### 1b. `MLAttention` + `indexer_mode` (DSA lightning indexer)

- `mla_attn.py:151-170`:
  ```python
  assert indexer_mode in (None, "full", "shared")
  self.indexer_mode = indexer_mode
  self.index_n_heads = index_n_heads
  self.index_head_dim = index_head_dim
  self.index_topk = index_topk
  self.index_norm_eps = index_norm_eps
  ...
  self.index_kpool = index_kpool
  self.index_kpool_tail = index_kpool_tail
  self.idx_plane_dim = (
      None if indexer_mode != "full" else
      index_head_dim * 2 if index_kpool else index_head_dim
  )
  ```
- "full" layers must additionally own these submodules (`mla_attn.py:255-275`,
  `assert q_lora_rank is not None, "DSA indexer queries project from the q_a latent"`):
  `idx_wq_b` (queries from the q_a latent), `idx_wk` (key head, fp16),
  `idx_k_norm` (biased LayerNorm), `idx_weights` (per-head scoring weights, fp16). kpool adds
  raw tensors `idx_kpool_ape` / `idx_kpool_gate` loaded in `load_local` (`mla_attn.py:317-324`).
- Runtime gates: `mla_attn.py:797-800` — `assert causal`, `assert host_seqlens is not None`,
  `sparse = max(host_seqlens) + seqlen > self.index_topk` (dense below the budget is
  bit-equivalent to the sparse path; see `tests/test_mla_dsa.py:226` "T <= index_topk: the sparse
  machinery must stand down and reproduce dense MLA"). "shared" layers assert a selection is
  present: `mla_attn.py:875-878` `indices = params.get("dsa_topk_indices"); assert indices is not
  None, "shared-indexer DSA layer found no top-k selection in params"`.
- BC eligibility, `bc_mla.py:598-618`: `_is_pow2(m.index_head_dim)`,
  `m.qk_rope_head_dim <= m.index_head_dim`, for kpool `PAGE_SIZE % m.index_kpool == 0 and
  m.index_topk % m.index_kpool == 0`, fp16 `idx_wk`/`idx_weights` weights, biased `idx_k_norm`,
  and `layer.get_idx() is not None`.

### 1c. `DSV4Attention` + `indexer` (CSA)

- `dsv4.py:413-416`: `self.compressor = None; self.indexer = None; self.idx_wq_b = None;
  self.idx_weights = None`; built only `if layer_type == "csa"` (`dsv4.py:514-519`, a
  `DSV4Compressor` at width `index_head_dim` with overlapping-window scheme). Scalars are ctor
  args `index_n_heads / index_head_dim / index_topk / compress_rate` (`dsv4.py:354-358, 388-392`),
  passed from the arch config (`deepseek_v4.py:47-49, 163-165`).
- `bc_dsa.py:55` `self.has_idx = m.indexer is not None`; `bc_dsa.py:66-67` raises
  `"indexer BC missing"` if `m.indexer.bc is None` (the BC companions are mandatory when armed).
  The C++ graph receives `m.index_n_heads, m.index_head_dim, m.index_topk` (`bc_dsa.py:162-164`).

---

## 2. Cache planes and who populates them

### QSA (`cache/qsa.py`)
`QSAPlanes` mixin adds two fp16 side planes to the ordinary K/V layer
(`cache/qsa.py:28-31`):
- `raw_k` `(num_pages, PAGE_SIZE, index_head_dim)` — per-token **raw indexer keys, unnormed and
  unroped** (`qsa_indexer.py:136-137`).
- `pooled` `(num_pages, PAGE_SIZE // compress_ratio, index_head_dim)` — per-block keys: fp32
  mean → k_layernorm → rope at the **block start** position, written once a block completes
  (`qsa_indexer.py:146-162`, `cache/qsa.py:13-18`).

Concrete layers `CacheLayer_qsa` (fp16 KV) and `CacheLayer_qsa_quant` (`cache/qsa.py:62, 86`).
The mapping is requested by the module itself: `Attention.cache_layer_type`
(`attn.py:981-993`) swaps the requested `CacheLayer_fp16/quant` for the planes-carrying variant,
called from `cache/cache.py:149-158`. **Who fills the planes:** `QSAIndexer.update_planes`
(`qsa_indexer.py:487-544`) on every cached forward — fused qk GEMM → stage kernel → rope →
`_mla_plane_update_kernel` raw append → `_qsa_pool_update_kernel` pool (re)build; runs *even in
the dense regime* so history is complete when sparsity engages (`attn.py:1070-1074`). The BC
graph maintains them in-graph instead (`bc_attn.py:240-242`: "the caller must not also run the
eager plane upkeep"), via `bc.set_qsa(... raw_plane = qsa_layer.raw_k.view(-1, idx.head_dim),
pool_plane = qsa_layer.pooled.view(-1, idx.head_dim) ...)` (`bc_attn.py:249-260`).

### MLA / lightning indexer (`cache/mla.py`)
- `k_idx` `(pages, PAGE_SIZE, idx_dim)` — "indexer keys, roped" (`cache/mla.py:71`), where
  `idx_dim = idx_plane_dim = index_head_dim` (or `2 * index_head_dim` under kpool, where each row
  packs `[k || gate_scores]`, `mla_attn.py:479-483`).
- `k_pool` `(pages, PAGE_SIZE // kpool, index_head_dim)` — "pooled indexer keys"
  (`cache/mla.py:72`), i.e. the **`index_kpool` pooled-key variant**: `index_kpool` consecutive
  tokens per entry; pools never straddle pages (`PAGE_SIZE % kpool == 0` asserted in
  `bc_mla.py:602-608`). This is the GLM5.3 k-pool compression of the lightning indexer:
  selection runs over pooled entries (`index_topk // P` pools), then expands back to raw indices.
- Accessors `get_idx() / get_pool()` and appends `update_idx_direct()` (`cache/mla.py:131-139`),
  `update_pool_direct()` (`cache/mla.py:149-154`, pool_seqlens = `cache_seqlens // kpool`).
  Populated by `MLAttention._attend`: `idx_layer.update_idx_direct(...)` at `mla_attn.py:852`
  (unconditionally on the cached path, `mla_attn.py:845-854`) and
  `self._update_pool_plane(...)` at `mla_attn.py:854` (method at `:581-612`; complete pools are
  immutable, written exactly once). BC graph equivalents: `k_plane_append`/`k_gate_append`/
  `k_pool_update` kernels (`bc_mla.py:362-386`), planes passed at `bc_mla.py:130-155`
  (`layer.get_idx()`, `layer.get_pool()`).

### DSA pools (`cache/dsa.py`)
- `pool_idx` `(num_pages, PAGE_SIZE // compress_rate, D_i)` with
  `self.D_i = attention.index_head_dim if attention.layer_type == "csa" else 0`
  (`cache/dsa.py:84`, allocated `:109-110`) — one indexer-key pool entry per compressed window,
  addressed through the job's ordinary block table (`epp = PAGE_SIZE // m`, `cache/dsa.py:77`).
  Populated by the CSA `indexer.forward_fused(...)` window compressor writing into
  `dest_b = pool_idx` (eager NC path `dsv4.py:1137-1143`; cached path via
  `rsl.idx_buf_kv / idx_buf_gate / idx_ovl` rings and the pool scatter in the BC graph,
  `bc_dsa.py:155-158`, `cache/dsa.py:345-349`).

---

## 3. How the selection is applied; the forced tail block

**Production form: a per-query-row index list consumed by a gathered attention kernel** — not a
materialized mask and not a pre-gather of KV:
- Selection = `dsa_indexer_scores` (relu(q·k) summed over index heads with per-head weights,
  causal bounds in-kernel) → `ext.dsa_topk` → (kpool only) `_dsa_pool_expand_kernel` expanding
  pool ids to raw indices. Output: `(bsz * seqlen, K_pad) int32, -1-padded`
  (`qsa_indexer.py:366-390` `select_indices`, `mla_attn.py:487-578` `_indexer_topk`).
- Application = gathered GQA kernels that read K/V through the block table only at selected
  positions: QSA `qsa_sparse_attend_rows` (`qsa_indexer.py:674-712` `sparse_attend`: "nothing
  S x L is ever materialized"); MLA `_attend_sparse` (`mla_attn.py:930-939`: "the indexer's
  causal bound keeps the selection causal, so the kernel needs no mask of its own"); DSV4
  `dsa_attn(indices = indices, k_len = k_len, ...)` over `[sliding ring ++ selected/dense pool
  entries]` (`dsv4.py:1148-1156`). BC decode: fewq score → topk → `_dsa_pool_expand_kernel` →
  `_dsa_attn_split/_qsa_sparse_split` + combine, all in one captured graph
  (`bc_dsa.py:215-236`, `bc_attn.py:473-549`).
- Eager reference form (QSA only, kept for parity tests): the selection is **ANDed into the
  causal mask** — `qsa_indexer.py:20-22` "The selection is ANDed into the causal mask of the main
  attention", implemented in `token_mask` (`qsa_indexer.py:204-238`:
  `mask |= token_sel & (kv_pos <= abs_pos)`), consumed via `build_mask` (`:448-465`).

**Forced tail block:** each query row always keeps, in addition to its top-k selected blocks/
pools, the **incomplete block/pool covering its most recent tokens** — the row's partial tail
is never subject to top-k and can never be excluded:
- QSA: "each query keeps the top token_budget / compress_ratio blocks plus, **always, the
  incomplete tail block**" (`qsa_indexer.py:20-22`); mask form `:217-219`
  (`kv_pos >= nb_q * cr` — everything from the tail block start onward is admitted);
  kernel form `TAIL = 1` in `_dsa_pool_expand_kernel` (`qsa_indexer.py:362`,
  `bc_attn.py:505`), whose docstring is "Expand selected pools to raw token indices ... and
  append the query's incomplete tail pool as raw tokens" (`dsa_triton.py:806-808`, tail region
  `:824-830`).
- MLA kpool: the same idea as an option — `index_kpool_tail` (checkpoint key
  `index_kpool_always_select_tail`, `glm5_next.py:77`) appends up to `P-1` raw tail tokens
  (`mla_attn.py:641-653` `append_tail`; `TAIL = 1 if m.index_kpool_tail else 0` at
  `bc_mla.py:444`).

**Regime switch (dense ⇄ sparse).** Sparse engages only once top-k could actually exclude
something; below that the dense path is exact and is used instead:
- QSA: `sparse_threshold() = 4 * self.block_topk + 3` — "Highest query position for which dense
  attention is still exact + 1" (`qsa_indexer.py:473-475`).
- MLA: `sparse = max(host_seqlens) + seqlen > self.index_topk` (`mla_attn.py:800`);
  BC regime `regime = 1 if t_total > self.index_topk else 0` (`bc_mla.py:529`).
- DSV4: `if T > self.index_topk: indices, k_len = self._indexer_topk(...)` (`dsv4.py:1144-1146`);
  BC regime `bc_dsa.py:316`. Sparse BC slots are decode-only (QSA: single-job, causal —
  `bc_attn.py:572-577` returns `None` for `bsz > 1 or not causal`, falling back to eager).

**Cross-layer sharing.** A "full" layer publishes its selection into the forward params for
"shared" layers: `params["dsa_topk_indices"] = indices` (`mla_attn.py:873`,
`bc_mla.py:549-550`), consumed at `mla_attn.py:875-878` / `bc_mla.py:537-541`. In the BC graph
the `indices` static is deliberately shared across layers ("layers run in order on one stream, so
a full layer's selection is in place when its shared consumers gather through it",
`bc_mla.py:432-434`).

---

## 4. Step-5: minimal change for the 23 `full_attention` layers

Today `step3_5.py` builds plain `Attention` for `full_attention` layers (`step3_5.py:138-170`)
and `SlidingAttention` for `sliding_attention` (`:172-203`); layer kinds come from
`config.layer_types` (`:60, :123`). Findings:

- **`SlidingAttention` is a standalone `Module`** (`sliding_attn.py:243`), not an `Attention`
  subclass: no `qsa_indexer` slot, no `cache_layer_type`, no indexer hooks anywhere in the file.
  This matches every existing arch: window layers never carry an indexer (DSV4 sliding layers,
  MLA "shared" layers), and `build_bc_swa` explicitly refuses indexed modules
  (`bc_attn.py:687`: `getattr(m, "qsa_indexer", None) is None`). Leave the sliding layers alone.
- **`Attention` already carries the whole QSA integration surface** — constructor slot
  (`attn.py:192`), cache-layer mapping (`attn.py:981-993`), eager + cached forward branches
  (`attn.py:884-922, 1079-1131`), BC arming (`bc_attn.py:243-260`), TP placement
  (`attn.py:1199-1206` `max_devices = 1`, export/import `attn.py:1278, 1347-1357`), autosplit
  measurement (`attn.py:996-1045`). So: **extend `Attention`, do not build a new module** —
  two sub-cases:
  1. If Step-5's indexer is QSA-shaped (block top-k, per-block pooled keys): construct a
     `QSAIndexer` per full layer and pass `qsa_indexer=...` exactly like
     `qwen4_exp.py:173-185` does. Nothing else changes; `cache_layer_type` upgrades the cache
     to `CacheLayer_qsa*` automatically. Constraints: checkpoint keys
     `{key}.self_attn.indexer.{index_qk_proj,q_layernorm,k_layernorm}`, `kv_heads == 1`
     (`qsa_indexer.py:60`), `PAGE_SIZE % compress_ratio == 0`.
  2. If it is the DeepSeek lightning indexer (token top-k `index_topk`, per-head `weights_proj`,
     which is what the port title says): the QSA block machinery is the wrong shape. The
     reusable code is the **MLA/DSA selection stack**, which is nearly projection-agnostic:
     `mla_attn._indexer_topk/_indexer_topk_kpool` (`mla_attn.py:487-763`) need only `q_resid`
     (from `idx_wq_b`) and `x` (for `idx_weights`); on plain `Attention` the query source can be
     `x` directly. Port that + `_attend_sparse`-style gathered attention (`dsa_attn` /
     `qsa_sparse_attend_rows`) into `Attention` behind the same optional submodule slot, and add
     a `k_idx`-style plane via `cache_layer_type` (the `QSAPlanes` mixin pattern,
     `cache/qsa.py:11-59`, is the template; MLA's `k_idx/k_pool`, `cache/mla.py:51-72`, is the
     shape). If the per-layer indexer tables exist only on some layers, copy the
     `"full"/"shared"` + `params["dsa_topk_indices"]` pattern (`mla_attn.py:148-155, 873-878`)
     so a later full layer's selection can serve neighbors.
- BC/graph decode is opt-in by shape and declines safely (`build_bc_attn` returns `None`), so
  the eager path is the correctness baseline — get it right first, per the MLA test pattern
  (`tests/test_mla_dsa.py`: dense equivalence at `T <= index_topk`, selection parity against the
  torch reference).
- TP: an indexed layer runs **whole on one rank** (`attn.py:1199-1206`,
  `attn.py:1349` "QSA attention layers run whole on one device"; same for MLA, `bc_mla.py:621`).

---

## 5. Existing knobs / flags gating the path

Env vars:
- `EXL3_BC_DSA=0` — disable the DSV4 whole-step graph decode (eager fallback);
  `EXL3_BC_DSA_DEBUG=1` — raise instead of declining on build failure (`bc_dsa.py:25-26`).
- `EXL3_BC_ATTN=0` — disable BC graph decode for Attention **and** MLA (shared flag,
  `bc_attn.py:42`, `bc_mla.py:7-23`); `EXL3_BC_ATTN_TRACE=1` prints build/decline per layer
  (`bc_attn.py:44-50`).
- `EXL3_BC_MLA=0` (`mla_attn.py:27`).
- `EXL3_QSA_SCORE_TILE` (default 8192, `qsa_indexer.py:253`) and `EXL3_DSA_SCORE_TILE`
  (default 32768, `mla_attn.py:36`) — selection scoring tile width.
- `EXL3_DSA_DEBUG_BOUNDS=1` (`dsa_triton.py:38`), `EXL3_DSA_QC_STAGE*` (`dsa_triton.py:842-844`,
  packed-pool prefill staging).
- `EXL3_AUTOSPLIT_WORSTCASE=0` skips the indexer-aware VRAM measurement
  (`attn.py:997`).

Caps / model-level: MLA declares `caps["kv_cache"] = True` (`mla_attn.py:277-279`); DSV4 declares
`caps["recurrent_states"] = True` + `recurrent_state_cls = DSV4State` (`deepseek_v4.py:270-279`);
a Step-5 port with paged planes needs only `kv_cache` (planes ride the ordinary cache layer).

Config / construction asserts: `deepseek_v4.py:40`
`assert self.num_kv_heads == 1, "DeepseekV4: expected shared-KV MQA"`;
`glm5_next.py:79-80` `assert all(t in ("full", "shared") for t in self.indexer_types)`;
`cache/dsa.py:75` `assert m and PAGE_SIZE % m == 0`; `cache/qsa.py:26`
`assert PAGE_SIZE % self.compress_ratio == 0`; `cache/mla.py:40`
`assert max_num_tokens % PAGE_SIZE == 0`; `bc_mla.py:602-608` (kpool divisibility) and
`bc_attn.py:622-625` (QSA geometry / no window-softcap-sinks).

Reference harnesses: `tests/test_mla_dsa.py` (dense⇔sparse equivalence, topk parity),
`tests/test_tp_export_attention.py:76-94` (TP placement with `qsa_indexer`),
`tests/bench_dsv4_attn_sweep.py:102-104` (reads `index_n_heads / index_head_dim / index_topk`).

---

## Required integration points, in order (checklist)

1. Arch config: read `index_n_heads`, `index_head_dim`, `index_topk` (+ `compress_rate` /
   `index_kpool`, `index_kpool_tail`) — pattern `deepseek_v4.py:47-49`, `glm5_next.py:74-80`.
2. Attention module: an optional indexer submodule slot on `Attention`
   (`qsa_indexer`-style, `attn.py:192, 213-214`) carrying `indexer_mode/index_*` attrs
   (`mla_attn.py:151-170`); gate asserts `indexer_mode in (None,"full","shared")`,
   `PAGE_SIZE % compress == 0`.
3. Indexer weights: `idx_wq_b` (queries), `idx_wk`, `idx_k_norm`, `idx_weights`
   (+ `idx_kpool_gate/ape` for pooled keys) — `mla_attn.py:255-275, 317-324`;
   or QSA's single `index_qk_proj` + two RMSNorms (`qsa_indexer.py:78-88`).
4. Cache planes: `cache_layer_type` override (`attn.py:981-993`) mapping the K/V layer to a
   planes-carrying variant (`cache/qsa.py:21-31`, `cache/mla.py:51-72`): per-token raw key
   plane `k_idx/raw_k` (+ `k_pool/pooled` for `index_kpool`), fp16, page-aligned.
5. Plane upkeep on **every** cached forward, dense regime included:
   `update_idx_direct` / `update_pool_direct` (`mla_attn.py:845-854`, `cache/mla.py:131-154`)
   or `QSAIndexer.update_planes` (`qsa_indexer.py:487-544`).
6. Forward hook: compute roped indexer queries + per-head weights, run
   `dsa_indexer_scores` + `ext.dsa_topk` (+ `_dsa_pool_expand_kernel` with `TAIL=1` forced tail)
   when `max(host_seqlens) + seqlen > index_topk` — `mla_attn.py:797-800, 856-882`.
7. Sparse apply: gathered attention over the `-1`-padded int32 index list
   (`qsa_triton.qsa_sparse_attend_rows` / `dsa_attn`), causal by construction; publish
   `params["dsa_topk_indices"]` for shared layers (`mla_attn.py:873-878`).
8. SlidingAttention: no indexer (match `bc_attn.py:687`); leave `SlidingAttention` untouched.
9. BC decode (optional): arm via `bc.set_qsa` / `bc.set_indexer` + `configure_slot_qsa/_dsa`
   (`bc_attn.py:243-260, 541-549`, `bc_mla.py:129-164`); must decline cleanly when unarmed.
10. TP + VRAM: whole-layer placement `max_devices = 1` (`attn.py:1199-1206`), indexer in
    `tp_export/tp_import` (`attn.py:1278, 1347-1357`), planes counted in `storage_size` +
    autosplit measurement (`attn.py:996-1045`).
