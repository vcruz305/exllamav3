# exllamav3 lightning indexer — exact forward math (port reference for Step-5 CSA)

Scope: the DSA lightning indexer already implemented in exllamav3, in three flavors:

1. **MLA / GLM-5.2 style** (`mla_attn.py` `MLAttention` with `indexer_mode in {"full","shared"}`) —
   raw-token indexer keys. This is the one Step-5's tensors should map onto.
2. **BC (CUDA-graph) mirror of (1)** (`bc_mla.py` + `libtorch/mla_attention.cpp`) — same math,
   fused into the graph-captured decode block.
3. **DeepSeek-V4 CSA** (`dsv4.py` `DSV4Attention` + `bc_dsa.py`) — same scoring kernel, but
   keys are *window-pooled compressor entries*; documented at the end because Step-5's
   `compression_method` may point there.

All line numbers are from branch `feat/step5-mtp-indexer` of `exllamav3-wt`.

---

## 1. Scoring formula

The canonical eager reference (cache-less k-pool parity path), `mla_attn.py:744-746`:

```python
sc = torch.einsum("shd,pd->shp", q_idx[b].float(), pool_keys)          # (S, H_i, P)
sc = F.relu(sc * (D_i ** -0.5))                                        # ReLU with 1/sqrt(D_i) INSIDE
sc = torch.einsum("sh,shp->sp", w[b].float() * (H_i ** -0.5), sc)      # learned head weights, 1/sqrt(H_i)
```

So for query row `t` (batch row `b`, chunk row `r`, `t = q_pos0 + r`) and key/pool entry `s`:

```
I(t, s) = H_i^(-1/2) · Σ_h  w[t,h] · ReLU( D_i^(-1/2) · (q_idx[t,h] · k_idx[s]) )
```

- **ReLU: yes.** `acc += tl.maximum(logits, 0.0) * wh[...]` (`dsa_triton.py:653`,
  `dsa_triton.py:726`). No softmax over the `s` axis anywhere — the raw score goes to top-k.
- **Learned per-head weight `w`: yes**, `w[t,h] = (x_t · W_w)[h]` (see §2). `w` multiplies
  *after* the ReLU and is **not** ReLU'd or softmaxed itself, so it can be negative.
- **1/sqrt(d) scale: yes, two of them.** The fused kernel constant is
  `scale = D_i ** -0.5 * H_i ** -0.5` (`dsa_triton.py:618`, `dsa_triton.py:680`,
  `dsa_triton.py:1126-1127`; compiled the same way at `bc_mla.py:425`,
  `bc_dsa.py:233`, `bc_dsa.py:543`). The kernel applies it once after the ReLU-weighted
  reduction (`acc = tl.sum(...) * scale`, `dsa_triton.py:726`), the eager reference folds
  `D_i^-0.5` inside and `H_i^-0.5` outside (`mla_attn.py:745-746`). These are mathematically
  identical (`ReLU(c·x) = c·ReLU(x)` for `c > 0`); fp rounding differs trivially.
  Kernel docstring: `scores[r, s] = sum_h w[r, h] * relu(q[r, h] . k[s]) * scale`
  (`dsa_triton.py:625`).
- Reduction is fp32 (`acc` is `tl.float32`, `dsa_triton.py:647`; comment
  "runs the relu-weighted reduction in fp32, matching the reference", `mla_attn.py:500-501`);
  scores are stored fp16 (`dsa_triton.py:658`, `dsa_triton.py:730`).
- QSA (Qwen3.8-Flash-Next) is the same formula with uniform head weights:
  "QSA's relu(q.k).sum(heads) * dk**-0.5 is the DSA weighted-relu score with uniform head
  weights" (`qsa_triton.py:5-6`) — i.e. `w_h ≡ 1`, `scale = D_i^-0.5` only.

**Diff vs DeepSeek's published form** `I(t,s) = Σ_h w_{t,h}·ReLU(q_{t,h}·K^s_C)`:
exllamav3 additionally scales by `D_i^-0.5` (inside ReLU) and `H_i^-0.5` (outside). The
activation (ReLU), the learned per-head `w`, and the un-normalized sum over heads match the
paper. **No softmax over scores, no `ssmax`, no per-query score normalization exists in the
implemented path** (see §7).

## 2. Indexer tensors — shapes, computation, norms, biases

All four tensors are per-layer, under checkpoint prefix
`{key}.{key_indexer}` with `key_indexer = "indexer"` by default (`mla_attn.py:110`, `:137`).
`Linear` stores its weight **(in_features, out_features)** after a transposed checkpoint load
(checkpoint `(out, in)` → `weight.T`, `linear.py:198-221`, pad orientation
"in (in_features, out_features)", `linear.py:123`). Bias is loaded only if the checkpoint
provides `{key}.bias` (`linear.py:199`); the indexer projections are used as pure GEMMs —
only `k_norm` carries weight **and** bias (construction below; the BC graph passes weight-only
handles for wk/weights_proj, `bc_mla.py:141-144`, and `_proj_ok` demands `bias is None` for
the quantized path, `bc_mla.py:554-562`). [Inference: GLM-5.2 checkpoints have no
`.bias` for wq_b/wk/weights_proj; nothing in exllamav3 would reject one on the eager path.]

Construction (`mla_attn.py:255-275`):

| tensor | checkpoint key | checkpoint shape | computes | notes |
|---|---|---|---|---|
| `idx_wq_b` | `{key}.{key_indexer}.wq_b` `.weight` | `(H_i·D_i, q_lora_rank)` | `q_idx = W · q_resid` → `(bsz, seqlen, H_i, D_i)` head-major | quantizable (EXL3), sits in the optimizer group (`mla_attn.py:299-305`) |
| `idx_wk` | `{key}.{key_indexer}.wk` `.weight` | `(D_i, hidden_size)` | `k = W · x` → `(bsz, seqlen, D_i)` | **single** key head shared by all `H_i` indexer heads (out_features = `index_head_dim`, `mla_attn.py:265-268`); unquantized fp16 ("router-like … stay unquantized", `mla_attn.py:263-264`) |
| `idx_k_norm` | `{key}.{key_indexer}.k_norm` `.weight`, `.bias` | `(D_i)`, `(D_i)` | **biased LayerNorm** over the `D_i` dims of `k` | eps = `index_norm_eps` (`mla_attn.py:269-271`); "k_norm is a biased LayerNorm applied before the rotation" (`mla_attn.py:472`) |
| `idx_weights` | `{key}.{key_indexer}.weights_proj` `.weight` | `(H_i, hidden_size)` | `w = W · x` → `(bsz, seqlen, H_i)` | input is the **raw module input `x`** (hidden states), not the q latent (`mla_attn.py:502`); unquantized fp16, `pad_to = 1` |

Where the inputs come from:

- `q_resid = q_a_layernorm(q_a_proj(x))` — i.e. **RMSNorm** (`mla_attn.py:227-229`) on the
  q_a latent; `idx_wq_b` projects from `q_lora_rank` ("DSA indexer queries project from the
  q_a latent", `mla_attn.py:256`; eager: `q_idx = self.idx_wq_b.forward(q_resid, …)`,
  `mla_attn.py:498`; BC graph: `exl3_gemm_gr(s.q_a, idx_wq_b->trellis, s.qidx, …)`,
  `mla_attention.cpp:596-598`, where `s.q_a` holds the normed latent, `mla_attention.cpp:354`).
  **No indexer-specific q norm** exists in the MLA path (contrast: QSA has an RMSNorm on q,
  `qsa_triton.py:50-60`; DSV4 has none either).
- `k = idx_k_norm(idx_wk(x))` — full per-token LayerNorm of the raw wk projection, applied to
  **every** `D_i` dim (mean/var over `D_i`, `x̂·w + b`): eager `mla_attn.py:473-474`,
  kernel `_mla_idx_norm_kernel` "Biased LayerNorm for the DSA indexer keys (GLM-5.2 k_norm),
  fp32 math" with `y = xn * w + b` (`mla_triton.py:181-200`); BC passes
  `k_norm_w`/`k_norm_b` (`bc_mla.py:142-143`, launch at `mla_attention.cpp:479-489`).

Norm summary: **q side — only the MLA `q_a_layernorm` (RMSNorm) before `wq_b`; no norm on
`q_idx` itself. k side — biased LayerNorm (weight+bias, eps `index_norm_eps`) on the raw key,
before RoPE. `w` — no norm, no bias, raw linear from `x`.**

## 3. RoPE on the indexer

- **Both q and k get the same RoPE** (same table, same absolute positions):
  `q_idx` roped at `mla_attn.py:499`, keys roped in `_indexer_keys` at `mla_attn.py:477`;
  BC: `rope_gr(s.kidx4, …)` (`mla_attention.cpp:496-498`) and `rope_gr(s.qidx4, …)`
  (`mla_attention.cpp:604-606`).
- **Partial rope over the FIRST `qk_rope_head_dim` dims** of the `index_head_dim`-wide
  vector: `v = x4[..., : self.qk_rope_head_dim]` (`mla_attn.py:455`); BC uses
  `.narrow(3, 0, qk_rope_head_dim)` (`mla_attention.cpp:250`, `:280`). "Only the first
  `qk_rope_head_dim` dims rotate (interleaved pairing, same table as the main attention)"
  (`mla_attn.py:471-472`). GPT-J interleaved pairing, in-place, trailing-slice view
  (`mla_attn.py:446-450`).
- Widths: `qk_rope_head_dim <= index_head_dim` is required (`bc_mla.py:599`);
  `D_i` must be a power of two (`bc_mla.py:599`). With `qk_rope_head_dim == 0` (NoPE,
  GLM-5.3) the rope stage is skipped entirely (`mla_attn.py:452-453`,
  `bc_mla.py:74-79`: "NoPE models (GLM5.3: qk_rope_head_dim 0)").
- The rope instance/table is the **main attention's** (`params["inv_freq"]` override,
  `mla_attn.py:458`; main `self.rope`, `mla_attn.py:326-333`). Same `rope_style`,
  `attn_factor`, `rotate_dims` (`mla_attn.py:462-466`).
- Order on keys: **wk GEMM → LayerNorm → RoPE** ("k_norm is a biased LayerNorm applied
  before the rotation", `mla_attn.py:472`; graph order `hgemm → k_idx_norm → rope_gr →
  plane append`, `mla_attention.cpp:478-510`). On q: **wq_b GEMM → RoPE** (no norm).

## 4. Top-k selection

- **Per query token, not per block, shared by all attention heads**: "Returns -1-padded int32
  indices, (bsz * seqlen, K_pad); selection is per query token, shared by all attention
  heads" (`mla_attn.py:492-493`). The MLA sparse attention then gathers latent rows for all
  `num_q_heads` through the same list (`_attend_sparse`, `mla_attn.py:930-958`).
- **K**: `k_sel = min(index_topk, visible)` (`mla_attn.py:525`, `:505`), output width
  `K_pad = ceil(min(index_topk, t_max)/32)*32` eager (`mla_attn.py:505`) /
  `ceil(index_topk/32)*32` in the graph (`bc_mla.py:346`), -1 padded
  (`dsa_topk.cu:272-275`).
- **Causal**: the scorer writes `-inf` for every entry at/after the per-row causal bound
  `bound[r] = min((q_pos0 + r + 1) // compress_rate, bound_max)`
  (`dsa_triton.py:656-657`, `:728-729`; docstring `dsa_triton.py:1123-1124`).
  `compress_rate = 1` for raw-token keys (MLA), `= index_kpool` for pooled keys
  (`bc_mla.py:424`, `dsa_indexer_scores` default path). The top-k kernel treats `-inf`
  (`KEY_NEG_INF`) as non-selectable (`dsa_topk.cu:92`, `:122-123`, `:136`).
- **No causal mask is ANDed later.** "the indexer's causal bound keeps the selection causal,
  so the kernel needs no mask of its own" (`mla_attn.py:934-936`); "causality lives in the
  selection" (`mla_attention.cpp:677`). The gathered `dsa_attn` runs mask-free over the
  selected rows. The batched top-k variant additionally scans only the row's own causal span
  ("rows only scan their own causal region, so no -inf backfill … is needed",
  `dsa_topk.cu:86-88`).
- **Kernel**: radix-histogram top-k (`dsa_topk.cu:77-275`), exact — per-tile top-k merge is
  exact because any global top-k member is in its own tile's top-k (`mla_attn.py:508-511`).
  Output is the selected **key indices in ascending index order** (ordered compaction,
  `dsa_topk.cu:181-184`; ties at the threshold resolved in ascending global index order,
  `dsa_topk.cu:282-284`), `-1` padded. If fewer than k finite entries exist, all are taken
  (`dsa_topk.cu:136`, `:141-146`).
- **Tiling/slabbing** (eager only): 256-row slabs × `EXL3_DSA_SCORE_TILE` (default 32768) key
  tiles with a running (score, index) top-k merge (`mla_attn.py:507-577`) — exact, changes
  nothing semantically.
- **Dense regime fallback**: while `max(host_seqlens) + seqlen <= index_topk`, no selection
  runs at all — the dense path is used and is "bit-equivalent" (selection would be
  all-inclusive) (`mla_attn.py:795-802`, `bc_mla.py:528-529`).
- **Forced tail**: the plain MLA indexer has **no** forced tail block / window
  ("no window, no sinks", `mla_attn.py:932-933`). The only forced inclusion is the
  **k-pool tail**: with `index_kpool` and `index_kpool_tail = True`, the query's incomplete
  final pool (the last `vis % P` raw tokens, up to `P-1`) is appended as raw token indices
  after the selected pools (`mla_attn.py:641-653`, `_dsa_pool_expand_kernel` TAIL region
  `dsa_triton.py:824-832`, "append the query's incomplete tail pool as raw tokens",
  `dsa_triton.py:806-807`). DSV4 CSA instead force-includes a sliding window
  (`win_len`, `HAS_WINDOW = True`, `bc_dsa.py:265`; "V3.2-on-MLA: off" window,
  `dsa_triton.py:7`).
- Selections are produced once per forward by the nearest preceding "full" layer and reused
  (see §5). In the BC decode block the top-k runs in-graph
  (`dsa_topk_gr(s.scores, s.indices, index_topk, graph)`, `mla_attention.cpp:650-672`).

### k-pool mode (`index_kpool`, GLM-5.3 style) — if Step-5 uses `compression_method`

Keys are pooled `P = index_kpool` tokens at a time; the cached plane holds `[k || gate]`
(`idx_plane_dim = 2*index_head_dim`, `mla_attn.py:157-170`):

- `gate = x · W_gate` (`index_kpool_compress_gate`, shape `(D_i, hidden)`, used via
  `F.linear`, `mla_attn.py:482-483`; transposed to `(hidden, D_i)` for the hgemm,
  `bc_mla.py:134-135`), `ape = index_kpool_compress_ape` `(P, D_i)` learned in-pool
  position embedding (`mla_attn.py:317-324`).
- Pool key = **softmax over pool members of (gate + ape), weighted mean of member keys**
  (`mla_attn.py:604-607`: `probs = softmax(gates + ape, dim=1); pool_keys = (probs*keys).sum(1)`;
  kernel `_dsa_pool_update_kernel`, `dsa_triton.py:747-751` — "Per-dim softmax over the
  present members": the softmax is over the pool-member axis, **per key dimension**, i.e.
  each of the `D_i` dims has its own gate softmax). Partial pools are written but never
  selected ("the causal bound admits only complete pools", `dsa_triton.py:749-751`).
- Scoring/top-k then run over pool entries with `compress_rate = P`
  (`mla_attn.py:676-684`, `bc_mla.py:424`), `k_sel = min(index_topk // P, pools)`
  (`mla_attn.py:671`), and selected pools expand to `P` raw indices each
  (`mla_attn.py:720-725`, `dsa_triton.py:820-822`) + the tail (above).

## 5. `indexer_mode`: "full" vs "shared"

`assert indexer_mode in (None, "full", "shared")` (`mla_attn.py:151-152`). Comment
(`mla_attn.py:148-150`): *"full" layers score and select index_topk tokens per query and
publish the selection; "shared" layers reuse the nearest preceding full layer's selection.
None = plain dense MLA.*

- **full**: owns the whole indexer (`idx_wq_b`, `idx_wk`, `idx_k_norm`, `idx_weights`,
  `mla_attn.py:255-275`), allocates the per-token indexer-key cache plane
  (`idx_plane_dim` non-None only for full, `mla_attn.py:166-170`), appends keys every step
  (even while dense, so history is complete when sparsity activates, `mla_attn.py:845-854`),
  runs scoring + top-k, and publishes `params["dsa_topk_indices"]`
  (`mla_attn.py:873`; BC: `params["dsa_topk_indices"] = self.slot_indices[…]`,
  `bc_mla.py:549-550`, one shared `bcm_dsa_idx` static per device so consumers read it in
  layer order, `bc_mla.py:432-435`).
- **shared**: no indexer tensors, no key plane; reads `params.get("dsa_topk_indices")` and
  asserts it exists (`mla_attn.py:874-878`; BC `set_indexer(mode = 2, wq_b = None, …)`,
  `bc_mla.py:156-162`, `ext_indices = params.get("dsa_topk_indices")`, `bc_mla.py:537-541`).
  Because layers execute in order, "the nearest preceding full layer's selection" is simply
  the last one published. TP constraint: "DSA 'shared' layers must share the device of the
  'full' layer whose selection they reuse" (`mla_attn.py:115-117`, `tp_affinity`).
- Graph-side mode ids: `1 = full, 2 = shared`
  (`mla_attention.cpp:104-106`: "BC_MLAttention: indexer mode must be 1 (full) or 2 (shared)").

## 6. BC path (bc_mla.py / mla_attention.cpp) — stage order per decode step

1. q/latent projections, q_a RMSNorm (`mla_attention.cpp:349-354`).
2. kv staging (kv_a RMSNorm, rope-key split) + partial RoPE on `q_pe`/`k_pe`
   (`mla_attention.cpp:382-406`).
3. W_UK absorb, cache append (`mla_attention.cpp:408-468`).
4. **Indexer keys (full layers, every step, both regimes)**: copy `x` → `x_st`, `kidx = x·W_wk`
   (hgemm, `mla_attention.cpp:478`), biased LayerNorm (`:479-489`), partial rope on the
   `qk_rope_head_dim`-wide head (`:491-499`), append to the paged `kidx` plane (`:500-519`);
   k-pool: gate hgemm + gate append + pool rebuild (`:520-570`).
5. **Sparse regime only** (`t_total > index_topk`, `bc_mla.py:528-529`):
   `qidx = q_a · W_wq_b` (exl3 GEMM, `mla_attention.cpp:596-598`), rope (`:599-607`),
   `wts = x_st · W_w` (hgemm on the **padded** static — zero rows don't affect scores,
   `:608-609`, `bc_mla.py:195-197`), `k_fewq` scoring with
   `scale = Di ** -0.5 * Hi ** -0.5` (`bc_mla.py:420-429`; args `mla_attention.cpp:612-645`),
   top-k (`:647-672`; k-pool: pool top-k + `_dsa_pool_expand_kernel`), gathered
   `dsa_attn_split/combine` over the selected latent rows (`:676-719`).
6. Shared layers: step 5's scoring is skipped; `ext_indices` (the producer's index tensor)
   is patched into the gather (`mla_attention.cpp:679-682`, `:692`).

`wts`/`wq_b` GEMMs take **`x` and `q_a` (normed latent)** respectively — `w` is from raw
hidden `x`, `q_idx` from the q_a latent. That split is the same on both eager and BC paths.

## 7. DeepSeek-V4 CSA flavor (dsv4.py / bc_dsa.py) — differences to be aware of

Same scoring kernel (`_dsa_indexer_fewq_kernel`, `bc_dsa.py:236`, `:546`;
`scale = Di ** -0.5 * Hi ** -0.5`, `bc_dsa.py:233`, `:543`), same `idx_wq_b`
(`{key}.indexer.wq_b`, `q_lora_rank → H_i·D_i`, `dsv4.py:525-535`) and `idx_weights`
(`{key}.indexer.weights_proj`, `hidden → H_i`, "Router-like scoring head: 4096 -> 64 (one
logit per indexer head), unquantized", `dsv4.py:537-546`). Differences:

- **Keys are pooled**, produced by a `DSV4Compressor` named `{key}.indexer.compressor`
  (`dsv4.py:515-523`): `wkv`/`wgate` projections of width `2·index_head_dim` ("the Ca / Cb
  overlapping-window scheme", `dsv4.py:96-97`), pool key
  `comp = (kv * gate.softmax(dim = 2)).sum(dim = 2)` over a window of `compress_rate` tokens
  (`dsv4.py:292`), then **RMSNorm** (`self.norm`, `dsv4.py:134`, `:293`) — not LayerNorm —
  then RoPE. Learned `ape` `(compress_rate, proj_width)` (`dsv4.py:135`, `:270`).
- **RoPE placement is the LAST `rope_head_dim` dims**, not the first:
  `comp.view(...)[..., -self.rope_dim:]` (`dsv4.py:295`), query likewise
  `q_idx[..., -self.rope_head_dim:]` (`dsv4.py:1183`), and the indexer uses the **compress
  rope table** (`inv_freq_compress`, `dsv4.py:1183`: "The indexer query rope uses the compress
  table … CSA layers rope with the compress table").
- Selection is over **pool entries** (`compress_rate` tokens each), `k = min(index_topk, ec)`
  (`dsv4.py:1185-1190`), and attention force-includes the sliding window
  (`win_len = w`, `dsv4.py:1148-1153`; `HAS_WINDOW = True`, `bc_dsa.py:265`).
- `bc_dsa.py` also shows the `m.indexer.wkv/wgate` "x-side fan" mgemm wiring
  (`bc_dsa.py:113-114`, `:405-409`) and the shared `idx_w = m.idx_weights.inner.weight`
  requirement (fp16, `bc_dsa.py:78-84`).

## 8. `ssmax` / `region_block_size` / `provider_group` / `block_compress`

Repo-wide search (`*.py`, `*.cpp`, `*.cu`, `*.h`, `*.md`):

- `region_block_size`: exactly one hit — `exllamav3/architecture/step5_robotics.py:54`
  `self.sparse_region_block_size = sc.get("region_block_size")` (Step-5 config descriptor).
- `ssmax`: exactly one hit — `exllamav3/architecture/step5_robotics.py:64`
  `self.sparse_ssmax_granularity = sc.get("sparse_indexer_ssmax_s_granularity")`.
- `provider_group`, `block_compress`: **zero hits anywhere in the repo.**
- **No softmax-over-s / ssmax-like mechanism exists in the implemented indexer.** The only
  softmax in any indexer path is the pool-member gate softmax (§4 k-pool; §7 DSV4
  compressor). `dsa_indexer_scores` emits raw ReLU-weighted sums
  (`dsa_triton.py:1123-1124`). If Step-5's `sparse_indexer_softmax_variant` /
  `sparse_indexer_ssmax_s_granularity` imply a per-query softmax (or softmax-max) over the
  `s` axis, that is **not implemented** and needs new code.
- The Step-5 descriptor (`step5_robotics.py:50-67`) reads these `text_config.sparse_config`
  keys: `enabled, topk, region_block_size, proxy_dim, sparse_indexer_num_heads,
  sparse_indexer_num_k_heads, sparse_indexer_rope_dim, sparse_indexer_use_rope,
  sparse_indexer_q_norm_type, sparse_indexer_k_norm_type, sparse_indexer_csa_z_norm_type,
  sparse_indexer_softmax_variant, sparse_indexer_ssmax_s_granularity, compression_method,
  attention_impl, apply_to_layer_types`. Note `sparse_indexer_num_k_heads` (exllamav3's MLA
  indexer hard-codes **one** key head, `mla_attn.py:265-268`) and the three
  `*_norm_type` knobs (exllamav3 hard-codes: q = none beyond q_a RMSNorm, k = biased
  LayerNorm) — both are gaps when mapping Step-5 tensors.

## 9. Quick tensor-role cheat sheet (MLA / GLM-5.2 flavor)

| role | tensor | shape (checkpoint) | input | output | norm | rope |
|---|---|---|---|---|---|---|
| indexer query | `…indexer.wq_b.weight` | `(H_i·D_i, q_lora_rank)` | `q_a_layernorm(q_a_proj(x))` | `(…, H_i, D_i)` | q_a RMSNorm only | first `qk_rope_head_dim` dims |
| indexer key | `…indexer.wk.weight` | `(D_i, hidden_size)` | `x` | `(…, D_i)` (1 key head) | biased LayerNorm `…indexer.k_norm.{weight,bias}` | first `qk_rope_head_dim` dims |
| head weights | `…indexer.weights_proj.weight` | `(H_i, hidden_size)` | `x` | `(…, H_i)` | none | none |
| k-pool gate* | `…indexer.index_kpool_compress_gate` | `(D_i, hidden_size)` | `x` | gate logits | none | none |
| k-pool ape* | `…indexer.index_kpool_compress_ape` | `(P, D_i)` | — | in-pool pos. emb | — | — |

\* only when `index_kpool > 0` (GLM-5.3-style compression).

Score: `I(t,s) = (D_i·H_i)^(-1/2) Σ_h w[t,h]·ReLU(q[t,h]·k[s])`, top-k per query token over
causal entries (`-inf` past `min((q_pos0+r+1)//compress_rate, bound_max)`), selection shared
by all attention heads, output ascending-index int32, `-1` padded to a multiple of 32.
