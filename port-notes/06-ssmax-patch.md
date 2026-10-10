# 06 — `ssmax_s` integration patch plan (Scalable-Softmax logit scale)

Companion to `03-ssmax-provider-groups.md` §1 (semantics research) and
`05-step5-tensor-ground-truth.md` §1/§4 (tensor shapes). Deliverable of this note:
a reviewable, line-anchored plan for wiring `self_attn.ssmax_s` into exllamav3's main
attention softmax scale. **No existing file has been edited**; the only code shipped
alongside this note is the new self-contained module
`exllamav3/modules/ssmax.py` (both candidate formulas, switchable).

> **UNVERIFIED — flagged open item.** The runtime formula is `s` vs `s * log(n)`
> (see 03 §1). `ssmax.py` defaults to the paper-faithful `s * log(n)` and exposes
> `variant="constant"` as a one-line switch. Nothing here should be considered
> numerically validated against the StepFun reference. Also flagged: what exactly
> `n` counts per path (§5 below).

---

## 1. Where the softmax scale lives today (verified line anchors)

The scale is a **single scalar per Attention module**, `self.sm_scale`, produced in
one place and fanned out to every kernel:

| file:line | what it is |
|---|---|
| `exllamav3/modules/attn.py:161` | `sm_scale: float | None = None` ctor parameter |
| **`exllamav3/modules/attn.py:210`** | **`self.sm_scale = sm_scale or self.head_dim ** (-0.5)`** — the only place `1/sqrt(head_dim)` is materialized for the main attention. **This is the primary anchor.** |
| `exllamav3/modules/attn.py:931` | no-cache/prefill dispatch: `attn_dispatch(..., sm_scale = self.sm_scale, ...)` |
| `exllamav3/modules/attn.py:1146` | cached dispatch: `attn_dispatch(..., sm_scale = self.sm_scale, ...)` |
| `exllamav3/modules/qsa_indexer.py:444` | sparse prefill (`sparse_attend_nc`): `qsa_sparse_attend_rows(..., indices, attn.sm_scale)` |
| `exllamav3/modules/qsa_indexer.py:708` | sparse decode (`sparse_attend`): `qsa_sparse_attend_rows(..., indices, attn.sm_scale, ...)` |
| `exllamav3/modules/attention_fn/common.py:17` | `AttnArgs.sm_scale: float` — scalar in the dispatch NamedTuple |
| `exllamav3/modules/attention_fn/torch.py:47` | SDPA `scale = args.sm_scale` |
| `exllamav3/modules/attention_fn/xformers.py:73, 81` | `scale = args.sm_scale` |
| `exllamav3/modules/attention_fn/triton_paged.py:331, 399–400, 453` | paged wrapper #1: `softmax_scale: float | None`, default `1.0 / math.sqrt(head_dim)`, passed `float(softmax_scale)` |
| `exllamav3/modules/attention_fn/triton_paged.py:479, 543–544, 601` | paged wrapper #2 (same pattern) |
| `exllamav3/modules/attention_fn/triton_paged.py:115, 233, 943` | Triton kernels: `scores = tl.dot(q_tile, k_tile) * scale` — `scale` is a **scalar constexpr** |
| `exllamav3/modules/attention_fn/bc_attn.py:128` | BC graph capture copies the scalar: `self.sm_scale = module.sm_scale` |
| `exllamav3/modules/attention_fn/bc_attn.py:302, 307` | `"scale"` is a **constexpr** baked at capture: `scale = float(self.sm_scale)` (kernels are the same triton_paged kernels compiled AOT — bc_attn.py:15–16) |
| `exllamav3/modules/attention_fn/qsa_triton.py:224` | gathered sparse kernel: `scores = tl.dot(q_tile, k_tile) * scale` |
| `exllamav3/modules/attention_fn/qsa_triton.py:267, 318` | `sm_scale: float` arg → `scale = float(sm_scale)` constexpr |
| `exllamav3/modules/attn.py:1254` | TP export carries `"sm_scale": self.sm_scale` |

**Not in scope:** `exllamav3/modules/qsa_indexer.py:70` (`self.scale = 1.0 /
math.sqrt(head_dim)` for the *indexer scoring* relu-sum) and the DSA/QSA scorer
kernels' `scale=` arguments (qsa_indexer.py:316, 321). Per 03 §1 the placement
evidence says `ssmax_s` acts on the **main** attention logits, not the indexer
scores; the `sparse_indexer_softmax_variant` config key name is the only
counter-evidence and is treated as config-namespace grouping (03 §1). Do not
touch the indexer scale until the reference runtime settles this.

## 2. The new module

`exllamav3/modules/ssmax.py` (new, self-contained, torch-free at import):

- `ssmax_logit_scale(s, softmax_len, variant)` — scalar formula:
  - `variant="log_n"` (DEFAULT, paper-faithful): **`scale = s * log(n)`**
  - `variant="constant"`: `scale = s`
  (`log` is the natural log, per arXiv 2501.19399: `n^(s·z) = e^(s·log(n)·z)`.)
- `ssmax_per_head_scale(num_heads, head_dim, ssmax_s, softmax_len, variant)` —
  returns the `[num_heads]` scale tensor from the checkpoint parameter
  (Step-5: `num_heads=64`, `head_dim=192`, `ssmax_s` = `[64]` F32, softmax_len = `n`).
  `head_dim` is validation-only: when SSMax is active it **replaces**
  `1/sqrt(head_dim)`, never multiplies it. (For Step-5 the ratio is
  0.08496 / 0.07217 ≈ 1.177 before any `log(n)`.)
- `fold_ssmax_into_q(q, per_head_scale)` — applies the scale by folding into q
  (see §3).

Validated by `py_compile` + a torch-free unit run (scalar/sequence/fake-tensor
paths, error cases, fold broadcast shapes all pass on this machine; real-tensor
behavior needs a torch box).

## 3. Integration strategy A (RECOMMENDED): fold the per-head scale into q

Math identity — for head `h` with scalar scale `s_h`:

```
logits_h = s_h * (q_h · k) = (s_h * q_h) · k
```

so a per-head (or per-head-per-row) scale can be applied by multiplying q after
RoPE/QK-norm and before the attention call. **No kernel, dispatcher, or
`AttnArgs` change is needed**; SDPA, xformers, triton_paged and the QSA gathered
kernels all keep receiving the scalar `sm_scale`. Softcap (if ever used) applies
to the already-scaled logits, so folding stays correct.

### 3.1 `attn.py` changes (the only file that MUST change)

1. **`attn.py:210`** — keep the line as the fallback, but neutralize it when SSMax
   is on. Add ctor/attribute plumbing next to `key_sinks` (`attn.py:168, 229–230`):

   ```python
   # attn.py ~210 (after existing line)
   self.ssmax_s = None            # [num_q_heads] F32, set at load time; None = off
   self.ssmax_variant = "log_n"   # "log_n" | "constant" — UNVERIFIED (port-notes/03 §1)
   ```

2. **Loading** — mirror the `sinks` pattern at `attn.py:442–444`
   (`self.config.stc.get_tensor(f"{self.key}.{self.key_sinks}", device, no_defer = True)`):

   ```python
   # attn.py, model-side key: "model.layers.N.self_attn.ssmax_s"
   # (plain parameter, NOT ".weight" — see 05 §1)
   self.ssmax_s = self.config.stc.get_tensor(f"{self.key}.ssmax_s", device, no_defer = True)
   if self.ssmax_s is not None:
       assert self.ssmax_s.numel() == self.num_q_heads   # [64], q_head granularity
       self.sm_scale = 1.0   # ALL scaling now rides the q-fold; every scalar
                             # fan-out site (below) then passes 1.0 harmlessly
   ```

   Setting `self.sm_scale = 1.0` at load makes **every** existing call site
   (`attn.py:931`, `attn.py:1146`, `qsa_indexer.py:444`, `qsa_indexer.py:708`,
   `bc_attn.py:307`) correct with zero edits, because the fold already happened.

3. **The fold itself** — in both forwards, immediately after the RoPE block
   (`attn.py:901–913` prefill / `attn.py:1109–1121` cached) and before the
   `qsa_q_idx` / `qsa_sparse` / `attn_dispatch` branches (`attn.py:921`,
   `attn.py:1129`):

   ```python
   from .ssmax import ssmax_per_head_scale, fold_ssmax_into_q
   if self.ssmax_s is not None:
       n = <softmax_len for this call — see §5, FLAGGED>
       scale = ssmax_per_head_scale(
           self.num_q_heads, self.head_dim, self.ssmax_s, n,
           variant = self.ssmax_variant,
       )
       q = fold_ssmax_into_q(q, scale)   # q: (bsz, seq, num_q_heads, head_dim) here
   ```

   Placement after RoPE is required (RoPE is head-orthogonal and commutes with a
   per-head scalar, but folding after it keeps q_norm/rope numerics untouched and
   matches "scale the logits").

4. **TP export** — `attn.py:1246–1264`: add `"ssmax_s"`/`"ssmax_variant"` to the
   exported kwargs and slice `ssmax_s` to the worker's head shard alongside the
   existing head splits (per-head tensor ⇒ TP-sharded like q heads).

5. **BC decode path (`attn.py:951` `bc_attn_step`)** — the CUDA-graph path does
   projections→attention inside the captured graph (bc_attn.py:13–19), so the
   Python-side fold does NOT run there, and `scale` is a frozen constexpr
   (`bc_attn.py:307`). Interim fix with existing machinery: have the BC arming
   check decline modules with `self.ssmax_s is not None` (there is precedent —
   bc_attn.py:33–35 "declines the BC path outright"), which falls back to the
   regular python path where §3.1.3 applies. Permanent fix = Strategy B.

## 4. Integration strategy B (uniform, invasive): per-head scale inside the kernels

Only needed if the BC graph path must keep ssmax layers, or if per-row `n` must be
computed inside the kernel. Replace the scalar `scale` constexpr with a runtime
`*fp32` per-head (or per-row-per-head) pointer and change exactly one line per
kernel:

- `triton_paged.py:115`, `:233`, `:943` — `scores = tl.dot(q_tile, k_tile) * scale`
  → `* scale_h` with `scale_h = tl.load(scale_ptr + q_head)` (q_head is already
  a program axis).
- `qsa_triton.py:224` — same edit.
- `bc_attn.py:302, 307` — `"scale"` moves from the constexpr dict to the runtime
  `sig` dict (`"scale": "*fp32"`), pointer patched per replay like
  `block_table`/`cache_seqlens` (bc_attn.py:17–18).
- Wrapper plumbing: `triton_paged.py:331/399–400/453` and `:479/543–544/601`
  accept `softmax_scale: torch.Tensor | None`; `common.py:17` `AttnArgs.sm_scale`
  becomes `float | torch.Tensor`; the SDPA/xformers fallbacks (torch.py:47,
  xformers.py:73, 81) **cannot** take per-head scale — SDPA's `scale=` is scalar,
  so those paths must either fold into q (Strategy A) or use `attn_bias`.

Recommendation: land Strategy A first (one file, reviewable, all Python paths +
QSA sparse correct), keep BC declined for ssmax layers, and only do Strategy B if
decode-through-BC performance demands it.

## 5. `softmax_len` = n — per-path definition (FLAGGED, unverified)

The SSMax paper defines `n` as the softmax input size. For Step-5 the honest
candidate definitions, per path:

| path | candidate n | source of the count |
|---|---|---|
| decode, q_len = 1 | keys in cache incl. current = `cache_seqlens[b]` (or +1) | `cache_seqlens` (attn.py:1144) |
| prefill, causal | per-row: `row + 1` keys attended (⇒ per-row scale, fold shape `[bsz, seq, heads]`) | `cu_seqlens`/`max_seqlen` (attn.py:928–929) |
| sparse (top-512) | keys actually selected per row (`K_pad` valid count), OR full context length | `select_indices` output (qsa_indexer.py:439) |
| sliding layers | **n/a — `ssmax_s` exists only on the 23 full_attention layers (3,7,…,91)** | 05 §2 |

The weights cannot distinguish these (03 §1). Under `"constant"` variant the
question evaporates. Until the reference runtime answers it, pick one, isolate it
behind a single `softmax_len_for(...)` helper so it is a one-line fix, and log the
choice. Suggested default: **decode n = cache_seqlens, prefill n = per-row causal
length** (the literal "softmax input size" reading).

## 6. What the patch must NOT change

- `qsa_indexer.py:70` indexer scoring scale (see §1 last paragraph).
- The `1/sqrt(head_dim)` defaults in `triton_paged.py:400/544` — they remain the
  fallback for non-ssmax modules; `attn.py:210` keeps serving them.
- Sliding-attention layers (no `ssmax_s` tensor) and MTP layers 92–94.

## 7. Verification checklist (needs a torch/CUDA box)

1. Loader test: `ssmax_s` loads `[64]` F32 on layers 3,7,…,91 only; `sm_scale`
   flips to 1.0 there; other layers keep `1/sqrt(192)`.
2. Equivalence test (constant variant): compare against a reference that scales
   logits explicitly (`softmax(s * q·k)`) — folded and unfolded must match to
   fp16 tolerance on all dispatch paths (torch SDPA, triton_paged, qsa gathered).
3. Variant smoke: `log_n` at n = 512 vs 1024 must produce visibly different
   entropy in attention distributions (this is the paper's point: larger n →
   sharper attention), and `log_n` at fixed n equals `constant` up to the `log(n)`
   factor.
4. Long-context probe (the GGUF note's observed failure mode): with and without
   `ssmax_s` on a >512-token continuation — "correct output but different
   long-context behaviour" is exactly what dropping the tensor causes.
5. Only after matching the StepFun reference numerics: flip the default variant /
   `n` semantics and delete the UNVERIFIED banner.
