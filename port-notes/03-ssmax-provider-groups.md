# 03 — `ssmax`, `num_provider_groups`, `region_block_size`, `sparse_gqa`

Research-only note (web + checkpoint-tensor measurement). No code changed. Complements
`01-mla-indexer-math.md` §8 (repo-side: these strings appear nowhere in exllamav3 except the
Step-5 config descriptor) and `02-dsv4-compressor.md` (the `csa_block_compress` / `z` mapping).

All claims marked **measured** were produced for this note by byte-range-fetching safetensors
headers/payloads from `rene98c/Step-5-Preview-BF16` (mirror of the gated `stepfun-ai` repo;
`model.safetensors.index.json` + `model-00001/2/12/23.safetensors` headers). Claims from the web
carry their URL. Where nothing public exists, it says so.

Per-layer tensors on the 23 full-attention layers (layers 3,7,…,91), **measured** shapes:

| tensor | shape | dtype |
|---|---|---|
| `self_attn.sparse_indexer_q.weight` | [4096, 4096] | BF16 (16 heads × 256 proxy) |
| `self_attn.sparse_indexer_q_norm.weight` | [256] | F32 (RMSNorm, per-head 256) |
| `self_attn.sparse_indexer_k.weight` | [256, 4096] | BF16 (single k head — matches `num_k_heads: 1`) |
| `self_attn.sparse_indexer_k_norm.{weight,bias}` | [256] | F32 (LayerNorm) |
| `self_attn.sparse_indexer_w.weight` | [16, 4096] | F32 (16 per-head score weights, DSA-`w^I`-style) |
| `self_attn.sparse_indexer_z.weight` | [256, 4096] | BF16 (CSA compression gate, see 02) |
| `self_attn.ssmax_s` | [64] | F32 (**64 = main q-heads**, not 16 indexer heads) |

There is **no tensor per provider group / per region** — the only per-group-shaped knob is the
config int itself. That constrains all interpretations below.

---

## 1. `sparse_indexer_softmax_variant: "ssmax"` + `ssmax_s_granularity: "q_head"`

**Best-supported reading: SSMax = Scalable-Softmax (Nakanishi 2025), with a learnable scale
parameter `s` per query head, stored as `self_attn.ssmax_s` — and the tensor belongs to the MAIN
attention softmax, not the 16-head indexer.** Confidence: **high** on the identity "ssmax =
Scalable-Softmax"; **medium** on where exactly it multiplies.

Evidence:

- SSMax is a named softmax replacement published Jan 2025:
  `SSMax(z)_i = n^(s·z_i) / Σ_j n^(s·z_j) = softmax over (s·log(n)·z_i)`, `s` a learnable
  scalar ("scaling parameter", typically per attention head), `n` = softmax input size.
  arXiv:2501.19399, https://arxiv.org/html/2501.19399v1 — the name match is exact and it is
  explicitly pitched at attention softmax with a learned per-head `s`. A 2026 follow-up uses
  exactly this combination in a sparse-attention retriever ("BlockSearch-SSMax … with s
  initialized to 0.43 at every attention layer", https://arxiv.org/html/2607.01538v1), so
  ssmax-inside-a-sparse-indexer is a known pattern, not a stretch. No other expansion of
  "ssmax" (stable/sum-of/…-softmax-max) exists in any attention literature found.
- Placement: the tensor is `self_attn.ssmax_s`, NOT `self_attn.sparse_indexer_*` (every other
  sparse tensor carries the `sparse_indexer_` prefix), and its shape **[64] = num main q-heads**,
  while the indexer has 16 heads. `"q_head"` granularity therefore resolves against the 64 main
  heads. The config key's `sparse_indexer_` prefix looks like config-namespace grouping, not a
  claim about the indexer math. (Counter-evidence kept honest: the key literally says
  `sparse_indexer_softmax_variant`, and `01` §8 notes no softmax exists in any implemented
  indexer path — an ssmax-in-the-indexer reading cannot be fully excluded.)
- The GGUF conversion side reached the same placement conclusion: dropping `*.ssmax_s` makes the
  23 layers "correct output but … different long-context behaviour", and the scale is compared
  against `1/sqrt(192)` — i.e. it is read as a main-attention logit scale replacing the standard
  inverse-sqrt head dim. https://huggingface.co/SHSLab/Step-5-Preview-GGUF and
  https://huggingface.co/vcruz305/StepFun-5-Preview-GGUF ("its constant softmax scale (0.08496
  against llama.cpp's `1/sqrt(192)` = 0.07217)"). No public runtime implements it
  (llama.cpp drops the tensors; vLLM/SGLang have no `step3p5v` sparse path).

**Measured values (new):** `ssmax_s` = **0.08495759963989258** (f32) — **bit-identical across all
64 heads, and identical on layers 3, 7, 47, 91**. So despite `q_head` granularity the checkpoint
carries one frozen constant; per-head shape is structural, not learned variation. This means the
parameter was almost certainly written as a constant (formula of config values or a training-side
fixed scalar), not trained per head.

**Open question for the implementer (flagged):** does the runtime softmax use
`scale = ssmax_s[head]` directly (a fixed logit temperature ≈ 1.177 × 1/√192), or the full SSMax
`scale = ssmax_s[head] · log(n)` with `n` = softmax length (e.g. the number of selected keys)?
The weights cannot distinguish these — with `s` constant, `s·log(n)` is "constant" at fixed `n`,
which is exactly how the GGUF-side note describes it. Under SSMax the `log(n)` factor is the whole
point (it is what fixes attention fading), so the pure-constant reading is only safe for ≤512
contexts. Numerology for the constant is inconclusive: 0.0849576 is not an obvious closed form of
the config values (1/0.0849576² = 138.55; 0.0849576·√192 = 1.1772 ≈ √(2·ln 2) = 1.1774 but not
equal). Validate against the reference runtime before committing to either formula.

## 2. `num_provider_groups: 4`

**Best-supported reading: the 4 GQA KV groups (== `num_attention_groups: 4`), each of which
"provides" its own KV stream and gets its own top-k selection; selection is shared by the 16
q-heads inside each group.** Confidence: **low-medium** — the numeric identity is strong, the
mechanism is inference. **No public source uses the term "provider group" anywhere** (web +
repo-wide search; only hits are the config.json mirrors). Flagged as unresolved nomenclature.

Evidence for the reading:

- `num_provider_groups: 4` equals `num_attention_groups` / `num_attention_groups_full_attention`
  = 4 exactly (measured in config; k/v proj [768, 4096] = 4×192 confirmed in `05` §4), and
  `attention_impl: "sparse_gqa"` says the sparse path is expressed in GQA terms. A KV group is the
  natural "provider" of keys/values in GQA.
- Per-GQA-group selection with per-group top-k is exactly what the nearest published designs do:
  MiniMax Sparse Attention "independently selects a Top-k subset for each GQA group" and its index
  branch keeps "a dedicated index query head [per group] and shares a single index key head across
  all groups" — the latter matches `sparse_indexer_num_k_heads: 1` precisely
  (https://www.alphaxiv.org/abs/2606.13392); HySparse aggregates scores within each query group so
  "all heads in the same group share identical sparse indices"
  (https://arxiv.org/html/2602.03560v1); TensorRT-LLM's sparse GQA path takes "a precomputed token
  list for each KV head … query heads in the same KV group share that list"
  (https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html).
- Consistency check with tensors: `sparse_indexer_w.weight [16, 4096]` (16 query-dependent head
  weights, DSA-`w^I`-shaped) could produce 4 per-group scores by summing the weighted head scores
  in groups of 4 (16 heads / 4 groups). That is the only grouping consistent with the existing
  tensors — there are no per-group weights, so any per-group split must partition the 16 heads
  (or reuse one shared score and vary only the budget).

Alternatives considered and weaker: splitting the top-512 budget across 4 quota groups (128 each)
— no mechanism visible in tensors; grouping sequence regions — conflicts with `region_block_size`
already covering that; multi-query "providers" as separate selection layers (e.g. IndexShare
provider layers) — 23 sparse layers don't divide by 4.

**Open (flagged):** exact partition of the 16 indexer heads onto 4 groups, and whether `topk: 512`
is per group (4×512) or global (4×128). Nothing public answers this.

## 3. `region_block_size: 8` (vs DeepSeek-V4 CSA `compress_rate` m=4)

**Best-supported reading: a "region" is a block of 8 consecutive tokens, and selection/compression
operates at region granularity — i.e. Step-5's CSA-style block compression merges 8 tokens per
unit where DeepSeek-V4's CSA compresses 4.** Confidence: **medium-high** for "8-token block
granularity"; **low** for the top-k unit (512 tokens vs 512 regions).

Evidence:

- `compression_method: "csa_block_compress"` + `sparse_indexer_z.weight` (the per-dimension
  compression gate of DeepSeek-V4 CSA, see `02`) + `region_block_size: 8` together say: block
  compression over 8-token blocks. DeepSeek-V4's CSA is the template and uses `compress_rate_csa
  m=4` with overlapping windows before the indexer selects top blocks
  (https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4); Step-5 substitutes 8 and
  drops the second branch (`z` is one [256, 4096] tensor, V4 has Z^a/Z^b).
- StepFun's own launch text: "Sparse GQA with **block-wise token merging** … cuts indexer and
  top-k selection costs to approximately **one-eighth** of a denser baseline" and "merges heavily
  overlapping top-k selections of neighboring tokens"
  (https://huggingface.co/TypeSafeAI/Step-5-Preview-BF16, https://pandaily.com/stepfun-step-5-preview-600b-moe-1m-context).
  The one-eighth figure is exactly `region_block_size: 8` (one scoring/selection unit per 8
  tokens), and the merge of neighboring selections confirms the region is a local 8-token run.
- The phrase "top-k selection over compressed KV blocks" from the GGUF conversion notes
  (https://huggingface.co/SHSLab/Step-5-Preview-GGUF) says selection runs over the compressed
  blocks, not raw tokens.

**Open (flagged):** whether `topk: 512` counts tokens (= 64 regions of 8) or compressed region
entries (= 512 regions ≈ 4096 raw tokens). Third-party write-ups read the config as "keeps the
top 512 keys" (https://muhammad-ahmed.com/blog/ai-brief-a-terabyte-of-proprietary-weights/), but
none of them saw the code. Also open: whether the 8-token merge is key-side (compressed proxy
entries, cf. `02`) and/or query-side (8 neighboring queries share one selection — the launch text's
"neighboring tokens" phrasing leans query-side). Both readings put the boundary at multiples of 8.

## 4. `attention_impl: "sparse_gqa"` — selection per q-head or per kv-group?

**Best-supported reading: selection is per KV group (4 groups), shared by the 16 q-heads of each
group — not per q-head.** Confidence: **medium**.

Evidence:

- Per-KV-group sharing is the standard sparse-GQA contract in every implementation found:
  MiniMax MSA (per-GQA-group top-k, https://www.alphaxiv.org/abs/2606.13392), HySparse ("all
  heads in the same group share identical sparse indices", https://arxiv.org/html/2602.03560v1),
  TensorRT-LLM ("query heads in the same KV group share that list",
  https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html).
- Per-q-head selection (64 lists) would need per-q-head scores; the indexer produces 16 head
  scores (`sparse_indexer_q.weight` 16×256) with DSA-style learned head weights `w` that reduce
  heads to a scalar — DeepSeek's own DSA shares ONE selection across all heads
  (https://www.emergentmind.com/topics/deepseek-sparse-attention-dsa). Nothing in the tensors
  scales to 64.
- The counterweight is DSA-style global sharing (one selection for all 64 heads); `sparse_gqa`
  + `num_provider_groups: 4` is the only evidence pushing away from it, so "4 groups" vs "1
  shared" is the real open axis, not "64 lists".

**Open (flagged):** whether all 4 groups share one selection (DSA-like, `num_provider_groups`
meaning something else) or each group selects independently. Deciding this needs the reference
runtime; the checkpoint alone can't settle it (no per-group tensors to prove or disprove).

---

## Summary of confidence

| # | Question | Reading | Confidence |
|---|---|---|---|
| 1 | `ssmax` | Scalable-Softmax (arXiv 2501.19399); `ssmax_s` per main q-head, constant 0.0849576, acts on the main sparse-attention logits | high (identity) / medium (placement) |
| 2 | `ssmax` formula | `s` vs `s·log(n)` multiplier on logits | **unresolved — flag** |
| 3 | `num_provider_groups` | the 4 GQA KV groups, each with its own selection | low-medium (no public source; name unattested) |
| 4 | group partition / topk unit | 16 indexer heads → 4 groups; topk per-group vs global | **unresolved — flag** |
| 5 | `region_block_size` | 8-token region: block-merge/compress unit (Step-5's analogue of DSV4 CSA m=4); "1/8 cost" matches | medium-high |
| 6 | topk unit | 512 tokens vs 512 regions | **unresolved — flag** |
| 7 | `sparse_gqa` | per-KV-group selection shared by that group's q-heads (not per q-head) | medium |
