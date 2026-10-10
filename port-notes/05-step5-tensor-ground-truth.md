# Step-5-Preview — Tensor Ground Truth (from safetensors headers)

Source: https://huggingface.co/rene98c/Step-5-Preview-BF16 (public mirror, byte-identical BF16 source).
Method: every `.safetensors` shard was probed with HTTP `Range` requests; only the 8-byte header
length prefix + JSON header were downloaded (no tensor payloads). Headers were parsed for **all**
26 files in the repo (model-00001..00024, model-mtp3-full-00001, model-mtp3-00001,
model-vit-00001, vit.safetensors). Every claim below is read from those headers; the index
`model.safetensors.index.json` (2449 entries) was cross-checked name-for-name against the parsed
headers — **2449/2449 names matched, 0 shard-placement mismatches**.

Dtype notation in this doc: `BF16` = `BF16`, `F32` = `F32` (safetensors dtype strings).

---

## 1. Sparse-indexer tensor families — exact shapes and dtypes

All 8 families are **uniform across every layer that carries them** (verified on all 23 layers,
sampled in full at layers 3, 11, 47, 91):

| tensor (suffix under `model.layers.N.self_attn.`) | shape | dtype | notes |
|---|---|---|---|
| `sparse_indexer_q.weight`        | `[4096, 4096]` | BF16 | full hidden→hidden projection |
| `sparse_indexer_q_norm.weight`   | `[256]`        | F32  | RMSNorm over proxy_dim=256 (config: q_norm_type=rmsnorm) |
| `sparse_indexer_k.weight`        | `[256, 4096]`  | BF16 | hidden→proxy_dim |
| `sparse_indexer_k_norm.weight`   | `[256]`        | F32  | LayerNorm over 256 (config: k_norm_type=layernorm → weight+bias) |
| `sparse_indexer_k_norm.bias`     | `[256]`        | F32  | |
| `sparse_indexer_w.weight`        | `[16, 4096]`   | **F32** | per-indexer-head scalars; 16 = sparse_indexer_num_heads |
| `sparse_indexer_z.weight`        | `[256, 4096]`  | BF16 | hidden→proxy_dim (csa_z_norm_type=none → no z_norm tensors) |
| `ssmax_s`                        | `[64]`         | F32  | **not a `.weight`** — plain parameter; 64 = num_attention_heads (ssmax_s_granularity=q_head) |

That is the complete sparse-related inventory: exactly 8 tensors per sparse layer, 8×23 = 184
tensors total. There are **no** other sparse/indexer/ssmax tensors anywhere in the checkpoint
(searched all shard headers).

Key dimensional anchors:
- `proxy_dim = 256` (q_norm/k_norm/z dims, k/z out features)
- `sparse_indexer_num_heads = 16` (w rows), `sparse_indexer_num_k_heads = 1`
- `sparse_indexer_rope_dim = 32` (no tensors — runtime-computed)
- `ssmax_s` is per-q-head (64 = `num_attention_heads`), F32.

## 2. Which layers carry the indexer

**Exactly 23 layers: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75,
79, 83, 87, 91** (i.e. every 4th layer, `4k+3`, for k=0..22).

Cross-check against `config.text_config.layer_types` (length **95** — 92 body layers + 3 MTP
layers):
- `layer_types[i] == "full_attention"` ⇔ i ∈ {3, 7, …, 91} → **23 entries, exact match with the
  tensor presence. No disagreement.**
- Layers 92/93/94 (MTP) are `"sliding_attention"` in `layer_types`, and carry **no** indexer
  tensors — consistent with `sparse_config.apply_to_layer_types = ["full_attention"]`.
- All other body layers are `"sliding_attention"` and carry no indexer tensors.

## 3. Full MTP inventory (`model.layers.{92,93,94}.*`)

All 51 MTP tensors live in `model-mtp3-full-00001.safetensors` per the index. Each of the three
layers has the **identical** 17-tensor set (shapes verified for 92, 93, 94 — all identical):

| tensor | shape | dtype |
|---|---|---|
| `model.layers.N.eh_proj.weight`                        | `[4096, 8192]`    | BF16 |
| `model.layers.N.enorm.weight`                          | `[4096]`          | F32  |
| `model.layers.N.hnorm.weight`                          | `[4096]`          | F32  |
| `model.layers.N.input_layernorm.weight`                | `[4096]`          | F32  |
| `model.layers.N.post_attention_layernorm.weight`       | `[4096]`          | F32  |
| `model.layers.N.mlp.gate_proj.weight`                  | `[13824, 4096]`   | BF16 |
| `model.layers.N.mlp.up_proj.weight`                    | `[13824, 4096]`   | BF16 |
| `model.layers.N.mlp.down_proj.weight`                  | `[4096, 13824]`   | BF16 |
| `model.layers.N.self_attn.q_proj.weight`               | `[12288, 4096]`   | BF16 |
| `model.layers.N.self_attn.k_proj.weight`               | `[768, 4096]`     | BF16 |
| `model.layers.N.self_attn.v_proj.weight`               | `[768, 4096]`     | BF16 |
| `model.layers.N.self_attn.o_proj.weight`               | `[4096, 12288]`   | BF16 |
| `model.layers.N.self_attn.g_proj.weight`               | `[64, 4096]`      | BF16 |
| `model.layers.N.self_attn.q_norm.weight`               | `[192]`           | F32  |
| `model.layers.N.self_attn.k_norm.weight`               | `[192]`           | F32  |
| `model.layers.N.transformer.shared_head.norm.weight`   | `[4096]`          | F32  |
| `model.layers.N.transformer.shared_head.output.weight` | `[128896, 4096]`  | BF16 |

- **Each MTP layer has its own `transformer.shared_head.{norm,output}`** — confirmed present for
  all of 92, 93, 94 (17×3 = 51 tensors). "shared_head" is shared *within* a layer's decode path,
  not across MTP layers. `shared_head.output` is a full vocab head (`128896 × 4096`), same shape
  as `lm_head.weight`.
- **MTP MLP is DENSE**: `mlp.{gate,up}_proj: [13824, 4096]`, `mlp.down_proj: [4096, 13824]` with
  intermediate_size 13824. **No `moe.*` and no `share_expert.*` tensors exist under layers
  92–94.** (`moe_layer_list` = 3..90, so 91 and MTP layers are dense; layers 0–2 also dense.)
- `eh_proj: [4096, 8192]` = concat(h, emb) 4096+4096 → 4096.
- No indexer/sparse tensors in any MTP layer.

### Redundant shard copies (watch out when loading)
- `model-00024.safetensors` **also physically contains** the 17 layer-92 MTP tensors
  (byte-identical names/shapes) plus `lm_head.weight [128896,4096]` (BF16),
  `model.embed_tokens.weight [128896,4096]` (BF16), `model.norm.weight [4096]` (F32).
  The index routes layer-92 MTP tensors to `model-mtp3-full-00001.safetensors`; `lm_head`,
  `embed_tokens`, `model.norm` route to `model-00024.safetensors`.
- `model-mtp3-00001.safetensors` is a standalone 10-tensor subset (not in the index): the
  `{eh_proj, enorm, hnorm, shared_head.norm, shared_head.output}` of layers 93+94, shapes
  identical to `model-mtp3-full-00001`.
- `vit.safetensors` ≡ `model-vit-00001.safetensors` (same 667 `vision_model.*` names, identical
  shapes/dtypes; vit.safetensors has 668 header entries incl. `__metadata__`). The index routes
  all `vision_model.*` to `model-vit-00001.safetensors`.

## 4. Body attention shapes (GQA geometry confirmed)

Identical for full-attention and sliding-attention layers (sampled layers 3 and 7; also 11, 47, 91):

| tensor | shape | dtype | geometry |
|---|---|---|---|
| `self_attn.q_proj.weight` | `[12288, 4096]` | BF16 | 64 heads × head_dim 192 = 12288 |
| `self_attn.k_proj.weight` | `[768, 4096]`   | BF16 | 4 groups × 192 = 768 |
| `self_attn.v_proj.weight` | `[768, 4096]`   | BF16 | 4 groups × 192 = 768 |
| `self_attn.o_proj.weight` | `[4096, 12288]` | BF16 | |
| `self_attn.g_proj.weight` | `[64, 4096]`    | BF16 | head-wise attn gate (use_head_wise_attn_gate=true) |
| `self_attn.q_norm.weight` | `[192]`         | F32  | per-head RMSNorm, head_dim 192 |
| `self_attn.k_norm.weight` | `[192]`         | F32  | per-group, head_dim 192 |

→ GQA geometry **confirmed**: hidden 4096, num_attention_heads 64, num_attention_groups 4
(both full and sliding), head_dim 192 (= true_head_dim; no padding). k/v are 768 = 4×192.

For contrast, MoE body layer 7 (context for the port):
`moe.gate.weight [352,4096] BF16`, `moe.router_bias [352] F32`,
`moe.{gate,up}_proj.weight [352,1536,4096] BF16`, `moe.down_proj.weight [352,4096,1536] BF16`,
`share_expert.{gate,up}_proj [1536,4096]`, `share_expert.down_proj [4096,1536]` — i.e. 352 experts,
moe_intermediate_size 1536, top-k 8.

## 5. `config.text_config.sparse_config` (verbatim) and nextn fields

```json
{
 "enabled": true,
 "proxy_dim": 256,
 "sparse_indexer_rope_dim": 32,
 "sparse_indexer_use_rope": true,
 "sparse_indexer_num_heads": 16,
 "sparse_indexer_num_k_heads": 1,
 "sparse_indexer_q_norm_type": "rmsnorm",
 "sparse_indexer_k_norm_type": "layernorm",
 "sparse_indexer_csa_z_norm_type": "none",
 "sparse_indexer_softmax_variant": "ssmax",
 "sparse_indexer_ssmax_s_granularity": "q_head",
 "num_provider_groups": 4,
 "topk": 512,
 "region_block_size": 8,
 "compression_method": "csa_block_compress",
 "attention_impl": "sparse_gqa",
 "apply_to_layer_types": [
  "full_attention"
 ]
}
```

MTP/nextn fields (the only one; no other `mtp_*`/`nextn_*` keys anywhere in config):
```json
"num_nextn_predict_layers": 3
```

Other config values relevant to the port (verbatim values):
- `model_type: "step3p5v"` (top level), `text_config.model_type: "step4"`,
  `text_config.architectures: ["Step4ForCausalLM"]`, top-level `architectures:
  ["MMGPTStepRoboticsForCausalLM"]`, `auto_map: {"AutoConfig":
  "configuration_step_robotics.StepRoboticsConfig"}`
- `hidden_size 4096`, `intermediate_size 13824`, `num_hidden_layers 92`, `vocab_size 128896`
- `num_attention_heads 64`, `num_attention_groups 4`, `head_dim 192`,
  `head_dim_full_attention 192`, `head_dim_sliding_attention 192`,
  `num_attention_groups_full_attention 4`, `num_attention_groups_sliding_attention 4`,
  `num_sliding_attention_heads 64`, `att_impl_type "GQA"`, `use_head_wise_attn_gate true`
- `sliding_window 512`, `yarn_only_types ["full_attention"]`, `rope_scaling
  {"rope_type":"llama3","factor":1.0,"original_max_position_embeddings":1048576,
  "low_freq_factor":1.0,"high_freq_factor":32.0}`, `max_seq_len 1048576`
- `rope_theta`: per-head-dim list of 1024+ entries — pattern `[10000,10000,10000,10000000]`
  repeating (4 entries per head-dim slot, matching partial_rotary_factors pattern
  `[1,1,1,1/3]` repeating)
- `torch_dtype "bfloat16"`, `norm_dtype "float32"` (all norm tensors F32 in the checkpoint —
  confirmed), `fp32_residual_connection true`
- MoE: `use_moe true`, `moe_num_experts 352`, `moe_intermediate_size 1536`, `moe_top_k 8`,
  `num_experts_per_tok 8`, `share_expert_dim 1536`, `moe_layers 3..90` (dense: 0–2, 91, MTP),
  `need_fp32_gate true`, `norm_expert_weight true`, `moe_router_scaling_factor 3.0`,
  `use_moe_router_bias true`, `swiglu_limits [7.0 × 92]`, `swiglu_limits_shared [7.0 × 92]`
- `rms_norm_eps 1e-05`

## Sanity totals
- index `weight_map`: 2449 tensors; headers of the 24 body shards + mtp3-full + model-vit cover
  all 2449 with matching shard placement.
- 23 sparse layers × 8 indexer tensors = 184; 3 MTP layers × 17 = 51 (51 in mtp3-full file).
