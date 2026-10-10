# Step-5 EXL3 re-encode — readiness

_Status as of the MTP + CSA indexer port. Supersedes the "text-body-only" scope in
`work/plans/PORT_SPEC.md` §5 for the encode step._

## Done

| Item | State |
|---|---|
| MTP module (layers 92–94) | `exllamav3/architecture/step5_robotics_mtp.py`, commit `13c1280`, branch `feat/step5-mtp-indexer` |
| MTP tensor map | 51/51 verified against real safetensors headers (`port-notes/05`) |
| MTP geometry | `eh_proj [4096,8192]`, `shared_head.output [128896,4096]`, dense `mlp` `intermediate_size=13824` — all match |
| Config plumbing | `step5_robotics.py` reads `sparse_config` + emits `index_*` / `ssmax_*` geometry |
| Source recovery | 6 byte-identical HF mirrors of the deleted bucket; `step5_source_download.sh` pulls `rene98c/Step-5-Preview-BF16` |
| Source verification | `step5_verify_source.py` — index<->header agreement, 2449 tensors, expected byte total, `--repair` for the overlapping shard sets |
| Encode launcher | `step5_encode_launch.sh` — pins `feat/step5-mtp-indexer`, refuses to run on an unverified source |
| Research notes | `port-notes/01..05` — indexer math, compressor, ssmax/provider-groups, integration, tensor ground truth |

## In flight

- `exllamav3/modules/step5_csa_indexer.py` — the lightning indexer
- `exllamav3/modules/ssmax.py` + attention-scale patch plan
- `z` → DSV4 compressor mapping + the norm-bypass gap

## Not done

1. **Wire the indexer into `Attention`.** Per `port-notes/04` the right move is to extend
   `Attention` (it has the whole hook surface) and leave `SlidingAttention` alone —
   `build_bc_swa` explicitly rejects indexed modules. Only the 23 `full_attention`
   layers (3,7,11,…,91) get an indexer.
2. **`ssmax_s` in the main-attention softmax.** Separate feature from the indexer.
3. **Tensor coverage check** 2449/2449 zero unmatched, once the modules are wired.
4. **Numerical gate**: sparse-vs-dense logits at `topk=512`.

## Recipe impact — read this before re-sizing

`expand_recipe_exl3.py` allocates only Linears with `qmap is not None and qbits_key == "bits"`;
everything else is either driven by a flag (`mtp_bits` → `-mb`, `head_bits` → `-hb`) or left
unquantized. Unrecognized budgeted Linears get `default_k` — they do **not** fail the expand.

Consequences:

- **MTP is outside body-bpw accounting** (per `SAGE-EXL3/AGENTS.md`). It is driven by
  `-mb/--mtp_bits`, default 4. The existing 2-spark / 4-spark body recipes stay valid
  as size targets for the *body*; MTP bytes are additive and must be budgeted separately.
  At 5.03 GB BF16, MTP is ~1.9 GB at 3 bpw / ~3.8 GB at 6 bpw.
- **The indexer is body budget.** 23 layers × `sparse_indexer_q [4096,4096]` ≈ 387 M
  params, plus `z [256,4096]` ≈ 24 M. ~411 M params total ≈ 154 MB at 3 bpw, 308 MB at
  6 bpw. Small against 227 GB / 451 GB, but it does consume the ceiling.
- Which indexer tensors are quantized follows `mla_attn.py`'s precedent:
  `q` is quantized (`qmap` set, `qbits_key="bits"`), while `wk`/`weights_proj` are
  deliberately **unquantized** (`qmap=None`, "router-like: tiny, and selection noise is
  coherent across every layer"). `z` is the open question — treat as quantized unless the
  compressor mapping says otherwise.

## Hessian path — RESOLVED, no extra capture needed

The 368 teacher Hessians cover exactly four qmap groups on 92 text layers:
`block.attn.input`, `block.attn.o`, `block.mlp.input`, `block.mlp.down`
(`step5_teacher_hessians.py:24`).

`Attention` assigns `qmap + ".input"` to its q/k/v projections (`attn.py:252/274/288`) and
`qmap + ".o"` to o_proj (`attn.py:317`). The indexer's `q` and `z` project from the **same**
post-`input_layernorm` hidden as q/k/v, so they belong to the same input-activation
distribution.

**Decision: `qmap = "block.attn.input"` for the indexer's q/z.** They reuse the existing
bank. No new Hessian capture, no identity-H fallback. This is the same reasoning behind
`qsa_indexer` in `qwen4_exp.py:173` using `qmap = "block.attn"`.

`idx_k` and `idx_w` keep `qmap = None` and stay unquantized, matching `mla_attn.py`'s
stated rationale ("router-like: tiny, and selection noise is coherent across every layer
sharing it"). `z` is quantized alongside `q`.

## Open items that need the StepFun reference or a validation run

From `port-notes/03`:

| Question | Current stance | Confidence |
|---|---|---|
| `ssmax` = `s` or `s·log(n)` | default `s·log(n)` (the paper's definition) | unverified |
| `num_provider_groups: 4` | probably the 4 GQA KV groups | low |
| `topk: 512` unit | raw tokens (not 8-token regions) | low |
| `region_block_size: 8` | key-side block merge | medium |
| indexer `qmap` name | must be chosen before Hessian capture | — |

`ssmax_s` is measured as a frozen constant `0.08495759963989258` across all 64 entries
and all 23 layers, so at least its *value* is not in doubt.

## Re-encode execution plan

1. Integrate indexer + ssmax into `Attention`; wire the 23 layers.
2. Coverage check 2449/2449.
3. Sparse-vs-dense numerical gate on one layer before committing GPU hours.
4. Decide the Hessian path above; if (1), run the short capture.
5. `step5_source_download.sh` → `step5_verify_source.py --repair`.
6. `step5_encode_launch.sh` for each recipe. Two independent chains, parallel.
   Measured: 88 MoE layers × ~230–250 s + compile. ~6.4–6.6 h per pack.
   **2× A100 SXM 80 GB ≈ $25 for both** (best value); 2× RTX PRO 6000 ≈ $27.
7. Verify shards/bytes, then upload to `darkmatterlab1/Step-5-Preview-EXL3`.
