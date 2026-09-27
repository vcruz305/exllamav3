# Recurrent-SWA repair: CPU/source candidate, default OFF

## Outcome

The measured retained MiMo configuration now passes initialization **without changing cache mode**, and ordinary rounds actually use the cost selector. The incremental implementation changes only `exllamav3/generator/draft_cost.py`; `generator.py` is byte-identical to the original two-file candidate. No neural, cache, target-verification, page-allocation, or sampling implementation changed.

`evidence/local01-summary.json` and the independent fresh-label rerun `evidence/final02-summary.json` record complete successful verifier runs:

- Original candidate: all **240** existing tests pass, unchanged.
- Repaired candidate, combined-patch applied tree, and incremental-patch applied tree: **254** tests each (240 existing + 14 recurrent regression methods, many parameterized subcases).
- Clean pinned baseline: 187 page + 20 native + 12 sampler tests pass.
- Independent measured-profile reconstruction: 1 passing consistency test.
- Parent regression: intended failure reproduced (3 tests, 1 failure, 0 errors). New source-grounded constructor regression against original: 1 intended failure, 0 errors.
- Both patches passed `git apply --check --whitespace=error-all`, applied into separate disposable LF trees, and **every resulting file** matched the candidate byte-for-byte.
- All baseline archive members match a freshly generated exact-pin Git archive. Baseline Git status remained clean. All **3219** sealed original artifacts and **956** other sealed dependency files rehashed unchanged.

The source-executed progression test emitted:

```
ACTUAL_HOST_PROGRESS {'rounds': 253, 'eligible': 252, 'fallback': 1,
 'new_tokens': 2019, 'position': 2041, 'effective_max_rq': 2025}
```

These are **synthetic CPU host-control results**, NOT measured model acceptance or GPU throughput. The test starts with the retained 23-token input and requested `max_rq_tokens=1`, executes the real queue-time alignment and full producer/receive-sample control methods, and reaches the real effective requeue cutoff. It does not merely substitute requested budget 1. Warm calibration labels and neural outputs are synthetic test fixtures. Observed profile costs plus synthetic all-accepted probabilities select k=7 on the ordinary round; separate synthetic cost rankings prove k=0/1/3/7 go through the complete unchanged native block8 producer.

## Root cause and narrow allow-list

The original policy rejected every non-None recurrent cache in both `from_environment()` and `select()`. Its passing full-constructor test used `model.caps={}`, which is not the retained MiMo runtime.

Source trace against exact pin `ca4a880e8918e1985fd25e06c6aff561666d3f14`:

| Source | Relevant contract |
|---|---|
| `model_init.py:74,115,262-301` | Default SWA is recurrent. Retained launch has `-rcs 0.25`, no `-swa_full`; loader passes that mode to the model and derives max_history=7 for native block8/request7. Q4/4096 target cache, max_batch_size=1. |
| `architecture/mimo_v2.py:77-171,188-362` | Actual config fields, exact 48-layer hybrid pattern; nine global-attention layers and 39 sliding layers. `not swa_full` selects SlidingAttention and sets recurrent_states=True, default checkpoint interval=2048, recurrent_state_cls=SWAState. |
| `cache/cache.py:145-171` | `cache.layers` contains paged global KV only: nine Q4 layers here. `cache.recurrent_layers` separately contains 39 SWALayerState rings; Q4 is NOT applied to these rings. |
| `modules/sliding_attn.py:41-93,148-179,266-310,486-490,845-872` | SWAState bookkeeping and rollback, half-precision ring allocation, window=127, overprocessing=512, rounded ring size=768, exact SlidingAttention/SWALayerState types and decode page shifts. This differs from GDN, DSA, short-convolution and other recurrent architectures. |
| `generator/generator.py:232-252` | RecurrentCache is created and slots reset; retained snapshot budget=256 MiB, interval=2048, prefill interval=32768. |
| `generator/job.py:265-266,308-309,1110-1123,1565-1582` | `job.checkpoint` is a banned-string hold, initially None; it is **not** the recurrent checkpoint/state. `recurrent_state` and `last_recurrent_checkpoint_pos` are distinct. Requeue alignment is recurrent interval, not requested token budget. Decode checkpoints are relative to `cached_pages*256`. |
| `generator/generator.py:545-553,1112-1131,1241-1312` | Snapshot check precedes drafting; checkpoint acceptance cutoff, rejection rollback, requeue and EOS paths remain unchanged. |

The new initialization gate uses exact runtime class identities, not matching class-name strings: MiMoV2Model, SWAState, SlidingAttention, SWALayerState and RecurrentCache. It requires the retained config/ring dimensions, exact global/recurrent layer-key partition, no layer-map or TP, slots/history/reserve 1/7/7, checkpoint intervals 2048/32768 and 256 MiB snapshot cache. All original profile/context/digest/runtime/Q4 checks remain. The original nonrecurrent compatibility path is preserved; it is **not presented as measured MiMo evidence**.

The per-round gate is bound to that validated model/cache/snapshot-cache identity. It requires the exact SWAState class, matching owning cache and sequence/state position, an ordinary single-sequence post-prefill neutral-greedy job, and valid cached-prefix metadata. It preserves DDS for banned-string checkpoints, rewound/finished jobs, stop strings/loop detectors, all requeued continuations, altered recurrent interval and unknown state types.

For decode phase `(kv_position - cached_pages*256) % 2048`, policy falls back at phase zero or when the maximum candidate verification envelope, including bonus, can reach the next checkpoint. The interval arithmetic is tested against the complete source `Job.is_checkpoint_boundary()` across current and future positions, including a nonzero cached-prefix offset. Existing max-new/requeue envelope guards remain unchanged. Unexpected token EOS cannot be predicted by this selector; it still runs through the untouched serial target verifier. The objective is a heuristic, not realized usable-token reward.

## Strict vertical TDD evidence

1. `RED00-parent.log`: copied parent repro fails its intended enabled-MiMo assertion; disabled/nonrecurrent controls pass.
2. `RED01-init.log` / `RED01b-narrow-init.log` -> `GREEN01-init.log`: source-constructed MiMo/cache enablement plus rejection controls; no selector change yet.
3. `RED02-eligible.log` -> `GREEN02-eligible.log`: initialization alone was insufficient; ordinary retained round returned None. Added instance-bound eligibility and proved selector reaches native producer.
4. `RED03-checkpoints.log` -> `GREEN03-checkpoints.log`: source-derived checkpoint envelopes failed before adding the conservative boundary guard.
5. `RED04-lifecycle.log` -> `GREEN04-lifecycle.log`: invalid state/lifecycle/requeue controls failed before adding their fallbacks.
6. Existing 22 cost tests reran green after each slice; the full 240-test inherited suite and all new tests ran on candidate and both independently applied trees in `local01`.

Preservation tests additionally execute actual checkpoint truncation/stash eligibility, effective requeue and continuation, max-new, token EOS, EOS-over-requeue precedence, mismatch rollback, ring page-shift bookkeeping and native assigned-page coverage for selected 0/1/3/7 windows. Original inherited tests were not edited and no obsolete rejection expectation was removed.

Two harness issues were preserved honestly: `HARNESS01.log` initially lacked the DraftAdapter cache-discovery seam; `PRESERVATION01.log` inspected fresh opaque page records as if already prefetched. They were fixed in test boundaries only. Neither is counted as an intended production RED.

## Executed source vs. explicit boundaries

`retained_harness.py` compiles source ASTs without importing the runtime:

- Complete MiMoV2Config constructor/qkv-dequant factory and MiMoV2Model constructor; complete SlidingAttention constructor and `_decode_state_prep`; complete Cache constructor, reset/get-new-state/get-all-recurrent-layers; complete SWAState class; complete RecurrentCache constructor; complete recurrence-advance function; complete checkpoint predicate/stash method.
- Actual model_init parser declaration **slices** for seven retained flags and the loader's **max_history + cache-construction slice**. Actual Job constructor **lifecycle declaration slice**, not whole Job initialization.
- Inherited native harness executes complete Generator constructor and DFlash producer, draft config/input/sample/KV-update paths, queue/reservation/requeue methods, actual calibrator, full receive_sample. Its target verification is the disclosed `iterate_gen` slice from outcome-list initialization through calibration feedback, **not** the full target forward/postprocessing/cleanup loop.
- Explicit substituted boundaries: base Config file/weight collection, generic neural module constructors, target Model base/discovery helpers, global Attention and QuantLayer constructors, Cache attachment/model loader, tokenizer/sampler responses, page-table allocation, prefill/restore, tensor/kernel arithmetic. SWA allocation records shape/dtype metadata only. Ring decode control uses NumPy with head/lane axes collapsed; no real half-precision ring is executed. Stash tests observe host stash calls, not actual checkpoint tensor contents.
- Requeue runs the complete method, which reinitializes the same job, but its CPUJob constructor and cache allocation/prefill/restore are explicit seams. GPU state restoration and EOS cache release remain unproved.
- A narrowly scoped import substitution supplies AST-executed classes for the policy's exact-type checks. `safe_run.py` blocks real torch/exllamav3 imports. No tensor devices, weights, SSH, network, GPU or live services were accessed.

## Profile and provenance

`profile.json` and `context.example.json` are byte-identical to the original. The profile is observed instrumented **whole-round host** cost data pooled from **606 warm / 191 cold** rounds, not a speed prediction. Raw means, counts, cold costs, source hashes, drafter revision `d50ead3c6a3dec221e9a595fbdc103ef60db594e`, and target-weight-identity limitation are untouched.

No target weight hash was discovered or invented. `context.example.json` remains `attest_same_round8_weights_and_runtime: false`. Synthetic true attestation files are explicitly TEST_ONLY, created in temporary directories inside this overlay and deleted at test completion; none is a ready-for-live configuration. Structural class/config validation does not authenticate weights, runtime binaries or kernels.

## Artifacts and reproduction

All files created here are under `dflash-cost-aware-recurrent-repair/`; original candidate and all sibling trees were read-only.

- `incremental.patch`: original candidate -> repaired candidate; one file.
- `combined.patch`: pristine exact pin -> repaired candidate; original two-file scope.
- `candidate/`, `full-files/`: complete source and changed-file copies.
- `verify_repair.py`, `safe_run.py`, `retained_harness.py`, `test_recurrent.py`: independently runnable local validation.
- `test_cost.py`, `test_profile.py`, `build_profile.py`, `run_native_controls.py`: exact copied inherited helpers. Some import/read sealed sibling evidence; preserve the supplied `mimo-tune` directory layout.
- `sealed-original-manifest.json`, `sealed-inputs-manifest.json`, `baseline-source-manifest.json`: immutable evidence/input/baseline hashes.
- `evidence/`: exclusive-label RED/GREEN, full-suite transcripts and summaries.
- `GPU_PLAN.md`: deferred A/B/A gates, separated from native16/kernel trials.

Run from Windows bash, using a fresh label (labels cannot be reused):

```bash
"C:/Users/Victor Cruz/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe" -B \
  "C:/Users/Victor Cruz/AppData/Local/hermes/cache/scratch/mimo-tune/dflash-cost-aware-recurrent-repair/verify_repair.py" \
  --label parent02
```

The verifier needs Python/NumPy, local Git and the sealed sibling source/evidence trees; it does not fetch or install anything. It writes only label-specific evidence/disposable trees and idempotent package artifacts in this overlay.

### SHA-256

- Incremental patch: `5f2ff5f766072d5e8d7a8769d1dde65df0bc4523e70fa79be828370628f2b25a`
- Combined patch: `93a653fd430a8d23cab3c01c1e5ad3352d010c02fdd45080b35991f0c89f2528`
- Repaired draft_cost.py: `543f8acb15cd0047985ea4efe70ae93566bb50fdc426c3b863d3778edcb2c334`
- Unchanged candidate generator.py: `30da9fd112d04ee1c8aae48819ce0c6af0833d7b26b2f03092aa1fd965cf0e58`
- Unchanged observed profile: `798adbc9c69156524d16f1ae028731974825b01322998613f03faa8ad6efa7e6`

**Remaining gates:** independent weight/runtime attestation and source integration review; actual GPU ring/paged-cache/restore/EOS correctness; ordered cold/warm A/B/A quality/token parity and end-to-end timings including selector overhead. No TPS, speedup, deployment or GPU-readiness claim.
