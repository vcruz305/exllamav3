# Local CPU review: two independent, default-off candidates

Pinned source: `ca4a880e8918e1985fd25e06c6aff561666d3f14`.
No SSH, CUDA initialization, GPU execution, deployment, model loading, publishing or server edits were performed.

## Candidates

1. `patches/mixedk-dead-readback.patch`
   - Enable **only** with `EXL3_MOE_MIXEDK_ELIDE_HANDLED=1` before importing the module.
   - Suppresses the unused `mixedk_handled` nonzero/list readback only when `expert_count_list is None`, where the fallback loop runs zero iterations.
   - Retains `counts_fused` readback, exact active count, launch arguments, MTILE tiering, scratch/gather tables, buffers and kernels. Default retains both original readbacks.
   - The broader `counts_fused` elision was deliberately NOT included. There is no claim of numerical identity or speedup on a GPU.

2. `patches/target-round-diagnostics.patch`
   - Construct an isolated diagnostic `Generator(..., record_draft_stats=True)`; default stays False.
   - `generator.draft_diagnostics` is a separate bounded CPU-scalar output, NOT HTTP usage. It counts actual calls at the target `model.forward` site in `iterate_gen`, including a call that raises (incomplete record).
   - Records the actual proposed window reaching target verification, target input shape, per-job sampled calls, accepted-prefix increments, actual emitted `token_ids` counts, net new-token delta, EOS, requeue and rewind.
   - Finalizes AFTER the final receive_sample event and BEFORE job removal/reinitialization. The generator-owned record survives requeue.
   - At most 256 passes, each at most the generator's active batch size; includes a dropped-pass count and lifetime target-decode invocation count. The existing `job.draft_stats` list is also capped to its latest 256 entries when enabled. This is an intentional diagnostic-only API behavior change; do not treat that legacy list as a lifetime history.
   - No capacities, draft windows, page reservations, sampler behavior, CUDA operations, timers, prefill or drafter paths changed. No expert-union census was added.

## Verify offline

Requires Python with NumPy and git; neither PyTorch nor pytest is required. From this directory:

```bash
python verify.py --label review-unique-01 --upstream ../dflash-pages-candidate/upstream
```

Every label must be new. `verify.py` checks the manifest, asserts the exact clean source pin, captures real subprocess return codes and unittest counts, requires baseline assertion failures with ZERO harness errors, runs preservation controls, applies each patch alone and both together in new disposable trees, compares every archive member byte-for-byte, and reruns tests. Logs are immutable `evidence/<label>-*.txt`, with a machine-readable summary JSON. Git blob bytes are preserved; archive creation and application use local `core.autocrlf=false` options only. The application additionally sets `core.whitespace=cr-at-eol` for portable whitespace checking.

The package contains the pinned `baseline.tar`, unmodified `baseline/`, modified `candidate/`, source-only patches, CPU harness/tests, and copied earlier generator/job files under `prior-source/` solely for reproducing the previous bug. Nothing reads or modifies the running/shared runtime except an optional read-only pin/status check.

## Export the diagnostic output

In the **generator-owning thread**, after the isolated job has completed (not the HTTP thread):

```python
# Supply record_draft_stats=True in Generator construction, not as an environment flag.
# Existing production serve_native.py does not pass this option and remains unchanged.
import json
with open("target-rounds.json", "x", encoding="utf-8") as f:
    json.dump(generator.draft_diagnostics, f, indent=2)
```

The ring is per generator, not per logical request. Use a fresh generator for a bounded single-stream experiment or associate integer `serial` plus `pass_id`. A batched target call counts once, not once per job. On failures, export in a caller finally block; do not interpret incomplete records as successful rounds. Do not mutate `record_draft_stats` after construction.

### Diagnostic semantics

- `role=target_decode`, `scope=target_decode_only`: **excludes target prefill, drafter forward, drafter prefill and cache-update work**. This is not a total-model-call count or pure target timing.
- `proposed_window`: columns of the supplied `draft_tokens` tensor after truncation, not native drafter capacity, configured maximum, or target rows (the latter are separately recorded).
- `accepted_prefix`: acceptance-branch increments, not IID acceptance probability. EOS/requeue/checkpoint exits before comparison are unresolved, not retroactively inferred matches.
- `sampled_tokens`: actual receive_sample completions. Includes stop tokens and tokens later rewound; token healing can also contribute.
- `emitted_token_ids`: actual IDs attached to stream events during this pass. Stop token may emit zero; held IDs from earlier passes can emit later. This is token-ID emission, not retokenized visible text.
- `new_tokens_delta`: actual net job counter change, potentially zero/negative after a rewind.
- No `tokens_per_pass=1+accepted` shortcut and no timing claims. If dropped_passes is nonzero, retained-pass averages are explicitly a tail sample, not whole-job averages.

## Scope and limitations

The tests execute complete `BlockSparseMLP.forward`, `Generator.__init__`, `_staging`, `iterate_gen`, `Job.__init__`, `prepare_for_queue`, `receive_sample`, `prepare_for_requeue`, and input-ID methods via AST; decorators are removed, method bodies are NOT sliced. The CPU boundaries are documented in the harness. CUDA launches are an assignment/slot oracle with poisoned scratch, not CUDA math, scheduler, race, or performance validation. NumPy tests do not establish GPU tensor identity.

A legacy-group initialization followed by unified dispatch fails in the ORIGINAL and candidate because `_mkd_bufs` exists without `_mkd_fused_rows`. The regression documents that unchanged pre-existing hazard; it does not claim to repair it. Do not switch these eligibility modes in a single loaded module. Batched reconstruction kernels, TP collectives, CPU expert offload, real samplers and GPU recurrent-state/cache behavior are not validated here. For the narrow optimization use the already-unified, small-row path after separate GPU verification.

See `AUDIT.md` for rejected arithmetic/patch claims and `GPU_VALIDATION.md` for the explicitly unexecuted local GPU acceptance protocol and rollback.
