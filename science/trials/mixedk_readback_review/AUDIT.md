# Critical audit

## Conclusions that the earlier evidence does NOT support

- The performance target is **60+ tokens/s, single-stream, GB10, MiMo 2.50 bpw**. The reported comparison host at **194 tokens/s is also single-stream** (task-provided context, not remeasured here). Calling it a batching advantage is incorrect. This CPU review establishes neither rate nor a roof.
- `U(8)=56.7` was an unmeasured routing/independence estimate, NOT an observed expert union. Temporal routing correlation, token-dependent probabilities, local-TP membership and varying mixed-K expert sizes matter. Even a measured union count alone is not bytes fetched: layer-specific expert sizes, cache reuse and repeated weight accesses also matter. No union census was smuggled into the default path.
- A round latency obtained by dividing generation wall time by an estimated number of rounds is **not pure target verification time**. It includes drafter work, sampling, host enqueue/readbacks, scheduling and potentially cache updates. The earlier approximately 150 ms figure cannot be used as a measured target-forward cost. The new diagnostics deliberately contain no such timing estimate.
- `draft_accept = accepted/(accepted+rejected)` is an aggregate accounting ratio. Rejected includes unresolved EOS/budget positions in this implementation. It is not a conditional IID probability. For a truly constant conditional match probability p and fixed window w, the ideal uncensored expected sampled span would be `1 + sum(p**j for j in range(1,w+1))`, not `1+w*p`. If a *reported aggregate ratio* is used with a genuinely fixed window and complete accounting, `w*ratio` can equal mean accepted positions algebraically; this does not establish IID behavior, dynamic-window behavior, visible emitted tokens, or pure verify latency. With dynamic windows and censored/held/rewound output, measure each count directly.
- Router-permuted expert renaming preserves the selected expert tensors and therefore their byte cost. Renaming cannot reduce bandwidth without changing what is computed/read; re-layout/cache-locality hypotheses require their own measurements.
- Widening pinned buffers/reservation counters does not implement multi-block DFlash. The earlier patch has no target-conditioned chaining loop. It cannot supply the necessary target-conditioned K/V state, rollback/EOS semantics, or page-bound proof for multiple native blocks. Its warning to raise an environment variable to chain blocks is misleading. **Do not deploy the old DFlash patch. Neither repaired candidate changes DFlash capacity or implements chaining.**

## Mixed-K source findings

Reviewed original `block_sparse_mlp.py` routing/dispatch and scratch construction (especially 1195-1606), `exl3_moe.cu` host grid and gather, and `exl3_moe_kernel.cuh` device count filtering/scheduler.

1. `expert_count_list is None` already forces the fallback range to zero. Building `mixedk_handled` through `.nonzero().tolist()` then feeds no consumer. Opt-in suppression of only this operation requires no count estimate and leaves the kernel's active count/geometry intact.
2. Removing `_ec[_m].tolist()` as well is not the same change. The C++ mixed-K wrapper sizes `num_groups=min(concurrency,MOE_MAX_GROUPS,num_active)` for positive active count and derives group width from `target_blocks/num_groups`. When actual active count is below concurrency, substituting concurrency can change reduction geometry. The previous assumption that every multi-row verify pass has enough active experts is unmeasured. The new patch does NOT substitute an estimate or `-1`.
3. The device kernel scans only local experts, ignores zero/over-cap/out-of-tier counts, and uses a ticket scheduler. The sentinel bucket is outside `num_experts`; gather additionally bounds experts by the sliced local tables. Our oracle checks coverage and scratch-slot bounds, not ticket-scheduler liveness or CUDA reduction identity.
4. MTILE splits use real host counts and remain unchanged. Oversize counts still reach fallback. Both atomic and deterministic paths are exercised under stubs, including actual row-cap boundaries and all-sentinel slices.
5. Existing legacy-to-unified first-use hazard: the shared `_mkd_bufs` may already exist without `_mkd_fused_rows`/`_mkd_mtile_ok`; the unified initialization guard does not repair that. Executed on both trees and intentionally reported, not hidden behind a nominal green suite. Candidate scope does not include runtime switching between legacy-grouped and unified initialization.
6. The larger mixed-K/batched-reconstruction path deserves separate review: `_run_batch_recon` initializes its scratch slot offset from homogeneous fused buffers, while unified mixed-K slot tables can reserve a mixed-K prefix. This is a source-level concern outside the elided branch; it was NOT proved or repaired by these tests. Do not read small-row coverage as approval of arbitrary prefill/grouped reconstruction shapes.

## Earlier telemetry failure: reproduced, not hypothetical

`Job.receive_sample` constructs the final EOS event before `Generator.iterate_gen` appends that round to `job.draft_stats`. The earlier helper was read inside receive_sample, so it missed the last round. `prior_telemetry_repro.py` executes the copied earlier complete methods:

- max_new_tokens=1: job history has one round, but final result `draft_rounds` is absent (`None != 1`).
- max_new_tokens=5 across two target passes: job history has two rounds, but final result says one (`1 != 2`).

Both are assertion failures, not import/stub errors; immutable transcript: `evidence/prior-final-round-red-01.txt`.

Additional earlier integration issues:

- Existing `serve_native.py` Generator construction does not enable `record_draft_stats`.
- `worker.py` 107-110 explicitly allowlists event fields, excluding the proposed telemetry.
- `serve_native.py` 58-68 derives a fixed usage dict; adding job fields cannot automatically reach HTTP. `protocol.py` only validates include_usage, not arbitrary passthrough.
- `prepare_for_requeue` reinitializes the Job; adding aggregate keys to rq_state without restoring/maintaining the corresponding round history does not preserve full statistics.

The repaired telemetry is an explicitly separate generator-owned, bounded diagnostic object, enabled through the real constructor flag and exported only after the job on the owner thread. No HTTP integration or production-default claim is made.

## Occupancy-report proposal: not approved here

The proposed CUDA report is not included. Its comment says once per `(kernel, shape)` but its guard is only `occupation_reported[device]`; a prefill/first-shape report may hide the verify shape of interest. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` gives a residency ceiling for that kernel/resource request, not achieved occupancy or throughput. BPS is launch arithmetic, not proof that two blocks reside usefully. Static shared-memory formulas alone do not establish achieved occupancy. The proposed fallback `cudaGetLastError()` also warrants scrutiny rather than blindly clearing errors in a measurement patch. No CUDA compile, occupancy query or GPU proof was performed.

## TDD provenance

- `mixedk-red-01.txt` is an INITIAL HARNESS ERROR (missing tuple-assigned MTILE constants), not accepted RED evidence. It is retained for transparency. Constants extraction was repaired before source changes.
- `mixedk-red-02.txt`: original source executes real dispatch; exactly the dead-readback assertion fails, preservation controls pass. Then the minimal source change makes it green.
- `telemetry-red-01.txt`: final-pass export assertion fails on original source; no-draft default control passes. Then direct target-call diagnostics make it green.
- `telemetry-cap-red-01.txt`: after the first telemetry slice, 260 actual rounds expose the uncapped legacy list. Only then was the cap added; subsequent run is green.
- The first apply-verifier attempt (`audit-01`) caught a byte-identity problem before application: Windows `git archive` inherited `core.autocrlf=true`, producing CRLF archive text from LF Git blobs. The initial AST tests were semantically valid but did not establish exact-blob applicability. `repair_archive_eol.py` rebuilt the isolated trees using `git -c core.autocrlf=false archive`, retained all earlier bytes in `historical-crlf/`, and kept candidate edits. Later verifier runs compare the baseline critical files with binary `git show` and rerun all evidence against the corrected exact-pin LF source.
- Later tests extend preservation/edge coverage without inventing a failing state for existing behavior. Final verifier records fresh original-source regression failures and all candidate/apply results with real subprocess return codes.

CPU passing tests establish host control-flow evidence only. The changed sync behavior can expose device lifetime/race assumptions; captured same-input GPU output equality and uninstrumented single-stream timing remain mandatory before approval.
