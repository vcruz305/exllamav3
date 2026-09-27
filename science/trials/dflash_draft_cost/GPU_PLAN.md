# Deferred GPU gate — cost-aware native block8 only

**NOT EXECUTED. This task authorizes LOCAL CPU/source work only.** No launch, SSH, GPU reservation, model load, service mutation, benchmark, commit, deployment or publication is authorized by this plan. A human owner must independently authorize any later live work.

## 0. Provenance and ownership before any load

- Obtain exclusive GPU/runtime ownership and record service/process state before changing anything. Stop if ownership is ambiguous.
- Review the exact combined/incremental patch, source pin `ca4a880e8918e1985fd25e06c6aff561666d3f14`, and `verify_repair.py` output from a fresh label.
- Independently establish target **weight** identity. The observed profile lacks a target hash/revision; a path is not authentication. Obtain owner-verified unchanged retained round8 pack identity and current hashes before issuing live attestation. Do not reuse temporary test attestation. Leave the shipped example false.
- Verify drafter bytes/revision `d50ead3c6a3dec221e9a595fbdc103ef60db594e`, 4.0bpw EXL3, native block8, tap_shift=0 and learned mask metadata. No inference-config width change.
- Record exact actual source/runtime/kernel hashes, target/draft cache structures, flags, sampler settings and profile digest. The hardcoded allow-list is structural, not weight/kernel authentication.

## 1. Freeze the measured retained configuration

Keep target Q4 **paged global KV plus the existing recurrent SWA rings**. Do not use `-swa_full`, `rcs=0`, a nonrecurrent fixture, native16, replacement kernels or different model weights as a workaround.

Fixed settings: context4096, chunk4096, max_batch1, native block8/request7, DDS enabled, draft confidence0.6, recurrent cache0.25 GiB, recurrent decode/prefill intervals2048/32768, original CUDA kernels, handled-elision1, counts-elision0, batch-verify0, retained UMA reserve8192 and split106. Keep identical neutral greedy sampler and request ordering. Check at runtime that cache.layers has the retained nine global Q4 instances and cache.recurrent_layers has 39 SWALayerState rings with the reviewed class/shape identities.

Run initialization with policy disabled first. Only after provenance is verified, supply a separately reviewed live profile digest/context attestation and confirm enabled initialization succeeds. Confirm no unauthorized mode/flag changed to get past a guard.

## 2. Device-semantic gate before timings

Use bounded separately reviewed diagnostics, preserving production outputs. Preflight instrumentation on tiny real tensors and align diagnostic device/dtype before comparisons; a harness crash is not a model/correctness verdict.

Observe all of these in both disabled and enabled arms:

- Actual native drafter input/output shape8/7 and full native physical writes at selected k=0/1/3/7. Confidence truncation must not be mistaken for narrower native cache writes.
- Assigned physical paged-cache coverage, not just block-table width. Validate write addresses/canaries/reference tensor contents, including neighboring storage, page ends and tiny max-new. A changed row alone is not proof of complete or exclusive write coverage.
- SWA ring page shifts, position, window base, rollback state, and actual tensor contents across match/mismatch, bonus, k=0 and all supported short verification windows.
- Nonzero selector-hit counters on ordinary post-prefill retained-profile rounds after real calibration burn-in. No-data/sparse bins must retain DDS. Record maximum eligible cap and actual chosen k separately. A construct-only success or an always-None selector is a failed eligibility gate.
- Guard fallbacks at current/approaching 2048-token checkpoints and nonzero cached-prefix offsets. Observe real checkpoint stash, restore and replay; do not infer tensor correctness from scalar position equality.
- Request EOS/max-new, unexpected token EOS, and simultaneous EOS/requeue precedence; no unresolved tokens emitted after terminal state. Verify cleanup/release as well as the host receive-sample slice.
- Actual requeue: with a 23-token input, requested max_rq1 is effectively2025 in this configuration. Budget max-new high enough to actually requeue. Observe first physical job termination near new_tokens2019 under the source's reserve7 rule, replacement job continuation and cache bounds. Requeued continuations must retain DDS. Do not call max-new1 a requeue test.
- Unknown recurrent architecture/class, altered geometry, cache mode, intervals, TP/layer mapping or owner identity must be rejected/fallback, never quietly accepted.

Any kernel error, out-of-range write, nonfinite result, replay/restore discrepancy, unexpected policy use at a protected boundary or output/terminal corruption blocks the performance phase. Preserve failures and instrumentation issues separately.

## 3. Ordered A/B/A experiment

A1 = unchanged DDS (policy off); B = reviewed cost policy on; A2 = DDS off again. Use a clean load/reset for each arm and the same ordered cold-then-warm sequence, because calibration history is stateful. Preserve cold results separately; do not discard them or carry the warm calibrator between arms accidentally.

Use exactly saved prompt bytes and request settings, with both code and prose workloads, plus the boundary cases above in a separate correctness set. Save token IDs, text, reasoning output, terminal reasons, generated/emitted token counts, per-round target/native forwards, actual verified q/k, acceptance/rejection counts, calibration state, selector hits/fallback reasons and cache metadata. Preserve all A1/B/A2 records without overwriting one arm.

Report output-identical and output-changed subsets separately. Speculative semantics and unchanged target verification are not a guarantee of identical floating-point output. Token/EOS/quality parity must be observed rather than asserted by construction.

Measure end-to-end emitted-token throughput and request latency including the new selector's CPU work. Keep cold/warm populations and distribution summaries separate. Record whole-round host spans and disjoint device intervals without summing overlapping host waits and CUDA events. Never treat expected-token/cost argmax as measured reward/TPS. Repeated A/B/A drift must be quantified before attributing any difference to policy.

## 4. Profile status and acceptance

The existing profile is provisional observed whole-round host costs from606 warm/191 cold rounds. State/policy selection bias remains, q8 warm has no prose observations, and the trace lacks per-position confidence/proposal IDs for a counterfactual replay. Do not fill missing cells, alter original measured values, or turn the profile into a speed prediction.

A new runtime/cache/model mode or broader eligibility needs fresh provenance and measurements, not a forged extension to existing attestation. Keep original raw evidence and hashes. Any proposed promotion requires clean device correctness, quality/terminal controls, reproducible ordered cold/warm A/B/A evidence and explicit owner approval. There is no performance promise or automatically passing threshold here.

## Separate experimental lanes

Native16 widths, kernel launch/readback changes, sampler batching, counts-elision changes, compiler/kernel rebuilds and neural/cache implementation changes are **separate trials** with their own source/device proofs and ownership. They must not share the B arm above; otherwise attribution is invalid. The inherited CPU native16 tests are preservation controls only, not authorization or GPU validation for a wider checkpoint.
