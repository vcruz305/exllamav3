# Integrated three-stage trial package — local gates complete

**NOT COMPILED / NOT GPU VALIDATED.** No model-throughput result. No remote operation has been performed for this integration.

## What is integrated

`package/` is all 2,309 byte-identical files from the final trial-hardening package, including the q1..8 timing fix and unchanged original/five-stage/three-stage CUDA source trees. `guarded_deferred.py` is the separately verified repaired runner plus a minimal `main()`-only patch:

- Bind the exact final deferred/build/reference/scalar/manifest hashes; refuse old helper bindings.
- Call the pinned, read-only all-root path preflight BEFORE opening the shared owner lock or creating receipts.
- Protect the guard and runner files as well as all declared source/serving roots and packaged source trees.
- Record the accepted protected-root inventory in invocation provenance.

All seven lifecycle functions are unchanged, including actual workload direct-child ownership and dedicated subreaper custody. Original packages remain sealed.

## Actual verification

- Parent reran helper hardening: 30 candidate tests + 30 exactly applied tests GREEN; five original preservation controls GREEN; 43 intended old-helper failures/subtests across 25 tests with zero harness errors.
- `suite-parent01/summary.json`: 42 expected outcomes reproduced, comprising 22 process cases (one intentional original-runner RED), five prerequisite-refusal cases, and 15 integrated CLI cases.
- `replay-parent01.json`: strict Git apply check/application, exact runner bytes and all 2,309 package files; only runner `main()` differs from the sealed repair.
- The disposable sibling `mixedk-three-stage-trial-integration-applied-parent01` reran the same 42-case suite under label `applied01`; all expected outcomes reproduced.
- `final-local-verification.json`: independent receipt/hash checks; real WSL procfs found none of 170 unique recorded PID/starttime identities still present. Repaired process cases required no emergency harness workload cleanup.
- Original 7,794 candidate/guard inputs, 1,312 sealed runner outputs, and 10,116 hardening input hashes were rechecked. These overlapping inventories are not additive unique-file totals.

The integrated CLI actually parses and admits the final package, acquires the existing shared lock, and invokes the real guard/custody. Tests explicitly replace memory samples and the GPU command with a small inert process. They do not execute CUDA, Torch, the compiler, or model inference. Invalid destinations, incomplete inventories, reused caches, nested receipt/output paths, old bindings, and five tampered helper/manifest files are rejected before invocation writes. All invalid-path mutations are confined to new test fixtures.

TDD evidence: `evidence/pins-RED01`, `pins-GREEN01`, `paths-RED01`, `paths-GREEN01`. Initial test formatting was corrected after syntax lint, before credited RED. The first final-audit attempt tried to parse a deliberately corrupted JSON fixture; the audit now parses only named process receipts while still hashing all files. A WSL cross-filesystem hash audit timed out without a surviving owned audit process; the successful audit hashes natively on Windows and queries actual WSL procfs separately. Neither issue changed production code or weakened assertions.

## Pins

- Runner: `2caeb7d4da4813d0c1aa725486c8aa6a26425f94c11fdc23c150c8b3a31f4b47`
- Runner patch: `32ae2b8daad15f9c48799ff15a5f4f192cb88549cfc9a8850accae094041a3dd`
- Guard: `eef5706b327cf20007dd49087b6d70b97ec178fcdbde616374188a0bc916d24d`
- Deferred: `29a870a4e11c55d185066df788402c283a46409ea08601758d721bb3eeeb1500`
- Build: `a4f10a7fc85596a57bccb9d764f573531ab735844ed23a4d160cfe3540eced0f`
- Reference: `0cde9ed13e945cc4f79a5e4a8366a310c3820643fb44cfbb94fb85a235dd48eb`
- Source manifest: `cf6a7cdb9392c50e771d651df94c9c13e7bf8cbbc27dbcf0fdd6900801bd6086`
- Scalar: `c503fc9c3ef82f056b553c359e95bc6928c24dfb689fd057081d78c50823c7d5`

## Future authorized invocation, not executed

First obtain host-specific exclusive operator authorization. Independently verify retained service/process identity, no competing controller, rollback readiness, and a healthy idle request queue. Stop only the identified owned model through its unchanged guard. Verify exact old identities gone, actual host/cgroup memory headroom and OOM counters, working sampler, available pidfd/subreaper support and actual toolchain. Do not modify original runtime, serving configuration, CUDA extension, target/drafter weights or other users' files.

Set `BUNDLE` to the byte-verified transferred integration directory, `GUARD` to the exact reviewed guard source, `OWNER_LOCK` to the real stable shared owner lock, and `RUN` to a new prefix outside every protected tree. Set absolute `CC`, `CXX`, `CUDA_HOME`, a PATH including Ninja, and freeze the compiler/SDK/dependencies for the trial. The lock only coordinates cooperating launchers; it does not prove sole ownership.

Cache, output and receipt must be separate NONEXISTENT sibling paths. **Do not use `--output "$RUN/receipt/trial"`: receipt/output containment is intentionally refused.** The existing regular shared owner lock is permitted; do not generate a new lock per trial.

```
EXL3_TRIAL_MODEL_STOPPED=YES EXL3_THREE_STAGE_AUTHORIZED=YES MAX_JOBS=4 \
python3 -B "$BUNDLE/guarded_deferred.py" \
  --guard "$GUARD" \
  --guard-sha256 eef5706b327cf20007dd49087b6d70b97ec178fcdbde616374188a0bc916d24d \
  --candidate "$BUNDLE/package" \
  --deferred-sha256 29a870a4e11c55d185066df788402c283a46409ea08601758d721bb3eeeb1500 \
  --receipt "${RUN}-receipt" --owner-lock "$OWNER_LOCK" \
  --seconds 1800 --memory-policy uma \
  --sole-gpu-owner --model-stopped --headroom-verified \
  -- --mode numeric --cache "${RUN}-cache" --output "${RUN}-result" \
  --protected-root "$BUNDLE" \
  --protected-root "$ORIGINAL_RUNTIME_ROOT" \
  --protected-root "$ORIGINAL_SERVING_SOURCE_ROOT" \
  --protected-root "$TARGET_MODEL_ROOT" \
  --protected-root "$DRAFTER_MODEL_ROOT" \
  --protected-roots-complete
```

The listed root variables are not a complete inventory by themselves; add every other actual source/serving root. Flags are operator assertions, not proof. Preserve full outer stdout/stderr/exit status and runner receipts.

Initially run ordinary numeric only, with its compile deadline and memory limits unchanged. Preserve failures; never weaken tolerances or silently increase time/memory. Poison numeric, memcheck, initcheck and profiler are separate later invocations with fresh caches. Timing requires all prerequisite receipts and covers q1..8; synthetic kernel timings are not model TPS. Do not combine sanitizer with CUPTI profiling. No kernel deployment/full-model candidate inference is authorized merely by this runbook.

Whether the bounded trial passes or fails, confirm its exact process identities are gone and restore the retained block8 quantized-drafter/Q4/handled-only profile under the original guard, with explicit healthy readback and original hashes. Ambiguous identity, another owner's activity, OOM counters, or missing actual headroom must fail closed.

The polling guard is not a kernel-hard OOM guarantee. Custodian destruction, uninterruptible tasks, fast allocation bursts, adversarial namespace/filesystem changes and missing receipt storage remain outside local proof.
