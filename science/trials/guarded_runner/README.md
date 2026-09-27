# Guarded bounded-trial runner

A standalone, non-deployed entrypoint that supervises the three-stage GPU trial so that a
bounded run cannot outlive its budget, leak the device, or touch artifacts it should not.
It replaces `bounded_deferred.py` as the launch entrypoint and must not be wrapped inside
the old guard CLI.

Nothing here was compiled, run on a GPU, or deployed. The proof is local, CPU-only and
process-topology based: real short-lived Linux processes with an inert command standing in
for the GPU workload.

## Process topology

```text
entrypoint (waits; SIGINT/SIGTERM -> STOP file)
  custodian (dedicated Linux subreaper; watches entrypoint pidfd)
    exact guard worker (dedicated subreaper; calls the unchanged supervise())
      actual deferred_gpu.py (the guard's direct child / session leader)
        compiler/workload descendants
```

## The safeguards

- **Exactly-once stop.** A `STOP` file is created once and never re-armed; a second
  signal path cannot re-enter cleanup.
- **Direct-child guard.** The guard's real child is the selected package's
  `deferred_gpu.py`, not a timeout wrapper, so the guard's memory monitoring and
  `max_seconds` apply to the actual workload.
- **Subreaper custody.** The repair's direct-child change alone is insufficient: the
  unchanged guard can stop waiting when the group leader exits and leave a TERM-ignoring
  descendant. The worker drains adopted descendants on every handled return/exception; a
  surviving custodian does the same if the worker dies. Cleanup enumerates **only direct
  children of these dedicated processes**, repeats as grandchildren are adopted, pins each
  signal target with pidfd + `/proc` starttime, and reaps exact PIDs. No `pkill`, no
  name-based matching, no all-host scan.
- **Owner lock.** The trial acquires the real pre-existing shared owner lock used by every
  competing launcher. The lock coordinates cooperating users; it is not proof of sole
  ownership, and the runner never invents a per-run lock name.
- **Path preflight.** A pinned, read-only check of every protected root, the cache, the
  output and the receipt paths runs before anything is written.
- **Operator confirmations.** `--sole-gpu-owner`, `--model-stopped` and
  `--headroom-verified` (plus `EXL3_TRIAL_MODEL_STOPPED=YES` /
  `EXL3_THREE_STAGE_AUTHORIZED=YES`) are required, and are recorded as assertions — not
  as proof. The guard's own memory sampler is the executing monitor.

Exit contract: workload status is preserved; guard `time_limit` is **124**; memory STOP,
STOP file, SIGINT/SIGTERM, monitor failure or incomplete cleanup is **125**; preflight
refusal is nonzero before any `torch` import.

## The two documented pre-stop scan defects and their fixes

Both defects were in the checks that run *before* the trial starts, and both are
evidenced by an explicit RED/GREEN pair in `integration/evidence/`. They were found and
fixed in the integration overlay (`integration/runner-integration.patch`), which patches
only `main()` — all seven lifecycle functions are unchanged.

### 1. The protected-path scan ran too late, and was incomplete

**Defect.** The overlap check between the write paths (cache, output, receipt, shared
lock) and the protected roots ran *after* the shared owner lock had been opened and the
receipt tree created, and it covered only the declared source/serving roots. A refused
invocation therefore still took the shared lock and wrote files, and the guard and runner
files themselves were not protected.

**Evidence (`paths-RED01`).** `invoked: 1`, `files_unchanged_before_receipt: false`,
`passed: false` — the workload was invoked even though the output overlapped protected
`other-source/`.

**Fix.** Call the pinned read-only `build_trial.preflight_trial_paths(...)` **before**
opening the owner lock or creating receipts, pass the guard and the runner itself
(`a.guard.resolve()`, `Path(__file__).resolve()`) into the protected set alongside every
declared root, and record the accepted protected-root inventory in `invocation.json`.
Receipt/output containment is refused outright rather than resolved.

**Evidence (`paths-GREEN01`).** `invoked: 0`,
`system_exit: "Output/cache overlaps protected tree: .../other-source/run"`,
`files_unchanged_before_receipt: true`, `passed: true`, and the lock identity is
unchanged.

### 2. The pre-invocation hash scan accepted stale helper bindings

**Defect.** Only two helpers (`build_trial.py`, `gpu_reference.py`) were hash-pinned, and
`--deferred-sha256` still accepted a second, historical deferred-script hash. A stale
helper/manifest binding could therefore be admitted and run.

**Evidence (`pins-RED01`).** The old-pin invocation did not produce the expected refusal
(`invoked: 0`, `system_exit: "2"`, `passed: false`).

**Fix.** Pin all four helper artifacts (`deferred_gpu.py`, `build_trial.py`,
`gpu_reference.py`, `source-manifest.json`, `scalar_reference.py`) and restrict
`--deferred-sha256` to the single final hash, refusing every old binding before the lock
or receipt is touched. Tampered helper/manifest files are rejected.

**Evidence (`pins-GREEN01`).** The valid invocation is admitted and the exactly-bound
package runs under the inert replacement command.

Also documented, and *not* a pre-stop scan defect: `red01-nested` was an initial
**harness** wrong-log-path assertion (not the credited B1 RED); `red02-nested` fixes only
that harness path and fails on the actual live orphan.

## Verification recorded in the source packages

- `../mixedk_three_stage/` — runner-repair overlay: 29/29 expected outcomes (28 GREEN
  plus one deliberately RED original-runner regression), every repair case with zero
  emergency workload signals and all recorded workload/supervisor PIDs gone.
- `integration/` — composed overlay that actually ran on the device:
  `suite-parent01/summary.json` reproduces 42 expected outcomes (22 process cases with
  one intentional original-runner RED, five prerequisite-refusal cases, 15 integrated CLI
  cases), plus `replay-parent01.json` proving strict git apply, exact runner bytes and all
  2,309 package files.

## Limits

- No GPU compilation, numerics, memory high-water bound or performance is proved here.
- The guard is a polling early-stop monitor, not kernel OOM containment. Fast allocation
  bursts, root-cgroup versus effective-cgroup policy and threshold calibration were not
  redesigned.
- Direct-child ownership alone does not guarantee arbitrary grandchildren; the tested
  guarantee relies on the added custody/subreaper layer. Adversarial namespace changes,
  uninterruptible kernel tasks, fork storms and killing the custodian itself are outside
  the proof — a privileged dedicated cgroup/job manager would be needed for more.
- The runner refuses to start when it already has children; it is not a reusable
  in-process library for applications with unrelated subprocesses or threads.
- Receipts may be incomplete after disk failure or simultaneous supervisor death. No
  synthetic successful result is ever written to hide a missing guard receipt.

## Files

- `guarded_deferred.py` — the repaired entrypoint.
- `runner-repair-README.md` — the runner-repair package's own README and runbook
  (preserved; this file supersedes it as the directory README).
- `seal.py`, `verify.py`, `process_fixture.py` — sealing, local verifier and the inert
  process fixture.
- `test_processes.py`, `test_preflight.py`, `test_cli_inert.py` — CPU process,
  preflight-refusal and inert-CLI tests.
- `integration/` — the composed overlay that ran on the device, with its own README,
  `runner-integration.patch`, tests, verifier and RED/GREEN evidence.
