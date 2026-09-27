# Deferred trial supervision repair (LOCAL OVERLAY, NOT DEPLOYED)

Only this new directory was written. The old candidate, guard source, parent audit,
independent review and timing-coverage package were not edited. There was no GPU,
Torch/exllamav3 import, model load, compiler, SSH, network or deployment operation.

## Recorded final verification

`suite-final03/summary.json`: **29/29 expected outcomes**, comprising 28 GREEN
cases and one deliberately RED original-runner regression. Every repair process
case had no emergency harness workload cleanup and all recorded workload and
supervisor PIDs were gone. Raw logs are retained alongside the summary. The
native-Windows platform-refusal control also passed.

`manifest-release01.json` seals input/output hashes, the final runner hash, the
suite result and the original-root immutability comparison. The manifest excludes
itself. A fresh rerun must use a new label and a new manifest.

## What is repaired

`guarded_deferred.py` is a standalone replacement entrypoint. Do **not** launch
`bounded_deferred.py`, and do not wrap this entrypoint inside the old guard CLI.
The entrypoint loads the exact retained guard's `supervise()` function. Its actual
child is `[python, -B, SELECTED_PACKAGE/deferred_gpu.py, ...]`, not a timeout wrapper.
The guard itself supplies memory monitoring and `max_seconds`.

The direct-child change closes independent-review B1, but is insufficient for B2:
the unchanged guard can stop waiting when the group leader exits, leaving a
TERM-ignoring descendant. The overlay therefore adds narrowly scoped custody:

```text
entrypoint (waits; SIGINT/SIGTERM -> STOP file)
  custodian (dedicated Linux subreaper; watches entrypoint pidfd)
    exact guard worker (dedicated subreaper; calls unchanged supervise)
      actual deferred_gpu.py (guard's direct child/session leader)
        compiler/workload descendants
```

The worker drains adopted descendants on every handled return/exception. A
surviving custodian does the same if the worker dies, or requests STOP if the
entrypoint dies. Cleanup enumerates **only direct children of these dedicated
processes**, repeats as grandchildren are adopted, pins each signal target with
pidfd + `/proc` starttime, and reaps exact PIDs. No `pkill`, name-based matching,
all-host process scan, serving-process operation or host-global configuration is
used. Subreaper and signal settings belong only to the newly created processes.
The runner refuses pre-existing children; it is not a reusable in-process library
for applications with unrelated subprocesses or threads.

The custodian's last-resort deadline is `seconds + 3` from worker creation, or 2
seconds after detected entrypoint death/custodian cancellation; it drains for at
most another 3 seconds. This is a guard-failure backstop, not the old detached
workload timeout. Normal workload deadlines are enforced by the exact guard.
The overlay does not promise a kernel-hard absolute walltime under all failures.

## Exit and receipt contract

- Ordinary workload exit: preserve its status (`0`, `7`, etc.; negative signal
  status is mapped to `128 + signal`).
- Guard `time_limit`: **124**.
- Memory STOP, STOP file, SIGINT/SIGTERM, monitor failure, supervisor failure or
  incomplete cleanup: **125**.
- External SIGKILL of the entrypoint: its caller sees SIGKILL (`-9` in Python;
  normally 137 in a shell); surviving custody still writes its own receipt.
- Preflight refusal: nonzero before any Torch import.

`RECEIPT/invocation.json` binds selected paths, hashes and operator confirmations;
`headroom.json` records actual prelaunch headroom. Under `RECEIPT/guard/`:

- `server.log`: preserved, combined workload stdout **and** stderr;
- `memory.jsonl`, `pid.json`, `result.json`: unchanged guard outputs when its
  execution reaches those writes;
- `guard-worker.log`: supervisor stdout/stderr;
- `exit.json`: worker result, primary error and its cleanup receipt;
- `custody-pids.json`, `custody.json`: supervisor identities and final custody
  result, including cleanup actions/remaining children;
- `failure.txt` / `custody-failure.txt`: available exception traces.

A killed guard cannot write its own `result.json`; the custody receipt deliberately
reports guard death instead of fabricating a guard result. Preflight failures
before receipt creation report to stderr; capture the outer invocation's stdout,
stderr and exit status too. Filesystem failure can prevent receipt persistence.

## Reproduce locally (safe, inert, CPU-only)

From Windows/Git Bash, use a **new** label:

```bash
wsl.exe --exec /usr/bin/python3 -B \
  '/mnt/c/Users/Victor Cruz/AppData/Local/hermes/cache/scratch/mimo-tune/mixedk-three-stage-runner-repair/verify.py' \
  --label my-new-proof
```

`verify.py` writes `suite-LABEL/summary.json` and raw per-case stdout/stderr.
Per-process observations are under `evidence/LABEL-CASE/receipt.json`. It includes
one expected RED against the byte-identical original nested runner, plus GREEN
repair/process/preflight/CLI cases. Every inert workload has a hard 12-second
SIGALRM expiry; the assertion observes it much earlier. The harness records the
state **before** its emergency finally cleanup. All repair cases require zero
emergency workload signals, absent/reaped workload PIDs, and gone supervisors.
Deliberate original REDs are cleaned using exact identities in finally.

Real Linux coverage includes memory-triggered stop, timeout, STOP file, guard
SIGTERM, monitor exception, normal/nonzero exit, TERM-ignoring descendants,
ordinary leader exit before its child, separate-session descendants, entrypoint
SIGKILL/SIGTERM/SIGINT, guard SIGKILL, raised KeyboardInterrupt, and a SIGSTOP-frozen
guard. Frozen-guard cleanup completes before the inert fixture expires.

`test_cli_inert.py` exercises real entrypoint preparation for both selected
packages, then explicitly substitutes only (1) memory samples and (2) the prohibited
GPU command with the inert fixture. It records both commands and does **not** claim
to execute the GPU script. `test_preflight.py` installs a forbidden GPU import hook.
Native Windows refusal is additionally recorded at
`evidence/native-windows-platform/`.

### Strict RED/GREEN trail

- `red02-nested` -> `green01-direct`: original actual guard/nested runner leaves
  the inert workload alive; direct guard child does not.
- `red03-descendant` -> `green02-descendant`: direct child alone still leaks a
  TERM-ignoring descendant; subreaper cleanup fixes that.
- `red04-wrapper-death` -> `green04-wrapper-death`: worker cleanup alone dies with
  its caller; surviving custody fixes this. That RED's source is archived as
  `evidence/red04-wrapper-death/runner-under-test.py`.
- `red05-cli-authorization` through `red10-cli-valid`: vertical CLI authorization,
  platform, ownership, hash, headroom and valid invocation slices, with matching
  numbered GREEN receipts.
- `red11-timing-overlay` -> `green12-timing-overlay`: explicit immutable package
  binding, rather than silently accepting a changed workload script.
- `red01-nested` is an initial **harness** wrong-log-path assertion, not the credited
  B1 RED. `red02` fixes only that harness path and fails on the actual live orphan.
- Initial manifest creation succeeded but its Windows-path-key print raised a
  KeyError; the subsequent independent hashes succeeded. Neither incident changed
  any input artifact.

## Exact future invocation (NOT AUTHORIZED OR EXECUTED HERE)

This command is a future Linux owner runbook, **not permission to run a GPU test**.
Set `R` to the absolute directory containing the immutable artifacts on the
independently authorized Linux GPU host. Set `RUN` to a fresh writable directory
outside every protected source/package tree. `OWNER_LOCK` must be the real shared
owner lock used by all competing GPU/model launchers, not an arbitrary new file.
The stopped-model and sole-owner state must be independently verified by that
owner immediately before invocation. A lock only coordinates cooperating users.

For the completed timing-coverage package:

```bash
# Assign these absolute paths after a separate authorization; no SSH is implied.
R=/absolute/path/to/immutable/artifacts
RUN=/absolute/path/to/new-trial-receipt
OWNER_LOCK=/absolute/path/to/the-real-shared-owner.lock

EXL3_TRIAL_MODEL_STOPPED=YES EXL3_THREE_STAGE_AUTHORIZED=YES MAX_JOBS=1 \
python3 -B "$R/mixedk-three-stage-runner-repair/guarded_deferred.py" \
  --guard "$R/guard-host-oom-candidate/guard_uma.py" \
  --guard-sha256 eef5706b327cf20007dd49087b6d70b97ec178fcdbde616374188a0bc916d24d \
  --candidate "$R/mixedk-three-stage-timing-coverage/package" \
  --deferred-sha256 00500630f3f875610d9e9f70d0a80d8f4237f2cf42cb626f6152f08c3d7ad83e \
  --receipt "$RUN" --owner-lock "$OWNER_LOCK" --seconds 1800 --memory-policy uma \
  --sole-gpu-owner --model-stopped --headroom-verified \
  -- --mode numeric --cache "${RUN}-fresh-cache" --output "$RUN/trial"
```

No `EXL3_THREE_STAGE_GUARD_ACTIVE` assertion is consumed or needed. The memory
guard is the executing monitor, not an environment flag. UMA preflight retains the
exact guard's conservative 102-GiB available/cgroup and 2-GiB free load-headroom
rule; physical policy retains 110 GiB. This is stricter than a small fixture's
allocation budget and is not silently relaxed. In local WSL, the actual sampler
fails closed because `/sys/fs/cgroup/memory.events` is absent; process proofs
therefore explicitly inject samples and do not certify that sampler on a GPU host.

For the old package, only replace `--candidate` with
`$R/mixedk-three-stage-candidate` and `--deferred-sha256` with
`6eedae4b8c4afe86f340d58b63a4a9178d3c76cb7bee36cf4e02cb55eaa71b31`.
Use separate fresh receipts/caches for numeric, profile and timing modes and obey
the package's prerequisites. Existing build/GPU flags and MAX_JOBS/RTLD checks
remain in force. The script requires explicit writable paths outside the selected
package and uses the selected script's package-relative source roots.

**Integration with later parent overlays:** this release pins `build_trial.py`
(`c6f5bea9a9fa956824eac2efd7c88e0ceda18a1833c42e8be88c57702c17b1b0`)
and `gpu_reference.py`
(`b4864bf94989b5982847cf73baf260592ca5fe052f1a1b3c1553bfb92c4776ae`).
A later C1/cache-identity/partial-row package changing these helpers or the deferred
script must get an explicit reviewed pin update in a **new integration overlay**,
with the inert CLI binding tests rerun. Do not weaken/remove hash checks or modify
this overlay/source package in place. No C1/build/kernel fix is included here.

## Limits and remaining gates

- No GPU compilation, numerics, memory high-water bound or performance is proved.
- The independent review assessed the old runner, not this repair; this overlay
  needs its own review before runtime authorization.
- The unchanged guard is a polling early-stop monitor, not kernel OOM containment.
  Fast allocation bursts, root-cgroup vs effective-cgroup policy, original sampler
  corner cases and threshold calibration were not redesigned.
- Direct-child ownership alone does **not** guarantee arbitrary grandchildren;
  the tested guarantee uses the added custody/subreaper layer. Adversarial
  namespace changes, uninterruptible kernel tasks, fork storms and killing/freezing
  the custodian itself (or all supervisors together) are outside this proof. A
  privileged dedicated cgroup/job manager would be needed for stronger containment.
- Custody cleanup is bounded and reports remaining children with nonzero exit;
  it cannot promise a hung kernel task disappeared. No broad kill is used.
- Ownership/model-stopped flags are operator confirmations, not independent device
  discovery. Real sole GPU ownership, stopped model, actual headroom and the shared
  lock protocol remain external preconditions. This local task granted none of them.
- Receipts may be incomplete after disk failure or simultaneous supervisor death.
  No synthetic successful result is written to hide a missing guard receipt.

Reusable lesson (kept here to honor write-only-overlay scope): test the real
supervisor/workload process topology, not a guard-active flag; process-leader exit
is not descendant termination. Use exact identities and a dedicated surviving
owner, and prove the failure paths with short inert Linux processes before any GPU
workload. No user-global skill file was changed.
