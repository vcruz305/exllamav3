# GAPS.md — everything this package does NOT prove

The local verification in `verification/<label>/summary.json` is real but its
scope is narrow: **real Linux processes, real `/proc`/`pidfd`/`flock`
boundaries, real filesystem ownership, an ephemeral loopback HTTP server**. It
never touched a GPU, a model, torch, exllamav3, the network, the real runtime
root, or port 8096. Every item below is a production-specific fact that this
package either substitutes or cannot observe. Nothing here is hypothetical: each
gap says what was done instead, and what the device run must therefore confirm.

Read this file before authorising a device run. If an item below is not
confirmed on the host, the trial is not ready.

---

## G-A / G-B — no device, no model, no CUDA, no 8096 (structural)

* Every "server" is `inert_width_stack.py` (an HTTP responder over an ephemeral
  loopback port chosen by the kernel, with an assertion that it is never 8096).
* Every "guard" is the same inert script; every "launcher" is one of its modes.
  No `serve_native.py`, no `guard_uma.py`, no `start_quant_readback.py`,
  no exllamav3, no `torch`, no safetensors, no CUDA context is ever loaded.
* The `dso` identity in local receipts is a real mapping of the running server
  process — `libc.so` — because no `*.cpython-312-aarch64-linux-gnu.so` exists
  here. The sealed adapter's own inert fixture used the same stand-in.
* Consequence: nothing in this package validates the production launch argv,
  the model config, the EXL3 drafter build, UMA memory behaviour on GB10, or the
  real health document of a live serve.

## G-FIXTURE-EVIDENCE — the probe stream is synthesized

`fixture_width_trial.py` writes `events.jsonl` from
`_probe_records()`: **15 records whose kind/field names and native16 values are
copied from the sealed diagnostic's `emit()` calls** (installed, job_init,
input_metadata, input, assigned_pages, state, qkv ×2, native_cache_writes,
pristine_draft_head, native_samples, proposal_truncation, target_forward,
received, terminal).

* No real native16 probe capture exists anywhere in this package. The only
  captured evidence in the repository is the **failed** round-8 attempt
  (`round8-cost-width/mirror/events.jsonl`), which stopped before the neural
  path and is used here only as the RED fixture for the diag contract.
* Consequence: the *contract* is tested; the *values on device* are not. The
  first device run is what produces the first real native16 stream, and the
  sealed diagnostic may emit a kind `trial_contract.PROBE_KINDS` does not know —
  in which case `require_trial_state` refuses and the failure is loud, not
  silently accepted.

## G-PID-EXEC-RACE — pid.json may be written before the child has exec'd

Observed locally (one flake in ~10 integration runs): the sealed adapter's new
process identity check read `/proc/<server_pid>/cmdline` while the child was
forked but not yet `exec`'d, and the sealed `proc_snapshot` correctly refused
with `empty/unknown argv`.

* The inert stack now waits (bounded, 10 s) for a non-empty cmdline before
  writing `pid.json`, so the local suite is deterministic — but that is a
  **fixture** fix.
* The sealed `linux_adapter.py` and the production launcher are unmodified and
  have no bounded exec wait. **Unverified on the host**: whether the production
  `guard_uma.py` / `start_quant_readback.py` handshake ever exposes that window
  (it depends on the real fork/exec timing and the model load time).
* Consequence: if the first device preflight refuses with `empty/unknown argv`,
  that is expected behaviour, not corruption. Do NOT auto-retry: re-run the same
  stage explicitly (each invocation is a separate authorized act).

## G-WIDTH-APPLICATION — native16 is proven by source inspection, not on device

`dflash-native-width-study` (sealed, composed here as `width/candidate/*`) proves
by source inspection that the top-level and nested `block_size` 8 → 16 change
makes the source producer create **15 mask positions**, with 15 masks of right
bound 15, `taps [0,11,23,35,47]`, `tap_shift: 0`. `round8-width-harness-repair`
reproduced the expected failing regression (native8/request12 RED) and passed 20
new + 187 original CPU tests.

* Locally, the trial's `width.expect` geometry (input/state `[1,16,4096]`,
  q `[1,16,64,128]`, k `[1,16,8,128]`, samples `[1,16]`, 15 proposals, span 16)
  is only checked against the synthesized stream.
* **Nothing here validates that the on-device masks, taps, cache writes and
  proposal counts are the native16 shapes**, that acceptance improves, or that
  the target verification cost is unchanged.

## G-NO-PERF — no throughput claim of any kind

No tok/s, no acceptance rate, no draft-window efficiency, no memory
high-water mark, and no statement about the ~41 tok/s code / ~24 tok/s prose
baseline or the 60 TPS target can be derived from this package. The device run
is a **correctness and recoverability** trial first.

## G-DRVFS-PREDICATE — the mode predicate is bypassed for local fixtures only

`check_permissions` requires a file owned by root/self with no group/other write
bits. The Windows drive mounted into WSL (`/mnt/c/...`) reports **0777 for every
file**, so no fixture tree can satisfy it.

* `witness/inert_trial_io.py` rebinds `linux_adapter.check_permissions` **inside
  the witness process only** (the same thing the adapter's own local fixtures did
  with `mock.patch`). The sealed source is byte-identical (asserted in
  `verify.py` and `test_trial_linux`), and the CLI refuses the witness module
  unless the trial document says `inert_fixture: true`.
* Every local receipt therefore records `io_hook: witness.inert_trial_io`.
* **Unverified**: the production tree's real ownership/mode bits, and the
  production-specific profile gates (root of the artifact tree, `login`, `C12`).
  Those have local unit controls only; the adapter's parent review explicitly
  forbade treating them as device-validated.

## G-MEMORY-SAMPLE — headroom is a fixed local number

The witness answers `memory()` with a constant sample
(`available_gib: 110, cgroup_headroom_gib: 110, free_gib: 3, host_oom_kill: 0,
cgroup_oom: 0, cgroup_oom_kill: 0`). The sealed release/restore path uses
`require_load_headroom(..., memory_policy='uma')` and the pre-launch headroom
gate; on the real GB10 those numbers come from `/proc/meminfo` and the UMA
cgroup.

* **Unverified**: the real 128 GB unified-memory headroom policy, the actual
  reserve (`EXL3_UMA_RESERVE_MB: 8192` in the retained config), and the OOM
  counters. A device run must confirm the headroom gate passes with the real
  serve stopped, and that no OOM kill delta appears.
* Substituted locally in the same module: `guard_parent_pid()` (the inert guard
  is a child of the test process, not of init; `PR_SET_CHILD_SUBREAPER` makes
  the reparented guard's lineage match production).

## G-HEALTH-DOCUMENT — the health body shape is assumed

`trial_readiness` requires `HTTP 200` with a JSON body of
`{'healthy': true, 'requests': 0, 'max_active_requests': 1}`. That shape comes
from the round-9 live readback scripts (`parent_verify_round9_live.py`,
`parent_verify_round9b_live.py`), not from a width-trial serve.

* **Unverified**: that the real serve on 8096 returns exactly those keys with
  those types after a retained launch. If it does not, R5 refuses — correctly —
  and the operator must reconcile the difference explicitly rather than relax
  the check.

## G-TIME-REALISM — timeouts are inert-scale

Local config/harness timeouts are 10–30 s; a real retained launch of MiMo 2.50bpw
plus its 4.0bpw drafter is minutes on GB10, and the sealed adapter's own
`launch/ready/http` windows are the production ones. No local evidence supports
any particular production timeout value; the device documents carry explicit
numbers (`timeouts.stop/headroom/launch/ready/http`) that the operator chooses,
and nothing in this package validates those choices.

## G-PROTECTED-ROOTS — the inventory is operator-declared

`--protected-root` must be supplied (repeatedly) and
`--protected-roots-complete` must be declared explicitly; the trial's
`protected.mandatory_roots` must be a subset. The package cannot discover the
real protected set.

* **Unverified**: that an operator's inventory is actually complete. A missing
  root means a model directory could be classified as a removal destination.
  Compare the inventory against the real tree on the host, in writing, before
  authorizing anything.
* Round-9's protected-root inventory is deliberately **not** copied here.

## G-AUTHORIZATION-DEVICE-GATE — the host gate checks everything except liveness

`device_docs.py --check` validates the four documents against the sealed
contract, the pins link and the cross-document hashes **on any machine**.
What only the device can check (and the CLI does, inside
`trial_contract.validate_authorization`): `boot_id`, `uid`, the operator lease
(`flock` on `operator.lock`), `issued_at ≤ now ≤ expires_at`, `failure_at`
ordering, and ingress blocking.

* Local coverage is unit tests only. **Unverified**: a real lease held by a real
  operator on the host, and a real `boot_id`/`uid` match.

## G-RECOVERY-NO-RETRY — one launch, by construction

There is no retry, no backoff, no candidate launcher and no fallback path: a
refusal aborts the stage and requires a fresh authorized invocation. That is the
intended posture, but it means a transient failure (e.g. G-PID-EXEC-RACE, or a
port still in TIME_WAIT) costs one operator round trip. Nothing here implements
or validates a retry policy, and none may be added without a new review.

## G-WIDTH-LAUNCH-UNWIRED — this package does not perform the width attempt

The CLI's stages are `preflight`, `release`, `restore`. The native16 width
**launch** (installing the block-16 drafter config, starting the serve with
`ROUND8_WIDTH=1` / `ROUND8_WIDTH_OUT`, running the sealed diagnostic, collecting
`events.jsonl`) is performed by the already-audited harness launcher path —
composed here as read-only copies under `width/candidate/`, `width/baseline/` and
`width/retained/` — not by this package's CLI.

* Consequence: the trial's own execution is only partly wrapped. The package
  gates *before* the width attempt (preflight, release) and *after* it (restore
  + readiness), and it consumes the probe stream the attempt produced. The
  operator is responsible for running the width attempt exactly once, and for
  recording the outcome (`passed` / `failed` plus the observed error text) that
  the restore authorization must carry — a mismatch between the recorded outcome
  and the probe evidence on disk is refused, by design.

## G-INERT-STACK-DIVERGENCE — the test stack is a copy, not the thing

`inert_width_stack.py` mirrors the sealed `round8-width-recovery-adapter/
inert_stack.py` shape (three modes, `flock` guard, `LAUNCHED` receipt line,
STOP/`result.json` contract) and adds the width-probe writes. It is a separate
file: divergence is possible in principle and is covered only by this package's
local suite.

## G-ROLLBACK-COMPLETENESS — recovery is proven to the health door, not beyond

The restore sequence proves: identity of the old owner, exact-PID STOP, old
processes proven gone (no zombie accepted), bounded run removal, headroom, ONE
retained launch, authenticated `HTTP 200` health, and the R1–R12 readiness
readback (active pointer, pid record, identity, prior gone, health, retained
configuration, retained drafter configs, absence of the width env, pinned
identities, process inventory, memory, protected roots).

* **Unverified**: that the restored serve actually generates tokens at the
  expected rate, that the model is the same model, and that the width run left no
  other side effect (cache files, drafter artifacts, logs) outside the runtime
  root. Readiness is a door check, not a quality check.

## G-SINGLE-OPERATOR / G-LEASE — advisory exclusion only

`operator.lock` is an advisory `flock`. A second operator who ignores the
convention can still act on the same root. Nothing in this package enforces
exclusion against a human.

## G-LABEL-EVIDENCE — evidence is local and single-shot

`verify.py` requires a **fresh** label and refuses to reuse one
(`FileExistsError` on the evidence directory). Evidence under
`verification/<label>/` is local to this machine and is not published anywhere.
`final-local-verification.json` in the sibling package is not reproduced here:
the summary lives next to the run that produced it.

## G-INERT-FIXTURE-DIRECTORY — fixtures are written beside the package

Local tests create `fixture-*` / `device-docs-*` temporary directories inside
this package directory (the adapter's own fixtures did the same, to keep the
worktree self-contained) and remove them on exit. `verify.py` excludes them from
the mutation copy and asserts no leftover process; it does not assert that no
leftover *directory* exists. On a read-only or shared mount the fixtures must be
redirected first.
