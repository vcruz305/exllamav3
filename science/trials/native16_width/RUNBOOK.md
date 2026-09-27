# RUNBOOK.md — the ONE device sequence for the native16 width trial

Everything below is done by **one named operator who holds exclusive ownership**
of the host. Every command is a single authorized act: there is no retry, no
loop, no auto-recovery, and no command in this runbook may be run twice to "see
if it works". If a stage refuses, stop, read the refusal, fix the named cause,
and run the stage again as a fresh act.

Read `GAPS.md` first: it lists what this package cannot prove.

Placeholders used below:
* `PKG` — the package directory on the host, e.g.
  `/workspace/mimo-tune/round8-width-trial-integration`
* `PY` — `/usr/bin/python3` (Linux 3.12; the sealed CLI requires `-I -B`)
* `DOCS` — the directory holding your four filled documents

---

## Phase 0 — preconditions (no model, no GPU)

**0.1 Exclusive ownership.** Confirm in writing that you are the only operator,
that no other session can start, stop or reconfigure the serve, and that ingress
is blocked (`ingress_blocked: true` is part of the authorization; the CLI does
not check the network, you do).

**0.2 Stage the package.** Copy `PKG` to the host. Do **not** edit any file: the
CLI verifies all 44 pinned artifact hashes before it reads anything else, and
refuses on a single byte of drift:

    EXL3_WIDTH_TRIAL_AUTHORIZED=YES $PY -I -B width_trial.py --help   # exit 0

**0.3 Confirm the pinned production identities exist on the host.** These are
fixed in the sealed adapter (`PRODUCTION_PINS`); a mismatch is a refusal, not a
warning:

| role | path | pin |
|---|---|---|
| guard | `/workspace/mimo-tune/guard_uma.py` | `eef5706b…916d24d` |
| launcher | `/workspace/mimo-tune/round6-quant-readback/start_quant_readback.py` | `7441d4e0…47b515` |
| server | `/workspace/mimo-tune/round6-quant-readback/serve_native.py` | `205ee1c0…cb2d83` |
| source | `/workspace/mimo-tune/60tps-round3/python-shadow/exllamav3/modules/block_sparse_mlp.py` | `61653605…f9e2a706` |
| dso | `…/exllamav3_ext.cpython-312-aarch64-linux-gnu.so` | `02b0ae5b…6f80207` |

`verify.py` re-hashes the 18 pinned **source** witnesses too
(`witness/source-provenance.json`) when you run it on the host.

**0.4 Install the native16 drafter config, and leave the native8 one alone.**
The trial names two drafter configs with different paths and different hashes;
the sealed contract refuses if they are equal, and the restore re-verifies both
after the retained launch (readiness `R7`).

* native16 (width) config — must hash to the pinned
  `259759b66599137681e9ef2c62b4047a2331f21466ca5dd33276b0e2ce8efd52`
  (`trial_contract.WIDTH_CONFIG_SHA256`; the pristine copy is in the package at
  `width/candidate/width_config.json`).
* retained native8 config — untouched, hash recorded in your documents.

**0.5 Device preflight (micro, before anything else).** This is the repaired,
metadata-first, device-safe reference check. It imports `torch` itself
(deliberately deferred; module scope is torch-free) and runs the slice-assignment
contract over `(cpu, cuda:0) × (cpu, cuda:0)` output/mask devices and two dtype
pairs, then asserts that a wrong value and a NaN are still rejected:

    $PY -B $PKG/width/device_preflight.py --authorize-device-probe \
        --diag $PKG/width/candidate/width_diag.py

Expected: eight `{"status": "device_contract_pass", …}` lines and a final
`MICRO_PREFLIGHT_ONLY: not neural/page/stream/kernel correctness or performance
proof`. **This is a micro-preflight, not a width trial.** It does not prove the
native16 masks, the page/cache writes, or performance. Missing flag → argparse
error; CUDA unavailable → `RuntimeError('CUDA unavailable; do not load model')`.

---

## Phase 1 — author the four documents, then gate them on the host

    $PY -I -B $PKG/device_docs.py --write-templates $DOCS     # once; never overwrite
    # fill in $DOCS/{config,trial,receipt,authorization}.json by hand
    $PY -I -B $PKG/device_docs.py --check $DOCS               # must exit 0

`--check` refuses any leftover `<placeholder>`, any field that is not in the
sealed contract, a non-production config or trial (`inert_fixture` must be
`false`), a trial whose `integration_pins_sha256` is not this package's
`integration-pins.json`, and any cross-document hash mismatch. On success it
prints the four `--*-sha256` values and the exact command to run.

**The receipt is the trust anchor.** `receipt.json` must be the trusted
PRE-FAILURE capture of the live retained serve — PID/starttime/argv plus scopes
for exactly the two owned processes — not a reconstruction from memory. On this
host the only locally validated capture path is the recovery adapter's
(`recover_linux.py`); see `INTEGRATION.md` and `GAPS.md`
(G-AUTHORIZATION-DEVICE-GATE).

For the release stage the authorization carries
`{"width": {"status": "failed", "error": "not_run_before_staging",
"phase": "before_width"}, "stage": "release"}`. For the restore stage it carries
the **observed** outcome (`phase: "after_width"`, `status` `passed` or `failed`,
and the error text when failed) and `stage: "restore"`.

---

## Phase 2 — preflight stage (read-only)

    EXL3_WIDTH_TRIAL_AUTHORIZED=YES $PY -I -B $PKG/width_trial.py \
      --stage preflight --authorize-width-trial \
      --config $DOCS/config.json --config-sha256 <…> \
      --receipt $DOCS/receipt.json --receipt-sha256 <…> \
      --authorization $DOCS/authorization.json --authorization-sha256 <…> \
      --trial $DOCS/trial.json --trial-sha256 <…> \
      --protected-root <root1> --protected-root <root2> --protected-roots-complete \
      --preflight-out preflight-receipt.json

Checks and what they refuse on:

| check | refuses when |
|---|---|
| `P1_profile` | the config is not the production profile / not this source pin / not port 8096 |
| `P2_documents` | any document hash, pin, lease, boot id, uid or time window fails |
| `P3_protected_roots` | the declared inventory is incomplete, nested without acknowledgement, or a destination sits inside a protected root |
| `P4_retained_inputs` | the retained launch inputs / identity pins do not match the pinned witness |
| `P4L_live_owner` (release) | the live owner is not the retained run, its pids/starttimes/argv/scopes do not match the receipt |
| `P4R_live_width_owner` (restore) | the live owner is not the width run with `ROUND8_WIDTH=1` + `ROUND8_WIDTH_OUT`, or the run already has STOP/result |
| `P5_previous_width_attempt` | any process (other than the current owner) is still running with `ROUND8_WIDTH=1`, or a previous width attempt is not proven dead |
| `P6_probe_state` | a probe path already exists before staging (release), or the probe evidence is absent/incomplete/inconsistent (restore) |
| — | container guard: `ROUND8_WIDTH` / `ROUND8_WIDTH_OUT` must not be present in this process's environment |

Expected: `"status": "preflighted"`, exit 0, and a written preflight receipt.
**Abort branch:** any refusal → do not proceed. Fix the named cause and re-run
the same stage as a fresh act. Never "continue anyway": the release stage will
re-check and refuse identically.

---

## Phase 3 — release stage (stop the retained serve, bounded)

Keep the run id, both PIDs, both starttimes and the argv of the live serve
written down **before** this stage; they are your recovery anchor.

    EXL3_WIDTH_TRIAL_AUTHORIZED=YES $PY -I -B $PKG/width_trial.py \
      --stage release --authorize-width-trial <same document arguments>

Sequence, all inside the sealed recovery core: verify identity → write the STOP
record for the exact guard → wait (bounded) for both owned processes to be
**gone** (a zombie is refused, not accepted) plus the guard's `result.json` →
bounded run removal → headroom check.

Expected: `"status": "released"`, `restore.report.processes_gone` listing both
old PIDs, `memory` with the real headroom sample, and **no model running**.

**Abort/restore branches**

* `released process still present` / `empty/unknown argv` / `zombie is not
  proven gone` → the sealed core refuses and hands off
  (`… keep exclusive ownership, inspect exact PID/starttime/argv and request
  state; do not broad-kill or retry candidate`). Do exactly that: inspect the
  exact PIDs yourself, do **not** pkill, do not restart the serve. Then decide
  whether to re-run `--stage release` (a fresh authorized act) or to abandon the
  trial and restore the serve by the retained procedure.
* Headroom refusal → the host is not ready; free memory under the operator's own
  control and re-run. Never edit the threshold.
* After a successful release the width attempt has not happened yet; the only
  legal next actions are Phase 4 or an explicit restore.

---

## Phase 4 — the ONE width attempt (native16)

Performed by the already-audited harness launcher path, not by this package's
CLI (see `GAPS.md` G-WIDTH-LAUNCH-UNWIRED): install the native16 drafter config,
start the serve with the sealed diagnostic enabled
(`ROUND8_WIDTH=1`, `ROUND8_WIDTH_OUT=$ROOT/round8-cost-width/width/ndt1`), run
exactly one bounded generation, and let `events.jsonl` accumulate in the probe
directory named by the trial.

* The probe directory must not pre-exist (Phase 2 proved it did not).
* Record the exact outcome: `passed`, or `failed` **with the observed error
  text**. The restore stage refuses a recorded outcome that contradicts the
  probe evidence on disk.
* Do not restart the width serve "to get a clean stream": the diagnostic
  APPENDS, and a second attempt invalidates the evidence.

---

## Phase 5 — restore stage (ONE retained launch, then readiness)

    EXL3_WIDTH_TRIAL_AUTHORIZED=YES $PY -I -B $PKG/width_trial.py \
      --stage restore --authorize-width-trial <same document arguments, new hashes>

Sequence: the same STOP path as Phase 3 for the width run → the width outcome is
preserved independently of recovery → **exactly one** retained launch from the
pinned launcher → wait for authenticated `HTTP 200` health with `requests == 0`
→ readiness `R1…R12`.

Expected: `"status": "restored"`, a status receipt in the receipt directory, and
`checks` naming all twelve: `R1_active_pointer`, `R2_pid_record`, `R3_identity`,
`R4_prior_gone`, `R5_health`, `R6_configuration`, `R7_drafter_configs`,
`R8_no_width_env`, `R9_identities`, `R10_process_inventory`, `R11_memory`,
`R12_protected_roots`.

**Abort/restore branches**

* Any refusal during the restore leaves `rollback: "failed"` with
  `rollback_error` and the stage exits non-zero. The width evidence is kept.
  Inspect the refusal before doing anything else; the most likely real-host
  causes are the health document shape (`GAPS.md` G-HEALTH-DOCUMENT), the
  headroom gate, or a port still in TIME_WAIT.
* `retained_not_ready` → the serve answered something other than
  `200/healthy/requests==0`. Record the exact body. Do **not** launch a second
  time from this package: a second launch is out of scope for this trial and
  requires a new review.
* `R10_process_inventory` → a process other than the two owned ones is under the
  runtime root. Identify it before stopping anything.
* `R12_protected_roots` → a protected root changed. Stop and find out why; the
  trial's removals are supposed to be bounded to the run directory.

---

## Phase 6 — close-out

1. Re-read health (`200`, `healthy`, `requests == 0`) and record it with the
   restoration receipt.
2. Archive, per run: the probe `events.jsonl`, the status receipt
   (`receipts/width-trial-status-<sha>.json`), both authorization documents, and
   the preflight receipt.
3. Measure throughput **separately** if you want it. Nothing in this package
   makes a performance claim (`GAPS.md` G-NO-PERF).
4. Run the local verifier on the host once, with a fresh label, to preserve the
   package's own evidence next to the run:

       $PY -B $PKG/verify.py --label device-<date>

   It expects a *fresh* label and refuses to reuse one.

---

## Failure taxonomy at a glance

| symptom | meaning | action |
|---|---|---|
| `pinned artifact hash mismatch: <file>` | the package tree was edited | restore the file from the pinned copy; never edit and re-pin to "make it pass" |
| `EXL3_WIDTH_TRIAL_AUTHORIZED=YES is required` | no authorization act | set the env token *and* `--authorize-width-trial` |
| `explicit trusted document hash mismatch` | a document changed after `--check` | re-run `device_docs.py --check` and use its printed command |
| `authorization/config hash mismatch` | the authorization pins different document bytes | regenerate the authorization from the final documents |
| `the inert IO hook is refused for a production trial document` | someone tried to run the device stage with the test witness | never do this; the witness is test-only |
| `identity_changed` | the live PID/starttime/argv moved under you | stop; you no longer own the exact processes you hold a receipt for |
| `empty/unknown argv` | a PID was read before exec (see G-PID-EXEC-RACE) | re-run the same stage once, explicitly — no auto-retry exists |
| `probe evidence missing` | the width attempt produced no stream | the recorded outcome must be `failed` with the error text, and the probe directory must still exist |
