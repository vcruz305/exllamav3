# round8-width-trial-integration

A local-only, fail-closed **integration package** that composes the repaired
native16 width-harness artifacts with the sealed Linux recovery adapter, so that
a *single bounded device run* can later be authorized with one reviewable
package instead of a chain of undocumented scripts.

It is deliberately small at the top and strict at the bottom: one CLI
(`width_trial.py`) with three stages, four hashed documents, 44 pinned artifact
hashes, and no retry path anywhere.

**Read `GAPS.md` before believing anything about the device.** This package
proves local mechanics; it does not prove device behaviour.

---

## What it does

| stage | what happens | what it proves |
|---|---|---|
| `--stage preflight` | read-only: profile, documents, protected roots, retained inputs, the live owner's identity, that no *previous* width attempt is alive, and the probe state | the trial is authorised and the world is in the state the trial assumes |
| `--stage release` | verify identity → STOP the exact guard → prove both owned processes gone (a zombie is refused) → bounded run removal → headroom check | the retained serve is stopped cleanly, with no model running |
| `--stage restore` | the same STOP path for the width run → preserve the width outcome → **ONE** retained launch → authenticated `HTTP 200` health with `requests == 0` → readiness `R1…R12` | recovery is complete to the health door |

Between `release` and `restore` the **width attempt itself** runs (native16
drafter config + the sealed diagnostic), performed by the already-audited
harness launcher path, not by this CLI — see `RUNBOOK.md` Phase 4 and
`GAPS.md` G-WIDTH-LAUNCH-UNWIRED.

## What it is made of

Four layers, none of them a rewrite:

1. **Sealed, copied byte-for-byte** — `linux_adapter.py`, `core/` (`recover.py`,
   `width_controller.py`), `recover_linux.py`, `runtime-pins.json`,
   `trust.schema.json`, the `witness/` copies, and the harness's
   `width/candidate/*`, `width/baseline/*`, `width/retained/*`,
   `width/device_preflight.py`. Every copy is proven byte-identical to its
   upstream origin by `make_pins.py` and re-proven by `verify.py`.
2. **Trial policy and wiring** — `trial_contract.py`, `trial_io.py`,
   `trial_preflight.py`, `trial_readiness.py`, `width_trial.py`.
3. **Device-document gate** — `device_docs.py` + `device-docs/*.json`
   (templates generated from the live contract).
4. **Local proof** — `device_contract.py` + the four test modules + the inert
   Linux stack (`inert_width_stack.py`, `fixture_width_trial.py`,
   `witness/inert_trial_io.py`) + `verify.py` + `make_pins.py`.

`INTEGRATION.md` lists every pinned hash and every inert substitution.

## Run the local verification

Linux / WSL is required (real child processes, `/proc`, `pidfd`, `flock`, an
ephemeral loopback HTTP server). Nothing else is needed — no GPU, no model, no
torch, no network, no SSH.

    /usr/bin/python3 -B verify.py --label <fresh-label>

The label must be fresh (the evidence directory is created with
`exist_ok=False`); evidence lands in `verification/<label>/`, and the run prints
a machine-readable PASS/FAIL summary plus `summary.json`.

What it does, in order: re-hash the 44 artifacts and their 71 upstream origins →
re-hash the sealed runtime pins, the 18 source witnesses and the permission
predicate → run the **sealed harness suite (37)** and the **sealed adapter suite
(36) in place** and prove neither original tree changed → run the **candidate
suite (63)** → prove the inert integration left **no process behind** → build a
**fresh, relocated mutation** of this package's own probe-consistency seam and
prove the intended test goes RED → prove the CLI refuses tampered artifacts →
prove the CLI fails closed without authorization, without its switch, and with
an unknown stage → re-hash everything again.

## What the local verification actually demonstrates

* **Candidate suite 63/63** — `test_trial_contract` 20,
  `test_device_contract` 16, `test_device_docs` 13, `test_trial_linux` 14.
  `test_trial_linux` runs the **real CLI** (`/usr/bin/python3 -I -B`) end to end
  in both stages: real `flock` guards, real child servers on ephemeral loopback
  ports (never 8096), real `STOP`/`result.json`, a real single retained launch,
  and the full `R1…R12` readiness readback.
* **Sealed suites 37 + 36, run in place, against the composed copies**, with
  **162 harness files and 124 adapter files re-hashed unchanged** around them.
* **71 upstream originals** verified byte-identical before *and* after.
* **A fresh mutation** (the whole runtime closure copied to
  `verification/<label>/mutation-mutant/`, `require_trial_outcome_consistent`
  gutted, the mutant's own pins regenerated) reproduces the intended failing
  test; the unmutated package passes it.
* **CLI fail-closed controls** — tampered `linux_adapter.py` → `pinned artifact
  hash mismatch`; tampered `runtime-pins.json` → same; no
  `EXL3_WIDTH_TRIAL_AUTHORIZED=YES` → refusal; token but no
  `--authorize-width-trial` → refusal; `--stage deploy` → argparse exit 2.
* **Zero leftover inert PIDs** (`INERT_CLEANUP` evidence in the candidate run):
  every fixture guard exited with `requested_stop` and every PID was proven
  gone before teardown returned.

## What it does NOT demonstrate

Device facts only. No GPU, no CUDA, no exllamav3, no model, no 8096, no real
`events.jsonl` from a native16 run, no throughput or acceptance claim. `GAPS.md`
is the exhaustive, itemised list — start with G-A/G-B, G-FIXTURE-EVIDENCE,
G-WIDTH-APPLICATION, G-PID-EXEC-RACE and G-HEALTH-DOCUMENT.

## Layout

```
width_trial.py            the only authorized entry point (3 stages, no retry)
trial_contract.py         width-trial policy: field sets, geometry, authorization
trial_io.py               WidthTrialIO / ReleaseIO: sealed boundaries + state gates
trial_preflight.py        P1..P6, P4L/P4R - every gate recorded
trial_readiness.py        R1..R12 post-restore readback (read-only)
device_docs.py            host gate for the four device documents
device-docs/*.json        templates generated from the live contract
device_contract.py        sealed finite()/inp() bodies executed on CPU, no torch
verify.py                 local end-to-end verifier (fresh label per run)
make_pins.py              pin generator (refuses origin drift)
witness/inert_trial_io.py test-only inert substitutions (2 facts + DrvFS predicate)
inert_width_stack.py      test-only guard/server/launcher
fixture_width_trial.py    test-only real Linux state builder
test_*.py                 candidate tests (63)
linux_adapter.py core/ witness/ width/ runtime-pins.json   sealed copies
integration-pins.json     44 artifact pins
input-pins.json           71 preservation pins
INTEGRATION.md RUNBOOK.md GAPS.md
```

## Posture

* **Fail closed**: no implicit defaults, no silent fallback, no auto-retry, no
  candidate launcher, no local/inert switch in the CLI.
* **Preserve**: every original tree is read-only; `verify.py` re-hashes them
  before and after.
* **One launch**: the restore stage performs exactly one retained launch; a
  second would require a new review.
* **No tolerance was weakened** to make anything pass, and no assertion was
  deleted: the failing cases in the sealed suites (native8/request12 RED, the
  leaked-active-state baseline, the historical operand RED, the pre-repair
  hoist RED) are preserved and reproduced.
