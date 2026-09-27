# Branch notes — `exp/native16-width-trial`

This branch is the **record and the code** of the native-16 draft-width trial harness: the
fail-closed, local-only composition of the repaired width harness, the Linux recovery adapter
and the retained-launcher rollback. It is deliberately **not** a source change to the runtime —
the width attempt is a config/launch experiment, and nothing here alters a default.

## Status

- Verified in its own workspace: `verify.py` → exit 0, `result: PASS`, 44/44 pinned artifacts,
  71 upstream originals re-hashed, 2 sealed core pins, 18 source witnesses, source pin
  `ca4a880e8918e1985fd25e06c6aff561666d3f14`; fail-closed controls behave (no token → 1,
  token without switch → 1, `--stage deploy` → 2, tampered pin → refusal).
- The contract suites themselves are self-contained: run from this directory, the four
  modules complete and pass. `python3 -B -m unittest -q test_trial_contract test_device_contract
  test_device_docs test_trial_linux` → `Ran 63 tests … OK`, exit 0, on three consecutive runs
  after this branch was committed. One earlier run of the same command failed a single test and
  did not reproduce (4 of 5 runs clean); the package's own `GAPS.md` attributes that class of
  flake to `G-PID-EXEC-RACE`, but the failing assertion was not captured, so treat that
  attribution as plausible, not proven.
- **It does not run from this branch alone.** The pin gate is the first thing it does, and it
  reads the sealed upstream trees that this package verifies. Observed from this directory:

  ```
  {"result": "FAIL", "error": "RuntimeError('upstream input missing: dflash-native-width-study/receipt.json')"}
  exit=1
  ```

  Vendoring those trees here would make the package verify copies of itself instead of the
  originals, which is the one thing it must not do, so they are referenced, not shipped. To
  use the harness, recreate the trial workspace layout and let `input-pins.json` validate it
  on the next run:

  ```
  <work>/
    dflash-native-width-study/          sealed width study + receipt
    round8-width-harness-repair/        sealed repaired harness (its own fixtures)
    round8-width-recovery-adapter/      sealed recovery adapter (124 files)
    round8-width-trial-integration/     this branch's tree (science/trials/native16_width/)
  ```

## Before any device run

Four gaps in `GAPS.md` gate it, and two of them are hard blockers:

| gap | effect |
|---|---|
| `G-WIDTH-LAUNCH-UNWIRED` | the CLI does `preflight`/`release`/`restore`; the width attempt itself is still a manual step through the older launcher path, so there is no single authorized command. |
| `G-PID-EXEC-RACE` | the adapter's process-identity check can read `/proc/<pid>/cmdline` in the fork-before-`exec` window and refuse on empty argv (~1 in 10 inert runs). Fixed in the inert fixture only; **production guard/launcher ordering is unverified**. |
| `G-WIDTH-APPLICATION` | native-16 correctness on device is proven by source inspection only. |
| `G-FIXTURE-EVIDENCE` | the probe event stream is a synthesized contract fixture, not device telemetry. |

The remaining 23 entries in `GAPS.md` are production facts the inert layer cannot establish.

## Scope and size policy

- No GPU, model, torch, exllamav3 import, SSH, network service or deployment is used or
  required by anything here; it runs as real Linux processes with an inert CLI/server on
  loopback.
- `verification/` (run evidence, ~1.3 MB) and every `__pycache__` are excluded from the commit:
  they are generated, not authored. This branch adds 52 files / 578 KB.
- No credentials, keys, tokens or host addresses are present.
- No performance, acceptance-rate or TPS claim of any kind is made here, and no device run has
  happened.
