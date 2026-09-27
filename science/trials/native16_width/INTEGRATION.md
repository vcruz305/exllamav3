# INTEGRATION.md — exactly what was composed, and every pinned hash

This package is an **integration of already-repaired artifacts**: it adds the
trial policy, the state/pointer wiring, the host document gate and the verifier
around code that was audited elsewhere. It changes no sealed byte — every copied
artifact is proven byte-identical to its upstream origin by `make_pins.py` at pin
time and by `verify.py`/`width_trial.py` at run time.

* `integration-pins.json` — 44 artifacts: the runtime closure the CLI loads,
  every sealed source copied into this tree, and the device-document templates.
  Each entry carries `sha256`, `bytes`, `origin` (the upstream path) and
  `origin_sha256`.
* `input-pins.json` — 71 upstream inputs (files read for context, plus every
  copied artifact's origin) with their hashes: the preservation record proving
  the original trees were not modified.
* `runtime-pins.json` — the sealed adapter's own load pin, kept intact: the CLI
  refuses to import `linux_adapter.py` unless its bytes match.
* `source_pin` — `ca4a880e8918e1985fd25e06c6aff561666d3f14` (shared with the
  sealed adapter and the trial document).

## Composed from

| tree | role |
|---|---|
| `round8-width-harness-repair` | the repaired candidate/ baseline diagnostic + controller + recovery + deferred device preflight, and the `source/` witnesses |
| `round8-width-recovery-adapter` | the sealed Linux recovery adapter, its core, its witness copies and its production template |
| `dflash-native-width-study` | the native16 config-as-differential proof and its expected values |
| `round8-width-parent-audit` | the two parent reviews whose gap lists this package closes or names |
| `round9-three-stage-gpu`, `round9b-three-stage-timing` | read-only shape reference for the controller/readiness/closeout pattern |
| `round8-cost-width` | read-only: the real failed attempt (`mirror/events.jsonl`) used as the RED fixture |
| `mixedk-three-stage-trial-integration` | read-only: the sibling package whose packagine shape this follows |

## The 44 pinned artifacts

| artifact | sha256 (16) | origin |
|---|---|---|
| `core/recover.py` | `ab2bd9a4e646c085…` | round8-width-recovery-adapter/core/recover.py |
| `core/width_controller.py` | `c80b438e78b6a917…` | round8-width-recovery-adapter/core/width_controller.py |
| `device-docs/authorization.json` | `31557d1d48f963ca…` | (in-package template) |
| `device-docs/config.json` | `fd93439aee4fbb60…` | (in-package template) |
| `device-docs/receipt.json` | `8bbe36d00471ad2a…` | (in-package template) |
| `device-docs/trial.json` | `64938bf5aa3d14d0…` | (in-package template) |
| `device_contract.py` | `8bb8a1767281d91e…` | (in-package source) |
| `device_docs.py` | `ffefe397620a8b4d…` | (in-package source) |
| `fixture_width_trial.py` | `089133c00b8ff494…` | (in-package source) |
| `inert_width_stack.py` | `36fa29467053e50c…` | (in-package test stack) |
| `linux_adapter.py` | `af775cdc2011ee4a…` | round8-width-recovery-adapter/linux_adapter.py |
| `production-template.json` | `31864c8b10936af9…` | round8-width-recovery-adapter/production-template.json |
| `recover_linux.py` | `8f847a35da01fa7b…` | round8-width-recovery-adapter/recover_linux.py |
| `runtime-pins.json` | `906eeac92d38ce18…` | round8-width-recovery-adapter/runtime-pins.json |
| `test_device_contract.py` | `2460c25353c37992…` | (in-package test) |
| `test_device_docs.py` | `f6eec6e2ddfdb88d…` | (in-package test) |
| `test_trial_contract.py` | `a07b67632921e33c…` | (in-package test) |
| `test_trial_linux.py` | `c08fb71733934338…` | (in-package test) |
| `trial_contract.py` | `b6b07d73fe9a54ce…` | (in-package source) |
| `trial_io.py` | `0ac22018566502bf…` | (in-package source) |
| `trial_preflight.py` | `670fef09f8071f0b…` | (in-package source) |
| `trial_readiness.py` | `0dfe05fcf905a712…` | (in-package source) |
| `trust.schema.json` | `9b7785a5dd3eed74…` | round8-width-recovery-adapter/trust.schema.json |
| `width/baseline/failed-server.log` | `bb656982c0fb4a00…` | harness-repair/baseline/failed-server.log |
| `width/baseline/recover.py` | `b8863ad517d18b70…` | harness-repair/baseline/recover.py |
| `width/baseline/width_controller.py` | `9d85dbd409b91e68…` | harness-repair/baseline/width_controller.py |
| `width/baseline/width_diag.py` | `16251941215aead7…` | harness-repair/baseline/width_diag.py |
| `width/candidate/width_config.json` | `259759b665991376…` | harness-repair/candidate/width_config.json |
| `width/candidate/width_contract.py` | `59889cfc4fcc818d…` | harness-repair/candidate/width_contract.py |
| `width/candidate/width_diag.py` | `7f537742bf2de7b5…` | harness-repair/candidate/width_diag.py |
| `width/candidate/width_expected.json` | `316058d2a67d8837…` | harness-repair/candidate/width_expected.json |
| `width/device_preflight.py` | `4e56a24895ce7cff…` | harness-repair/deferred_device_preflight.py |
| `width/retained/failed_launch.json` | `31b046b1dd2c246a…` | harness-repair/source/failed_launch.json |
| `width/retained/retained_launch.json` | `a9bbcf2ff1cefd79…` | harness-repair/source/retained_launch.json |
| `width/retained/retained_launcher.py` | `7441d4e0de981462…` | harness-repair/source/retained_launcher.py |
| `width_trial.py` | `bd053b406734c52c…` | (in-package CLI) |
| `witness/failed_launch.json` | `31b046b1dd2c246a…` | recovery-adapter/witness/failed_launch.json |
| `witness/guard_uma.py` | `eef5706b327cf200…` | recovery-adapter/witness/guard_uma.py |
| `witness/inert_trial_io.py` | `8a1d5cd3ff4992e3…` | (in-package test witness) |
| `witness/retained_launch.json` | `a9bbcf2ff1cefd79…` | recovery-adapter/witness/retained_launch.json |
| `witness/retained_launcher.py` | `7441d4e0de981462…` | recovery-adapter/witness/retained_launcher.py |
| `witness/retained_server.py` | `205ee1c08d7a7763…` | recovery-adapter/witness/retained_server.py |
| `witness/source-provenance.json` | `40b9440541ee685e…` | recovery-adapter/witness/source-provenance.json |
| `witness/supplementary-provenance.json` | `22e10da26cf2422a…` | recovery-adapter/witness/supplementary-provenance.json |

Two artifacts appear twice with identical hashes under different paths on
purpose: `witness/retained_launch.json` and `width/retained/retained_launch.json`
(`a9bbcf2f…`), and the two `retained_launcher.py` copies (`7441d4e0…`). The
sealed adapter pins the witness path (`witness/retained_launch.json`,
`a9bbcf2ff1cefd796abb241a876b560a63b35e5304f12078423bb80d711da32a`); the harness
tree pins the source-mirror path. Both are verified, and
`linux_adapter.validate_production` re-checks the witness one at run time.

## Sealed runtime pins re-checked by this package

| pin | value | checked by |
|---|---|---|
| adapter load pin (`runtime-pins.json`) | `af775cdc2011ee4acffb6a8441c6a7e9bc693fa960ee8cc48eaf106c29f1b416` | `recover_linux.py`, `width_trial.py` (all 44 artifacts before anything else) |
| `CORE_HASHES['recover.py']` | `ab2bd9a4e646c0852bc3aa3507014db516356ba0da49beec0a47ca4ab0420645` | `verify.py` |
| `CORE_HASHES['width_controller.py']` | `c80b438e78b6a917580dee55841d300cb06b209e413f5e246a919de9241b3d7f` | `verify.py` |
| `PRODUCTION_PINS['guard']` | `eef5706b327cf20007dd49087b6d70b97ec178fcdbde616374188a0bc916d24d` | `verify.py` vs `witness/guard_uma.py`; `validate_production` |
| `PRODUCTION_PINS['launcher']` | `7441d4e0de981462c03f05cca38be81ea3bb7c5965d4b83cd96e582d9147b515` | `verify.py` vs both launcher copies; `validate_production` |
| `PRODUCTION_PINS['server']` | `205ee1c08d7a7763522aa7b269eddeb6a8ebf5ae2674532d0c6d62ce89cb2d83` | `verify.py` vs `witness/retained_server.py` |
| `PRODUCTION_PINS['source']` | `61653605b432350db3cccddbaa00b12f121e38fa2b072fc9de2032adf9e2a706` | `verify.py` (pin constant intact) |
| `PRODUCTION_PINS['dso']` | `02b0ae5bc8414d335facca41083f561cf24f41d56ef6e1e73a8b41fb16f80207` | `verify.py` |
| source provenance (`witness/source-provenance.json`) | `pin ca4a880e…`, 18 files with `blob_sha256` | `verify.py` re-hashes `round8-width-harness-repair/source/<path>` |
| native16 width config | `259759b66599137681e9ef2c62b4047a2331f21466ca5dd33276b0e2ce8efd52` | `trial_contract.WIDTH_CONFIG_SHA256`; the device template uses it verbatim |
| sealed permission predicate | `st_mode & 0o022` still present in `linux_adapter.py` | `verify.py` (source assertion) |

## What this package adds (the integration surface)

1. **`trial_contract.py`** — the width-trial policy: exact field sets, canonical
   path rules, the native16 geometry (`native_block 16`, `retained_block 8`,
   15 proposals, span 16, taps `[0,11,23,35,47]`, `tap_shift 0`), the probe
   vocabulary, and the authorization rules (intent, operator lease, boot id,
   uid, time window, stage↔phase pairing, `ROUND8_WIDTH`/`ROUND8_WIDTH_OUT`
   never inherited by the CLI process).
2. **`trial_io.py`** — `WidthTrialIO(LinuxIO)`: the sealed adapter's real
   filesystem, `/proc`, `pidfd`, `flock`, `urllib` and `subprocess` boundaries,
   plus the trial state gates. It overrides exactly four things: the width-run
   state check (launch config present, no STOP/result), `launch_retained` (one
   launch only, protected-root containment re-checked on the *new* run),
   `require_trial_state` (the probe evidence must be on disk immediately before
   the launch), and the width-run/probe paths. `ReleaseIO` makes
   `launch_retained` a hard failure: the release stage can never start a model.
3. **`trial_preflight.py`** — the stage-split preflight (`P1…P6`, `P4L`/`P4R`),
   every gate recorded with its evidence, no optional checks, no defaults.
4. **`trial_readiness.py`** — the post-restore readback in the round-9 live
   shape: `R1…R12`, read-only, no CUDA init, no inference, no writes; the same
   code path runs on the host and in tests.
5. **`width_trial.py`** — the single authorized entry point
   (`--stage preflight|release|restore`), requiring the env token **and**
   `--authorize-width-trial`, `-I -B`, the four hashed documents, an explicit
   protected-root inventory, and all 44 artifact pins. It exits non-zero on any
   refusal and never retries.
6. **`device_docs.py` + `device-docs/*`** — the host-side gate for the four
   device documents, with templates generated from the live contract.
7. **`device_contract.py` + `test_device_contract.py`** — the sealed
   diagnostic's `finite`/`inp` bodies AST-executed on CPU with a `MockTensor`
   shim (no torch), including the historical device failure reproduced as RED,
   plus the deferred device preflight's own error paths driven through a stub
   `torch` module.
8. **`inert_width_stack.py` + `fixture_width_trial.py` + `witness/inert_trial_io.py`**
   — the real Linux test stack, the state builder, and the two inert
   substitutions (documented in `GAPS.md`).
9. **`verify.py` + `make_pins.py`** — the local end-to-end verifier and the pin
   generator.

## Inert substitutions (complete list)

`witness/inert_trial_io.py` changes exactly two device facts, both listed in
`GAPS.md`:

* `memory()` → a fixed headroom sample (`available_gib 110`, `cgroup_headroom_gib
  110`, `free_gib 3`, no OOM counters) because the real policy needs ~110 GiB;
* `guard_parent_pid()` → the calling process, because the inert launcher is a
  child of the test process; `PR_SET_CHILD_SUBREAPER` makes the reparented inert
  guard's lineage match production.

It also rebinds `linux_adapter.check_permissions` inside the witness process,
because the Windows drive mounted into WSL reports `0777` for every file. The
sealed source is untouched (asserted), the CLI refuses this module unless the
trial document says `inert_fixture: true`, and every hooked receipt records
`io_hook`. Everything else — `/proc` identity, pidfd absence, flock ownership,
listener inode, HTTP readiness, STOP/`result.json`, run-based removal — is the
sealed adapter's unchanged path.

## How to re-derive the pins after an intentional change

    /usr/bin/python3 -B make_pins.py          # rewrites integration-pins.json + input-pins.json

It refuses to write anything if a copied artifact differs from its origin, so an
accidental edit to a sealed copy cannot be papered over. `make_pins.py` honours
`MIMO_TUNE_ROOT` so a relocated copy (used for the fresh-mutation RED run) can
still pin against the real upstream trees.
