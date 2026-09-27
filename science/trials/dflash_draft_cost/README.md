# DFlash draft-window cost selector (experiment record)

**Status: opt-in, default-off, source/CPU verified only. Do not enable it live.**

## What the selector does

DFlash decode chooses a draft window per round — how many speculative tokens the drafter
proposes before the target verifies. This change adds an external, measured cost profile
so that choice can be made from recorded whole-round costs instead of a fixed setting:
`DFlashCostPolicy.from_environment` reads a profile, and at each round the generator asks
the policy for a window and applies it to the existing cut computation. The incumbent
behavior is untouched when the policy is absent.

## Opt-in contract

- **Default off.** The policy is constructed only when `EXL3_DFLASH_COST_AWARE=1`;
  a value other than `0` or `1` is a hard `ValueError`, and any initialization error is
  treated as initialization-only (the policy stays `None` and the original window logic
  runs).
- The profile comes from `EXL3_DFLASH_COST_PROFILE` (with an optional
  `EXL3_DFLASH_COST_PROFILE_SHA256` pin).
- Changed paths: `exllamav3/generator/draft_cost.py` (new) and
  `exllamav3/generator/generator.py`.

## The recurrent-SWA repair

The original candidate **rejected the real retained MiMo configuration**. Its constructor
fixture used a *non-recurrent* model, so the blanket `recurrent_cache is not None`
exclusion looked correct in test but refused the actual recurrent sliding-window profile
on the device.

The repair replaces that blanket exclusion with:

- an **exact class-identity and geometry allow-list** — model/state class identities,
  parsed geometry, the loader's cache-construction code, and separate paged versus
  recurrent cache collections, instead of a single "is there a recurrent cache" test; and
- a **per-round checkpoint-proximity fallback**, so a round that is not eligible for the
  cost policy falls back to the original behavior at round level rather than failing
  initialization.

The vertical RED/GREEN slices were kept separate: one proves the retained recurrent-SWA
configuration now initializes and gets a policy, the other proves an ordinary round
actually selects *k* and reaches the full native producer. A paged Q4 invariant is never
applied to half-precision SWA rings.

## What is actually verified

Source and CPU only:

- **254 tests per tree**, and **3219 + 956 sealed inputs unchanged**.
- Baseline, original-candidate, candidate, applied and incremental trees were each
  compared for exact byte equality after applying both the pristine-pin (`combined.patch`)
  and incremental patches under LF-safe Git settings.
- No real `torch` or `exllamav3` import, no GPU, and **no quality or TPS evidence at all**.
  Scalar agreement in the CPU harness is not real GPU cache correctness.

## What must happen before it is enabled live

All of the following, in order, none of which exists yet:

1. Weight and runtime attestation for the exact retained profile, plus an independently
   verified live attestation (the harness's transient `TEST_ONLY` attestation is not that).
2. Device-side correctness for the recurrent ring: ring update, restore, and EOS handling
   on the actual hardware.
3. An **ordered cold/warm A/B/A** measurement against the same configuration.

Until those exist this branch is a record of the change, not a validated improvement.

## Honest note on the shipped context

`context.example.json` still carries **`attest_same_round8_weights_and_runtime: false`**.
The run it describes used a *transient, test-only* attestation of target-weight identity;
it is not a verified runtime identity and is not presented as one. The file is also
deliberately not renamed or promoted to a "verified" context.

## Contents

- `report.md`, `WORKFLOW.md`, `GPU_PLAN.md` — findings, the reusable local workflow and
  the deferred device plan.
- `profile.json`, `build_profile.py` — the pooled round8 whole-round cost profile and the
  offline builder that produces it (a provisional instrumented observational profile, not
  a clean live calibration).
- `context.example.json` — the trial context, including the honest attestation flag above.
- `retained_harness.py`, `safe_run.py`, `run_native_controls.py` — CPU harness that
  forbids real `torch`/`exllamav3` imports, plus the native control runner.
- `test_cost.py`, `test_profile.py`, `test_recurrent.py`, `test_parent_repro.py` — the
  CPU/host RED/GREEN slices (cost policy, profile pooling, recurrent eligibility,
  parent repro).
- `verify_repair.py` — the offline source-only verifier.
- `incremental.patch` — the repair delta relative to the original candidate;
  `combined.patch` is the pristine-pin patch and is the one this branch's source change
  was applied from.
- `evidence/` — the small `*.log` and `*-summary.json` receipts from the final CPU runs.

**Not shipped, deliberately:** `baseline.tar`, the `sealed-*.json` manifests, the
`candidate/`, `full-files/` and every `applied-*` / `incremental-applied-*` tree. Those
are large derived artifacts, and `verify_repair.py` needs them plus the wider mimo-tune
sibling layout (`dflash-cost-aware-candidate`, `dflash-pages-candidate/upstream`,
`dflash-pages-candidate/tests/run_suite.py`, `dflash-native-width-study`). Running
`verify_repair.py` from this directory alone therefore fails at
`HERE/'sealed-original-manifest.json'`; it is shipped as the record of how the
verification was performed, not as a self-contained runnable check.
