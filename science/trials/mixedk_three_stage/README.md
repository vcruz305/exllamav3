# Mixed-K MoE three-stage pipeline (experiment record)

**Status: opt-in, numerically exact, and a measured speed regression. Keep it off.**

This directory is the archived evidence package for an experiment on the mixed-K MoE
decode path. The source change itself lives on this branch (see the eight paths listed
below); everything here is the trial's own harness, fixtures and receipts.

## What the three-stage pipeline is

The incumbent mixed-K MoE entry launches its GEMMs as separate projection-Y CTAs. This
candidate instead fuses the work into three ordinary, current-stream-ordered launches,
each CTA owning one full-K column tile:

| Stage | Work | Grid per eligible-expert ordinal |
|---|---|---|
| 0 | Original gather and independent gate/up input Hadamards | x=1, y=1, 256 threads |
| 1 | Gate GEMM, independent up GEMM, then this tile's original GUAD epilogue | x=8, y=1, 512 threads |
| 2 | Down GEMM, then this tile's original output Hadamard / weighted fused-slot store | x=16, y=1, 512 threads |

Stage 1's GUAD consumes only raw gate/up outputs written by its own CTA, so it can be
fused at the epilogue. Stage 2 needs the full transformed intermediate across Stage 1
CTAs, so it stays a separate launch, as does Stage 0. No cooperative GEMV,
last-arrival counter, device-wide spin wait, group barrier or cross-CTA raw-output
consumption is introduced, and the caller's deterministic gather is unchanged.

## Opt-in contract

Both a Python flag and the native explicit entry are required: **`EXL3_MK_THREE_STAGE=1`**.

- Default **off**. The original `exl3_moe_mixedk` entry always calls the shared
  implementation with `allow_three_stage=false`, even if the environment flag is set.
- Eligible shapes are **decode q1..q8** only, with a narrow production guard: no
  `prefill` key, no measurement/TP warmup, authentic `routing_dots` identity, complete
  256-expert local layer, H4096/I2048/top8, gated SiLU/mul1, no biases, padding,
  slicing or latent projections, neutral scalars. The native guard additionally checks
  sm121, m16, deterministic slot pointers, contiguity/dtypes and N256/pipeline tuning.
- Anything unsupported **falls back to the original entry and kernel** — never to
  truncated work. q>8 and prefill are unaffected.
- The new `exl3_moe_mixedk_three_stage` binding keeps the existing 35-argument
  ABI (same types and order).

Changed paths: `exllamav3/exllamav3_ext/bindings.cpp`,
`quant/exl3_gemm_inner.cuh`, `quant/exl3_moe.cu`, `quant/exl3_moe.cuh`,
`quant/exl3_moe_kernel.cuh`, `quant/exl3_moe_three_stage.cuh` (new),
`exllamav3/modules/block_sparse_mlp.py`, `exllamav3/modules/mixedk_three_stage.py` (new).

## It is numerically exact

On device (400-record trial on the GB10 sm_121 box, 2.50 bpw pack, EXL3 4.0 bpw drafter,
Q4 KV, chunk4096):

- **52 records** captured.
- **25 flagged `five_three` / `off_original` identity checks: ALL true.**
- **max |delta| = 1.71e-05.**
- Numerical tolerances were left **untouched** — nothing was relaxed to make this pass.

The default-off path is bitwise identical to the original, and the three-stage path is
bitwise identical to the five-stage control where full-K geometry matches.

## It is a measured speed regression — do not enable it

The same 400-record device timing run shows the three-stage path is slower, not faster:

- `three` / `original` **median 1.096** across **16 q-widths**.
- **q8-hot 1.153** and **q8-spread 1.131**.
- **13 of 16 widths slower.**
- Same-semantics `off` vs `original` control noise: **0.32% median**, so the regression
  is well outside measurement noise.

Serial gate/up inside one CTA halves their tile-level parallelism relative to separate
projection-Y CTAs; saving two launches does not pay for it.

**Keep the default off. The original kernel won.**

## What is and is not proven

- CPU source, layout, ownership and preservation controls pass; the ownership
  enumeration executes the actual scanner/partition/epilogue expressions over 94
  fixtures and 194,048 projection-Had128 ownership instances.
- No CUDA compilation, no profiler stages, no memcheck/initcheck and no kernel-level
  race proof were executed. The synchronization arguments are source contracts.
- Triggering the flag is unsafe on shapes the guard does not cover.

## Running the checks locally

The suite needs the three snapshot trees, which are **not committed** (see the size
policy below). Materialize them first, then run:

```sh
sh science/trials/mixedk_three_stage/materialize_controls.sh          # rebuilds the trees
cd science/trials/mixedk_three_stage
python -B run_cpu_tests.py --json /tmp/cpu.json                       # 26 tests, no torch
python -B verify.py --label <unique-label> --upstream <pinned-clone>  # full verification
```

Order matters: `materialize_controls.sh` is required before either of the others, because
`run_cpu_tests.py` reads `tree/` and `verify.py` reads `original/` and `tree/`. Nothing
needs a GPU, CUDA or `torch`; both entrypoints actively refuse to import `torch` or
`exllamav3`. `verify.py --upstream` wants the pristine pin clone checked out at
`ca4a880e8918e1985fd25e06c6aff561666d3f14` with a clean `git status`.

`materialize_controls.sh` records `BASE_SHA` and self-checks the sha256 of each tree it
builds (`sha256` over sorted `path<TAB>blob-sha1` lines, `__pycache__` excluded):

| tree | rebuilt from | tree sha256 |
|---|---|---|
| `original/` | `git archive $BASE_SHA` | `11f05c4760d0563abca13665cf9dbf7a38045aa7f6244c5c58007557505fe7c4` |
| `tree/` | `git archive $BASE_SHA` with this branch's `exllamav3/` overlaid | `6be6822a24cecabeb0631fa9db044ece595f99da00a68f367d2c88b533196b3f` |
| `five-stage-control/` | `git archive $BASE_SHA` + `five-stage-control.delta.patch` | `048c8d56715fdd8ba29f7229b61bfdc700caf2bb6a5035335eb7580b8d43f0bf` |

## What is NOT in this branch / size policy

The branch stays source-sized. The two 21 MB control trees are **materialized from git
history, not vendored**; `tree/` is derived rather than stored because the candidate patch
only ever touches `exllamav3/`.

- `original/` and `five-stage-control/` are **not committed**. They are rebuilt by
  `materialize_controls.sh` from `ca4a880e8918e1985fd25e06c6aff561666d3f14` and are listed
  in `.gitignore`, so a local run leaves `git status` clean.
- `tree/` is not committed either — it is this branch's own applied source.
- `BASE_SHA` (`ca4a880`) and `origin/master` (`74b6f5a`) are byte-identical under
  `exllamav3/`, so basing this branch on master does not change what the trial compares.
- **`five-stage-control/` is the one thing git history alone cannot reproduce.** That
  experiment was never committed anywhere, so it is shipped as
  `five-stage-control.delta.patch` (17,202 bytes, sha256
  `e0f33f1c72c8f7a60490649ad1e41301a841cd1990614dcb18a3fdae5e600028`) — a 4-path delta
  against `BASE_SHA` (`exl3_gemm_inner.cuh`, `exl3_moe.cu`, `exl3_moe_kernel.cuh`
  modified; `exl3_moe_phased.cuh` added). The script applies it and verifies the result.
- `fixtures/` holds exactly **one** file, `fixtures/expert-byte-table.json`, because that
  is all `test_guard.py` reads at run time — and it needs the whole file. The test pins its
  bytes with `sha256(raw) == census['sha256']`
  (`b861cd38b192c35729b83178885d6682f4c8fcc950685e113fa81d1e53199983`) *and* walks all
  47 layers x 256 experts x 3 projections (36,096 entries) of its `trellis` shapes. It is
  **kept whole and undisguised**: trimming it would either break that pinned digest or
  force an assertion to be weakened, and it cannot be regenerated here because it is a
  census of the quantized MiMo pack headers on the device, which are not in git.
- `verify-*/` receipt directories and the `mixedk-three-stage.patch` that `verify.py`
  re-derives are not committed; both are generated, and the patch's change is already the
  branch's own committed diff.

This is the difference between the package as it ran and the package as it ships: same
tests, same assertions, same fixtures, minus the derived trees.

## Contents

- `REPORT.md`, `DEFERRED_GPU.md` — the candidate's own report and the deferred GPU plan.
- `verify.py`, `run_cpu_tests.py` — CPU verifier and stdlib-only test runner (both
  refuse to import `torch`/`exllamav3`).
- `test_candidate.py`, `test_guard.py`, `test_harness.py`, `test_layout.py`,
  `test_native_guard.py`, `test_ownership.py`, `test_preservation.py` — CPU contracts.
- `scalar_reference.py`, `gpu_reference.py`, `build_trial.py`, `deferred_gpu.py` — the
  bounded, isolated GPU harness (its `--plan`/`--help` paths are CPU-only).
- `materialize_controls.sh`, `five-stage-control.delta.patch`, `.gitignore` — rebuild the
  three snapshot trees from git history; the delta patch is the only one that history
  cannot supply. See the size policy above.
- `fixtures/`, `manifest.json`, `metadata-census.json`, `provenance.json` — inputs and
  derived metadata.
- `evidence/` — the small RED/GREEN and attempt logs that were under 8 KB (larger
  transcripts and every `verify-*/` run receipt were left out).

`bounded_deferred.py` is **not** the entrypoint to use; see `../guarded_runner/` for the
repaired bounded-trial runner.
