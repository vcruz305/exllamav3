# Mixed-K handled-set readback elision (experiment record)

**Status: opt-in, default-off, CPU-reviewed. On device it is a real but tiny decode
gain. The 60 tok/s target was not reached, and this change does not reach it.**

This directory is the archived review package for a one-line-class change to the mixed-K
MoE decode path. The source change is on this branch; everything here is the review's own
harness, patches, receipts and audit notes.

## What the change is

`exllamav3/modules/block_sparse_mlp.py` gains one module constant and turns one `else:`
into `elif`:

```python
MIXEDK_ELIDE_HANDLED = os.environ.get("EXL3_MOE_MIXEDK_ELIDE_HANDLED", "0") == "1"
...
-                        else:
+                        elif not MIXEDK_ELIDE_HANDLED:
```

That `else:` branch builds `mixedk_handled` by `.nonzero().tolist()` — a device readback
that forces a host sync. It only runs when the unified mixed-K path is taken at all, and
when `expert_count_list is None` the fallback loop it guards executes **zero iterations**,
so the set it builds feeds no consumer. The elision removes exactly that readback's sync
and nothing else.

Deliberately **not** included: eliding the `counts_fused` readback. Removing
`_ec[_m].tolist()` as well is a different change, because the C++ mixed-K wrapper sizes
`num_groups = min(concurrency, MOE_MAX_GROUPS, num_active)` for a positive active count and
derives group width from `target_blocks/num_groups`; substituting an estimate (or `-1`)
can change reduction geometry. No estimate is substituted here.

Unchanged: `counts_fused` readback, the exact active count, launch arguments, MTILE
tiering, scratch and gather tables, buffers, kernels, capacities and draft windows.

## Opt-in contract

- **Default off.** The environment variable must be set before importing the module. Any
  value other than exactly `1` keeps both original readbacks.
- Fixing the path to `exllamav3/modules/block_sparse_mlp.py` alone is what this branch
  contains; the review's own claim is "there is no claim of numerical identity or speedup
  on a GPU" from the CPU side.

## The one device measurement (A/B/A)

Round 6 single-stream bracket, reported by the device session in
`round6-quant-readback/mirror/comparison.json` (A and A2 are the unmodified arm,
`EXL3_MOE_MIXEDK_ELIDE_HANDLED=0`; B is the elision arm, `=1`):

| arm | code warm mean (tok/s) | prose warm mean (tok/s) |
|---|---|---|
| A (=0) | 40.8467 | 23.9233 |
| **B (=1)** | **41.2933** | **24.2933** |
| A2 (=0) | 40.7333 | 23.8700 |

B versus the mean of the A/A2 bracket: **code +1.23396%, prose +1.65992%**. All 8
corresponding (case x repeat) texts, completion counts and acceptance ratios match across
A/B/A2 — bounded equality, **not** universal greedy identity and **not** a quality
certification.

Configuration: MiMo target 2.50 bpw with the original compiled extension, EXL3 4.0 bpw
drafter, Q4 KV, context and chunk 4096, native window 7, dynamic confidence 0.6,
`max-active-requests 1`, serial verify, greedy seed 42, max 320 tokens. Warm code hits the
intentional 320-token cap, so the code samples are not claimed as tested production code.

Fresh 3527-token prefill was slightly **lower** on B: A 709.72, B 702.86, A2 708.93 tok/s
(MAPLE-9362 correct in all three). The gain is therefore decode-only and sits close to the
bracket's own drift; the honest summary is "a measurable ~1.2%/1.7% decode gain, prefill
within noise, 60 tok/s not reached".

Because the two A arms bracket B, the bracket's spread (0.11% on code) is much smaller
than the reported gain, which is why the result is retained at all.

## What the CPU review does and does not establish

- The tests execute complete methods — `BlockSparseMLP.forward`, `Generator.__init__`,
  `_staging`, `iterate_gen`, `Job.__init__`, `prepare_for_queue`, `receive_sample`,
  `prepare_for_requeue` and the input-ID methods — via AST with decorators removed; method
  bodies are **not** sliced.
- CUDA launches are an assignment/slot oracle over poisoned scratch. That is not CUDA math,
  scheduler, race or performance validation, and NumPy tests do **not** establish GPU
  tensor identity.
- The changed sync behavior can expose device lifetime and race assumptions. Captured
  same-input GPU output equality plus uninstrumented single-stream timing remain
  **mandatory** before approval.
- `GPU_VALIDATION.md` is an explicitly **unexecuted** local acceptance protocol, not
  evidence. No SSH, CUDA initialization, GPU execution, deployment or model loading was
  performed by this review.
- Pre-existing hazard, reproduced on **both** the original and the patched tree:
  legacy-grouped initialization followed by unified dispatch fails because `_mkd_bufs`
  exists without `_mkd_fused_rows`. It is documented, **not** repaired; do not switch
  these eligibility modes inside one loaded module.
- `_run_batch_recon` initialises its scratch slot offset from homogeneous fused buffers
  while unified mixed-K tables can reserve a mixed-K prefix. This is a source-level concern
  outside the elided branch and was **not** proved or repaired. Small-row coverage is not
  approval for arbitrary prefill or grouped-reconstruction shapes.
- The audit also **rejects** several earlier claims (an unmeasured `U(8)=56.7` expert union,
  an approximate 150 ms round latency as pure target verification time, `draft_accept` as a
  conditional probability, router-permuted expert renaming as a bandwidth reduction, and a
  pinned-buffer widening patch as "multi-block DFlash"). See `AUDIT.md`.

## The second patch is NOT applied

`patches/target-round-diagnostics.patch` is a **separate** candidate shipped here only as
an unapplied reference. Nothing from it is in this branch's source, and you can see that
directly: running the full telemetry suite against this branch errors with
`AttributeError: 'Generator' object has no attribute 'draft_diagnostics'`, because that
attribute only exists once the diagnostics patch is applied.

Its companion finding, reported rather than hidden: `Job.receive_sample` constructs the
final EOS event **before** `Generator.iterate_gen` appends that round to `job.draft_stats`,
so the last round is lost. `tests/prior_telemetry_repro.py` executes the copied earlier
methods and fails on it (max_new_tokens=1 reports no `draft_rounds`; max_new_tokens=5 over
two target passes reports one). Immutable transcript: `evidence/prior-final-round-red-01.txt`.

## Honest RED/GREEN provenance

- `evidence/mixedk-red-01.txt` is an **initial harness error** (missing tuple-assigned MTILE
  constants), **not** accepted RED evidence. It is retained for transparency.
- `evidence/mixedk-red-02.txt` is the real RED: original source executes real dispatch,
  exactly the dead-readback assertion fails, and the preservation controls pass. The
  minimal source change then makes it green.
- The first apply-verifier attempt caught a byte-identity problem before application:
  Windows `git archive` inherited `core.autocrlf=true` and produced CRLF archive text from
  LF git blobs. Trees were rebuilt with `-c core.autocrlf=false` (see the package's
  `repair_archive_eol.py`, not shipped here) and the earlier bytes were kept aside in
  `historical-crlf/` (also not shipped).

## How the checks were run here

```sh
cd science/trials/mixedk_readback_review/tests
# GREEN against this branch's source (the elision applied):
REVIEW_SOURCE_ROOT=<repo-root> python -B test_driver.py test_mixedk
REVIEW_SOURCE_ROOT=<repo-root> python -B test_driver.py test_telemetry.Preservation
# RED against the pristine pin (unpatched), which is what proves the test bites:
REVIEW_SOURCE_ROOT=<pin-ca4a880-clone> python -B test_driver.py test_mixedk.Elision
REVIEW_SOURCE_ROOT=<pin-ca4a880-clone> python -B test_driver.py test_mixedk.Preservation test_telemetry.Preservation
```

`run_tests.py` is the same runner with immutable `evidence/<label>.txt` transcripts; it
needs a new label each time.

## What is NOT in this branch / size policy

- `patches/target-round-diagnostics.patch` **ships unapplied**, by design.
- `baseline.tar` (20 MB), `baseline/`, `candidate/`, `prior-source/`, `verified/`,
  `historical-crlf/` and `repair_archive_eol.py` are **not** shipped. They are the
  disposable apply trees and the pinned baseline archive.
- Consequence, stated plainly: **`verify.py` cannot run from this directory as shipped.**
  `MANIFEST.json` pins sha256 digests for those excluded artifacts, so `verify.py` stops at
  the first missing one — `FileNotFoundError: .../mixedk_readback_review/baseline/
  exllamav3/generator/generator.py`. It is shipped as the record of how verification was
  performed, not as a self-contained check. The checks above drive `tests/test_driver.py`
  directly and do not need those trees.
- The package's own `README.md` is preserved here as **`source-README.md`**, so that this
  file could be the directory's `README.md`.

## Contents

- `AUDIT.md`, `GPU_VALIDATION.md`, `MANIFEST.json`, `source-README.md` — the review's own
  critical audit, its explicitly unexecuted GPU protocol, the pinned manifest, and its
  package README.
- `patches/mixedk-dead-readback.patch` (the source change, applied),
  `patches/target-round-diagnostics.patch` (unapplied reference).
- `verify.py`, `run_tests.py`, `build_manifest.py`, `build_patches.py` — the review's
  offline verifier, runner and package builders.
- `tests/` — `test_driver.py` plus the harnesses (`source_harness.py`,
  `generator_harness.py`), the suites (`test_mixedk.py`, `test_telemetry.py`) and
  `prior_telemetry_repro.py`.
- `evidence/` — the small `*.txt` receipts and the `*-summary.json` machine-readable
  summaries from the original run.
