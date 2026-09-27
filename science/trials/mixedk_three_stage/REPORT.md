# Three-stage mixed-K launch-fusion source candidate

**Outcome: implemented a distinct, default-off candidate, with a real cumulative
patch and passing CPU source/ownership/layout regressions. NOT COMPILED. NOT GPU
VALIDATED. No performance or deployment-readiness claim.**

Artifact root **A**:
`C:/Users/Victor Cruz/AppData/Local/hermes/cache/scratch/mimo-tune/mixedk-three-stage-candidate`.

## Deliverables and verified identity

- `mixedk-three-stage.patch`: **29,610 bytes**, SHA256
  `757f5b998f936a825abe95006a77d15732c2c29303722e7d05d35f2a279bd703`.
- `tree/`: complete candidate repository snapshot, **769 files**. Eight changed/new
  source paths; not a stub or a patch depending on the historical phased mode.
- `original/`: **767 complete exact Git blobs**, every byte checked against literal
  pin `ca4a880e8918e1985fd25e06c6aff561666d3f14` by `git cat-file --batch`.
  `provenance.json` records all baseline hashes.
- `five-stage-control/`: separately copied historical reference sources. Not shipped
  by the cumulative patch and not presented as this new candidate.
- `verify-final/summary.json`: canonical verification receipt. Strict LF
  `git apply --check --whitespace=error-all`, actual disposable application, and
  **all-file applied/candidate byte equality** passed. Shared upstream clean before
  and after, at the exact pin.
- `run_cpu_tests.py`, `verify.py`, `test_*.py`, corrected `scalar_reference.py`:
  runnable CPU evidence. Imports of torch and exllamav3 are explicitly blocked.
- `build_trial.py`, `gpu_reference.py`, `deferred_gpu.py`, `bounded_deferred.py`,
  `DEFERRED_GPU.md`: isolated bounded future build/replay controls and instructions.
  Only their CPU-safe paths/AST/metadata checks ran.

Changed paths under `tree/`:

1. `exllamav3/exllamav3_ext/quant/exl3_moe_three_stage.cuh` — actual three-stage kernel.
2. `quant/exl3_gemm_inner.cuh` — default-false full-K option, same pipeline/stores,
   and explicit full-K-only return drain/barrier.
3. `quant/exl3_moe_kernel.cuh` — existing independent runtime-K dispatch forwards
   the default-false full-K template parameter.
4. `quant/exl3_moe.cu` — original/shared host implementation, new explicit entry,
   narrow guard, four owned tensors, three ordinary launches, debug poison option.
5. `quant/exl3_moe.cuh` — new entry declaration; existing ABI unchanged.
6. `exllamav3/exllamav3_ext/bindings.cpp` — new binding; original binding unchanged.
7. `exllamav3/modules/block_sparse_mlp.py` — cached metadata qualification and
   guarded entry selection, identical 35 call arguments and existing gather.
8. `exllamav3/modules/mixedk_three_stage.py` — pure-Python fail-closed selection.

Paths 2–5 are relative to `exllamav3/exllamav3_ext/`.

## What actually changed

| Stage | Work | Grid per eligible-expert ordinal |
|---|---|---|
| 0 | Original gather and independent gate/up input Hadamards | x=1, y=1, 256 threads |
| 1 | Gate GEMM, independent up GEMM, then this tile's original GUAD epilogue | x=8, y=1, 512 threads |
| 2 | Down GEMM, then this tile's original output Hadamard/weighted fused-slot store | x=16, y=1, 512 threads |

This is **three ordinary current-stream-ordered launches**. No cooperative GEMV,
last-arrival counter, device-wide spin wait, group barrier or cross-CTA raw-output
consumption is introduced. The caller's deterministic gather is unchanged.

Both the Python flag and native explicit entry are required:
`EXL3_MK_THREE_STAGE=1`. The original `exl3_moe_mixedk` entry always calls the shared
implementation with `allow_three_stage=false`, even if the environment flag is set.
This avoids accidentally enabling tiny prefills that the C++ ABI cannot distinguish.
The new `exl3_moe_mixedk_three_stage` has the same 35 argument types/order.

Production guard: decode q1..8, no `prefill` key (including false), no measurement/TP
warmup, authentic `routing_dots` identity, complete 256-expert local layer, H4096,
I2048, top8, gated SiLU/mul1, no biases/padding/slicing/latent projections, neutral
linear pre/post/weight scalars and softcap. Native guard additionally checks sm121,
m16, deterministic slot pointers, selected contiguity/dtypes, ordinary N256/pipeline
and tuning settings. Unsupported calls use the original entry/kernel, not truncation.
Activation clamp value is passed unchanged to the incumbent helper. Tier/overflow
checks and all-excluded early return remain intact. Direct low-level callers still
must supply valid routing/count/slot/device metadata, as with the baseline ABI.

No `EXL3_MK_PHASED` dispatch is included in this patch. Four buffers remain owning
per-call tensors allocated on the guarded current device/stream; no singleton scratch
or cross-stream shared cache was added.

## Why the two fusions are source-feasible

### Tile and row ownership

The full-K partition is read directly from `exl3_gemm_inner.cuh`:
`floor(tiles_n * blockIdx.x / gridDim.x) * tiles_k` through the next boundary.
With `grid.x=N/256`, one CTA owns exactly one 256-column tile and all its K slices.
Gate/up N=2048 gives eight tiles; down N=4096 gives sixteen. Every 128-column
Hadamard group is contained within exactly one such tile.

The fused epilogue enumerates `r=row+w/2`,
`col=blockIdx.x*256+(w%2)*128`, with `w < min(16,rows-row)*2` and sixteen physical
warps. Rows advance by sixteen, including duplicate-route synthetic counts above
q. It never processes inactive tail rows. The scanner advances the sorted-assignment
prefix even over excluded experts. Output uses `fused_base[expert]+r` while route
weight uses `weight_sorted[start+r]`; compact output bases are not confused with
sorted starts.

Stage1's GUAD consumes only raw gate/up outputs written by its own CTA. It overwrites
only that CTA's `ig` groups using the existing in-place helper contract. Stage2 needs
the full transformed intermediate across Stage1 CTAs, so it **remains a separate
launch**. Stage2's output Hadamard consumes only its own raw down tile. Stage0 also
remains separate because each GEMM consumes the whole rotated hidden vector.

### Shared lifetime and visibility

The existing full-K reduction does the same CTA sum and half `write_sum_gl`, then
`__syncthreads`. The candidate adds `cp_async_wait<0>(); __syncthreads();` at the
full-K helper return: gate/up/row calls and epilogues may safely reuse its dynamic
shared span only after every thread has drained outstanding copy groups. The
incumbent/default-false helper does not acquire this new return code.

After each tile epilogue, a uniform CTA barrier includes inactive warps before the
next row chunk reuses shared memory. Output Hadamard additionally completes each
warp's shared reads with `__syncwarp` before reusing that warp's span. These are
source synchronization contracts, **not an executed CUDA race proof**.

The unchanged Hadamard helpers internally add `blockIdx.y*32` to scale indices.
All new stages therefore have y=1/blockIdx.y=0, and pass explicit column-offset
scale pointers. Retaining the five-stage gate/up projection-Y grid here would be
incorrect.

### Arithmetic boundaries and resource accounting

The complete Hadamard/activation/output helpers, codebook and decoder bytes are
unchanged. Input multiplication still rounds to half before Hadamard; sm121 MMA
still accumulates FP32; GEMM outputs still store half; gate/up transforms, postscales,
SiLU, clamp, gating, down prescale and down-input transform retain their half-rounded
ordering. Output still receives `R_SCALE * float(route_weight)` before svh scaling.
Gate/up/down pointer and K arrays remain independent. Merely changing a cooperative
GEMV accumulator was explicitly not used.

Exact largest GEMM dynamic footprint at these parameters is **44,032 bytes**;
the retained host reservation is **60,416 bytes**. Output Hadamard at 512 threads
needs **16*128*4 = 8,192 bytes**, not the old 256-thread allocation of 4,096 bytes.
Because the spans are reused sequentially, the launch uses their **maximum**, not
an invented sum; 60,416 covers both. Static scanner storage is two ints, separate
from that dynamic reservation. Compiler registers, spills and occupancy are unknown.

New global scratch is at most **64*(2*4096+2*2048)*2 = 1,572,864 bytes**, exactly the
prior bounded four-tensor footprint, in addition to unchanged caller buffers/slots.
This calculation is not a measured memory high water or an OOM guarantee.

## Executed CPU evidence

Canonical receipt: `verify-final/summary.json`.

- **26/26 candidate tests pass**, no errors/skips; **26/26 pass on the independently
  applied copy**. All forbidden-import lists empty.
- **9/9 original preservation/layout controls pass**.
- Original missing-fusion regression: **one intended failure, no harness error**.
- Separate missing-column-offset mutation fails the ownership domain assertion.
- Earlier vertical RED/GREEN receipts: missing fused dispatch, missing Python guard,
  missing actual-intermediate poison control, padded/sliced-input acceptance and
  raw-pointer dtype acceptance. Original failure transcripts were not overwritten.
- Ownership enumeration executes actual scanner/partition/epilogue expressions:
  **94 fixtures, 194,048 projection-Had128 ownership instances**. Producer and
  consumer match in the same CTA exactly once; the independent expected domain
  proves no jointly omitted work. q1..8 hot/spread/duplicates, every count1..64,
  partial row chunks, mixed live/excluded tiers, sentinel-only and all-excluded cases.
- Corrected layout reader executes labeled reconstruct stores, including j2=(8,0)
  and j4=(0,8) at lane0, not merely a bijection. All eleven supported integer/half-bit
  formats, four seeded payloads each, and all 65,536 mul1 codebook indices are checked.
- Actual nested original/candidate Python callers execute against the same opaque
  inputs and preserve all 35 argument objects. Scalar argument9 is not a tensor.
  Native eligibility expression runs with metadata-only CPU objects; unsupported
  architecture/shape/activation/tuning/contiguity and int32 pointer tables reject.
- The derived local header table was copied, hashed and recounted: 47 MoE entries,
  36,096 matrices, 33 mixed layers; K2 counts4:21381,6:12049,8:2665,10:1. The unique
  K2=10 down matrix is layer46/expert148. This is **derived metadata, not a fresh raw
  header or payload audit**. All existing K formats remain supported.

Preserved harness issues, not CUDA failures: `ownership-attempt01.txt` records an
expression whitespace/IndentationError; `preservation-attempt01.txt` records an
over-wide source slice starting at the uniform entry's early return;
`verify-initial/candidate.txt` records a source assertion using `args` rather than
its actual parser variable `a`. Fixed only this candidate's new CPU harness files.
The final applied-tree path control also uses the invariant artifact root instead
of assuming its disposable parent contains `original/`.

Reproduce from A:

```
python -B verify.py --label NEW_UNIQUE_LABEL
python -B run_cpu_tests.py --json NEW_RESULT.json
python -B deferred_gpu.py --plan
```

## Deferred gates and tradeoffs

`DEFERRED_GPU.md` specifies isolated original/off/five/three controls, independent
small real heterogeneous captures, bitwise gates where geometry/math match, intact
historical numerical thresholds otherwise, actual profiler-stage names, poison
build, repeated lifetime/compact gather, separate memcheck/initcheck and bounded
q1 versus multirow timing. No CUDA compiler output is claimed or fabricated.

Five-stage versus three-stage must be bitwise on all covered identical full-K cases.
Original default geometry with group8 also has complete-K boundaries; other small
active-group geometries can split K, so they retain the existing numeric thresholds
rather than a false universal original-bitwise claim.

Serial gate/up inside one CTA halves their tile-level parallelism relative to
separate projection-Y CTAs, may change register pressure and could regress. Two fewer
launches alone do not imply speed. The historical five-stage candidate's actual
round3 microbench gains/regressions and round4 negligible q1-only model change are
not inherited by this source candidate.

Parent's mid-turn round8 update was retained as **parent-provided context**: native
multirow calls dominate warm code and substantial prose; q1-only tuning is not the
objective, native16 remains unvalidated, and owner release grants no GPU permission.
This candidate remains q1..8 with unchanged q>8/prefill fallback. No service, model,
GPU, network, original sources, installed extension, quantization payload, or other
role's artifacts were modified. No commit/publication/deployment occurred.
