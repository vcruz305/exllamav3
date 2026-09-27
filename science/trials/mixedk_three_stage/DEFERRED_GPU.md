# Deferred GB10 gates — NOT COMPILED / NOT GPU VALIDATED

This directory is a **source candidate**, not deployment or performance evidence.
Nothing here has imported torch/exllamav3, used a GPU, SSH, a remote service or model
weights during local verification. All GPU commands below are a future operator plan.
Only the authorized GPU owner may execute them, with the model stopped and the existing
memory guard supervising the process. Do not replace/install the serving extension.

## Four independent controls

- `original/`: complete exact-pin `ca4a880e8918e1985fd25e06c6aff561666d3f14` source.
- `tree/`, original entry or flag off: unchanged original dispatch/35-argument ABI.
- `five-stage-control/`: separate historical five-stage reference, **not** in the new patch.
- `tree/`, `EXL3_MK_THREE_STAGE=1`, new explicit entry: new three-stage candidate.

`build_trial.py` copies sources to content-addressed cache directories, uses unique
extension names and hidden C++/nvcc host symbols, rejects RTLD_GLOBAL, and puts caches
outside source directories. No fixed-name module alias or installation is performed.
Flags, compiler-toolchain Torch/CUDA identity and source bytes enter the cache digest.
Each standalone extension has 46 TUs including its binding. Actual build output,
registers/spills, DSO independence and compiler success remain unknown.

CPU-only preview:

```
python -B deferred_gpu.py --plan
python -B build_trial.py --source tree --list-sources
```

Future Linux operator setup (these environment assertions do not themselves stop a
model, establish exclusive ownership, or prove a memory guard is active):

```
export EXL3_TRIAL_MODEL_STOPPED=YES EXL3_THREE_STAGE_AUTHORIZED=YES
export EXL3_THREE_STAGE_GUARD_ACTIVE=YES MAX_JOBS=1
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1
# Put the authorized interpreter's ninja and CUDA bin on PATH.
```

Use a NEW receipt/output directory for every run. `bounded_deferred.py` enforces a
maximum 1800-second deadline, propagates failures and kills only its own new process
group on timeout; raw logs/process exit status survive. It does not replace the
memory guard. Cache paths must remain outside all serving/source trees.

```
python -B bounded_deferred.py --receipt receipts/numeric01 --seconds 1800 -- \
  --mode numeric --output evidence/numeric01 --cache isolated-build-cache
python -B bounded_deferred.py --receipt receipts/poison01 --seconds 1800 -- \
  --mode numeric --poison --output evidence/poison01 --cache isolated-build-cache
python -B bounded_deferred.py --receipt receipts/profile01 --seconds 1800 -- \
  --mode profile --output evidence/profile01 --cache isolated-build-cache
```

The poison variant is a distinct compile-flag/digest. It fills the **actual four new
internal tensors** with NaNs before pointer substitution. Poisoning the caller's
legacy buffers alone is not intermediate coverage. Never time the poison variant.

After warming/building the same immutable DSOs, run separate Compute Sanitizer passes
with the numeric mode, **never** the profiler mode. For example, under the same guard:

```
compute-sanitizer --tool memcheck --error-exitcode 98 --target-processes all \
  python -B bounded_deferred.py --receipt receipts/memcheck01 --seconds 1800 -- \
  --mode numeric --output evidence/memcheck01 --cache isolated-build-cache
PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool initcheck \
  --error-exitcode 98 --target-processes all \
  python -B bounded_deferred.py --receipt receipts/initcheck01 --seconds 1800 -- \
  --mode numeric --output evidence/initcheck01 --cache isolated-build-cache
```

Preserve sanitizer stdout/stderr as fresh external logs as well as child receipts.
CUPTI profiler and Compute Sanitizer must be separate invocations. An initcheck run
with allocator caching disabled can report zero allocator peak counters; this is not
a zero-memory footprint. Racecheck/synccheck and concurrency/lifetime stress are
additional gates, not claimed by memcheck/initcheck or the local ownership model.

## Numerical and dispatch gates

The bounded numeric driver retains the historical thresholds:

- Original versus three-stage: NRMSE <=0.002, peak/RMS <=0.025 where incumbent geometry
  splits K differently (e.g. synthetic duplicate/hot-small-active adversaries).
- Independent reconstruct/half pipeline: NRMSE <=0.005, peak/RMS <=0.06.
- Original versus candidate-off: **bitwise**, no relaxed fallback threshold.
- Five-stage versus three-stage: **bitwise for all covered cases**, because both use
  identical complete-K tiles, row chunks and half helpers; a mismatch blocks this
  scheduling-only claim rather than silently becoming a tolerance comparison.
- Normal q1..8 hot/spread complete-K original geometry: **bitwise**.

Coverage includes all q1..8, independent per-projection K (including odd half-rate
codes and K2=10), sentinel, zero weights, duplicate64, partial row/tier exclusions,
live/excluded mixtures, distinct compact bases with excluded prefixes, all-excluded
no-op and unsupported activation/codebook/q9 fallback. Repeated invocations exercise
allocator/lifetime reuse. The CPU suite additionally enumerates every count1..64.
Standalone numeric tests cannot validate Python's production guard; a later module
capture must prove its guard reached the new entry without wrapping `routing_fn`.

The separate profile mode must report the real three `exl3_moe_three_stage_kernel`
instantiations, no persistent/five-stage fallback in the new arm, the five old stages
in the reference arm, and persistent kernels for original/off. Environment flags or
trace messages alone do not count as dispatch proof.

After all gates pass, ordinary-build timing can be run with `--mode timing`; it keeps
q1 and q8 hot/spread rows separate and records original/off/five/three/original
brackets. Timings include host dispatch and gather; they are not model TPS. The
harness does not automatically authorize timing based on a receipt: the owner must
verify prior numeric/profiler/sanitizer records and all real-capture gates first.

## Private selected-real-capture interface

Use at most six packs and stay below 1.5GiB live tensor budget/2GiB allocator cap.
`--capture PATH` is repeatable in each mode. No full model is loaded by this driver.

Each small trusted tensor-only `.pt` pack contains:

- `x`: FP16 `[q,4096]`, q1..8;
- `selected_experts`: int64 `[q,8]`, sorted-remapped expert IDs into `pool`;
- `routing_weights`: FP16 `[q,8]`, captured values, not regenerated;
- `pool`: list in **ascending original expert-ID order**, each item three independent
  `(K2, trellis, suh, svh)` tuples for gate/up/down. `K2` uses half-bit ABI units;
  trellis int16 `[k/16,n/16,K2*8]`, suh/svh FP16 `[k]`/`[n]`.

The companion `PATH.pt.json` must contain `pin` (the exact comparison-source pin),
`capture_sha256`, `layer` (27/43/47, or rare-outlier layer46), `q`, and
`sorted_original_expert_ids`. Also retain native-versus-prefix provenance, original
installed source/DSO hashes, original route/count/sorted token/weight/scalar values,
per-payload hashes/header offsets and the original expert mapping in that sidecar.
The driver checks the pack hash, source pin, bounded shapes, sorted unique mapping
and required fields; **it does not independently recover those historical facts**.

The producer must capture identical module input and routes below guarded function
identities. Verify sorted remapping preserves tie/order semantics against actual
captured extension arguments. Scalar ABI argument9 is activation, not a tensor;
K arrays are int32 and pointer arrays int64. Gather takes `expert_start` **before**
`slot_base`. Compare the installed original DSO against exact-pin original and
candidate-off in a separately controlled replay before candidate-on. This driver
loads isolated DSOs, not the installed serving DSO. Do not loosen existing thresholds
or cast incompatible arguments to hide capture defects.

## Explicit limits

CPU expression/ownership execution is not CUDA execution, a synchronization proof,
a compiler check, a full floating-point simulation, a sanitizer result or model
quality certification. The future harness itself has only CPU/AST/plan checks so far.
The existing dense reference has tolerance-based accumulation and route-scale
association; the strict scheduling identity oracle is five-stage versus three-stage,
not a claim that dense Torch reconstruction is bitwise the incumbent.

Serializing gate/up within one CTA may reduce parallelism or raise register pressure.
Fewer launches are not evidence of lower latency. Full-model fresh-prefill/decode
A/B/A, memory high water and QoR are separate future promotion gates. Parent-reported
round8 native-q distribution reinforces covering q2..8, not q1-only claims; native16
is unvalidated and this candidate does not enable it. No GPU authorization is implied
by the previous owner's release.
