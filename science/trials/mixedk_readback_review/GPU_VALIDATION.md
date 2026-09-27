# GPU acceptance protocol — NOT EXECUTED

This task explicitly forbids GPU/SSH use. The following is a **future local-only reviewer protocol**, not evidence that these candidates are GPU-correct or fast. No command here was run on a GPU. Use a disposable authorized local checkout, never production or a shared tree.

## 1. Freeze the comparison

Record the pin and patch SHA256, model/quant file checksums, CUDA/PyTorch/extension build, GPU identity, deterministic mode, shapes, and EVERY kernel environment setting. Keep N/BPS/CMAX/MTILE identical in both arms. Do not combine this candidate with a simultaneous grid/occupancy experiment. No multi-block DFlash flags or enlarged draft windows.

In the disposable checkout (commands to be run only after separate authorization):

```bash
git -c core.autocrlf=false -c core.whitespace=cr-at-eol apply --check --whitespace=error-all /path/to/review/patches/mixedk-dead-readback.patch
git -c core.autocrlf=false -c core.whitespace=cr-at-eol apply --whitespace=error-all /path/to/review/patches/mixedk-dead-readback.patch
```

Obtain actual MoE calls from the already-approved local single-stream validation driver. Do not generate random inputs and describe them as captured model inputs. Keep capture output private: tensors can encode prompt information. The following function is an explicit same-input before/after protocol to call at a captured target-MoE boundary in that driver, on its owner thread. `m`, `x`, `params` must be the real module/call arguments, not reconstructed formulas. This restricted first gate excludes CPU offload, collectives and alternate residual routing. Test those paths separately before expanding eligibility.

```python
# Future authorized GPU driver only; never run by this CPU review.
import importlib
from pathlib import Path
import torch
sparse = importlib.import_module("exllamav3.modules.block_sparse_mlp")

@torch.inference_mode()
def capture_pair(m, x, params, path):
    path = Path(path)
    assert not path.exists(), "never overwrite evidence"
    assert x.is_cuda
    assert m.mixedk_unified and m.fused_mode_buffers is None
    assert m.routing_gate is not None and m.routing_device is None
    assert not m.tp_reduce and not m.cpu_offload and m.cpu_split_first is None
    assert not m.alt_residual_channel
    rows = x.numel() // m.hidden_size
    assert 1 <= rows <= 8
    cap = getattr(m, "_mkd_fused_rows", sparse.TEMP_ROWS_FUSED)
    assert rows * m.num_experts_per_tok <= cap
    assert sparse.FUSED_DET, "first gate requires deterministic slot/gather"
    assert not params.get("tp_warmup") and not params.get("autosplit_measure")
    original_route = m.routing_fn
    original_flag = sparse.MIXEDK_ELIDE_HANDLED
    saved = []
    x0 = x.detach().clone()
    def capture_route(*args, **kwargs):
        ids, weights = original_route(*args, **kwargs)
        saved.append((ids.detach().clone(), weights.detach().clone()))
        return ids, weights
    try:
        sparse.MIXEDK_ELIDE_HANDLED = False
        m.routing_fn = capture_route
        before = m.forward(x0.clone(), dict(params)).detach().clone()
        torch.cuda.synchronize(x.device)
        assert len(saved) == 1
        ids, weights = saved[0]
        m.routing_fn = lambda *args, **kwargs: (ids.clone(), weights.clone())
        sparse.MIXEDK_ELIDE_HANDLED = True
        after = m.forward(x0.clone(), dict(params)).detach().clone()
        torch.cuda.synchronize(x.device)
        # Save actual input, selection, weights and BOTH outputs before asserting.
        record = dict(input=x0.cpu(), selected_experts=ids.cpu(),
                      routing_weights=weights.cpu(), before=before.cpu(), after=after.cpu(),
                      module_key=m.key, rows=rows, role="target", instrumented=True)
        with path.open("xb") as f:
            torch.save(record, f)
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    finally:
        m.routing_fn = original_route
        sparse.MIXEDK_ELIDE_HANDLED = original_flag
```

Capture at most 16 small-row pairs, then remove this callback. The module flag is changed only for the paired diagnostic invocation and restored even on failure. Never do this while other requests execute. The callback intentionally clones/synchronizes/copies: **its latency is invalid for performance comparisons**. The extra replay must occur only in the disposable driver, not in production generation.

This direct module replay verifies the tested routed layer only. It does not prove end-to-end cache, speculative acceptance, whole-model output, tensor lifetime or multistream correctness. Also replay the same captured calls in separate baseline/candidate processes using the actual workload driver before approval, and retain both output files.

## 2. Coverage and bounds before enabling broadly

- Capture real target rows 1 and native verification rows up to 8, including actual active expert count below concurrency (no estimate). Keep distinct target, drafter and prefill labels. The new target-round diagnostic excludes drafter/prefill; do not fill those categories with inferred numbers.
- Force/capture repeated-expert, sparse-active and full-width selections. Verify sentinel/local slices, empty local contribution, row-cap boundaries and MTILE tiers in a separate approved kernel/module test; the restricted snippet above deliberately does not claim those shapes.
- Count assignment coverage using device counts/routed selections only in this instrumented test. Every local expert-row pair must contribute once; sentinel selections must not read an unwritten scratch slot. Run compute-sanitizer memcheck/racecheck on the approved bounded test driver before asserting buffer/lifetime safety.
- Do not switch a live module from legacy-grouped initialization into unified mode; the CPU harness reproduced the existing missing-attribute failure. Investigate/fix mixed-K plus batched-reconstruction scratch-offset concerns independently before using those shapes.
- Candidate mixed-K optimization allocates no new GPU buffers and changes no indices, capacities or launch arguments. Existing FP16 fused buffers cost `4*C*R*(H+I)` bytes across gate/up state and intermediates. All-fused deterministic scratch costs `4*A*H` bytes, with `A=tokens*top_k`; sentinel slots may exist but are not read by gather. This is a storage-accounting bound, NOT a proof of CUDA writes.
- A saved pair contains one input plus two outputs and the routing tensors. Bound it by the observed `numel()*element_size()` for each tensor, reject a pair exceeding the driver's memory budget, and cap total capture count. Do not retain whole model outputs or unbounded per-layer histories.
- New round telemetry has no GPU tensors or extra CUDA readbacks. Storage is O(256*max_batch_size) scalar dictionaries plus O(256) legacy tuples per Job. Temporary result scans are bounded by the existing per-pass result list. Integer counters grow only with total calls. After the job, serialize on the owning thread and release the generator when finished.

## 3. End-to-end correctness and uninstrumented timing

Run independent single-stream requests with identical prompt IDs, seed, sampler and native draft geometry, covering max_new_tokens=1, stop/EOS, mismatch first/last, full acceptance, dynamic truncation and requeue/page boundaries. Compare emitted IDs, terminal events and K/V/page lifetime checks—not just response text or an acceptance ratio. For deterministic output, require exact tensor/token equality; investigate rather than silently loosening tolerance. Atomic paths need a predeclared numerical policy and separate tests.

For performance use fresh processes and disable all capture callbacks and round statistics. The only difference is:

```bash
# Prefix the SAME already-approved local single-stream benchmark command:
EXL3_MOE_MIXEDK_ELIDE_HANDLED=0 <approved-local-benchmark-command>
EXL3_MOE_MIXEDK_ELIDE_HANDLED=1 <approved-local-benchmark-command>
```

The benchmark command is intentionally supplied by the operator; this package does not invent a model path, local GPU allocation or deployment command. Alternate A/B order, warm both, repeat enough to characterize variance, and report single-stream generated/emitted tokens plus wall-time definition. Instrumented diagnostics and captured-output timings must be reported separately. No expected speedup or 60 tokens/s claim is justified by the CPU tests.

## 4. Rollback

Stop the isolated process; unset `EXL3_MOE_MIXEDK_ELIDE_HANDLED` (or set exactly `0`) and construct the generator with `record_draft_stats=False` (default). A fresh process resets module-import constants and all scratch state. To remove the candidates in the disposable checkout:

```bash
git -c core.autocrlf=false -c core.whitespace=cr-at-eol apply -R --check /path/to/review/patches/target-round-diagnostics.patch
git -c core.autocrlf=false -c core.whitespace=cr-at-eol apply -R /path/to/review/patches/target-round-diagnostics.patch
git -c core.autocrlf=false -c core.whitespace=cr-at-eol apply -R --check /path/to/review/patches/mixedk-dead-readback.patch
git -c core.autocrlf=false -c core.whitespace=cr-at-eol apply -R /path/to/review/patches/mixedk-dead-readback.patch
```

Reverse only patches actually applied and inspect `git diff` afterward. Never reset unrelated edits. Prefer deleting the disposable checkout after retaining private evidence. Production defaults and server sources require no rollback because this review did not change them.
