# 07 — Step-5 `sparse_indexer_z` → exllamav3 CSA compressor: mapping decision + norm bypass

Scope: how the Step-5-Preview compression projection `model.layers.N.self_attn.sparse_indexer_z`
maps onto exllamav3's fused DSV4 compressor (`exllamav3/exllamav3_ext/dsv4_compress.cu`), the
resolution of the `z_norm_type: "none"` gap, and the code this decision produced.

Prerequisites: `port-notes/02-dsv4-compressor.md` (compressor pipeline, kernel contract) and
`port-notes/05-step5-tensor-ground-truth.md` (verified checkpoint inventory). All `file:line`
cites refer to the tree **before** the norm-flag patch below unless marked "post-patch"; the
patch itself is reproduced verbatim in §2.3.

---

## 0. Decision summary

| question | answer | confidence |
|---|---|---|
| z mapping | **Hypothesis A**: z = the single content projection, `hd = W = proxy_dim = 256`, non-overlapping, `m = region_block_size = 8`, degenerate zero gate (=> **mean pooling** per 8-token block), `ape = zeros`, rope on trailing 32 cols at `w * 8`, **no norm** | HIGH (~85%), INFERRED — must be validated against the gated reference |
| kernel norm for `z_norm_type: "none"` | **Cannot be disabled in the stock kernel**; `norm_w = ones` is NOT equivalent. Fixed with a `no_norm` flag threaded `dsv4_compress(.cuh/.cu)` → kernel (patch applied to the tree, §2.3) | CONFIRMED (source) |
| `m` | `region_block_size = 8` is the compression unit | MEDIUM-HIGH, INFERRED |
| rope | GPT-J interleaved pairs on trailing `sparse_indexer_rope_dim = 32` cols, at the block's first-token position `w * 8`, post-pool (kernel convention) | MEDIUM, INFERRED (whether Step-5 ropes the compressed key at all is unconfirmed) |

Deliverables produced: `exllamav3/modules/step5_csa_compress.py` (module), this note, and the
kernel patch below.

---

## 1. The z mapping: hypothesis A (mean pooling) over B (self-gated)

Recall the DSV4 compressor's learned inputs (port-notes/02 §1, §2): **two** projections
(`wkv` content + `wgate` pooling logits, dsv4.py:126-133), a learned per-slot positional bias
`ape (m, W)` (dsv4.py:135, 629) and a norm weight `norm_w` (dsv4.py:134). Per-column softmax
pooling over the window: `comp = (kv * gate.softmax(dim=window)).sum(window)` (test:41,
dsv4.py:292), then RMSNorm (test:42, dsv4.py:293), then rope.

Step-5 provides **exactly one** compression tensor per sparse layer —
`sparse_indexer_z.weight [256, 4096]` BF16 — and the full sparse inventory is 8 tensors/layer
with **no gate projection, no ape, no norm tensor** (05 §1; `sparse_indexer_csa_z_norm_type:
"none"` confirms the absence of z-norm tensors). The two hypotheses from note 02 §4 (I3/I3b):

- **(A)** degenerate gate: `gate_rows = 0` → softmax = `1/m` → `comp = mean(z rows of the block)`.
- **(B)** self-gated: `gate_rows = kv_rows = z(x)` — softmax weights from z's own values.

**A is chosen.** The argument, strongest first:

1. **Tensor accounting.** DSV4's compressor consumes two independent learned projections plus
   `ape` plus `norm_w` (dsv4.py:126-135, 629). Step-5 ships one (05 §1, exhaustive over all
   shard headers, 2449/2449 index cross-check). Mapping A uses exactly that one tensor and
   fills the gate role with a **constant** (zero) — no invented parameters. Hypothesis B does
   not reduce the parameter count below A's; it instead requires inventing a *weight-sharing
   rule* (gate := kv) that exists nowhere in DSV4 and has **no config knob** (`sparse_config`
   in 05 §5 has no `gate_type`/`share_gate`/temperature field). A is the minimal-assumption
   mapping; B is A plus an extra architectural fiction.

2. **Config semantics.** The only norm knob on the compression path is
   `sparse_indexer_csa_z_norm_type`, i.e. a norm on **z's output** — and it is `"none"`: z is
   used raw. Under A the whole path is then parameter-free and deterministic
   (`entry_w = mean_e z(x_{8w+e})`). Under B, z's raw bf16 linear output would double as
   per-column softmax logits with no norm and no temperature — an unnormalized, scale-fragile
   gate that would need its own knob to be a sane design. Nothing in the config expresses it.

3. **Linearity property (why mean is the natural "block compress").** z is linear, so under A
   `entry_w = z(mean_e x_{8w+e})` — the block key is exactly the z projection of the block's
   mean hidden state: a stable, deterministic block summary, which is what block-level
   candidate selection (`topk: 512` over `2,048 blocks × 8`, PORT_SPEC §3) wants. "csa_block_compress"
   reads as "compress each `region_block_size` block to one entry"; mean pooling is the canonical
   block summary. Self-gating makes the summary a nonlinear, contrast-dependent function of z.

4. **Consistency of `"none"` across the path.** A has zero learned scale anywhere on the
   compression path (no gate, no ape, no norm gain) — `z_norm_type: "none"` is exactly the
   right description. B introduces implicit nonlinear weighting while claiming "no norm".

**Counter-consideration (stated honestly).** DeepSeek-V4's CSA family *is* softmax-gated, so if
"csa" in `csa_block_compress` implies the same pooling family, the gate logits must come from
somewhere, and with one tensor that means self-gating. If the gated reference turns out to be B,
the switch is one line in the module (`gate = z` instead of zeros — both hypotheses are plain
input tensors to the kernel, `kv_new`/`gate_new`, no further kernel change) plus the same norm
bypass. Pick-by-equivalence-test against the reference when it is available.

**Confidence: ~85% for A.** It is an inference, not source-confirmed (the Step-5 modeling code
is gated, PORT_SPEC §3). Validate before wiring into scoring.

### Companion resolutions

- **`m = region_block_size = 8`** (note 02 I2, was MEDIUM): chosen. The block is the compression
  unit ("csa_block_compress") and the selection granularity; `m = 8` divides PAGE_SIZE 256
  (assert dsa.py:75; `epp = 32`, dsa.py:77, constants.py:2). The competing `m = 4` reading
  would make `region_block_size` a pure selection knob while the compress unit stays DSV4's —
  less consistent with the method name. MEDIUM-HIGH.
- **`hd = W = 256`, non-overlapping** (note 02 I1): stands. Compressed block keys must be
  score-compatible with the indexer proxy space (q per-head 256, k 256 — 05 §1), so `hd = 256`;
  z outputs 256 cols ⇒ `W = hd` ⇒ kernel non-overlap branch (`overlap = W == 2 * hd`,
  dsv4_compress.cu:413-414; window entries `E = m` at :118; `sc = c` at :146).
- **`ape = zeros((8, 256))` fp32** (note 02 I6): stands — the kernel requires the tensor
  (`const at::Tensor&`, dsv4_compress.cuh:36; dtype check .cu:398; shape check .cu:405) but
  Step-5 has none, and zeros make the additive bias term vanish exactly (.cu:159).
- **rope on the trailing 32 cols at `w * 8`** (note 02 I5): implemented as the kernel does
  (post-pool, GPT-J interleaved pairs, `.cu:190-202`, `c0 = hd - rd` :192, `theta =
  inv_freq[p] * (ec0 + w) * m` :196). OPEN: whether Step-5 ropes the *compressed* key at all —
  `sparse_indexer_use_rope` most plausibly governs the per-token q/k rope; roping the pooled
  key at the block start is DSV4's convention, not confirmed for Step-5. MEDIUM.

---

## 2. Norm bypass (`z_norm_type: "none"`)

### 2.1 Can the kernel's norm be disabled? No.

Verified against the sources:

- The windows kernel applies the RMS rescale **unconditionally** after pooling —
  dsv4_compress.cu:170-186, specifically :184 `float rmr = rsqrtf(sh_red[0] / (float) hd + eps);`
  and :185 `float normed = comp * rmr * (active ? __half2float(norm_w[c]) : 0.0f);`. There is
  no branch on any flag; the value flows straight into the rope stage (:191 `float out = normed;`)
  and the store (:207-210).
- No bypass exists at the API level either: the arg lists (dsv4_compress.cuh:10-31 `dsv4_compress_gr`,
  :33-53 `dsv4_compress`) carry no norm-type switch, and `norm_w` is mandatory — dtype check
  dsv4_compress.cu:399 (`TORCH_CHECK_DTYPE(norm_w, kHalf)`) and shape check :415
  (`norm_w.size(0) == hd`).
- The torch reference applies the same stage unconditionally (tests/test_dsv4_compress_kernel.py:42),
  and `DSV4Compressor.forward` hardwires `self.norm` (dsv4.py:134 default `RMSNorm`, :293).
- **`norm_w = ones` is not a bypass**: the output becomes
  `comp * rsqrt(mean(comp²) + eps)`, a per-window rescale to unit RMS — different values from
  `comp` for every window whose mean square differs from `1 - eps`. With mean-pooled z rows this
  would additionally destroy the `z(mean hidden)` property of §1. Algebraically impossible to
  neutralize with a constant weight.

### 2.2 Chosen fix: `no_norm` flag (applied to the tree)

A `bool no_norm` is threaded `dsv4_compress(.cuh/.cu)` → `dsv4_compress_windows_kernel`, default
`false` (all existing semantics unchanged). In the kernel the norm stage is skipped entirely —
not faked with `norm_w = ones` — so `out = pooled comp`, then the unchanged rope/store stages.
The `if (!no_norm)` branch is uniform across the block (kernel-wide flag), so the two
`__syncthreads()` inside it remain legal. `norm_w` stays a required argument (still type/shape
checked at :399/:415, post-patch) but is unused when the flag is set.

Alternatives considered:

- **`norm_w` as `c10::optional<at::Tensor>`, `None` = no norm**: zero arity change, no
  bindings.cpp edit, implicit semantics ("missing weight means no norm") — rejected for
  explicitness; recorded here because it is strictly lower-risk if the reviewer prefers it.
- **Pure-torch fallback only** (never call the kernel for Step-5): correct but throws away the
  fused cached path; kept as the *fallback* inside `step5_csa_compress.py` (§3), not the fix.
- **Post-hoc divide-out of `rmr`**: would need `mean(comp²)` recomputed outside; strictly more
  work than the pool itself. No.

### 2.3 Exact diff (applied to `dsv4_compress.cuh` / `dsv4_compress.cu`)

```diff
diff --git a/exllamav3/exllamav3_ext/dsv4_compress.cuh b/exllamav3/exllamav3_ext/dsv4_compress.cuh
--- a/exllamav3/exllamav3_ext/dsv4_compress.cuh
+++ b/exllamav3/exllamav3_ext/dsv4_compress.cuh
@@ -27,7 +27,9 @@ void dsv4_compress_gr
     const c10::optional<at::Tensor>& slot_ids = {},
     const c10::optional<at::Tensor>& pool_bt = {},
     int pool_epp = 0,
-    bool stage_rel = false          // dest_a = per-job staging rows [0, nw) (see kernel)
+    bool stage_rel = false,         // dest_a = per-job staging rows [0, nw) (see kernel)
+    bool no_norm = false            // skip the RMS-norm stage entirely (norm_w unused;
+                                    // Step-5 z_norm_type "none")
 );
 
@@ -49,7 +51,8 @@ void dsv4_compress
     const c10::optional<at::Tensor>& slot_ids,
     const c10::optional<at::Tensor>& pool_bt,
     int pool_epp,
-    bool stage_rel
+    bool stage_rel,
+    bool no_norm = false
 );
```

```diff
diff --git a/exllamav3/exllamav3_ext/dsv4_compress.cu b/exllamav3/exllamav3_ext/dsv4_compress.cu
--- a/exllamav3/exllamav3_ext/dsv4_compress.cu
+++ b/exllamav3/exllamav3_ext/dsv4_compress.cu
@@ -30,8 +30,9 @@ Layout/invariants (see cache/dsa.py):
   - Gate softmax runs per COLUMN over the window entries, fp32, matching
     (kv * gate.softmax(dim = 2)).sum(dim = 2).
-  - Norm is a weighted RMSNorm over hd; rope is GPT-J pairs on the trailing rope_dim
-    columns at theta = inv_freq * (ec0 + w) * m.
+  - Norm is a weighted RMSNorm over hd (skipped entirely when no_norm is set, e.g. Step-5
+    z_norm_type "none"; norm_w is then unused but still type/shape-checked); rope is GPT-J
+    pairs on the trailing rope_dim columns at theta = inv_freq * (ec0 + w) * m.
@@ -78,6 +79,9 @@ void dsv4_compress_windows_kernel
     const bool stage_rel                 // dest_a is a per-JOB staging buffer of this step's
                                          // entries at rows [0, nw) (packed-pool quantization
                                          // follows); dest_b / pool_bt unused
+    , const bool no_norm                 // skip the RMS-norm stage: out = pooled comp, norm_w
+                                         // unused (Step-5 z_norm_type "none"). Uniform across
+                                         // the block, so the contained syncs stay legal
 )
@@ -167,22 +171,28 @@ void dsv4_compress_windows_kernel
     }
     float comp = active ? acc / l : 0.0f;
 
-    // Weighted RMS norm over the hd columns
-    float sq = comp * comp;
-    for (int offset = 16; offset > 0; offset >>= 1)
-        sq += __shfl_down_sync(0xffffffffu, sq, offset);
-    if ((c % 32) == 0) sh_red[c / 32] = sq;
-    __syncthreads();
-    if (c < 32)
+    // Weighted RMS norm over the hd columns. no_norm skips the stage entirely: out is the
+    // pooled comp and norm_w is unused. The branch is uniform across the block (no_norm is a
+    // kernel-wide flag), so the contained __syncthreads() remain legal
+    float normed = comp;
+    if (!no_norm)
     {
-        sq = c < t_warps ? sh_red[c] : 0.0f;
+        float sq = comp * comp;
         for (int offset = 16; offset > 0; offset >>= 1)
             sq += __shfl_down_sync(0xffffffffu, sq, offset);
-        if (c == 0) sh_red[0] = sq;
-    }
-    __syncthreads();
-    float rmr = rsqrtf(sh_red[0] / (float) hd + eps);
-    float normed = comp * rmr * (active ? __half2float(norm_w[c]) : 0.0f);
+        if ((c % 32) == 0) sh_red[c / 32] = sq;
+        __syncthreads();
+        if (c < 32)
+        {
+            sq = c < t_warps ? sh_red[c] : 0.0f;
+            for (int offset = 16; offset > 0; offset >>= 1)
+                sq += __shfl_down_sync(0xffffffffu, sq, offset);
+            if (c == 0) sh_red[0] = sq;
+        }
+        __syncthreads();
+        float rmr = rsqrtf(sh_red[0] / (float) hd + eps);
+        normed = comp * rmr * (active ? __half2float(norm_w[c]) : 0.0f);
+    }
     if (active) sh_comp[c] = normed;
     __syncthreads();
     if (!active) return;
@@ -385,7 +395,8 @@ void dsv4_compress_gr
     const c10::optional<at::Tensor>& slot_ids,
     const c10::optional<at::Tensor>& pool_bt,
     int pool_epp,
-    bool stage_rel
+    bool stage_rel,
+    bool no_norm
 )
@@ -490,7 +501,7 @@ void dsv4_compress_gr
             seq, m, buf_rows, ovl_depth, W, hd, rd, Wa,
             overlap,
             slot_ids_ptr, ring_stride, ovl_stride, da_stride, db_stride,
-            pool_bt_ptr, bt_stride, pool_epp, stage_rel
+            pool_bt_ptr, bt_stride, pool_epp, stage_rel, no_norm
         );
@@ -531,7 +542,8 @@ void dsv4_compress
     const c10::optional<at::Tensor>& slot_ids,
     const c10::optional<at::Tensor>& pool_bt,
     int pool_epp,
-    bool stage_rel
+    bool stage_rel,
+    bool no_norm
 )
@@ -554,6 +566,7 @@ void dsv4_compress
         pool_bt,
         pool_epp,
-        stage_rel
+        stage_rel,
+        no_norm
     );
 }
```

C++ callers are source-compatible: `dsv4_compressor.cpp:78-80` (`BC_DSV4Compressor::run_gr`)
calls `dsv4_compress_gr` positionally and picks up the `no_norm = false` default.

### 2.4 REQUIRED companion diffs (NOT applied — review items)

The pybind entry `m.def("dsv4_compress", &dsv4_compress, "dsv4_compress");` (bindings.cpp:108)
binds the raw function pointer with **no `py::arg` defaults**, and pybind11 does not inherit C++
default arguments. Consequences after rebuilding with §2.3 alone:

- Python arity goes 18 → 19: **`DSV4Compressor.forward_fused` (dsv4.py:241-245) breaks** until
  the binding gains trailing defaults. The §2.3 patch must land with the bindings change below.
- `tests/test_dsv4_compress_kernel.py:73-75` passes only **17** args today (it omits
  `stage_rel`) — against the current 18-parameter binding that call raises `TypeError`; the test
  appears stale w.r.t. the `stage_rel` addition. The defaults below also repair this.

```diff
--- a/exllamav3/exllamav3_ext/bindings.cpp
+++ b/exllamav3/exllamav3_ext/bindings.cpp
@@ -108,1 +108,6 @@
-    m.def("dsv4_compress", &dsv4_compress, "dsv4_compress");
+    m.def("dsv4_compress", &dsv4_compress, "dsv4_compress",
+        py::arg("kv_new"), py::arg("gate_new"), py::arg("ring_kv"), py::arg("ring_gate"),
+        py::arg("ovl"), py::arg("ape"), py::arg("norm_w"), py::arg("rms_norm_eps"),
+        py::arg("inv_freq"), py::arg("dest_a"), py::arg("dest_b"), py::arg("position"),
+        py::arg("position_tensor"), py::arg("m"), py::arg("slot_ids") = py::none(),
+        py::arg("pool_bt") = py::none(), py::arg("pool_epp") = 0,
+        py::arg("stage_rel") = false, py::arg("no_norm") = false);
```

Also pending (not blockers):

- Extend `tests/test_dsv4_compress_kernel.py` with a Step-5 case
  (`hd = 256, W = 256, m = 8, overlapping = False`, zero gate + zero ape, `no_norm = True`) and a
  reference that skips test:42; the existing `torch_reference` needs an `apply_norm` switch.
- `BC_DSV4Compressor` (dsv4_compressor.h:17-115) has no `no_norm` member; the C++-transition
  path can't express the bypass yet. The Python module therefore drives `ext.dsv4_compress`
  directly (2 transitions for the chunk), which is fine for the port; add a member if the BC
  path is needed later.
- The `dsv4_compress_gr` kernel launch in dsv4_compressor.cpp passes `no_norm = false` by
  default — correct for DSV4 (which always norms).

### 2.5 Resulting call contract

`ext.dsv4_compress(..., stage_rel, no_norm)` — 19 positional args after rebuild + bindings fix.
With `no_norm = True`: output entry `= pooled comp` (fp32 acc → fp16 store), rope unchanged,
`norm_w` unused (pass ones to satisfy :399/:415).

---

## 3. What was written: `exllamav3/modules/step5_csa_compress.py`

`Step5CSACompressor(Module)` — one instance per full-attention layer, wrapping the single z
projection and the compressor pipeline for mapping A:

- `key = "model.layers.N.self_attn.sparse_indexer_z"` → `Linear` loads `{key}.weight`
  (`[256, 4096]` BF16, transposed load). `hd = W = sparse_proxy_dim = 256`,
  `m = sparse_region_block_size = 8`, `rope_dim = sparse_indexer_rope_dim = 32`.
  Fails closed (`NotImplementedError`) on any `z_norm_type` other than `"none"`; asserts
  `PAGE_SIZE % m == 0` (like dsa.py:75) and `rope_dim <= hd`.
- **`forward(x, params, ...)`** — torch reference: z → mean of each 8-token block (fp32) →
  **no norm** → GPT-J interleaved-pair rope on trailing 32 cols at `(position // m + w) * m`
  (matches dsv4_compress.cu:190-202 and test:43-49) → `(bsz, nw, hd)` fp16. Stateless: complete
  windows only; assumes `position % m == 0` for cross-chunk use.
- **`forward_fused(x, params, buf_kv, buf_gate, dest_a, dest_b, position, ...)`** — cached path
  (single job): projects z, then `ext.dsv4_compress` with `gate = 0`, `ape = 0`,
  `norm_w = ones` (unused), `ovl = None`, `no_norm = True`. Same entry addressing as the kernel
  (entry `ec0 + w`, `pool_bt`/`pool_epp` block-table remap, `stage_rel` staging rows,
  ring store at `(position + j) % buf_rows`).
- **Fallback**: if the extension raises `TypeError` (build without the flag or without the
  §2.4 binding), `forward_fused` computes the identical chunk semantics in torch
  (`_forward_fused_torch`: window gather across the chunk/ring boundary per
  dsv4_compress.cu:148-158, mean pool, rope, scatter per :204-210, ring store per :270-275).
  It **never** calls the unpatched kernel for z: that would RMS-normalize the entries and
  silently produce wrong values.
- Buffers: `ape = zeros((8, 256))` fp32 (checkpoint has none), zero-gate scratch per call,
  `norm_w = ones((256,))` fp16 to satisfy the API checks. `make_fused(inv_freq)` arms the
  rope table (`len = rope_dim / 2 = 16`); the rope **theta for the sparse path is not yet
  determined** (open item, §4) — the caller supplies the table, mirroring `make_bc`.

Not yet wired into `Step5RoboticsModel` (the architecture module still runs those layers dense,
per step5_robotics.py:25-30); that wiring plus the cache-layer (pool_idx of
`CacheLayer_dsa`-style paged pools at `epp = 32`) is the next step and is deliberately out of
scope here.

---

## 4. Remaining risk

1. **A vs B (gate)** — the one substantive open question, ~15% residual. Equivalence-test against
   the gated reference as soon as it is reachable; the switch is `gate = z` in
   `forward`/`forward_fused` (and in the fallback) with no kernel change.
2. **`m = 8` vs `m = 4`** (MEDIUM residual): if `region_block_size` is only the selection
   granularity and the compress unit is DSV4-style `m = 4` (possibly overlapping), the module's
   `m` changes and — for overlapping — `W` would need to become `2 * hd = 512`, which z (256 out)
   cannot supply, so overlapping is effectively ruled out by the tensor shape; `m = 4`
   non-overlapping remains possible. Resolve against the reference.
3. **Compressed-key rope (MEDIUM)**: whether Step-5 ropes the pooled block key at all, and if so
   at which position and with which theta (main `rope_theta` is a per-head-dim list,
   05 §5; DSV4 uses `compress_rope_theta = 160000`). The module takes the table as an argument
   and follows the kernel convention (post-pool, block start); a pre-pool rope would instead be
   a mean of rotated rows (rotation is linear — `mean(rope(x))` roped at *differing* token
   positions is not the same as roping the mean at the block position).
4. **fp32 vs fp16 accumulation**: kernel accumulates fp32 and stores fp16; the torch reference
   matches (fp32 pool → fp16 store). Equivalence tolerance should follow the existing test
   (`tol = 2e-2` relative).
5. **Not built here**: no CUDA build on this machine — the §2.3 patch is compile-checked by
   inspection only (no nvcc); the module is `py_compile`-clean. First build must run
   `tests/test_dsv4_compress_kernel.py` (all existing cases must still pass with the default
   `no_norm = false`) plus the new Step-5 case.
6. **Bindings gap** (§2.4) — the .cuh/.cu patch alone changes the Python arity; land with the
   `bindings.cpp` diff or `dsv4.py` breaks at runtime.

---

## DECISION (supersedes the "kernel flag applied" state above): kernel reverted

The `no_norm` flag patch to `dsv4_compress.cu`/`.cuh` was **reverted**. Both files are
now byte-identical to `fork/master`.

Reason: the flag cannot land safely on its own. `exllamav3/exllamav3_ext/bindings.cpp:108`
binds the kernel as `m.def("dsv4_compress", &dsv4_compress, "dsv4_compress")` with **no
`py::arg` list**. pybind11 does not read C++ default arguments, so adding a 19th parameter
raises the binding's required arity and breaks `exllamav3/modules/dsv4.py:241` and `:1369`
(both call with 18 positional args) on the next extension rebuild. That would regress the
DeepSeek-V4 CSA compressor for every architecture in the fork, not just Step-5.

Not needed for the current goal: the re-encode only quantizes tensors and never runs the
compressor forward. And `Step5CSACompressor.forward_fused` already degrades to an exact
pure-torch chunked path (`_forward_fused_torch`) when the extension lacks the flag — verified
to mirror the kernel's entry addressing. So correctness is preserved with zero fork risk.

To land the kernel fast-path later, BOTH of these must go in together:

1. `dsv4_compress.cuh` / `.cu`: `bool no_norm = false` (as previously drafted).
2. `bindings.cpp:108`: declare the full `py::arg` list so the default is reachable from
   Python, e.g.
   `m.def("dsv4_compress", &dsv4_compress, "dsv4_compress",
      py::arg("kv_new"), py::arg("gate_new"), py::arg("ring_kv"), py::arg("ring_gate"),
      py::arg("ovl") = c10::nullopt, py::arg("ape"), py::arg("norm_w"),
      py::arg("rms_norm_eps"), py::arg("inv_freq"), py::arg("dest_a"),
      py::arg("dest_b") = c10::nullopt, py::arg("position"),
      py::arg("position_tensor") = c10::nullopt, py::arg("m"),
      py::arg("slot_ids") = c10::nullopt, py::arg("pool_bt") = c10::nullopt,
      py::arg("pool_epp") = 0, py::arg("stage_rel") = false,
      py::arg("no_norm") = false);`
   Check the `c10::nullopt` defaults convert cleanly before merging; if they do not, bind a
   small lambda wrapper taking 18 args and forwarding `no_norm` explicitly instead.
3. Re-verify `tests/test_dsv4_compress_kernel.py` — it is already stale (calls 17 args vs
   the 18 the kernel took before this work), so it should be fixed in the same change.
