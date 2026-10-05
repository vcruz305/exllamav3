"""
Whole-forward CUDA graphs for single-job decode / MTP verify / MTP draft steps (EXL3_FULL_GRAPH=1).

The per-module BC paths already replay one CUDA graph per attention / MLP block, but a decode
round still issues ~2000 launches from Python between them. This captures the entire
Model.forward_ls (all layers, the head and the MTP export hooks) as one torch CUDA graph per
shape key and replays it with only three small H2D copies (ids, block table, cache lengths)
per step. Inner BC graphs are spliced into the capture as child graphs (graph.cu), so their
patched values freeze at capture: everything they patch must be constant for the key:

  - pointers: the statics below plus graph-pool activations (stable across replays)
  - block-table width W (dense MLA split configuration): part of the key
  - DSA regime (dense below index_topk, sparse above): part of the key. Sparse-regime keys
    need EXL3_DSA_DEVPOS=1 (scan width / clamps / k-pool position read from device memory);
    without it the sparse regime stays eager
  - recurrent slots and history mode: part of the key

Host-side bookkeeping that forward() does after forward_ls (recurrent state advance) still
runs per replay. A key is only captured after WARM eager runs of the same key, so every lazy
allocation, autotune and inner-graph capture (which synchronizes) has already happened. A
capture that raises (a host sync somewhere in the forward) marks the key failed: that shape
stays eager for the rest of the process. EXL3_FULL_GRAPH_DEBUG=1 prints capture events.
"""
from __future__ import annotations
import os
import torch
from ..cache.recurrent_util import advance_recurrent_states

FULL_GRAPH = os.environ.get("EXL3_FULL_GRAPH", "0") == "1"
FULL_GRAPH_DRAFT = os.environ.get("EXL3_FULL_GRAPH_DRAFT", "1") != "0"
WARM = int(os.environ.get("EXL3_FULL_GRAPH_WARM", "3"))
DEBUG = os.environ.get("EXL3_FULL_GRAPH_DEBUG", "0") == "1"
MAX_GRAPHS = int(os.environ.get("EXL3_FULL_GRAPH_MAX", "64"))


def _devpos_on() -> bool:
    try:
        from ..modules.attention_fn.bc_mla import _dsa_devpos
        return _dsa_devpos
    except Exception:
        return False


class _Entry:
    __slots__ = ("graph", "ids", "x_in", "bt", "sl", "extra", "out", "export_states", "params", "replays")


class ForwardGraphs:
    """One per model (target or draft). run() returns the forward output, or None when the
    caller must run the eager path for this step."""

    def __init__(self, model, tag: str):
        self.model = model
        self.tag = tag
        self.entries = {}
        self.seen = {}
        self.failed = set()
        self.pool = None
        self.stats = {"replay": 0, "capture": 0, "eager": 0, "fail": 0}
        cfg = model.config
        self.index_topk = getattr(cfg, "index_topk", None) if getattr(cfg, "indexer_types", None) else None
        self.devpos = _devpos_on()
        self.device = None
        devs = set()
        for m in model.modules:
            d = getattr(m, "device", None)
            if d is not None and torch.device(d).type == "cuda":
                devs.add(torch.device(d))
        # Layer-split across several GPUs is not supported (one capture stream)
        self.device = next(iter(devs)) if len(devs) == 1 else None
        self.disabled = bool(getattr(model, "loaded_tp", False)) or self.device is None or \
            len(model._get_prefetch_layers) > 0 or bool(getattr(cfg, "moe_cpu_hosts", None))
        # Leading host-resident modules (the target's CPU embedding table) run eagerly every
        # step; the graph starts at the first GPU module and takes their output as a static
        self.n_pre = 0
        if not self.disabled:
            for module, instance, idx in model.fwd_modules:
                d = getattr(module, "device", None)
                if d is not None and torch.device(d).type == "cuda":
                    break
                self.n_pre += 1

    def _log(self, msg):
        if DEBUG:
            print(f" -- [fullgraph {self.tag}] {msg}", flush = True)

    def run(self, kind: str, input_ids: torch.Tensor, params: dict, host_len: int,
            extra: tuple = ()):
        """kind: "forward" (forward_ls + recurrent advance) or "prefill" (prefill_ls).
        params: the eager params dict (paged attn: block_table, cache_seqlens host or device).
        host_len: pre-append cache length of the (single) sequence. extra: names of further
        tensor params (e.g. target_hidden) that become statics. Returns the output (forward) or
        True (prefill) when the step ran graphed, None when the caller must run it eagerly."""
        if self.disabled:
            return None
        q = input_ids.shape[-1]
        if input_ids.shape[0] != 1:
            return None
        bt = params["block_table"]
        W = bt.shape[-1]
        regime = 0
        if self.index_topk is not None:
            regime = int(host_len + q > self.index_topk)
            if regime and not self.devpos:
                self.stats["eager"] += 1
                return None
        rs = params.get("recurrent_states")
        slots = tuple(r.slot for r in rs) if rs else ()
        hist = bool(params.get("recurrent_history"))
        ex_shapes = tuple((n, tuple(params[n].shape), params[n].dtype) for n in extra)
        key = (kind, q, W, regime, slots, hist, ex_shapes)
        if key in self.failed:
            self.stats["eager"] += 1
            return None
        e = self.entries.get(key)
        if e is None:
            n = self.seen.get(key, 0) + 1
            self.seen[key] = n
            if n <= WARM or len(self.entries) >= MAX_GRAPHS:
                self.stats["eager"] += 1
                return None
            e = self._capture(key, kind, input_ids, params, host_len, extra)
            if e is None:
                self.stats["eager"] += 1
                return None

        if self.n_pre:
            self._fill(e.x_in, self._run_pre(input_ids, params))
        else:
            self._fill(e.ids, input_ids)
        self._fill(e.bt, bt)
        self._fill(e.sl, params["cache_seqlens"])
        for name, st in e.extra:
            self._fill(st, params[name])
        e.graph.replay()
        e.replays += 1
        self.stats["replay"] += 1
        if kind == "forward":
            advance_recurrent_states(input_ids, params, self.model)
        if e.export_states is not None:
            params["export_states"] = e.export_states
        return e.out if kind == "forward" else True

    def _fill(self, dst: torch.Tensor, src: torch.Tensor):
        """Stream-ordered upload of a host tensor into a static. The generator mutates its pinned
        staging (draft cache lengths += 1) right after issuing a step, so host sources are first
        snapshotted into a private pinned ring slot (32 per shape, reused long after the round's
        sync point)"""
        if src.device.type == "cpu":
            ring = getattr(self, "_ring", None)
            if ring is None:
                ring = self._ring = {}
                self._ring_idx = {}
            k = (src.dtype, tuple(src.shape))
            slots = ring.get(k)
            if slots is None:
                slots = ring[k] = [torch.empty(src.shape, dtype = src.dtype, pin_memory = True) for _ in range(32)]
            i = self._ring_idx.get(k, 0)
            self._ring_idx[k] = (i + 1) % len(slots)
            buf = slots[i]
            buf.copy_(src)
            dst.copy_(buf, non_blocking = True)
        else:
            dst.copy_(src, non_blocking = True)

    def _run_pre(self, input_ids, params):
        x = self.model.prepare_inputs(input_ids, params)
        for module, instance, idx in self.model.fwd_modules[:self.n_pre]:
            params["layer_instance"] = instance
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x

    def _run_body(self, kind, x, params):
        mods = self.model.fwd_modules[self.n_pre:]
        if kind == "forward":
            for module, instance, idx in mods:
                params["layer_instance"] = instance
                if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                    x = x[..., -num:, :].contiguous()
                x = module.prepare_for_device(x, params)
                x = module.forward(x, params)
            return x
        for module, instance, idx in mods:
            params["layer_instance"] = instance
            pf = (idx, instance) == self.model.last_kv_module_idx_instance
            params["prefill"] = pf
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if pf:
                break
        del params["prefill"]
        return None

    def _bc_ready(self) -> bool:
        """Every MLA layer must run its graphed BC block: the eager dispatch path branches on host
        cache lengths (pool completion, regime) and would be frozen into the capture"""
        from ..modules.mla_attn import MLAttention
        for m in self.model.modules:
            attn = getattr(m, "attn", None)
            if isinstance(attn, MLAttention):
                bc = [v for k, v in attn.dispatch_cache.items() if isinstance(k, tuple) and k[0] == "bcm"]
                if not bc or not all(bc):
                    return False
        return True

    def _capture(self, key, kind, input_ids, params, host_len, extra):
        if not self._bc_ready():
            self.disabled = True
            print(f" !! [fullgraph {self.tag}] disabled: an MLA layer runs the eager dispatch path "
                  f"(BC declined; EXL3_BC_MLA_FP16_WQB=1 admits fp16 indexer wq_b)", flush = True)
            return None
        dev = self.device
        e = _Entry()
        e.ids = input_ids.to(dev).clone()
        e.bt = params["block_table"].to(dev).clone()
        e.sl = params["cache_seqlens"].to(dev).clone()
        e.extra = [(n, params[n].to(dev).clone()) for n in extra]
        e.replays = 0
        cp = {k: v for k, v in params.items() if k not in ("dev_cache", "export_states", "_mla_host_seqlens")}
        cp["block_table"] = e.bt
        cp["cache_seqlens"] = e.sl
        cp["positions"] = None
        cp["_mla_host_seqlens"] = [host_len]
        for n, st in e.extra:
            cp[n] = st
        e.x_in = None
        try:
            if self.n_pre:
                pre = self._run_pre(input_ids, dict(params))
                e.x_in = pre.to(dev).clone()
                x0 = e.x_in
                self.model.prepare_inputs(e.ids, cp)
            else:
                x0 = self.model.prepare_inputs(e.ids, cp)
            torch.cuda.synchronize(dev)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.device(dev), torch.cuda.graph(g, pool = self.pool):
                out = self._run_body(kind, x0, cp)
        except Exception as ex:
            self.failed.add(key)
            self.stats["fail"] += 1
            print(f" !! [fullgraph {self.tag}] capture failed for {key[:4]}: {ex!r}"[:600], flush = True)
            if DEBUG:
                import traceback
                traceback.print_exc()
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
            return None
        if self.pool is None:
            self.pool = g.pool()
        e.graph = g
        e.out = out
        e.export_states = cp.get("export_states")
        e.params = cp
        self.entries[key] = e
        self.stats["capture"] += 1
        self._log(f"captured {key[:4]} (graphs: {len(self.entries)})")
        return e
