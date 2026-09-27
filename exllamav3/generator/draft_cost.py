"""Experimental, opt-in DFlash verification cost policy; no device operations.

The product of calibrated conditional probabilities is a heuristic, not a
counterfactual acceptance observation or a performance guarantee. Sparse-bin
nearest-neighbour estimates are deliberately not used by this policy.
"""
from __future__ import annotations
import math
import hashlib
import json
import os
from pathlib import Path
import re


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate cost-profile JSON key: " + key)
        result[key] = value
    return result


class DFlashCostPolicy:
    @classmethod
    def from_profile(cls, profile, context):
        """Bind an externally attested deployment to one observational profile.

        This validates provenance and shape, not weight contents. Round8 did not
        record a target weight revision/hash; its owner must explicitly attest
        the identical retained pack before enabling this experiment.
        """
        required = {
            "source_revision": "ca4a880e8918e1985fd25e06c6aff561666d3f14",
            "model_pack": "MiMo-V2.6-Flash-RL-EXL3/2.50bpw",
            "drafter_revision": "d50ead3c6a3dec221e9a595fbdc103ef60db594e",
            "drafter_bpw": 4.0, "native_block": 8, "cache_bits": 4,
            "context_tokens": 4096, "chunk_size": 4096, "batch_size": 1,
            "gpu_split": 106, "reserve_mb": 8192, "dynamic_confidence": 0.6,
            "batch_verify": 0, "handled_elision": 1, "counts_elision": 0,
            "kernel": "original-CUDA", "hardware": "France-GB10",
            "target_weight_identity": "owner-attested unchanged round8 2.50bpw pack",
        }
        try:
            if (type(profile["schema"]) is not int or profile["schema"] != 1 or profile["units"] != "ms" or
                    profile["metric"] != "whole_round_host" or
                    profile["population"] != "pooled_warm_actual_q" or profile["provisional"] is not True):
                raise ValueError("Unsupported DFlash cost profile schema/units/population")
            if profile["context"] != context:
                raise ValueError("Cost profile does not match attested deployment context")
            if set(context) != set(required) | {"model_directory", "drafter_directory"}:
                raise ValueError("Incomplete cost profile context")
            if any(context[k] != v or type(context[k]) is not type(v) for k, v in required.items()):
                raise ValueError("Unsupported model/drafter/revision/cache configuration")
            if any(not isinstance(context[k], str) or not context[k] for k in ("model_directory", "drafter_directory")):
                raise ValueError("Missing model directory identity")
            sources = profile["sources"]
            if not sources or any(not isinstance(k, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for k, v in sources.items()):
                raise ValueError("Missing or malformed source hash provenance")
            rows = profile["q_costs"]
            if len(rows) != 8:
                raise ValueError("Cost profile must cover every native q=1..8")
            for q, row in enumerate(rows, 1):
                cost = row["mean_ms"]
                counts = row["case_counts"]
                low, high = row["min_ms"], row["max_ms"]
                if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (low, high)) or
                        type(cost) not in (int, float) or not low <= cost <= high):
                    raise ValueError("Invalid per-q cost range")
                if (type(row["q"]) is not int or row["q"] != q or
                        type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0 or
                        type(row["samples"]) is not int or row["samples"] < 1 or
                        set(counts) != {"code", "prose"} or
                        any(type(c) is not int or c < 0 for c in counts.values()) or
                        sum(counts.values()) != row["samples"]):
                    raise ValueError("Invalid per-q cost/sample coverage")
            if sum(r["samples"] for r in rows) != profile["warm_rounds"]:
                raise ValueError("Inconsistent warm-round sample count")
        except (KeyError, TypeError, AttributeError, IndexError) as e:
            raise ValueError("Malformed DFlash cost profile") from e
        obj = cls()
        obj.costs = tuple(r["mean_ms"] for r in rows)
        return obj

    @classmethod
    def from_environment(cls, g, batch_verify):
        """Initialization only; fail clearly instead of failing mid-request."""
        path = os.environ.get("EXL3_DFLASH_COST_PROFILE")
        digest = os.environ.get("EXL3_DFLASH_COST_PROFILE_SHA256", "")
        attestation = os.environ.get("EXL3_DFLASH_COST_CONTEXT")
        if not path or not re.fullmatch(r"[0-9a-f]{64}", digest) or not attestation:
            raise ValueError("Cost-aware DFlash requires PROFILE, PROFILE_SHA256 and CONTEXT")
        try:
            raw = Path(path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Untrusted DFlash cost profile hash")
            profile = json.loads(raw, object_pairs_hook=_unique_object)
            context = json.loads(Path(attestation).read_bytes(), object_pairs_hook=_unique_object)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError("Cannot read DFlash cost profile/context") from e
        if not isinstance(context, dict) or context.get("attest_same_round8_weights_and_runtime") is not True:
            raise ValueError("Cost-aware DFlash requires explicit runtime/weight attestation")
        obj = cls.from_profile(profile, context.get("context"))
        ctx = profile["context"]
        if (not g.dflash_draft or g.mtp_draft or g.draft_model.config.block_size != 8 or
                not 1 <= g.num_draft_tokens <= 7 or not g.dynamic_draft or g.draft_calibrator is None or
                g.draft_confidence != 0.6 or g.max_batch_size != 1 or g.max_chunk_size != 4096 or
                batch_verify or
                g.model.config.directory != ctx["model_directory"] or
                g.draft_model.config.directory != ctx["drafter_directory"]):
            raise ValueError("Unsupported cost-aware DFlash generator configuration")
        for cache in (g.cache, g.draft_cache):
            if (cache.max_num_tokens != 4096 or not cache.layers or
                    any(getattr(layer, "k_bits", None) != 4 or getattr(layer, "v_bits", None) != 4
                        for layer in cache.layers.values())):
                raise ValueError("Cost-aware DFlash requires the measured Q4/4096 caches")
        if (os.environ.get("EXL3_MOE_MIXEDK_ELIDE_HANDLED", "0") != "1" or
                os.environ.get("EXL3_UMA_RESERVE_MB") != "8192"):
            raise ValueError("Unsupported cost-aware DFlash runtime flags")
        if g.recurrent_cache is not None:
            obj._bind_mimo_swa(g)
        return obj

    def _bind_mimo_swa(self, g):
        """Allow only the retained MiMo window ring, not arbitrary recurrence.

        Q4 applies to cache.layers (global attention), NOT recurrent_layers:
        SWA uses half-precision per-slot rings. This is a structural check, not
        weight/runtime authentication; the independent attestation remains required.
        """
        from exllamav3.architecture.mimo_v2 import MiMoV2Model
        from exllamav3.modules.sliding_attn import SWAState, SWALayerState, SlidingAttention
        from exllamav3.cache.recurrent import RecurrentCache

        m, c = g.model, g.cache
        cfg = m.config
        global_layers = {0, 5, 11, 17, 23, 29, 35, 41, 47}
        pattern = [0 if i in global_layers else 1 for i in range(48)]
        try:
            valid = (
                type(m) is MiMoV2Model and not m.loaded_tp and m.swa_full is False and
                m.recurrent_state_cls is SWAState and c.recurrent_state_cls is SWAState and
                c.model is m and cfg.layer_map is None and cfg.num_hidden_layers == 48 and
                cfg.hybrid_layer_pattern == pattern and cfg.sliding_window == 128 and
                cfg.head_dim == 192 and cfg.v_head_dim == 128 and cfg.num_kv_heads == 4 and
                cfg.swa_num_kv_heads == 8 and c.num_slots == 1 and c.max_history == 7 and
                g.draft_reserve_tokens == 7 and g.recurrent_checkpoint_interval == 2048 and
                g.recurrent_checkpoint_interval_pp == 32768 and
                type(g.recurrent_cache) is RecurrentCache and g.recurrent_cache.model is m and
                g.recurrent_cache_size == g.recurrent_cache.max_size == 256 * 1024**2 and
                set(c.layers) == {(i, 0) for i in global_layers} and
                set(c.recurrent_layers) == {(i, 0) for i in range(48) if i not in global_layers} and
                all(type(layer) is SWALayerState and type(layer.module) is SlidingAttention and
                    layer.module.layer_idx == key[0] and layer.module.sliding_window == 127 and
                    layer.module.sliding_window_overp == 512 and layer.module.kv_state_size == 768 and
                    layer.module.head_dim == 192 and layer.module.num_kv_heads == 8 and
                    layer.max_history == 7 and layer.max_batch_size == 1
                    for key, layer in c.recurrent_layers.items())
            )
        except AttributeError:
            valid = False
        if not valid:
            raise ValueError("Unsupported cost-aware recurrent configuration; requires retained MiMo SWA")
        self._mimo_swa = (m, c, g.recurrent_cache, SWAState)

    def select(self, generator, conf, cap):
        """Narrow singleton neutral-greedy guard; all other jobs retain DDS.

        Boundary rounds retain DDS rather than treating suppressed/terminal tokens
        as useful output. Only validated MiMo SWA interior rounds are admitted;
        checkpoint/requeue boundaries and banned-string rewinds are not modeled.
        Only already-read CPU scores are inspected here. Unexpected EOS still
        follows the unchanged target verifier; the objective is not realized yield.
        """
        g = generator
        if (not g.dflash_draft or g.mtp_draft or g.draft_model.config.block_size != 8 or
                not 1 <= g.num_draft_tokens <= 7 or len(g.active_jobs) != 1 or
                conf.shape != (1, 7)):
            return None
        if g.recurrent_cache is not None:
            bound = getattr(self, "_mimo_swa", None)
            if (bound is None or bound[0] is not g.model or bound[1] is not g.cache or
                    bound[2] is not g.recurrent_cache):
                return None
        job = g.active_jobs[0]
        if (len(job.sequences) != 1 or not job.is_prefill_done() or job.filters or
                job.forced_ids is not None or job.return_probs or job.return_top_tokens or
                job.return_logits or job.banned_strings or job.checkpoint is not None or
                job.new_tokens < 0 or not getattr(job.sampler, "supports_batch_verify", False) or
                getattr(job.sampler, "reqs_past_ids", True) or job.device_logit_mask is not None):
            return None
        q = cap + 1
        if g.recurrent_cache is not None:
            # is_checkpoint_boundary() measures decode checkpoints from cached
            # pages, not absolute position. Keep DDS at/currently approaching one,
            # including the bonus position; never mutate the job to probe it.
            pos = job.sequences[0].kv_position
            state = getattr(job, "recurrent_state", None)
            if (type(state) is not bound[3] or state.cache is not g.cache or state.position != pos or
                    getattr(job, "is_requeued", True) or getattr(job, "is_finished", True) or
                    getattr(job, "checkpoint_rewound", True) or job.stop_strings or job.loop_detector or
                    g.recurrent_checkpoint_interval != 2048 or
                    type(job.cached_pages) is not int or not 0 <= job.cached_pages * 256 <= pos):
                return None
            phase = (pos - job.cached_pages * 256) % g.recurrent_checkpoint_interval
            if phase == 0 or g.recurrent_checkpoint_interval - phase <= q:
                return None
        # receive_sample increments new_tokens before its strict requeue comparison.
        remaining = job.max_new_tokens - job.new_tokens
        rq_remaining = job.max_rq_tokens - g.draft_reserve_tokens - job.new_tokens + 1
        if remaining <= q or rq_remaining <= q:
            return None
        return self.choose(g.draft_calibrator, conf[0].tolist(), cap)

    def choose(self, cal, scores, cap):
        """Return proposal count, or None to preserve ordinary DDS unchanged."""
        if (cal is None or type(cap) is not int or not 0 <= cap <= 7 or
                len(scores) < cap or cal.total < max(64.0, cal.burn_in)):
            return None
        probabilities = []
        for score in scores[:cap]:
            if not math.isfinite(score):
                return None
            # Require the actual bin, not estimate()'s possibly distant fallback.
            b = cal.bins.get(math.floor(score / cal.bin_width))
            if b is None or b[0] < max(8.0, cal.min_count):
                return None
            p = cal.estimate(score)
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                return None
            probabilities.append(p)
        best = 0
        expected = survival = 1.0
        best_rate = expected / self.costs[0]
        for k, p in enumerate(probabilities, 1):
            survival *= p
            expected += survival
            rate = expected / self.costs[k]
            if rate > best_rate:  # exact ties prefer the least verification work
                best, best_rate = k, rate
        return best
