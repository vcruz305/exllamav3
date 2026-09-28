from __future__ import annotations
from typing_extensions import override
import os as _os
import weakref
import torch

from ..model.model import Model
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mimo_v2 import MiMoV2Config

"""
MTP (multi-token prediction / nextn) draft head for MiMo-V2 (Flash and Pro).

Checkpoint layout (model.mtp.layers.{k}.*, shipped separately as model_mtp.safetensors in the
source repo, never inside an EXL3 pack; point EXL3_MIMO_MTP_PATH at the file or its directory):

    enorm, hnorm            RMSNorm on the token embedding / the trunk hidden state
    eh_proj                 (2H -> H), input is cat(enorm(emb), hnorm(h)) -- embedding first
    input_layernorm         pre-attention norm
    self_attn.*             SWA-geometry attention (64 q / 8 kv heads, qk 192 / v 128, window 128,
                            swa_rope_theta, learned sinks), fused FP8 qkv_proj in the trunk's
                            TP-shard-interleaved layout
    pre_mlp_layernorm       pre-MLP norm
    mlp.{gate,up,down}_proj DENSE SwiGLU, intermediate_size (16384), FP8 128x128 block scales
    final_layernorm         norm before the shared lm_head

References: vllm/model_executor/models/mimo_v2_mtp.py (layer 0 only, EAGLE-style recursion on its
own post-final_layernorm output), sglang/srt/models/mimo_v2_nextn.py. The trunk state consumed by
hnorm is the trunk's post-final-norm output in both (sglang: return_hidden_states_before_norm=False
on the target worker; vLLM: the model's forward returns the normed state).

Two drafting modes (EXL3_MIMO_MTP_MODE):

  multi (default)  all num_nextn_predict_layers layers, DeepSeek-V3 MTP semantics: layer k
                   predicts t[i+k+2] from (h_trunk[i], t[i+k+1]); every layer has its own draft
                   cache layer (layer_idx k) and is fed TRUNK hidden states, not its own output.
                   Driven by Generator.iterate_draftmodel_mtp_multi_gen; draft length <= layers.
  chain            one layer (EXL3_MIMO_MTP_LAYER, default 0) re-applied per drafted position on
                   its own post-final_layernorm output (the vLLM/EAGLE recursion), driven by the
                   fork's existing Generator.iterate_draftmodel_mtp_gen.
EXL3_MIMO_MTP_TAP=pre switches hnorm's input to the trunk's pre-norm residual (default post).
The draft cache is a normal paged cache: attention runs with the checkpoint's sliding window.
"""

_MTP_HEAD_N = int(_os.environ.get("EXL3_MIMO_MTP_HEAD_N", "0"))


def mimo_v2_mtp_layer_index(config) -> int:
    idx = int(_os.environ.get("EXL3_MIMO_MTP_LAYER", "0"))
    assert 0 <= idx < max(1, config.num_mtp_layers), \
        f"EXL3_MIMO_MTP_LAYER={idx} out of range (checkpoint has {config.num_mtp_layers} MTP layers)"
    return idx


def build_mimo_v2_mtp_layer(config, key: str, slot: int) -> list:
    """[input combine, decoder block, final_layernorm] for one checkpoint MTP layer; `slot` is
    the draft-cache layer index"""
    from .mimo_v2 import _mimo_v2_qkv_dequant

    input_layer = Qwen3_5MTPInputLayer(
        config = config,
        key = f"{key}.input",
        key_pre_fc_norm_hidden = f"{key}.hnorm",
        key_pre_fc_norm_embedding = f"{key}.enorm",
        key_fc = f"{key}.eh_proj",
        hidden_size = config.hidden_size,
        rms_norm_eps = config.rms_norm_eps,
        native_draft_len = 1,
        out_dtype = torch.float,
        qbits_key = "mtp_bits",
        constant_bias = 0.0,
    )

    # SWA geometry. Paged cache with the window applied in the kernel (the draft cache is
    # a few layers, so the full-length allocation is cheap)
    attn = Attention(
        config = config,
        key = f"{key}.self_attn",
        layer_idx = slot,
        hidden_size = config.hidden_size,
        head_dim = config.swa_head_dim,
        v_head_dim = config.swa_v_head_dim,
        num_q_heads = config.swa_num_q_heads,
        num_kv_heads = config.swa_num_kv_heads,
        rope_settings = config.rope_settings_swa,
        sm_scale = None,
        key_fused_qkv = "qkv_proj",
        key_o = "o_proj",
        key_sinks = "attention_sink_bias" if config.add_swa_attention_sink_bias else None,
        qmap = "block.attn",
        out_dtype = torch.float,
        sliding_window = config.sliding_window - 1,
        qbits_key = "mtp_bits",
    )
    qkv_reader = _mimo_v2_qkv_dequant(
        config.qkv_ckpt_tp,
        config.swa_num_q_heads,
        config.swa_num_kv_heads,
        config.swa_head_dim,
        config.swa_v_head_dim,
    )
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
        proj.fdequant = qkv_reader
    attn.o_proj.weight_scale = config.attention_value_scale

    block = TransformerBlock(
        config = config,
        key = key,
        layer_idx = slot,
        attn_norm = RMSNorm(
            config = config,
            key = f"{key}.input_layernorm",
            rms_norm_eps = config.rms_norm_eps,
        ),
        attn = attn,
        mlp_norm = RMSNorm(
            config = config,
            key = f"{key}.pre_mlp_layernorm",
            rms_norm_eps = config.rms_norm_eps,
        ),
        mlp = GatedMLP(
            config = config,
            key = f"{key}.mlp",
            hidden_size = config.hidden_size,
            intermediate_size = config.intermediate_size,
            key_up = "up_proj",
            key_gate = "gate_proj",
            key_down = "down_proj",
            qmap = "block.mlp",
            interm_dtype = torch.half,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        ),
    )
    final_norm = RMSNorm(
        config = config,
        key = f"{key}.final_layernorm",
        rms_norm_eps = config.rms_norm_eps,
        out_dtype = torch.half,
    )
    return [input_layer, block, final_norm]


class MiMoV2MTPModel(Model):

    def __init__(
        self,
        config: MiMoV2Config,
        key_prefix: str = "model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.mtp_mode = _os.environ.get("EXL3_MIMO_MTP_MODE", "multi").strip().lower()
        assert self.mtp_mode in ("multi", "chain"), f"EXL3_MIMO_MTP_MODE={self.mtp_mode!r}"
        if self.mtp_mode == "chain":
            self.mtp_layers = [mimo_v2_mtp_layer_index(config)]
        else:
            self.mtp_layers = list(range(config.num_mtp_layers))
        assert self.mtp_layers, "MiMoV2 MTP: checkpoint has no MTP layers"

        self.modules = []
        self.layer_modules = []
        self.first_block_idx = 1
        for slot, ckpt_idx in enumerate(self.mtp_layers):
            mods = build_mimo_v2_mtp_layer(config, f"{key_prefix}.mtp.layers.{ckpt_idx}", slot)
            self.layer_modules.append(mods)
            self.modules += mods
        self.input_layer = self.layer_modules[0][0]
        self.final_norm = self.layer_modules[-1][-1]
        # Chain mode: generic prefill stops at the (only) block; multi mode never uses the
        # generic prefill/forward (see forward_layer)
        self.last_kv_module_idx = 1

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 3 if self.mtp_mode == "multi" else 2,
            "mtp_multi": self.mtp_mode == "multi",
            "mtp_num_layers": len(self.mtp_layers),
            "autosplit_load_fwd": False,
        })

        self.pre_norm_tap = _os.environ.get("EXL3_MIMO_MTP_TAP", "post").lower() == "pre"

        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        for mods in self.layer_modules:
            mods[0].attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        if self.pre_norm_tap:
            self.draft_verifier_params = {
                "export_state_layers": {self.config.num_hidden_layers - 1},
            }
        else:
            target_norm = target.modules[target.logit_layer_idx - 1]
            assert isinstance(target_norm, RMSNorm), "Expected target final RMSNorm immediately before lm_head"
            self.draft_verifier_params = {
                "export_state_norm_keys": {target_norm.key},
            }


    @torch.inference_mode()
    def forward_layer(self, slot: int, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Run MTP layer `slot` alone over input_ids (token t[j+slot+1] at position j) with
        params["target_hidden"] the trunk states h[j-1]; writes the layer's draft-cache K/V at
        cache_seqlens.. and returns the post-final_layernorm state (half)
        """
        x = self.prepare_inputs(input_ids, params)
        for m in self.layer_modules[slot]:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
        return x


    # ---- multi-layer (DeepSeek-V3 style) drafting ------------------------------------------------
    #
    # Per-job state (job.mtp_multi):
    #   h0, h   paired trunk hidden: h[:, i] is the trunk's final-norm state at position h0 + i - 1,
    #           i.e. the hnorm input for draft position h0 + i
    #   front   per layer k, first position whose layer-k draft-cache entry is not final yet. Layer k
    #           at position p consumes (h_trunk[p - 1], t[p + k]), so its entries are final only once
    #           t[p + k] is committed: after a round with n committed tokens, front[k] = n - k
    #
    # All positions of a layer between front[k] and the newest one are (re)computed in one
    # forward per round, the entries beyond the committed tokens from this round's own drafts.

    @staticmethod
    def multi_positions(front: int, n: int, k: int, upto: int | None = None) -> tuple[int, int]:
        """
        Inclusive position range layer k (re)computes. Draft round (upto None): [front, n - 1],
        the last position yields draft k + 1. Prefill (upto = last position with a paired hidden
        state): only entries that are final, and never layer 0's last position n - 1, which the
        first draft round must run to produce draft 1
        """
        if upto is None:
            return front, n - 1
        return front, min(upto, n - 1 - max(k, 1))

    @staticmethod
    def multi_token_ids(committed: torch.Tensor, drafts: list[int], p0: int, p1: int, k: int) -> torch.Tensor:
        """Token ids t[p + k] for p in [p0, p1]; ids at or past len(committed) come from drafts"""
        n = committed.shape[-1]
        a, b = p0 + k, p1 + k + 1
        ids = committed[..., a:min(b, n)]
        if b > n:
            extra = drafts[max(a, n) - n : b - n]
            assert len(extra) == b - max(a, n), "MiMoV2 MTP: missing drafted tokens"
            ids = torch.cat((ids, torch.tensor([extra], dtype = committed.dtype)), dim = -1)
        return ids

    def multi_push_hidden(self, job, p0: int, hidden: torch.Tensor):
        """Record paired hidden states for positions p0 .. p0 + T - 1 (overwrites any later ones)"""
        st = getattr(job, "mtp_multi", None)
        L = len(self.mtp_layers)
        hidden = hidden.half()
        if st is None or st["h"] is None or p0 < st["h0"] or p0 > st["h0"] + st["h"].shape[1]:
            h0 = p0
            h = hidden
            front = [p0] * L
        else:
            h0 = st["h0"]
            h = torch.cat((st["h"][:, :p0 - h0], hidden.to(st["h"].device)), dim = 1)
            front = [max(min(f, p0), h0) for f in st["front"]]
        job.mtp_multi = {"h0": h0, "h": h, "front": front}

    def _multi_trim(self, job, active: int | None = None):
        # Layers beyond the draft window never run; they must not pin the hidden history
        st = job.mtp_multi
        lo = min(st["front"][:active or len(st["front"])])
        if lo > st["h0"]:
            st["h"] = st["h"][:, lo - st["h0"]:]
            st["h0"] = lo

    def _multi_run_layer(self, k, job, seq, cache, p0, p1, drafts, params_extra = None):
        st = job.mtp_multi
        committed = seq.sequence_ids.torch_slice(0, None)
        ids = self.multi_token_ids(committed, drafts, p0, p1, k)
        hid = st["h"][:, p0 - st["h0"] : p1 - st["h0"] + 1]
        assert hid.shape[1] == ids.shape[-1] == p1 - p0 + 1, \
            f"MiMoV2 MTP: hidden/token range mismatch at layer {k}: {hid.shape[1]} vs {ids.shape[-1]}"
        params = {
            "attn_mode": "flash_attn",
            "block_table": seq.block_index_tensor,
            "cache": cache,
            "cache_seqlens": torch.tensor([p0], dtype = torch.int32),
            "target_hidden": hid,
        }
        if params_extra:
            params.update(params_extra)
        return self.forward_layer(k, ids, params), params

    @torch.inference_mode()
    def multi_prefill(self, job, seq, p0: int, paired_hidden: torch.Tensor, cache, window: int | None = None):
        """
        After a target prefill chunk: record the chunk's paired hidden states (positions
        p0 .. p0 + T - 1, T = chunk length + 1 incl. the carry) and finalize every layer's
        entries whose tokens are all committed
        """
        self.multi_push_hidden(job, p0, paired_hidden)
        st = job.mtp_multi
        upto = st["h0"] + st["h"].shape[1] - 1
        n = len(seq.sequence_ids)
        active = min(window or len(self.mtp_layers), len(self.mtp_layers))
        for k in range(active):
            a, b = self.multi_positions(st["front"][k], n, k, upto)
            if b < a:
                continue
            self._multi_run_layer(k, job, seq, cache, a, b, [])
            st["front"][k] = b + 1
        self._multi_trim(job, active)

    @torch.inference_mode()
    def multi_draft(self, job, seq, cache, window: int, calibrator = None):
        """
        One drafting round: returns (draft ids list, conf list or None). Layer k produces draft
        k + 1 from position n - 1, n = committed length
        """
        st = job.mtp_multi
        n = len(seq.sequence_ids)
        if st is None or st["h"] is None or st["h0"] + st["h"].shape[1] != n:
            # Out of step (e.g. a rewind dropped the carry): skip drafting this round; the next
            # plain target step re-seeds the paired states
            return [], None
        drafts, confs = [], []
        reach = None
        window = min(window, len(self.mtp_layers))
        for k in range(window):
            a, b = self.multi_positions(st["front"][k], n, k)
            extra = {"export_draft_conf": True} if calibrator is not None else None
            state, params = self._multi_run_layer(k, job, seq, cache, a, b, drafts, extra)
            ids = self.sample_from_state(state[:, -1:, :], params)
            drafts.append(int(ids.view(-1)[0].item()))
            if calibrator is not None and params.get("draft_conf") is not None:
                c = float(params["draft_conf"].view(-1)[-1].item())
                confs.append(c)
                e = calibrator.estimate(c)
                reach = e if reach is None else reach * e
                if k + 1 < window and reach < calibrator.confidence:
                    st["front"][k] = n - k
                    break
            st["front"][k] = n - k
        self._multi_trim(job, window)
        return drafts, (confs if calibrator is not None else None)


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        state = lm.prepare_for_device(state, params)
        if _MTP_HEAD_N > 0:
            ph = self._pruned_head(lm, state.device)
            if ph is not None:
                from ..ext import exllamav3_ext as ext
                tr, svh, n2 = ph
                inner = lm.inner
                b, q, k = state.shape
                x = state.reshape(b * q, k)
                if x.dtype != torch.half: x = x.half()
                x = x.contiguous()
                xh = torch.empty_like(x)
                y = torch.empty((b * q, n2), dtype = torch.half, device = x.device)
                ext.exl3_gemm(x, tr, y, inner.suh, xh, svh, -1, inner.mcg, inner.mul1, 0)
                if params.get("export_draft_conf"):
                    conf, ids = torch.max(y, dim = -1)
                    params["draft_conf"] = conf.view(b, q)
                    return ids.view(b, q)
                return torch.argmax(y, dim = -1).view(b, q)
        logits = lm.forward(state, params)
        if params.get("export_draft_conf"):
            logits = logits[..., :self.attached_model().config.vocab_size]
            conf, ids = torch.max(logits, dim = -1)
            params["draft_conf"] = conf
            return ids
        return torch.argmax(logits, dim = -1)


    def _pruned_head(self, lm, device):
        # Draft-only argmax over the first N vocab columns of the shared EXL3 lm_head
        # (EXL3_MIMO_MTP_HEAD_N); verification always uses the full head
        cached = getattr(self, "_pruned_head_cache", None)
        if cached is not None:
            return cached if cached is not False else None
        inner = getattr(lm, "inner", None)
        tr = getattr(inner, "trellis", None)
        if tr is None or getattr(inner, "bias", None) is not None or not hasattr(inner, "svh"):
            self._pruned_head_cache = False
            return None
        n_full = tr.shape[1] * 16
        n2 = min(_MTP_HEAD_N, n_full) // 128 * 128
        if n2 <= 0 or n2 >= n_full:
            self._pruned_head_cache = False
            return None
        tr2 = tr[:, :n2 // 16, :].contiguous().to(device)
        svh2 = inner.svh[:n2].contiguous().to(device)
        self._pruned_head_cache = (tr2, svh2, n2)
        return self._pruned_head_cache
