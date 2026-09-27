"""Default-off, decode-only selection for the three-stage source experiment.

Pure metadata: no device reads, allocations or routing changes. CUDA validates
architecture/ABI geometry too. Existing mixed-K dispatch is always the fallback.
"""
import os


def format_supported(layer):
    # Evaluated once after mixed-K metadata is loaded; never scan experts per token.
    if not (layer.is_quantized and layer.gated and layer.activation_fn == "silu" and
            layer.expert_size == 4096 and layer.intermediate_size == 2048 and
            layer.intermediate_size_padded == 2048 and layer.num_experts == 256 and
            layer.num_local_experts == 256 and layer.num_experts_per_tok == 8 and
            layer.latent_in is None and layer.latent_out is None):
        return False
    for linears, k, n in ((layer.gates, 4096, 2048), (layer.ups, 4096, 2048), (layer.downs, 2048, 4096)):
        if len(linears) != 256:
            return False
        for linear in linears:
            q = linear.inner
            if not (linear.quant_type == "exl3" and q.bias is None and q.mul1 and not q.mcg and
                    q.K in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8) and
                    linear.in_features == k and linear.in_features_unpadded == k and
                    not linear.is_sliced and linear.out_features == n and
                    linear.out_features_unpadded == n and
                    linear.pre_scale == 1.0 and linear.post_scale == 1.0 and
                    linear.weight_scale == 1.0 and linear.softcap == 0.0):
                return False
    return True


def select_entry(layer, params, rows, ext, routing_dots):
    original = ext.exl3_moe_mixedk
    if (os.environ.get("EXL3_MK_THREE_STAGE", "0") != "1" or
            not getattr(layer, "_mk_three_stage_ok", False) or
            not 1 <= rows <= 8 or "prefill" in params or
            params.get("autosplit_measure") or params.get("tp_warmup") or
            layer.routing_fn is not routing_dots):
        return original
    return getattr(ext, "exl3_moe_mixedk_three_stage", original)
