#!/usr/bin/env python3
"""Layer-by-layer MoE input comparison for DeepSeek-V4.1 exllamav3 port vs reference."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

EXL3 = os.path.expanduser("~/tp1/src/exllamav3-new")
sys.path.insert(0, EXL3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/tp1/v41port/model"), help="Model directory")
    parser.add_argument("--reference-logprobs", type=str, default=os.path.expanduser("~/tp1/v41port/ref_logprobs_4x512.safetensors"), help="Reference logprobs with tokens")
    parser.add_argument("--dump-dir", type=str, default=os.path.expanduser("~/tp1/v41port/ref_moe_in"), help="Directory with reference MoE input dumps")
    parser.add_argument("--layers", type=str, default="0,1,2,3,8,14,20,21,24,39", help="Comma-separated layer indices to compare")
    parser.add_argument("--engram-rows", type=str, default=os.path.expanduser("~/tp1/v41port/engram-rows.safetensors"), help="Engram rows file")
    parser.add_argument("--split", type=int, default=256, help="EXL3_MOE_CPU_SPLIT")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    # Set environment variables before importing exllamav3
    os.environ.setdefault("EXL3_MOE_CPU_SPLIT", str(args.split))
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
    os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
    os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if os.path.isfile(args.engram_rows):
        os.environ["EXL3_ENGRAM_ROWS"] = args.engram_rows

    layers_to_check = [int(x.strip()) for x in args.layers.split(",")]
    result = {"ok": False, "layers": []}

    try:
        import torch
        import torch.nn.functional as F
        from exllamav3 import Config, Model
        from safetensors import safe_open

        # Load reference tokens
        with safe_open(args.reference_logprobs, "pt") as f:
            ref_tokens = f.get_tensor("tokens").long()  # [4, 512]

        n_seq, seq_len = ref_tokens.shape

        # Load model
        t0 = time.time()
        cfg = Config.from_directory(args.model)
        model = Model.from_config(cfg)
        model.load("cuda:0", progressbar=False, verbose=False)
        device = torch.device("cuda:0")

        # Capture MoE inputs via hook
        captured = {}

        def make_hook(layer_idx):
            def hook(self, x, params=None, out_dtype=None):
                # x can be (seq*tokens, hidden) or (1, seq*tokens, hidden)
                if x.dim() == 3:
                    x_captured = x[0].float().cpu()  # (seq*tokens, hidden)
                else:
                    x_captured = x.float().cpu()  # (seq*tokens, hidden)

                if layer_idx not in captured:
                    captured[layer_idx] = []
                captured[layer_idx].append(x_captured)
                return None  # Don't override the actual forward
            return hook

        # Register hooks for requested layers
        from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
        original_forward = BlockSparseMLP.forward

        def hooked_forward(self, x, params, out_dtype=None):
            layer_key = self.key if hasattr(self, 'key') else None
            if layer_key:
                # Extract layer index from key like "layers.0.ffn"
                try:
                    parts = layer_key.split(".")
                    if "layers" in parts and "ffn" in parts:
                        layer_idx = int(parts[parts.index("layers") + 1])
                        if layer_idx in layers_to_check:
                            # Capture input
                            if x.dim() == 3:
                                x_cap = x[0].float().cpu()
                            else:
                                x_cap = x.float().cpu()
                            if layer_idx not in captured:
                                captured[layer_idx] = []
                            captured[layer_idx].append(x_cap)
                except (ValueError, IndexError):
                    pass
            # Call original forward
            return original_forward(self, x, params, out_dtype)

        BlockSparseMLP.forward = hooked_forward

        # Process sequences
        with torch.inference_mode():
            ids = ref_tokens.to(device)
            logits = model.forward(ids, {"attn_mode": "flash_attn_nc", "position": 0})

        torch.cuda.synchronize()

        # Load reference dumps and compute metrics
        hidden_dim = cfg.hidden_size
        n_experts = cfg.n_routed_experts
        topk = cfg.n_activated_experts

        layer_results = []
        for layer_idx in layers_to_check:
            dump_path = Path(args.dump_dir) / f"moe-input-L{layer_idx:02d}.safetensors"
            if not dump_path.exists():
                continue

            with safe_open(str(dump_path), "pt") as f:
                ref_hidden = f.get_tensor("hidden").to(torch.float32)  # (seq*tokens, hidden)
                ref_topk_ids = f.get_tensor("topk_ids").to(torch.int32)  # (seq*tokens, topk)

            # Get our captured hidden for this layer
            if layer_idx not in captured or len(captured[layer_idx]) == 0:
                continue

            exl_hidden = captured[layer_idx][0].to(torch.float32)

            # Align to reference shape: both should be (seq*tokens, hidden)
            expected_rows = n_seq * seq_len
            if ref_hidden.shape[0] != expected_rows:
                continue

            # Compute per-token relative error
            diff = exl_hidden - ref_hidden
            rel_err_per_token = (diff.pow(2).sum(dim=1).sqrt() /
                                  ref_hidden.pow(2).sum(dim=1).sqrt().clamp(min=1e-8))

            mean_rel_err = rel_err_per_token.mean().item()
            median_rel_err = rel_err_per_token.median().item()
            p90_rel_err = torch.quantile(rel_err_per_token, 0.9).item()

            # Cosine similarity per token
            cos_sim = F.cosine_similarity(exl_hidden, ref_hidden, dim=1)
            mean_cos_sim = cos_sim.mean().item()

            # Routing analysis: compute topk ids using the gate
            gate_weight = cfg.stc.get_tensor(f"layers.{layer_idx}.ffn.gate.weight", "cuda:0").float()  # [n_experts, hidden]
            gate_bias = cfg.stc.get_tensor(f"layers.{layer_idx}.ffn.gate.bias", "cuda:0").float()  # [n_experts]

            # Compute routing ids for both exl and ref
            def compute_routing_ids(hidden_input):
                # hidden: (seq*tokens, hidden)
                h_dev = hidden_input.to(device)
                logits = torch.nn.functional.linear(h_dev, gate_weight, gate_bias)  # (seq*tokens, n_experts)
                gate = torch.sqrt(F.softplus(logits)) + gate_bias  # Use formula from reference
                topk_vals, topk_ids_local = torch.topk(gate, k=topk, dim=1)  # (seq*tokens, topk)
                return topk_ids_local.cpu().to(torch.int32)

            exl_topk_ids = compute_routing_ids(exl_hidden)
            ref_topk_ids_formula = compute_routing_ids(ref_hidden)

            # Routing overlap: compare exl_topk_ids with reference dump
            # Check if each expert in exl is in the reference set
            overlap_count = 0
            for i in range(expected_rows):
                set_exl = set(exl_topk_ids[i].tolist())
                set_ref = set(ref_topk_ids[i].tolist())
                overlap = len(set_exl & set_ref)
                overlap_count += overlap

            mean_routing_overlap = overlap_count / (expected_rows * topk)

            # Formula sanity check: ref formula vs ref dump
            formula_overlap_count = 0
            for i in range(expected_rows):
                set_formula = set(ref_topk_ids_formula[i].tolist())
                set_dump = set(ref_topk_ids[i].tolist())
                overlap = len(set_formula & set_dump)
                formula_overlap_count += overlap

            formula_check = formula_overlap_count / (expected_rows * topk)

            layer_result = {
                "layer": layer_idx,
                "rel_err_mean": round(mean_rel_err, 5),
                "rel_err_p90": round(p90_rel_err, 5),
                "cosine_sim": round(mean_cos_sim, 5),
                "routing_overlap": round(mean_routing_overlap, 5),
                "formula_check": round(formula_check, 5),
            }
            layer_results.append(layer_result)
            result["layers"] = layer_results

        result["ok"] = True
        result["total_time_s"] = round(time.time() - t0, 1)

    except Exception:
        result["error"] = traceback.format_exc()[-2000:]
        traceback.print_exc()

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    # Print compact table
    if result.get("ok") and result.get("layers"):
        print("\n" + "="*80)
        print(f"{'Layer':<8} {'Rel_Err':<12} {'P90':<12} {'Cosine':<12} {'Routing':<12} {'Formula':<12}")
        print("-"*80)
        for r in result["layers"]:
            print(f"{r['layer']:<8} {r['rel_err_mean']:<12.5f} {r['rel_err_p90']:<12.5f} "
                  f"{r['cosine_sim']:<12.5f} {r['routing_overlap']:<12.5f} {r['formula_check']:<12.5f}")
        print("="*80)
        print("LAYER_DIFF_OK")
    else:
        print("LAYER_DIFF_FAIL")

    print(json.dumps(result, indent=2)[:2000])
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
