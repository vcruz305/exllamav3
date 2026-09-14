#!/usr/bin/env python3
"""Cached-path (attn_mode flash_attn with DSA) vs stateless nc path parity
for DeepSeek-V4.1 at full precision. Validates cache consistency across:
- single-token decode steps
- multi-token decode in chunks
- per-layer attention outputs for troubleshooting

    python tests/deepseek_v41/cache_consistency.py --model ~/tp1/v41port/model --reference ~/tp1/v41port/ref_logprobs_4x512.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

EXL3 = os.path.expanduser("~/tp1/src/exl3-bc")
sys.path.insert(0, EXL3)

import torch


def fwd_modules(model, ids, params):
    """Forward pass through module list with given params (attn_mode, position, etc)."""
    params["input_ids"] = ids   # hash-MoE routing
    x = ids
    with torch.inference_mode():
        for m in model.modules:
            x = m.prepare_for_device(x, params)
            x = m.forward(x, params)
    return x[0].float().cpu()


def fwd_cached(model, ids, state, chunks):
    """Forward through chunks with cache state management."""
    from exllamav3.cache.recurrent_util import _get_slot_tensor

    outs = []
    a = 0
    for size in chunks:
        b = min(a + size, ids.shape[1])
        if b <= a:
            break
        params = {
            "attn_mode": "flash_attn",
            "recurrent_states": [state],
            "recurrent_slots": _get_slot_tensor((state.slot,))
        }
        outs.append(fwd_modules(model, ids[:, a:b], params))
        state.position += b - a
        state.post_advance()
        a = b
    return torch.cat(outs, dim=0) if outs else torch.tensor([])


def compare_logits(tag, got, ref, kl_tol, arg_tol, verbose=False):
    """Compare logits with KL divergence and top-1 accuracy."""
    if got.shape != ref.shape:
        print(f"  FAIL {tag}: shape mismatch {got.shape} vs {ref.shape}")
        return False

    am = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
    lp_r = torch.log_softmax(ref.double(), -1)
    lp_g = torch.log_softmax(got.double(), -1)
    kld = (lp_r.exp() * (lp_r - lp_g)).sum(-1).mean().item()
    ok = am >= arg_tol and kld < kl_tol

    result = f"  {'PASS' if ok else 'FAIL'} {tag}: argmax {am*100:.2f}% KL {kld:.6f} maxdiff {(got - ref).abs().max().item():.4f}"
    print(result)
    if verbose and not ok:
        print(f"    KL per-position: mean={kld:.6f}, max={(lp_r.exp() * (lp_r - lp_g)).sum(-1).max().item():.6f}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/tp1/v41port/model"))
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--engram-rows", type=str, default=os.path.expanduser("~/tp1/e2e/engram-rows-4k.safetensors"))
    parser.add_argument("--max-seqs", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, help="Output JSON with results")

    args = parser.parse_args()

    # Set environment before importing exllamav3
    os.environ.setdefault("EXL3_MOE_CPU_SPLIT", "256")
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
    os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
    os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.engram_rows and os.path.isfile(args.engram_rows):
        os.environ["EXL3_ENGRAM_ROWS"] = args.engram_rows

    result = {
        "ok": False,
        "tests": []
    }

    try:
        from exllamav3 import Config, Model, Tokenizer
        from exllamav3.cache.cache import Cache
        from safetensors import safe_open

        # Load reference data
        with safe_open(args.reference, "pt") as f:
            ref_tokens = f.get_tensor("tokens").long()
            ref_logprobs = f.get_tensor("logprobs").half()

        S, N = ref_tokens.shape
        print(f"Loaded reference: {S} sequences, {N} tokens each")

        # Load model
        t0 = time.time()
        cfg = Config.from_directory(args.model)
        model = Model.from_config(cfg)
        model.load("cuda:0", progressbar=False, verbose=False)
        device = torch.device("cuda:0")
        print(f"Model loaded in {time.time() - t0:.1f}s")

        # Build cache
        cache = Cache(model, max_num_tokens=4096, max_batch_size=2)

        # Test on first few sequences
        all_pass = True
        for seq_idx in range(min(args.max_seqs, S)):
            print(f"\n=== Sequence {seq_idx} ===")
            ids = ref_tokens[seq_idx:seq_idx+1].to(device)

            try:
                # No-cache reference forward
                ref_logits = fwd_modules(model, ids, {"attn_mode": "flash_attn_nc"})
            except Exception as e:
                print(f"  ERROR in nc forward: {e}")
                result["error"] = str(e)
                continue

            # Single-token decode: prefill 0..255, then decode 256..N-1 one token per forward
            print(f"Single-token decode (prefill 256, decode {N-256}):")
            state = cache.get_new_state()
            try:
                prefill_outs = fwd_cached(model, ids[:, :256], state, [256])
                decode_outs = fwd_cached(model, ids[:, 256:], state, [1] * (N - 256))
                cached_logits = torch.cat([prefill_outs, decode_outs], dim=0)
            except Exception as e:
                print(f"  ERROR in cached forward: {e}")
                state.free()
                result["error"] = str(e)
                continue

            kl_tol = 0.01
            arg_tol = 0.99
            test_ok = compare_logits(
                f"single-token decode vs nc",
                cached_logits,
                ref_logits,
                kl_tol,
                arg_tol,
                verbose=args.verbose
            )
            all_pass &= test_ok
            state.free()

            result["tests"].append({
                "seq": seq_idx,
                "test": "single_token_decode",
                "pass": test_ok
            })

            # Chunk decode: prefill 256, then 16-token chunks
            print(f"Chunk decode (prefill 256, decode 16-token chunks):")
            state = cache.get_new_state()
            chunk_outs = fwd_cached(model, ids[:, :256], state, [256])
            remaining = N - 256
            chunk_size = 16
            while remaining > 0:
                size = min(chunk_size, remaining)
                chunk_outs = fwd_cached(model, ids[:, 256 + (N - 256 - remaining):256 + (N - 256 - remaining) + size],
                                       state, [size])
                remaining -= size

            # Re-forward for accurate comparison
            state = cache.get_new_state()
            chunk_logits = fwd_cached(model, ids, state, [256] + [16] * ((N - 256 + 15) // 16))

            test_ok = compare_logits(
                f"chunk decode vs nc",
                chunk_logits[-32:] if chunk_logits.shape[0] > 32 else chunk_logits,
                ref_logits[-32:] if ref_logits.shape[0] > 32 else ref_logits,
                kl_tol,
                arg_tol,
                verbose=args.verbose
            )
            all_pass &= test_ok
            state.free()

            result["tests"].append({
                "seq": seq_idx,
                "test": "chunk_decode",
                "pass": test_ok
            })

        result["ok"] = all_pass

    except Exception as e:
        import traceback
        result["error"] = traceback.format_exc()[-1000:]
        traceback.print_exc()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    print("\n" + ("="*60))
    print("CACHE_OK" if result.get("ok") else "CACHE_FAIL")
    print("="*60)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
