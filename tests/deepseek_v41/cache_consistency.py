#!/usr/bin/env python3
"""Cached-path validation for DeepSeek-V4.1: prefill 0-255, decode 256-319 single tokens
and 16-token chunks. Compares cached logits with no-cache reference (KL mean < 0.01,
top-1 >= 99%). Uses model.forward with proper generator-style params."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

EXL3 = os.path.expanduser("~/tp1/src/exl3-bc")
sys.path.insert(0, EXL3)

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/tp1/v41port/model"))
    parser.add_argument("--reference", type=str, required=True)
    parser.add_argument("--engram-rows", type=str, default=os.path.expanduser("~/tp1/e2e/engram-rows-4k.safetensors"))
    parser.add_argument("--output", type=str, help="Output JSON")
    parser.add_argument("--seq-idx", type=int, default=0, help="Sequence index to test")

    args = parser.parse_args()

    os.environ.setdefault("EXL3_MOE_CPU_SPLIT", "256")
    os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
    os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
    os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.engram_rows and os.path.isfile(args.engram_rows):
        os.environ["EXL3_ENGRAM_ROWS"] = args.engram_rows

    result = {"ok": False, "tests": []}

    try:
        from exllamav3 import Config, Model
        from exllamav3.cache.cache import Cache
        from safetensors import safe_open
        from torch.nn.functional import log_softmax

        # Load reference
        with safe_open(args.reference, "pt") as f:
            ref_tokens = f.get_tensor("tokens").long()
            ref_logprobs = f.get_tensor("logprobs").half()

        S, N = ref_tokens.shape
        print(f"Reference: {S} sequences, {N} tokens each")

        # Load model - BEFORE Cache (per test_dsv4_cached.py pattern)
        t0 = time.time()
        cfg = Config.from_directory(args.model)
        model = Model.from_config(cfg)
        device = torch.device("cuda:0")
        print(f"Model from_config done in {time.time() - t0:.1f}s")

        # Create cache BEFORE model.load (same model object)
        cache = Cache(model, max_num_tokens=8192, max_batch_size=1)

        # Now load model to device
        t0_load = time.time()
        model.load(device, progressbar=False, verbose=False)
        print(f"Model loaded in {time.time() - t0_load:.1f}s")

        ids = ref_tokens[args.seq_idx:args.seq_idx+1].to(device)

        # No-cache reference forward
        print(f"No-cache forward ({N} tokens)...")
        t0 = time.time()
        ref_logits = model.forward(ids, {"attn_mode": "flash_attn_nc"})
        nc_time = time.time() - t0
        ref_logits = ref_logits[0].float().cpu()
        print(f"  Done in {nc_time:.1f}s")

        # Cached: prefill 0-255, then 64 single-token decodes
        print(f"Cached prefill (0-255)...")

        # Get state INSIDE inference_mode (per test_dsv4_cached.py line 109-110)
        with torch.inference_mode():
            state = cache.get_new_state()

        # Build block table: need enough pages for all tokens
        # For V4.1, cache uses DSV4LayerState with epp (entries per page) = PAGE_SIZE/compress_rate
        # Allocate enough pages for N tokens worth of entries
        num_pages = (N // 256) + 2  # Conservative allocation
        block_table = torch.arange(num_pages, dtype=torch.int32, device=device).unsqueeze(0)

        t0 = time.time()
        with torch.inference_mode():
            model.forward(
                ids[:, :256],
                {
                    "attn_mode": "flash_attn",
                    "cache": cache,
                    "block_table": block_table,
                    "cache_seqlens": torch.tensor([0], dtype=torch.int32, device=device),
                    "recurrent_states": [state],
                    "positions": torch.arange(256, dtype=torch.int32, device=device),
                },
            )
        state.position = 256
        state.post_advance()
        prefill_time = time.time() - t0
        print(f"  Prefill done in {prefill_time:.1f}s")

        # Single-token decode loop
        print(f"Cached decode (256-319, single tokens)...")
        cached_logits = torch.zeros((N, cfg.vocab_size), dtype=torch.float32)
        cached_logits[:256] = ref_logits[:256]  # Use nc logits for prefilled range

        t0 = time.time()
        for pos in range(256, N):
            with torch.inference_mode():
                logits = model.forward(
                    ids[:, pos:pos+1],
                    {
                        "attn_mode": "flash_attn",
                        "cache": cache,
                        "block_table": block_table,
                        "cache_seqlens": torch.tensor([pos], dtype=torch.int32, device=device),
                        "recurrent_states": [state],
                        "positions": torch.tensor([pos], dtype=torch.int32, device=device),
                    },
                )
            cached_logits[pos] = logits[0].float().cpu()
            state.position += 1
            state.post_advance()

        decode_time = time.time() - t0
        print(f"  Decode done in {decode_time:.1f}s ({(N-256)/decode_time:.1f} tok/s)")

        # Compare cached vs nc
        lp_ref = log_softmax(ref_logits[256:N].double(), -1)
        lp_cached = log_softmax(cached_logits[256:N].double(), -1)
        kl = (lp_ref.exp() * (lp_ref - lp_cached)).sum(-1)
        top1 = (cached_logits[256:N].argmax(-1) == ref_logits[256:N].argmax(-1)).float()

        kl_mean = kl.mean().item()
        kl_max = kl.max().item()
        top1_mean = top1.mean().item()

        print(f"\nSingle-token decode vs nc:")
        print(f"  KL: mean={kl_mean:.6f}, max={kl_max:.6f}")
        print(f"  Top-1: {top1_mean*100:.2f}%")

        test_ok = kl_mean < 0.01 and top1_mean >= 0.99
        result["tests"].append({
            "test": "single_token_decode",
            "kl_mean": float(kl_mean),
            "kl_max": float(kl_max),
            "top1": float(top1_mean),
            "pass": test_ok,
        })

        # Cached with 16-token chunks
        print(f"\nCached decode (256-319, 16-token chunks)...")
        state.free()
        with torch.inference_mode():
            state = cache.get_new_state()

        cached_logits_chunks = torch.zeros((N, cfg.vocab_size), dtype=torch.float32)
        cached_logits_chunks[:256] = ref_logits[:256]

        # Prefill again
        with torch.inference_mode():
            model.forward(
                ids[:, :256],
                {
                    "attn_mode": "flash_attn",
                    "cache": cache,
                    "block_table": block_table,
                    "cache_seqlens": torch.tensor([0], dtype=torch.int32, device=device),
                    "recurrent_states": [state],
                    "positions": torch.arange(256, dtype=torch.int32, device=device),
                },
            )
        state.position = 256
        state.post_advance()

        # Decode in 16-token chunks
        t0 = time.time()
        for chunk_start in range(256, N, 16):
            chunk_end = min(chunk_start + 16, N)
            chunk_size = chunk_end - chunk_start
            with torch.inference_mode():
                logits = model.forward(
                    ids[:, chunk_start:chunk_end],
                    {
                        "attn_mode": "flash_attn",
                        "cache": cache,
                        "block_table": block_table,
                        "cache_seqlens": torch.tensor([chunk_start], dtype=torch.int32, device=device),
                        "recurrent_states": [state],
                        "positions": torch.arange(chunk_start, chunk_end, dtype=torch.int32, device=device),
                    },
                )
            cached_logits_chunks[chunk_start:chunk_end] = logits[0].float().cpu()
            state.position += chunk_size
            state.post_advance()

        chunk_time = time.time() - t0
        print(f"  Chunks done in {chunk_time:.1f}s ({(N-256)/chunk_time:.1f} tok/s)")

        # Compare chunks
        lp_ref_chunks = log_softmax(ref_logits[256:N].double(), -1)
        lp_cached_chunks = log_softmax(cached_logits_chunks[256:N].double(), -1)
        kl_chunks = (lp_ref_chunks.exp() * (lp_ref_chunks - lp_cached_chunks)).sum(-1)
        top1_chunks = (cached_logits_chunks[256:N].argmax(-1) == ref_logits[256:N].argmax(-1)).float()

        kl_mean_chunks = kl_chunks.mean().item()
        kl_max_chunks = kl_chunks.max().item()
        top1_mean_chunks = top1_chunks.mean().item()

        print(f"\n16-token chunks vs nc:")
        print(f"  KL: mean={kl_mean_chunks:.6f}, max={kl_max_chunks:.6f}")
        print(f"  Top-1: {top1_mean_chunks*100:.2f}%")

        chunks_ok = kl_mean_chunks < 0.01 and top1_mean_chunks >= 0.99
        result["tests"].append({
            "test": "chunks_16",
            "kl_mean": float(kl_mean_chunks),
            "kl_max": float(kl_max_chunks),
            "top1": float(top1_mean_chunks),
            "pass": chunks_ok,
        })

        result["ok"] = test_ok and chunks_ok
        state.free()

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print("\n=== FULL TRACEBACK ===")
        print(tb_str)
        print("=== END TRACEBACK ===\n")
        result["error"] = tb_str[-1000:]

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    print("\n" + ("="*60))
    print("CACHE_OK" if result.get("ok") else "CACHE_FAIL")
    print("="*60)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
