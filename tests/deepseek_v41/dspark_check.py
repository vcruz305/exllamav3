#!/usr/bin/env python3
"""
DSpark V4.1 drafter validation: greedy generation with and without drafter,
lossless check, and performance metrics (acceptance rate, tok/s).
"""

import sys
import os
import torch
import time
from typing import List, Tuple

# Add the repo to path
sys.path.insert(0, os.path.expanduser("~/tp1/src/exl3-dspark"))

from exllamav3.model import ExLlamaV3, ExLlamaV3Tokenizer
from exllamav3.cache import ExLlamaV3Cache
from exllamav3.generator import ExLlamaV3BaseGenerator


def load_model_and_cache(model_path: str) -> Tuple[ExLlamaV3, ExLlamaV3Cache]:
    """Load the V4.1 model and initialize cache with drafter if available."""
    print(f"Loading model from {model_path}...")
    model = ExLlamaV3.from_pretrained(model_path)

    # Check if drafter is available
    if hasattr(model, "draft_model"):
        print("DSpark drafter found and loaded")
    else:
        print("Warning: DSpark drafter not found")

    # Initialize cache with draft-enabled settings
    cache = ExLlamaV3Cache(model, batch_size=1, max_seq_len=2048)

    return model, cache


def generate_greedy(
    model: ExLlamaV3,
    tokenizer: ExLlamaV3Tokenizer,
    prompt: str,
    max_tokens: int = 128,
    use_drafter: bool = False,
    cache: ExLlamaV3Cache = None,
) -> Tuple[str, float, List[int], float]:
    """Generate tokens greedily with optional drafter."""
    input_ids = tokenizer.encode(prompt, add_bos=True).unsqueeze(0)

    if cache is None:
        cache = ExLlamaV3Cache(model, batch_size=1)

    start_time = time.time()

    # Generate loop
    generated_ids = []
    accepted_tokens = 0
    draft_steps = 0

    with torch.no_grad():
        for step in range(max_tokens):
            logits = model.forward(
                input_ids[:, -1:],
                cache=cache,
                use_draft=use_drafter and hasattr(model, "draft_model")
            )

            # Greedy sampling: take argmax
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())

            input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)

            if next_id.item() == tokenizer.eos_token_id:
                break

    elapsed = time.time() - start_time
    tok_per_sec = len(generated_ids) / elapsed if elapsed > 0 else 0

    # Decode back to text
    full_ids = torch.cat([
        tokenizer.encode(prompt, add_bos=True).unsqueeze(0),
        torch.tensor([generated_ids], dtype=torch.long)
    ], dim=1)

    text = tokenizer.decode(full_ids[0])

    return text, tok_per_sec, generated_ids, elapsed


def run_validation():
    """Main validation script."""
    model_path = os.path.expanduser("~/tp1/v41port/model")

    # Environment setup
    os.environ["EXL3_MOE_CPU_SPLIT"] = "256"
    os.environ["EXL3_MOE_CPU_THREADS"] = "16"
    os.environ["EXL3_MOE_CPU_SWAP"] = "0"
    os.environ["EXL3_MOE_STREAM_MIN_ROWS"] = "1000000000"
    # EXL3_ENGRAM_ROWS unset (disk tables)

    print("=" * 80)
    print("DeepSeek-V4.1 DSpark Drafter Validation")
    print("=" * 80)

    # Load model and tokenizer
    model, cache = load_model_and_cache(model_path)
    tokenizer = ExLlamaV3Tokenizer.from_pretrained(model_path)

    # Test prompts
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
    ]

    results_without = []
    results_with = []

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/3: {prompt[:30]}... ---")

        # Generate WITHOUT drafter
        print("Generating without drafter...")
        text_no_draft, tps_no_draft, ids_no_draft, time_no_draft = generate_greedy(
            model, tokenizer, prompt, max_tokens=128, use_drafter=False, cache=cache
        )
        results_without.append((text_no_draft, ids_no_draft, tps_no_draft, time_no_draft))
        print(f"  Tokens: {len(ids_no_draft)}, Tok/s: {tps_no_draft:.2f}, Time: {time_no_draft:.2f}s")

        # Generate WITH drafter
        if hasattr(model, "draft_model"):
            print("Generating with drafter...")
            cache.reset()  # Reset cache for fair comparison
            text_with_draft, tps_with_draft, ids_with_draft, time_with_draft = generate_greedy(
                model, tokenizer, prompt, max_tokens=128, use_drafter=True, cache=cache
            )
            results_with.append((text_with_draft, ids_with_draft, tps_with_draft, time_with_draft))

            # Check lossless
            if ids_no_draft == ids_with_draft:
                print(f"  ✓ LOSSLESS: Token sequences match")
                print(f"  Tokens: {len(ids_with_draft)}, Tok/s: {tps_with_draft:.2f}, Time: {time_with_draft:.2f}s")
                if len(ids_no_draft) > 0:
                    speedup = tps_no_draft / tps_with_draft if tps_with_draft > 0 else 0
                    print(f"  Speedup: {speedup:.2f}x")
            else:
                print(f"  ✗ NOT LOSSLESS")
                print(f"    Without: {ids_no_draft[:50]}")
                print(f"    With:    {ids_with_draft[:50]}")
        else:
            print("  Drafter not available")
            results_with.append(None)

    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)

    lossless_count = 0
    for i, (r_no, r_with) in enumerate(zip(results_without, results_with)):
        if r_with is not None:
            if r_no[1] == r_with[1]:  # Compare token sequences
                lossless_count += 1
                print(f"Prompt {i+1}: ✓ PASS (lossless)")
            else:
                print(f"Prompt {i+1}: ✗ FAIL (not lossless)")
        else:
            print(f"Prompt {i+1}: - (drafter not available)")

    print(f"\nLossless: {lossless_count}/{len(prompts)} prompts")

    # Aggregate stats
    if results_with and any(r is not None for r in results_with):
        mean_tokens_no_draft = sum(r[1] for r in results_without if r) / len(results_without)
        mean_tps_no_draft = sum(r[2] for r in results_without if r) / len(results_without)

        with_draft_valid = [r for r in results_with if r is not None]
        if with_draft_valid:
            mean_tokens_with_draft = sum(r[1] for r in with_draft_valid) / len(with_draft_valid)
            mean_tps_with_draft = sum(r[2] for r in with_draft_valid) / len(with_draft_valid)

            print(f"\nWithout drafter: {mean_tps_no_draft:.2f} tok/s ({mean_tokens_no_draft:.0f} tokens/prompt)")
            print(f"With drafter:    {mean_tps_with_draft:.2f} tok/s ({mean_tokens_with_draft:.0f} tokens/prompt)")


if __name__ == "__main__":
    run_validation()
