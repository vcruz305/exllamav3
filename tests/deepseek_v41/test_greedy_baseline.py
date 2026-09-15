#!/usr/bin/env python3
"""
Test greedy generation baseline for V4.1 to verify cached attention path works.
If this fails, the cached path is broken and drafter won't help.
"""

import sys
import os
import torch
import time

# Set environment before imports
os.environ.setdefault("EXL3_MOE_CPU_SPLIT", "256")
os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")

sys.path.insert(0, os.path.expanduser("~/tp1/src/exl3-dspark"))

from exllamav3 import Config, Model, Tokenizer, Cache

def main():
    model_path = os.path.expanduser("~/tp1/v41port/model")

    print("=" * 80)
    print("V4.1 Greedy Generation Baseline Test")
    print("=" * 80)

    print(f"Loading model from {model_path}...")
    config = Config.from_directory(model_path)
    model = Model.from_config(config, component="text")
    tokenizer = Tokenizer.from_config(config)

    print(f"Model: {config.arch_string}")
    print(f"Vocab size: {config.vocab_size}")

    # Load model to GPU
    print("Loading model to CUDA...")
    model.load("cuda:0", progressbar=False)

    # Create cache
    print("Creating cache...")
    cache = Cache(model, batch_size=1, max_seq_len=2048)

    # Test prompts
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Hello world",
    ]

    print("\n" + "=" * 80)
    print("Generation Test")
    print("=" * 80)

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/3: '{prompt}' ---")

        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_bos=True).unsqueeze(0)
        print(f"  Prompt tokens: {input_ids.shape[1]}")

        # Generate 128 tokens
        generated_ids = []
        start_time = time.time()

        cache.reset()
        with torch.no_grad():
            # Prefill phase
            logits = model.forward(
                input_ids,
                cache=cache,
                preprocess_only=False,
            )

            # Token generation loop
            for step in range(128):
                # Get last token logits
                last_logits = logits[:, -1, :]

                # Greedy: argmax
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
                generated_ids.append(next_token.item())

                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    print(f"  EOS at step {step + 1}")
                    break

                # Forward for next token
                logits = model.forward(
                    next_token,
                    cache=cache,
                    preprocess_only=False,
                )

        elapsed = time.time() - start_time
        tok_per_sec = len(generated_ids) / elapsed if elapsed > 0 else 0

        # Decode
        all_ids = torch.cat([input_ids, torch.tensor([generated_ids], dtype=torch.long, device="cpu")], dim=1)
        text = tokenizer.decode(all_ids[0])

        print(f"  Generated {len(generated_ids)} tokens in {elapsed:.2f}s ({tok_per_sec:.2f} tok/s)")
        print(f"  Text: {text[:100]}...")

    print("\n✓ Baseline test PASSED")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Test FAILED with error:")
        import traceback
        traceback.print_exc()
        sys.exit(1)
