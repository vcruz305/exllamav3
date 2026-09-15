#!/usr/bin/env python3
"""
Test V4.1 cached attention forward pass (single token generation).
If this works, the cached path is not broken.
"""

import sys
import os
import torch

# Set environment before imports
os.environ.setdefault("EXL3_MOE_CPU_SPLIT", "256")
os.environ.setdefault("EXL3_MOE_CPU_THREADS", "16")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("EXL3_MOE_STREAM_MIN_ROWS", "1000000000")

sys.path.insert(0, os.path.expanduser("~/tp1/src/exl3-dspark"))

from exllamav3 import Config, Model, Tokenizer, Cache

def main():
    model_path = os.path.expanduser("~/tp1/v41port/model")

    print("V4.1 Cached Forward Pass Test (single token)")
    print("=" * 60)

    # Load config and model
    config = Config.from_directory(model_path)
    model = Model.from_config(config, component="text")
    tokenizer = Tokenizer.from_config(config)

    print(f"Model: {config.arch_string}")
    print(f"Vocab: {config.vocab_size}")

    # Load to GPU
    print("Loading model to CUDA...")
    model.load("cuda:0", progressbar=False, verbose=False)

    # Create cache
    cache = Cache(model, batch_size=1, max_seq_len=2048)

    # Test with a single short prompt
    prompt = "Hello"
    input_ids = tokenizer.encode(prompt, add_bos=True).unsqueeze(0)

    print(f"\nPrompt: '{prompt}'")
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        # Prefill
        print("Prefill phase...")
        logits = model.forward(input_ids, cache=cache, preprocess_only=False)
        print(f"  Logits shape: {logits.shape}")

        # Single token generation
        print("Cached forward (1 token)...")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print(f"  Next token: {next_token.item()}")

        logits = model.forward(next_token, cache=cache, preprocess_only=False)
        print(f"  Logits shape: {logits.shape}")

        # Another token
        print("Cached forward (2 tokens)...")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print(f"  Next token: {next_token.item()}")

        logits = model.forward(next_token, cache=cache, preprocess_only=False)
        print(f"  Logits shape: {logits.shape}")

    print("\n✓ Test PASSED - cached forward path works")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
