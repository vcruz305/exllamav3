#!/usr/bin/env python3
"""
Unit test for grouped MoE dispatch with mixed-K experts.
Builds a single layer and compares grouped vs dense dispatch outputs.
"""
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from exllamav3.model import ExLlamaV3
from exllamav3.config import ExLlamaV3Config
from exllamav3.lora import ExLlamaV3Lora

# Configuration
MODEL_PATH = Path.home() / "hf_pack" / "deepseek-v4.1-gptq"  # Adjust if needed
BATCH_SIZE = 4
SEQ_LEN = 32
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
FLOAT_TOLERANCE = 1e-3  # Max absolute difference
RELATIVE_TOLERANCE = 1e-2  # Max relative error

def test_grouped_moe_dispatch():
    """Test that grouped dispatch produces identical output to dense dispatch."""

    print("=" * 70)
    print("Grouped MoE Unit Test")
    print("=" * 70)

    # Load model config
    print(f"Loading model from {MODEL_PATH}...")
    config = ExLlamaV3Config(str(MODEL_PATH))

    # Check if model has mixed-K MoE
    if not hasattr(config, 'num_experts') or config.num_experts <= 1:
        print("Model does not have MoE, skipping test")
        return True

    # Load model
    model = ExLlamaV3(config)
    model = model.to(DEVICE)
    model.eval()

    print(f"Model loaded. num_experts={getattr(config, 'num_experts', 'N/A')}")

    # Get first MoE layer
    moe_layer = None
    for name, module in model.named_modules():
        if hasattr(module, 'exl3_k_groups') and module.exl3_k_groups is not None:
            moe_layer = module
            layer_name = name
            break

    if moe_layer is None:
        print("No mixed-K MoE layer found, skipping test")
        return True

    print(f"Testing MoE layer: {layer_name}")

    # Create random test input
    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, config.hidden_size, dtype=torch.float16, device=DEVICE)

    # Test 1: Grouped dispatch (grouped ON)
    print("\n1. Running grouped dispatch (EXL3_MOE_GROUPED=1)...")
    os.environ['EXL3_MOE_GROUPED'] = '1'
    os.environ['EXL3_MOE_GROUPED_DEBUG'] = '1'
    moe_layer.grouped_launches = 0
    moe_layer.dense_loop_calls = 0

    try:
        output_grouped = moe_layer(hidden_states.clone())
        print(f"   Grouped dispatch succeeded")
        print(f"   Counter: grouped_launches={moe_layer.grouped_launches}, dense_loop_calls={moe_layer.dense_loop_calls}")

        if moe_layer.grouped_launches == 0:
            print("   WARNING: No grouped launches detected!")
            return False
        if moe_layer.dense_loop_calls > 0:
            print(f"   WARNING: Dense loop was called {moe_layer.dense_loop_calls} times (expected 0)")
            return False
    except Exception as e:
        print(f"   ERROR: Grouped dispatch failed with {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Dense dispatch (grouped OFF)
    print("\n2. Running dense dispatch (EXL3_MOE_GROUPED=0)...")
    os.environ['EXL3_MOE_GROUPED'] = '0'
    os.environ['EXL3_MOE_GROUPED_DEBUG'] = '0'
    moe_layer.grouped_launches = 0
    moe_layer.dense_loop_calls = 0

    try:
        output_dense = moe_layer(hidden_states.clone())
        print(f"   Dense dispatch succeeded")
        print(f"   Counter: grouped_launches={moe_layer.grouped_launches}, dense_loop_calls={moe_layer.dense_loop_calls}")
    except Exception as e:
        print(f"   ERROR: Dense dispatch failed with {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Compare outputs
    print("\n3. Comparing outputs...")
    max_diff = torch.abs(output_grouped - output_dense).max().item()
    mean_diff = torch.abs(output_grouped - output_dense).mean().item()

    # Relative error (avoid division by zero)
    denominator = torch.abs(output_dense) + 1e-6
    rel_error = (torch.abs(output_grouped - output_dense) / denominator).max().item()

    print(f"   Max absolute difference: {max_diff:.6e}")
    print(f"   Mean absolute difference: {mean_diff:.6e}")
    print(f"   Max relative error: {rel_error:.6e}")

    # Check tolerances
    if max_diff > FLOAT_TOLERANCE:
        print(f"   FAIL: Max absolute difference {max_diff} exceeds tolerance {FLOAT_TOLERANCE}")
        return False

    if rel_error > RELATIVE_TOLERANCE:
        print(f"   FAIL: Max relative error {rel_error} exceeds tolerance {RELATIVE_TOLERANCE}")
        return False

    print("   PASS: Outputs match within tolerance")

    print("\n" + "=" * 70)
    print("Unit test PASSED")
    print("=" * 70)
    return True

if __name__ == '__main__':
    try:
        success = test_grouped_moe_dispatch()
        exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
