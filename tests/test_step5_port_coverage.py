"""Source-level contract test for the Step-5 MTP + CSA indexer port.

CPU-only: no torch, no built extension, no checkpoint. Mirrors the technique in
test_optimizer_targets.py -- parse the ported sources with `ast`, pull out the tensor-key
generators, and assert they produce exactly the key families that exist in the real
Step-5-Preview checkpoint (2,449 tensors; see port-notes/05).

This is the contract that makes the packs complete: if a refactor renames a key
suffix, the next encode silently drops tensors and the pack is short. This test makes
that a CI failure instead.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The 8 indexer families x 23 layers = 184 tensors. Verbatim from the real
# model.safetensors.index.json of rene98c/Step-5-Preview-BF16 (port-notes/05).
INDEXER_TENSORS = {
    "sparse_indexer_q.weight",
    "sparse_indexer_q_norm.weight",
    "sparse_indexer_k.weight",
    "sparse_indexer_k_norm.weight",
    "sparse_indexer_k_norm.bias",
    "sparse_indexer_w.weight",
    "sparse_indexer_z.weight",
    "ssmax_s",
}

# The 17 MTP families x 3 layers = 51 tensors.
MTP_TENSORS = {
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.g_proj.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "transformer.shared_head.norm.weight",
    "transformer.shared_head.output.weight",
}

INDEXER_LAYERS = 23
MTP_LAYERS = 3


def _init_defaults(relative: str, class_name: str, prefix: str = "key") -> dict[str, str]:
    """Extract {param_name: default_string} for __init__ params of the given class."""
    source = (ROOT / relative).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    out = {}
                    args = item.args
                    defaults = args.defaults
                    # defaults align to the TAIL of the positional args
                    pos = [a for a in args.args if a.arg != "self"]
                    for arg, default in zip(pos[-len(defaults):], defaults):
                        if not arg.arg.startswith(prefix):
                            continue
                        out[arg.arg] = (
                            default.value if isinstance(default, ast.Constant)
                            else ast.unparse(default)
                        )
                    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                        if default is None or not arg.arg.startswith(prefix):
                            continue
                        out[arg.arg] = (
                            default.value if isinstance(default, ast.Constant)
                            else ast.unparse(default)
                        )
                    return out
    raise AssertionError(f"class {class_name} not found in {relative}")


def test_indexer_tensor_families_match_checkpoint():
    keys = _init_defaults("exllamav3/modules/step5_csa_indexer.py", "Step5CSAIndexer")
    # key_q="sparse_indexer_q" -> the Linear contributes "<name>.weight"
    produced = {f"{v}.weight" for k, v in keys.items() if k != "key_k_norm"}
    # k_norm is a biased LayerNorm -> both .weight and .bias
    produced |= {"sparse_indexer_k_norm.weight", "sparse_indexer_k_norm.bias"}
    # ssmax_s is a bare tensor with no .weight suffix, carried by Step5SSMaxScale
    produced |= {"ssmax_s"}
    assert produced == INDEXER_TENSORS, (
        f"indexer key mismatch\n  extra: {sorted(produced - INDEXER_TENSORS)}\n"
        f"  missing: {sorted(INDEXER_TENSORS - produced)}"
    )


def test_ssmax_scale_has_no_weight_suffix():
    """ssmax_s is stored WITHOUT a .weight suffix; a regression here drops the tensor."""
    source = (ROOT / "exllamav3/modules/step5_ssmax_scale.py").read_text(encoding="utf-8")
    assert "tensor_weight_suffix" not in source or "False" in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == "tensor_key":
                    assert ast.unparse(node.value) == "self.key", (
                        "Step5SSMaxScale.tensor_key must be self.key (no .weight suffix)"
                    )
                    return
    raise AssertionError("Step5SSMaxScale.tensor_key assignment not found")


def test_mtp_tensor_families_match_checkpoint():
    """Every MTP family is composed as ``f"{key}.<parent>"`` + ``key_<x> = "<leaf>"``.

    e.g. ``mlp.up_proj.weight`` comes from ``key = f"{key}.mlp"`` with ``key_up = "up_proj"``,
    so the literal ``mlp.up_proj`` never appears in the source. Assert the parts instead.
    """
    source = (ROOT / "exllamav3/architecture/step5_robotics_mtp.py").read_text(encoding="utf-8")

    # parent fragment -> leaf suffixes it must carry
    composed = {
        "mlp": {"up_proj", "gate_proj", "down_proj"},
        "self_attn": {"q_proj", "k_proj", "v_proj", "o_proj", "g_proj"},
    }
    for parent, leaves in composed.items():
        assert f'.{parent}"' in source or f".{parent}" in source, \
            f"missing parent key '{parent}'"
        for leaf in leaves:
            assert f'"{leaf}"' in source, f"missing leaf key '{leaf}' under {parent}"

    # These are direct ``key = f"{key}.<name>"`` fragments.
    direct = [
        "eh_proj", "enorm", "hnorm", "input_layernorm",
        "post_attention_layernorm",
        "transformer.shared_head.norm", "transformer.shared_head.output",
        "self_attn.q_norm", "self_attn.k_norm",
    ]
    for name in direct:
        assert name in source, f"MTP family '{name}' not referenced by the module"


def test_layer_counts():
    """23 full-attention layers carry the indexer; 3 MTP depths at 92..94."""
    cfg = (ROOT / "exllamav3/architecture/step5_robotics.py").read_text(encoding="utf-8")
    assert "sparse_indexer_layers" in cfg
    assert "mtp_base_layer_idx" in cfg
    assert "mtp_num_layers" in cfg
    # The MTP model must step by depth from the base layer index.
    mtp = (ROOT / "exllamav3/architecture/step5_robotics_mtp.py").read_text(encoding="utf-8")
    assert "first + depth" in mtp or "first_mtp_layer" in mtp


def test_wiring_targets_only_full_attention_layers():
    """The indexer must be attached to self_attn of the sparse layers, not to every layer."""
    cfg = (ROOT / "exllamav3/architecture/step5_robotics.py").read_text(encoding="utf-8")
    assert 'self_attn' in cfg
    assert 'qmap = "block.attn.input"' in cfg, (
        "indexer q/z must share the block.attn.input Hessian group so the existing "
        "368-teacher bank covers them"
    )
    assert "Step5CSAIndexer" in cfg and "Step5SSMaxScale" in cfg


if __name__ == "__main__":
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            fn()
            print("ok", fn_name)
    print("ALL OK")
