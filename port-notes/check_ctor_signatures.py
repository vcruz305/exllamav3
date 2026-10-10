#!/usr/bin/env python3
"""Static constructor-signature conformance check for the Step-5 port.

exllamav3 needs a CUDA torch + a built exllamav3_ext to import, so the module tree
cannot be instantiated on a CPU-only machine. This catches the next most likely
failure -- a keyword that does not exist on the target's __init__ -- before a GPU
is rented.

It parses the target classes' __init__ parameter names out of the source with `ast`,
then walks the ported files and checks every call to those classes.

This does NOT prove the modules build (dtypes, device, config.stc must exist at
runtime). It proves the call shapes match.
"""
import ast
import sys
from pathlib import Path

ROOT = Path("/Users/victorcruz/work/upstream/exllamav3-wt/exllamav3")

# class name -> file to read the __init__ from
TARGETS = {
    "Linear": "modules/linear.py",
    "RMSNorm": "modules/rmsnorm.py",
    "LayerNorm": "modules/layernorm.py",
    "Embedding": "modules/embedding.py",
    "TransformerBlock": "modules/transformer.py",
    "Attention": "modules/attn.py",
    "SlidingAttention": "modules/sliding_attn.py",
    "GatedMLP": "modules/mlp.py",
    "BlockSparseMLP": "modules/block_sparse_mlp.py",
    "Qwen3_5MTPInputLayer": "modules/arch_specific/qwen3_5_mtp.py",
    "Step5CSAIndexer": "modules/step5_csa_indexer.py",
    "Step5CSACompressor": "modules/step5_csa_compress.py",
    "Step5SSMaxScale": "modules/step5_ssmax_scale.py",
}

PORTED = [
    "architecture/step5_robotics.py",
    "architecture/step5_robotics_mtp.py",
    "modules/step5_csa_indexer.py",
    "modules/step5_csa_compress.py",
    "modules/step5_ssmax_scale.py",
]


def init_params(path: Path, cls: str) -> set[str] | None:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = {a.arg for a in item.args.args + item.args.kwonlyargs}
                    args.discard("self")
                    return args
    return None


def main() -> int:
    sig = {}
    for cls, rel in TARGETS.items():
        p = ROOT / rel
        if not p.exists():
            print(f"MISSING target source {rel} for {cls}")
            return 2
        params = init_params(p, cls)
        if params is None:
            print(f"NO __init__ found for {cls} in {rel}")
            return 2
        sig[cls] = params

    problems = []
    checked = 0
    for rel in PORTED:
        tree = ast.parse((ROOT / rel).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else \
                   (node.func.attr if isinstance(node.func, ast.Attribute) else None)
            if name not in sig:
                continue
            checked += 1
            given = {kw.arg for kw in node.keywords if kw.arg is not None}
            unknown = given - sig[name]
            if unknown:
                problems.append(f"{rel}:{node.lineno} {name}() unknown kwargs {sorted(unknown)}")
            pos = len(node.args) - (1 if name in sig else 0)
            # only flag if the call is purely positional and clearly over-long
            if node.args and not node.keywords:
                limit = len([a for a in sig[name]])
                if len(node.args) > limit:
                    problems.append(f"{rel}:{node.lineno} {name}() {len(node.args)} positional > {limit}")

    print(f"checked {checked} constructor calls across {len(PORTED)} ported files")
    for cls in TARGETS:
        print(f"  {cls:22} accepts {len(sig[cls])} params")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("  -", p)
        return 3
    print("\nSIGNATURES OK -- no unknown kwargs")
    return 0


if __name__ == "__main__":
    sys.exit(main())