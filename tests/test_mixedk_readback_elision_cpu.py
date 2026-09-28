"""Guard the unified mixed-K launch decision against silent host readbacks.

The unified mixed-K dispatch used to size its grid from a device->host readback of the per-expert
row counts (one `.tolist()` per MoE layer per forward), plus a second, unconsumed readback that
rebuilt the "handled experts" set for a fallback loop that never runs at that shape.

Nothing here imports the CUDA extension or torch: the extracted helper is pure Python and the
launch contract on the C++ side is checked as source text.

Point EXL3_TEST_SOURCE_ROOT at a checkout to test that tree (default: this repo). Running it
against a tree without the helper fails, which is the red state this test was written from.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(os.environ.get("EXL3_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[1]))


def _source(rel):
    return (ROOT / rel).read_text()


def _extract(name, rel="exllamav3/modules/block_sparse_mlp.py"):
    """Exec one top-level function from its real source, with its real defaults."""
    tree = ast.parse(_source(rel))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 f"<{name}>", "exec"), namespace)
    return namespace[name]


# ---------------------------------------------------------------- the helper itself

def test_launch_plan_uses_host_counts_when_they_exist():
    plan = _extract("_mixedk_launch_plan")
    counts, active = plan([3, 0, 5, 200, 1], 5, 128, 6, True)
    assert counts == [3, 5, 1]
    assert active == 3
    # row cap is exclusive at the top, so an oversized expert is not fused
    counts, active = plan([128, 129], 2, 128, 6, True)
    assert counts == [128] and active == 1


def test_launch_plan_defers_to_the_device_readback_when_disabled():
    plan = _extract("_mixedk_launch_plan")
    # (None, None) is the sentinel that keeps today's readback path bit-identical
    assert plan(None, 256, 128, 6, False) == (None, None)


def test_launch_plan_sizes_the_grid_from_concurrency_without_a_readback():
    plan = _extract("_mixedk_launch_plan")
    counts, active = plan(None, 256, 128, 6, True)
    assert counts is None
    assert active == 6
    # a zero active count would trip the kernel's early-out and skip a launch that has work
    assert plan(None, 256, 128, 0, True)[1] == 0


def test_elision_is_opt_in_and_default_off():
    src = _source("exllamav3/modules/block_sparse_mlp.py")
    assert 'os.environ.get("EXL3_MOE_MIXEDK_NO_READBACK", "0") != "0"' in src, \
        "the elision must default off, so the measured default profile is unchanged"


# ---------------------------------------------------------------- the launch contract

def test_grid_stays_preserved_while_active_count_covers_concurrency():
    """num_groups = MIN(concurrency, num_active); group_size = MIN(target_blocks//num_groups, 32).

    Reproduce the host arithmetic from exl3_moe.cu for the MiMo decode shape (48 SMs, one block
    per SM, MOE_SMS_PER_EXPERT = 8) and show that any num_active >= concurrency yields the same
    grid, hence the same per-expert fp reduction order.
    """
    cuda = _source("exllamav3/exllamav3_ext/quant/exl3_moe.cu")
    assert "num_groups = MIN(num_groups, num_active);" in cuda
    assert "group_size = MIN(target_blocks / num_groups, MOE_MAX_SMS_PER_EXPERT);" in cuda

    def grid(num_active, concurrency, target_blocks, sms_per_expert=8, max_sms=32):
        num_groups = min(concurrency, 64)
        group_size = sms_per_expert
        if num_active > 0:
            num_groups = min(num_groups, num_active)
            group_size = min(target_blocks // num_groups, max_sms)
        return num_groups, group_size

    for active in (6, 8, 40, 61, 64):
        assert grid(active, 6, 48) == (6, 8)
    # below the concurrency the grid does change: that is why the elision is documented as
    # grid-preserving only for the multi-row verify shapes and is opt-in
    assert grid(3, 6, 48) == (3, 16)
    assert grid(0, 6, 48) == (6, 8)          # num_active == 0 skips the kernel's grid branch


def test_fallback_loop_is_guarded_so_the_handled_set_is_unconsumed():
    """mixedk_handled reaches nothing when expert_count_list is None."""
    tree = ast.parse(_source("exllamav3/modules/block_sparse_mlp.py"))
    iters = [ast.unparse(n.iter) for n in ast.walk(tree) if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Call)
             and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range"]
    guarded = [t for t in iters if "expert_count_list is not None" in t and "else 0" in t]
    assert len(guarded) == 1, f"expected exactly one guarded per-expert loop, saw {guarded}"
    assert guarded[0].startswith("range(num_ex if expert_count_list is not None else 0)")


def test_no_second_device_readback_remains_on_the_no_host_count_path():
    """The old code built mixedk_handled from a device readback in the no-host-count case.

    That readback fed a loop the guard above skips, so the only `mixedk_handled.add` sites left
    must sit under a host count list.
    """
    tree = ast.parse(_source("exllamav3/modules/block_sparse_mlp.py"))
    adds = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "add"
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "mixedk_handled"]
    assert adds, "the handled set must still be populated on the host-count path"
    src = _source("exllamav3/modules/block_sparse_mlp.py")
    # the update() site belongs to the legacy per-K-group dispatch, which is gated on a host list
    assert "mixedk_handled.update(grp_experts)" in src
    assert "(_ec > 0) & (_ec <= _mkd_rows)).nonzero" not in src, \
        "the unconsumed device readback must be gone"


def test_tier_split_cannot_fire_without_per_expert_counts():
    src = _source("exllamav3/modules/block_sparse_mlp.py")
    for tier in ("t1 = sum(1 for c in counts_fused", "t2 = sum(1 for c in counts_fused"):
        line = next(ln for ln in src.splitlines() if ln.strip().startswith(tier))
        assert "if counts_fused else 0" in line


# ---------------------------------------------------------------- single-readback accounting

def _readbacks(src):
    """Count device->host syncs on the unified path: `.tolist()` on a tensor slice."""
    return [ln for ln in src.splitlines() if ".tolist()" in ln]


def test_unified_path_has_at_most_one_readback_site():
    src = _source("exllamav3/modules/block_sparse_mlp.py")
    unified = src.split("# === Mixed-K unified fused dispatch")[1].split(
        "# === Legacy Mixed-K per-K-group fused dispatch")[0]
    sites = [ln for ln in _readbacks(unified)]
    assert len(sites) == 1, f"expected the fallback readback only, found {sites}"
    assert "_ec[_m].tolist()" in sites[0]


def test_preservation_control_the_host_path_is_unchanged():
    """With expert_count_list present the decision is exactly the old list comprehension."""
    plan = _extract("_mixedk_launch_plan")
    counts = plan([2, 7, 0, 129, 1], 5, 128, 6, True)[0]
    assert counts == [c for c in [2, 7, 0, 129, 1][:5] if 0 < c <= 128]
