"""Pin the DFlash draft-window geometry and the over-wide -ndt clamp.

Two facts drive this test:
  * the drafter materialises exactly `block_size - 1` mask rows per forward
    (modules/arch_specific/dflash.py:119), so one round proposes one native block per drafter
    forward, while the round copies the *window* into draft_ids_pinned;
  * a window wider than the drafter can fill therefore reaches that copy with fewer source columns
    than destination slots (seen on two hosts: -ndt 8, 12 and 16 all fail with
    "a (12) must match b (7)"), which the server reports as engine_unavailable.

So the capacity must come from the drafter's geometry, the reserve (pinned width, cache guard)
must cover the chained capacity, and the copy must clamp instead of raising. This test extracts
that function from the real source and checks the invariants; it does NOT validate the drafter
forwards or acceptance - a chained round needs a GPU (and a drafter-KV append per block) to be real.
"""
import ast
import os
from pathlib import Path

ROOT = Path(os.environ.get("EXL3_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
GEN = (ROOT / "exllamav3/generator/generator.py").read_text()
DFLASH_IN = (ROOT / "exllamav3/modules/arch_specific/dflash.py").read_text()
ARCH = (ROOT / "exllamav3/architecture/dflash.py").read_text()
BLOCKS = (ROOT / "exllamav3/modules/block_sparse_mlp.py").read_text()


def _geometry():
    tree = ast.parse(GEN)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "dflash_draft_geometry")
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "<geometry>", "exec"), ns)
    return ns["dflash_draft_geometry"]


def test_mask_rows_are_the_native_block_minus_one():
    """The cap the whole item rests on, read from the drafter's input layer."""
    assert "noise_mask = torch.full((bsz, self.native_draft_len - 1), self.mask_token_id" in DFLASH_IN
    assert "native_draft_len = config.block_size," in ARCH
    assert '"default_draft_size": config.block_size - 1,' in ARCH


def test_capacity_is_blocks_times_the_native_block():
    geo = _geometry()
    assert geo(7, 7, 1, True) == (1, 7, 7)
    assert geo(7, 7, 2, True) == (2, 14, 14)
    assert geo(7, 7, 3, True) == (3, 21, 21)
    assert geo(7, 7, 0, True) == (1, 7, 7)          # k >= 1


def test_the_reserve_covers_the_chain_and_never_narrows_the_pin():
    """-ndt 12 with a native block of 7 keeps its 12-slot reserve; k = 2 widens it to 14."""
    geo = _geometry()
    assert geo(12, 7, 1, True) == (1, 7, 12)
    assert geo(12, 7, 2, True) == (2, 14, 14)
    assert geo(7, 7, 2, True) == (2, 14, 14)


def test_non_dflash_drafters_keep_the_old_reserve_contract():
    geo = _geometry()
    assert geo(4, None, 3, False) == (1, 4, 4)
    assert geo(0, None, 1, False) == (1, 0, 0)
    # an AR/MTP drafter has no native block, so the chain knob must not apply
    assert geo(4, 4, 8, False) == (1, 4, 4)


def test_pinned_width_comes_from_the_reserve():
    assert "(max_batch_size, self.draft_reserve_tokens)," in GEN
    assert "self.draft_blocks, self.draft_window_capacity, self.draft_reserve_tokens" in GEN


def test_the_copy_clamps_instead_of_raising():
    """The crash site itself: one clamp, one warning, before any tensor work."""
    copy_at = GEN.index("self.draft_ids_pinned[:batch_size, :window].copy_(new_ids[:batch_size, :window])")
    clamp_at = GEN.index("if window > new_ids.shape[1]:")
    assert clamp_at < copy_at
    assert "window = new_ids.shape[1]" in GEN
    # warned once per engine, not once per round
    assert 'if not getattr(self, "_draft_window_warned", False):' in GEN


def test_verify_row_count_constraint_is_recorded_where_k_is_chosen():
    """q = 1 + window; the fused decode kernels accept bsz <= MAX_BSZN, so k is bounded there."""
    assert "MAX_BSZN = 8" in BLOCKS
    assert "bszn_eligible = self.bc is not None and bsz <= MAX_BSZN" in BLOCKS
    # with the native block at 7, one block already spends the whole bszN budget
    geo = _geometry()
    assert geo(7, 7, 1, True)[1] + 1 == 8


def test_the_chain_knob_is_opt_in():
    assert '_os.environ.get("EXL3_DFLASH_DRAFT_BLOCKS", "1")' in GEN
