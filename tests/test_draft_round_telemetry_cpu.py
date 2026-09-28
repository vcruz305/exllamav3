"""Pin the per-round draft telemetry that turns acceptance into tokens-per-verify-pass.

`accepted_draft_tokens / (accepted + rejected)` is a ratio: the same 0.78 describes a 7-token
window that mostly hits and a 3-token window that mostly hits, and the difference is 2x in tokens
per weight pass. The bench records for this pack never captured the window, so every bytes/token
figure for the DFlash path is currently unconstrained. The job result must therefore carry the
round-level window and the derived tokens-per-pass, and the arithmetic must live in one place.

CPU-only: this validates the aggregation, NOT what the drafter actually proposed.
"""
import ast
import os
from pathlib import Path

ROOT = Path(os.environ.get("EXL3_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
JOB = (ROOT / "exllamav3/generator/job.py").read_text()


def _stats():
    tree = ast.parse(JOB)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "draft_round_stats")
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "<stats>", "exec"), ns)
    return ns["draft_round_stats"]


def test_empty_history_adds_no_keys():
    assert _stats()(None) == {}
    assert _stats()([]) == {}


def test_tokens_per_pass_is_one_plus_mean_accepted():
    # (new_tokens, window, accepted - 1) tuples, as recorded by the generator
    s = _stats()([(4, 8, 3), (4, 8, 3), (2, 8, 1), (8, 8, 7)])
    assert s["draft_rounds"] == 4
    assert s["draft_window_mean"] == 8.0
    assert s["draft_accepted_mean"] == (3 + 3 + 1 + 7) / 4
    assert s["tokens_per_pass"] == 1.0 + (3 + 3 + 1 + 7) / 4


def test_the_same_acceptance_ratio_at_two_windows_is_distinguishable():
    """The whole point: 0.75 acceptance with window 8 is not window 4."""
    stats = _stats()
    long_ = stats([(2, 8, 3) for _ in range(8)])       # 4 verified positions
    short = stats([(1, 4, 1) for _ in range(8)])       # 2 verified positions
    assert long_["draft_window_mean"] == 8.0
    assert short["draft_window_mean"] == 4.0
    assert long_["tokens_per_pass"] == 4.0
    assert short["tokens_per_pass"] == 2.0


def test_window_extremes_are_reported():
    s = _stats()([(2, 3, 1), (2, 8, 1), (2, 21, 1)])
    assert (s["draft_window_min"], s["draft_window_max"]) == (3, 21)


def test_both_result_dicts_carry_the_telemetry():
    """The streaming and non-streaming result dicts must not diverge."""
    assert JOB.count("**draft_round_stats(self.draft_stats),") == 1
    assert JOB.count("r.update(draft_round_stats(self.draft_stats))") == 1


def test_block_telemetry_stays_opt_in():
    """The per-round list is allocated by the job and filled only when the caller opted in."""
    GEN = (ROOT / "exllamav3/generator/generator.py").read_text()
    assert "record_draft_stats: bool = False," in GEN
    assert "if self.record_draft_stats:" in GEN
    assert "job.draft_stats.append((" in GEN
    assert "self.draft_stats = []" in JOB
    # one append site for the DFlash/MTP round record, and the aggregate is read-only
    assert GEN.count("job.draft_stats.append((") == 1
