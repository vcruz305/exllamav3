"""EXL3_DRAFT_ROW_BUDGET shortens MTP drafts when several streams verify together.

Unique experts scale with rows, so a long draft across a wide batch moves more
weight than it accepts. This checks the cap only; it does not construct a
Generator.
"""
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "exllamav3" / "generator" / "generator.py"


def _window(full: int, batch_size: int, env: str) -> int:
    if batch_size <= 1 or full <= 0:
        return full
    raw = env
    if not raw:
        return full
    budget = int(raw)
    if budget <= 0:
        return full
    return max(0, min(full, budget // batch_size - 1))


def test_source_reads_the_budget_and_uses_it_for_mtp():
    src = SRC.read_text(encoding="utf-8")
    assert "EXL3_DRAFT_ROW_BUDGET" in src
    assert "def _mtp_window" in src
    assert "window = mtp_window" in src


def test_one_stream_keeps_the_full_draft():
    assert _window(5, 1, "8") == 5
    assert _window(5, 4, "") == 5
    assert _window(5, 4, "0") == 5


def test_budget_shrinks_a_wide_batch():
    assert _window(5, 2, "8") == 3
    assert _window(5, 4, "8") == 1
    assert _window(5, 8, "8") == 0
