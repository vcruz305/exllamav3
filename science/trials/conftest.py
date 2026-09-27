# Archived experiment assets, deliberately NOT part of the repository test suite.
#
# science/trials/** keeps frozen copies of the trial packages, which embed snapshot
# trees (original/, five-stage-control/, tree/) that carry the upstream tests/ modules
# as fixtures, plus the trial's own test_*.py harnesses. Those are evidence artifacts
# for a specific device run, not repo regressions, so a repo-wide `pytest` must not
# collect them. `pytest tests ...` (what CI runs) is unaffected either way.
collect_ignore_glob = ["*"]
