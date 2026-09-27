# Archived experiment assets, deliberately NOT part of the repository test suite.
#
# science/trials/** keeps frozen copies of trial packages. Their test_*.py modules are
# evidence harnesses for one specific device/CPU session (they assert against absolute
# scratch paths and sealed manifests that are not shipped), not repo regressions, so a
# repo-wide `pytest` must not collect them. `pytest tests ...` (what CI runs) is
# unaffected either way.
collect_ignore_glob = ["*"]
