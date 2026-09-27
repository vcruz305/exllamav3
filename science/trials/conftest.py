# Archived experiment assets, deliberately NOT part of the repository test suite.
#
# science/trials/** keeps frozen copies of trial packages. Their test_*.py modules are
# evidence harnesses for one specific source pin (they read the reviewed tree through
# REVIEW_SOURCE_ROOT and are run through tests/test_driver.py, not pytest), so a
# repo-wide `pytest` must not collect them. `pytest tests ...` (what CI runs) is
# unaffected either way.
collect_ignore_glob = ["*"]
