"""Machine-readable unittest outcome from the real subprocess execution."""
import json
import sys
import unittest
suite=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
r=unittest.TextTestRunner(verbosity=2).run(suite)
print('TEST_RESULT '+json.dumps(dict(tests=r.testsRun, failures=len(r.failures),errors=len(r.errors),
                                    skipped=len(r.skipped),unexpected_successes=len(r.unexpectedSuccesses))))
sys.exit(0 if r.wasSuccessful() else 1)
