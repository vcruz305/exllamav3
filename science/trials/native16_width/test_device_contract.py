"""Candidate tests: the source-executed device contract and the device preflight.

Everything here runs on CPU with no `torch` import: the sealed diagnostic's
`finite` / `inp` / `forward` bodies are AST-extracted and executed against the
mock device contract in `device_contract.py`, and the REAL deferred device
preflight script is driven through a stub `torch` module.

No tolerance is relaxed and no assertion is deleted: the cross-device failure,
the value/nonfinite/shape rejections and the metadata-first property are the
harness's own regressions, re-executed against the trial's pinned copies.
"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import device_contract as dc  # noqa: E402
import linux_adapter as a  # noqa: E402

CANDIDATE = HERE / 'width/candidate/width_diag.py'
BASELINE = HERE / 'width/baseline/width_diag.py'
DEVICE_PREFLIGHT = HERE / 'width/device_preflight.py'
FAILED_LOG = HERE / 'width/baseline/failed-server.log'
DEVICES = [('cpu', 'cuda:0'), ('cpu', 'cpu'), ('cuda:0', 'cuda:0'), ('cuda:0', 'cpu')]
DTYPES = [('float16', 'bfloat16'), ('float32', 'float16')]
DEVICE_PROBE_ARGS = ['--authorize-device-probe']


class DiagnosticInputTests(unittest.TestCase):
    def test_sealed_hashes_are_the_audited_ones(self):
        self.assertEqual(a.digest(CANDIDATE.read_bytes()),
                         '7f537742bf2de7b506abaed464035f118c9c5e4f016083c2abbb78c8be9e798e')
        self.assertEqual(a.digest(BASELINE.read_bytes()),
                         '16251941215aead728fae60b116cb25d3001ab287dac3c6f15479f293646c834')

    def test_fix_is_present_in_candidate_and_absent_from_baseline(self):
        candidate = CANDIDATE.read_text()
        baseline = BASELINE.read_text()
        self.assertIn(dc.REPAIRED_OPERAND, candidate)
        self.assertNotIn(dc.HISTORICAL_OPERAND, candidate)
        self.assertIn('dm.input_layer.mask_embedding.to(y.dtype)' + dc.OPERAND_SUFFIX, baseline)
        self.assertNotIn('device=y.device', baseline)

    def test_cross_device_and_dtype_matrix_is_green(self):
        seen = 0
        for output_device, mask_device in DEVICES:
            for output_dtype, mask_dtype in DTYPES:
                row = dc.run_input_contract(CANDIDATE, output_device=output_device,
                                            mask_device=mask_device, output_dtype=output_dtype,
                                            mask_dtype=mask_dtype)
                self.assertTrue(row['row']['mask_equal'], (output_device, mask_device))
                self.assertTrue(row['row']['finite'])
                self.assertEqual(row['row']['input_shape'], [1, 16, 4096])
                self.assertTrue(row['result_is_output'])
                self.assertTrue(row['output_pointer_stable'])
                self.assertTrue(row['output_unchanged'])
                self.assertFalse(row['source_mutated'])
                self.assertEqual(row['output_device'], output_device)
                operands = dc.compare_operands(CANDIDATE, output_device=output_device,
                                               mask_device=mask_device, output_dtype=output_dtype,
                                               mask_dtype=mask_dtype)
                self.assertIsNotNone(operands)
                self.assertEqual(operands['left_device'], operands['right_device'],
                                 (output_device, mask_device))
                self.assertEqual(operands['left_dtype'], operands['right_dtype'])
                self.assertEqual(operands['left_shape'], operands['right_shape'])
                seen += 1
        self.assertEqual(seen, 8)

    def test_historical_operand_reproduces_the_real_device_failure(self):
        """RED: the exact pre-repair expression, executed from source.

        The expected error text is the one preserved verbatim in the device
        attempt's own server log (`width/baseline/failed-server.log`).
        """
        preserved = FAILED_LOG.read_text()
        self.assertIn('Expected all tensors to be on the same device', preserved)
        with self.assertRaises(RuntimeError) as caught:
            dc.run_input_contract(CANDIDATE, mutate=(dc.REPAIRED_OPERAND, dc.HISTORICAL_OPERAND),
                                  output_device='cpu', mask_device='cuda:0')
        message = str(caught.exception)
        self.assertIn('Expected all tensors to be on the same device', message)
        self.assertIn('but got other is on cuda:0', message)
        self.assertIn('different from other tensors on cpu', message)
        # The same mutation is accepted only when both operands already agree,
        # which is exactly the pre-repair blind spot.
        row = dc.run_input_contract(CANDIDATE, mutate=(dc.REPAIRED_OPERAND, dc.HISTORICAL_OPERAND),
                                    output_device='cuda:0', mask_device='cuda:0')
        self.assertTrue(row['source_mutated'])
        self.assertTrue(row['row']['mask_equal'])

    def test_metadata_is_emitted_before_the_first_assertion(self):
        row = dc.run_input_contract(CANDIDATE, output_device='cpu', mask_device='cuda:0')
        self.assertEqual(row['emits'][:2], ['input_metadata', 'input'])
        self.assertEqual(row['metadata']['anchor_shape'], [1, 1])
        self.assertEqual(row['metadata']['output_device'], 'cpu')
        self.assertEqual(row['metadata']['mask_device'], 'cuda:0')
        self.assertEqual(row['metadata']['mask_shape'], [4096])
        # A shape rejection must still have emitted the metadata row FIRST.
        broken = dc.run_input_contract(CANDIDATE, block=8, output_device='cpu',
                                      mask_device='cpu', on_failure='capture')
        self.assertEqual(broken['error_type'], 'AssertionError')
        # the repaired hook emits its metadata row FIRST, then rejects the shape
        self.assertEqual(broken['emits'], ['input_metadata'])
        self.assertEqual(broken['metadata']['input_shape'], [1, 8, 4096])
        self.assertEqual(broken['metadata']['mask_device'], broken['metadata']['output_device'])
        self.assertIsNone(broken['row'])
        # RED: the pre-repair hook emits its verdict with NO metadata row at all,
        # so a failing device attempt cannot be diagnosed from its own output.
        baseline = dc.run_input_contract(BASELINE, block=8, output_device='cpu',
                                        mask_device='cpu', on_failure='capture')
        self.assertEqual(baseline['error_type'], 'AssertionError')
        self.assertEqual(baseline['emits'], ['input'])
        self.assertIsNone(baseline['metadata'])
        # RED: with device-coincident operands the pre-repair hook silently accepts
        # a wrong-shaped output slice, which is the same blind spot.
        operands = dc.compare_operands(BASELINE, output_device='cpu', mask_device='cpu')
        self.assertIsNotNone(operands)
        self.assertEqual(operands['left_shape'], [1, 15, 4096])
        with self.assertRaises(RuntimeError):
            dc.compare_operands(BASELINE, output_device='cpu', mask_device='cuda:0')

    def test_invalid_outputs_are_still_rejected(self):
        for bad in (3.14159, float('nan'), float('inf')):
            with self.assertRaises(AssertionError):
                dc.run_input_contract(CANDIDATE, output_device='cpu', mask_device='cuda:0',
                                      output_corruption=(0, 1, 0, bad))

    def test_baseline_source_fails_the_same_contract(self):
        """RED: the pre-repair source, executed unchanged, fails the contract."""
        with self.assertRaises(RuntimeError):
            dc.run_input_contract(BASELINE, output_device='cpu', mask_device='cuda:0')

    def test_reversed_device_control(self):
        """The repaired operand is a real transfer, not a device label."""
        row = dc.run_input_contract(CANDIDATE, output_device='cuda:0', mask_device='cpu')
        self.assertTrue(row['row']['mask_equal'])
        with self.assertRaises(RuntimeError):
            dc.run_input_contract(CANDIDATE, mutate=(dc.REPAIRED_OPERAND, dc.HISTORICAL_OPERAND),
                                  output_device='cuda:0', mask_device='cpu')


class ActiveStateTests(unittest.TestCase):
    """The second repaired defect: `state['active']` must not survive a failure."""

    def test_candidate_resets_active_on_mapping_failure(self):
        row = dc.run_forward_lifetime(CANDIDATE)
        self.assertTrue(row['raised'].startswith('AssertionError'), row['raised'])
        self.assertEqual(row['emits'], [], 'the rejection must precede any capture')
        self.assertFalse(row['active_after'], 'the repaired diagnostic leaked active state')

    def test_baseline_source_leaks_active_state(self):
        row = dc.run_forward_lifetime(BASELINE)
        self.assertTrue(row['raised'].startswith('AssertionError'), row['raised'])
        self.assertTrue(row['active_after'], 'the baseline control did not reproduce the leak')

    def test_candidate_spans_preparation_with_try_finally(self):
        candidate = CANDIDATE.read_text()
        self.assertIn("finally:state['active']=False", candidate)
        before_try = candidate.split('try:', 1)[0]
        self.assertIn("state['active']=True", before_try)
        self.assertIn('assigned_positions(start,16,assigned,table)', candidate)


class DevicePreflightScriptTests(unittest.TestCase):
    """Drive the REAL deferred device preflight script with a stub torch.

    It is the repaired, metadata-first, device-safe reference check the operator
    runs before any width trial. It must refuse without its explicit permission
    switch, refuse without CUDA, and - with a satisfying torch stand-in - pass
    its full device/dtype matrix while still rejecting a mutated diagnostic.
    """

    @staticmethod
    def driver(diag=None, cuda=True):
        args = list(DEVICE_PROBE_ARGS) + (['--diag', str(diag)] if diag else [])
        return ('import sys, types, runpy;'
                'sys.path.insert(0, %r);'
                'import device_contract as dc;'
                'torch = dc.MockTorch();'
                'torch.cuda = types.SimpleNamespace(is_available=lambda: %r, synchronize=lambda: None);'
                'sys.modules["torch"] = torch;'
                'sys.argv = ["device_preflight"] + %r;'
                'runpy.run_path(%r, run_name="__main__")'
                % (str(HERE), cuda, args, str(DEVICE_PREFLIGHT)))

    def run_driver(self, driver):
        return subprocess.run([sys.executable, '-B', '-c', driver], capture_output=True, text=True)

    def test_module_scope_import_is_torch_free(self):
        source = DEVICE_PREFLIGHT.read_text()
        self.assertIn('import torch  # Deliberately deferred', source)
        check = subprocess.run([sys.executable, '-B', '-c',
                               'import ast,sys;tree=ast.parse(open(sys.argv[1]).read());'
                               'mods=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))];'
                               'print(sorted(a.name for n in mods for a in n.names))',
                               str(DEVICE_PREFLIGHT)], capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertNotIn('torch', check.stdout)

    def test_refuses_without_the_explicit_permission_switch(self):
        result = subprocess.run([sys.executable, '-B', str(DEVICE_PREFLIGHT)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('deferred: requires exclusive-owner permission and --authorize-device-probe',
                      result.stderr)

    def test_refuses_when_cuda_is_unavailable(self):
        result = self.run_driver(self.driver(cuda=False))
        self.assertEqual(result.returncode, 1)
        self.assertIn('CUDA unavailable; do not load model', result.stderr)

    def test_full_matrix_passes_using_the_mock_contract(self):
        result = self.run_driver(self.driver())
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()
                if line.startswith('{') and 'device_contract_pass' in line]
        self.assertEqual(len(rows), 8, result.stdout + result.stderr)
        self.assertIn('MICRO_PREFLIGHT_ONLY', result.stdout)

    def test_mutated_diagnostic_is_rejected_by_the_device_preflight(self):
        """RED: a fresh mutation of the sealed diagnostic must be caught here."""
        mutated = HERE / 'verification-tmp-mutant.py'
        try:
            text = CANDIDATE.read_text().replace(dc.REPAIRED_OPERAND, dc.HISTORICAL_OPERAND)
            self.assertNotEqual(text, CANDIDATE.read_text())
            mutated.write_text(text)
            result = self.run_driver(self.driver(diag=mutated))
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn('Expected all tensors to be on the same device', result.stderr)
        finally:
            mutated.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
