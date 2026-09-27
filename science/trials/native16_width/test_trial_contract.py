"""Candidate tests: the width-trial policy contract (no process, no device)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import linux_adapter as a  # noqa: E402
import trial_contract as tc  # noqa: E402

SHA = '0' * 64
WIDTH_CONFIG = HERE / 'width/candidate/width_config.json'


def width_config_doc(block):
    doc = json.loads(WIDTH_CONFIG.read_text())
    doc['block_size'] = block
    doc['dflash_config']['block_size'] = block
    return doc


def sample_trial(root, width_view, retained_view):
    return {
        'schema': 1, 'inert_fixture': True, 'source_pin': a.SOURCE_PIN, 'root': str(root),
        'width': {'native_block': 16, 'retained_block': 8,
                  'drafter_config': str(width_view), 'drafter_config_sha256': a.digest(width_view.read_bytes()),
                  'retained_drafter_config': str(retained_view),
                  'retained_drafter_config_sha256': a.digest(retained_view.read_bytes()),
                  'probe_dir': 'round8-cost-width/width/ndt1', 'events': 'events.jsonl',
                  'run_prefix': 'france-round8-width16-ndt1-', 'failed_run': None,
                  'failed_pids': {'guard_pid': 4242, 'server_pid': 4243},
                  'env': {'ROUND8_WIDTH': '1', 'ROUND8_WIDTH_OUT': str(root / 'round8-cost-width/width/ndt1')},
                  'expect': copy.deepcopy(tc.EXPECTED)},
        'protected': {'mandatory_roots': [str(root / 'protected-a')], 'nesting_acknowledged': False},
        'artifacts': {'integration_pins': 'integration-pins.json', 'integration_pins_sha256': SHA},
        'scope': 'unit-test fixture',
    }


def auth_doc(root, sha, stage, status, error, phase):
    now = 1_000_000.0
    return {'schema': 1, 'intent': 'stop_owned_then_launch_retained_once', 'root': str(root),
            'operator_lock': 'operator.lock', 'ingress_blocked': True, 'uid': 0, 'boot_id': 'boot',
            'config_sha256': sha, 'receipt_sha256': sha, 'trial_sha256': sha,
            'issued_at': now, 'failure_at': now - 10, 'expires_at': now + 10, 'width': {
                'status': status, 'error': error, 'phase': phase}, 'stage': stage,
            'scope': 'unit-test fixture'}


class TrialDocumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(HERE), prefix='unit-')
        self.root = Path(self.tmp.name)
        (self.root / 'protected-a').mkdir()
        self.width_view = self.root / 'width-view.json'
        self.retained_view = self.root / 'retained-view.json'
        self.width_view.write_text(json.dumps(width_config_doc(16)))
        self.retained_view.write_text(json.dumps(width_config_doc(8)))
        self.trial = sample_trial(self.root, self.width_view, self.retained_view)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, trial):
        return tc.validate_trial(trial, self.root)

    def test_valid_document_passes(self):
        validated = self.check(copy.deepcopy(self.trial))
        self.assertEqual(validated, self.trial)
        self.assertEqual(validated['width']['native_block'], 16)

    def test_missing_and_unknown_fields_refused(self):
        for mutate in (lambda t: t.pop('scope'), lambda t: t.update(extra=1),
                       lambda t: t['width'].pop('events'), lambda t: t['width'].update(extra=1),
                       lambda t: t['protected'].update(extra=1)):
            trial = copy.deepcopy(self.trial)
            mutate(trial)
            with self.assertRaisesRegex(a.Refusal, 'fields must be exactly'):
                self.check(trial)

    def test_width_must_stay_native16(self):
        for block, message in ((8, 'native16'), (32, 'native16')):
            trial = copy.deepcopy(self.trial)
            trial['width']['native_block'] = block
            with self.assertRaisesRegex(a.Refusal, message):
                self.check(trial)

    def test_retained_block_must_stay_native8(self):
        trial = copy.deepcopy(self.trial)
        trial['width']['retained_block'] = 16
        with self.assertRaisesRegex(a.Refusal, 'retained block'):
            self.check(trial)

    def test_retained_and_width_configs_must_differ(self):
        trial = copy.deepcopy(self.trial)
        trial['width']['retained_drafter_config'] = trial['width']['drafter_config']
        trial['width']['retained_drafter_config_sha256'] = trial['width']['drafter_config_sha256']
        with self.assertRaisesRegex(a.Refusal, 'must differ'):
            self.check(trial)

    def test_probe_dir_must_be_relative_and_env_must_match(self):
        trial = copy.deepcopy(self.trial)
        trial['width']['probe_dir'] = '/abs/probe'
        with self.assertRaisesRegex(a.Refusal, 'unsafe relative path'):
            self.check(trial)
        trial = copy.deepcopy(self.trial)
        trial['width']['probe_dir'] = '../escape'
        with self.assertRaisesRegex(a.Refusal, 'unsafe relative path'):
            self.check(trial)
        trial = copy.deepcopy(self.trial)
        trial['width']['env']['ROUND8_WIDTH_OUT'] = str(self.root / 'elsewhere')
        with self.assertRaisesRegex(a.Refusal, 'must equal root/probe_dir'):
            self.check(trial)

    def test_failed_pids_must_be_explicit(self):
        trial = copy.deepcopy(self.trial)
        del trial['width']['failed_pids']
        with self.assertRaisesRegex(a.Refusal, 'failed_pids'):
            self.check(trial)
        trial = copy.deepcopy(self.trial)
        trial['width']['failed_pids']['guard_pid'] = 0
        with self.assertRaisesRegex(a.Refusal, 'trusted PID'):
            self.check(trial)

    def test_expect_block_is_pinned(self):
        for key, value in (('proposal_count', 7), ('assigned_span', 8), ('input_shape', [1, 8, 4096]),
                           ('max_round', 4096)):
            trial = copy.deepcopy(self.trial)
            trial['width']['expect'][key] = value
            with self.assertRaisesRegex(a.Refusal, 'width.expect'):
                self.check(trial)

    def test_root_must_match(self):
        trial = copy.deepcopy(self.trial)
        trial['root'] = '/somewhere/else'
        with self.assertRaisesRegex(a.Refusal, 'does not match the owned root'):
            self.check(trial)


class WidthConfigTests(unittest.TestCase):
    def test_block_size_must_agree_at_both_levels(self):
        raw = json.dumps(width_config_doc(16)).encode()
        self.assertEqual(tc.validate_width_config(raw, 'width config', 16), (16, 16))
        disagreeing = width_config_doc(16)
        disagreeing['dflash_config']['block_size'] = 8
        with self.assertRaisesRegex(a.Refusal, 'disagree'):
            tc.validate_width_config(json.dumps(disagreeing).encode(), 'width config', 16)
        with self.assertRaisesRegex(a.Refusal, 'expected block_size 8'):
            tc.validate_width_config(raw, 'width config', 8)

    def test_taps_and_tap_shift_are_pinned(self):
        doc = width_config_doc(16)
        doc['dflash_config']['tap_shift'] = 1
        with self.assertRaisesRegex(a.Refusal, 'tap_shift'):
            tc.validate_width_config(json.dumps(doc).encode(), 'width config', 16)
        doc = width_config_doc(16)
        doc['dflash_config']['target_layer_ids'] = [0, 11, 23, 35]
        with self.assertRaisesRegex(a.Refusal, 'tap set changed'):
            tc.validate_width_config(json.dumps(doc).encode(), 'width config', 16)


class ProtectedRootTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(HERE), prefix='unit-')
        self.root = Path(self.tmp.name)
        (self.root / 'protected-a').mkdir()
        (self.root / 'protected-b').mkdir()
        (self.root / 'runs').mkdir()
        self.trial = sample_trial(self.root, WIDTH_CONFIG, WIDTH_CONFIG)
        self.destinations = [str(self.root / 'runs'), str(self.root / 'receipts')]

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_inventory_passes_and_records_nesting(self):
        roots = [str(self.root / 'protected-a'), str(self.root / 'protected-b')]
        result = tc.validate_protected_roots(roots, True, self.trial, self.destinations)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['nested'], [])

    def test_incomplete_or_undeclared_inventory_refused(self):
        with self.assertRaisesRegex(a.Refusal, 'declared complete'):
            tc.validate_protected_roots([str(self.root / 'protected-a')], False, self.trial, self.destinations)
        with self.assertRaisesRegex(a.Refusal, 'inventory is incomplete'):
            tc.validate_protected_roots([str(self.root / 'protected-b')], True, self.trial, self.destinations)
        with self.assertRaisesRegex(a.Refusal, 'duplicate protected root'):
            tc.validate_protected_roots([str(self.root / 'protected-a')] * 2, True, self.trial, self.destinations)

    def test_destination_inside_a_protected_root_refused(self):
        self.trial['protected']['mandatory_roots'] = [str(self.root)]
        roots = [str(self.root)]
        with self.assertRaisesRegex(a.Refusal, 'inside a protected root'):
            tc.validate_protected_roots(roots, True, self.trial, self.destinations)

    def test_nesting_requires_explicit_acknowledgement(self):
        outer, inner = str(self.root / 'protected-a'), str(self.root / 'protected-a/inner')
        with self.assertRaisesRegex(a.Refusal, 'nesting_acknowledged'):
            tc.validate_protected_roots([outer, inner], True, self.trial, self.destinations)
        self.trial['protected']['nesting_acknowledged'] = True
        result = tc.validate_protected_roots([outer, inner], True, self.trial, self.destinations)
        self.assertEqual(result['nested'], [(inner, outer)])


class AuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(HERE), prefix='unit-')
        self.root = Path(self.tmp.name)
        (self.root / 'protected-a').mkdir()
        self.trial = sample_trial(self.root, WIDTH_CONFIG, WIDTH_CONFIG)

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self, auth, stage):
        return tc.validate_authorization(auth, self.trial, self.root, SHA, SHA, SHA, 'boot', 0,
                                         1_000_000.0, stage)

    def test_release_stage_requires_the_labelled_not_run_encoding(self):
        auth = auth_doc(self.root, SHA, 'release', 'failed', tc.NOT_RUN_ERROR, 'before_width')
        self.assertIs(self.validate(auth, 'release'), auth)
        wrong = auth_doc(self.root, SHA, 'release', 'passed', None, 'before_width')
        with self.assertRaisesRegex(a.Refusal, 'binary schema'):
            self.validate(wrong, 'release')
        other = auth_doc(self.root, SHA, 'release', 'failed', 'some other error', 'before_width')
        with self.assertRaisesRegex(a.Refusal, 'binary schema'):
            self.validate(other, 'release')

    def test_restore_stage_requires_after_width(self):
        auth = auth_doc(self.root, SHA, 'restore', 'failed', 'device refusal text', 'after_width')
        self.assertIs(self.validate(auth, 'restore'), auth)
        before = auth_doc(self.root, SHA, 'restore', 'failed', tc.NOT_RUN_ERROR, 'before_width')
        with self.assertRaisesRegex(a.Refusal, 'only authorize the release stage'):
            self.validate(before, 'restore')
        release_with_after = auth_doc(self.root, SHA, 'release', 'failed', 'x', 'after_width')
        with self.assertRaisesRegex(a.Refusal, 'only authorize the restore stage'):
            self.validate(release_with_after, 'release')

    def test_stage_and_hash_and_expiry_are_enforced(self):
        auth = auth_doc(self.root, SHA, 'release', 'failed', tc.NOT_RUN_ERROR, 'before_width')
        auth['stage'] = 'restore'
        with self.assertRaisesRegex(a.Refusal, 'stage does not match'):
            self.validate(auth, 'release')
        auth = auth_doc(self.root, SHA, 'release', 'failed', tc.NOT_RUN_ERROR, 'before_width')
        auth['trial_sha256'] = 'a' * 64
        with self.assertRaisesRegex(a.Refusal, 'trial-document hash mismatch'):
            self.validate(auth, 'release')
        auth = auth_doc(self.root, SHA, 'release', 'failed', tc.NOT_RUN_ERROR, 'before_width')
        auth['expires_at'] = 1.0
        with self.assertRaisesRegex(a.Refusal, 'expired or future-dated'):
            self.validate(auth, 'release')
        auth = auth_doc(self.root, SHA, 'release', 'failed', tc.NOT_RUN_ERROR, 'before_width')
        auth['ingress_blocked'] = False
        with self.assertRaisesRegex(a.Refusal, 'quiescence'):
            self.validate(auth, 'release')

    def test_width_error_must_accompany_a_failed_outcome(self):
        auth = auth_doc(self.root, SHA, 'restore', 'failed', None, 'after_width')
        with self.assertRaisesRegex(a.Refusal, 'error must accompany'):
            self.validate(auth, 'restore')


class TokenTests(unittest.TestCase):
    def test_both_switches_are_required(self):
        import argparse
        args = argparse.Namespace(authorize_width_trial=True)
        with self.assertRaisesRegex(a.Refusal, 'EXL3_WIDTH_TRIAL_AUTHORIZED'):
            tc.require_authorization_token({}, args)
        args = argparse.Namespace(authorize_width_trial=False)
        with self.assertRaisesRegex(a.Refusal, 'authorize-width-trial'):
            tc.require_authorization_token({'EXL3_WIDTH_TRIAL_AUTHORIZED': 'YES'}, args)
        self.assertTrue(tc.require_authorization_token({'EXL3_WIDTH_TRIAL_AUTHORIZED': 'YES'},
                                                       argparse.Namespace(authorize_width_trial=True)))
        with self.assertRaisesRegex(a.Refusal, 'EXL3_WIDTH_TRIAL_AUTHORIZED'):
            tc.require_authorization_token({'EXL3_WIDTH_TRIAL_AUTHORIZED': 'yes'}, args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
