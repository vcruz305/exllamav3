"""Candidate tests: the host-side device-document gate (`device_docs.py`).

CPU-only, no torch/exllamav3, no process, no socket, no network: the gate is a
pure document check. It builds a COMPLETE production document set in a temp dir
(the same field sets the sealed contract demands), proves the gate accepts it
and prints the four stage hashes, then proves each individual defect is refused.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import device_docs  # noqa: E402
import linux_adapter as a  # noqa: E402
import trial_contract as tc  # noqa: E402

ROOT = '/workspace/mimo-tune'
FILLER64 = 'a' * 64


@contextlib.contextmanager
def filled_documents():
    """A complete, placeholder-free production document set in a temp dir."""
    with tempfile.TemporaryDirectory(dir=str(HERE), prefix='device-docs-') as tmp:
        target = Path(tmp)
        docs = device_docs.write_templates(str(target))

        def load(name):
            return json.loads((target / (name + '.json')).read_text())

        def store(name, payload):
            (target / (name + '.json')).write_text(json.dumps(payload, indent=2) + '\n')

        cfg = load('config')
        forged = {'interpreter': '/usr/bin/python3',
                  'target_config': '/workspace/MiMo-V2.6-Flash-RL-EXL3/2.50bpw/config.json',
                  'drafter_config':
                      '/workspace/mimo-exl3/models/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw/config.json',
                  'source_manifest': ROOT + '/60tps-round3/python-shadow/source-manifest.json'}
        for row in cfg['identities']:
            if row['role'] in forged:
                row['path'] = forged[row['role']]
                row['sha256'] = hashlib.sha256(forged[row['role']].encode()).hexdigest()
        store('config', cfg)

        trial = load('trial')
        trial['width']['drafter_config'] = '/workspace/mimo-tune/round8-cost-width/native16/config.json'
        trial['width']['retained_drafter_config'] = forged['drafter_config']
        trial['width']['retained_drafter_config_sha256'] = 'b' * 64
        trial['width']['failed_run'] = 'runs/france-round8-width16-ndt1-1774154000000000000'
        trial['width']['failed_pids'] = {'guard_pid': 4011, 'server_pid': 4012}
        trial['protected']['mandatory_roots'] = ['/workspace/mimo-exl3/models', ROOT + '/runs']
        trial['artifacts']['integration_pins_sha256'] = hashlib.sha256(
            (HERE / 'integration-pins.json').read_bytes()).hexdigest()
        store('trial', trial)

        receipt = load('receipt')
        receipt['boot_id'] = '11111111-2222-3333-4444-555555555555'
        receipt['uid'] = 1000
        receipt['captured_at'] = 1000.0
        receipt['owner'] = {'run': ROOT + '/runs/france-round6-quant-readback-1774153000000000000',
                            'pids': {'guard_pid': 3901, 'server_pid': 3902,
                                     'command': ['/workspace/mimo-exl3/venv/bin/python',
                                                 ROOT + '/round6-quant-readback/serve_native.py']},
                            'processes': [{'role': 'guard', 'pid': 3901, 'starttime': '99112233',
                                           'args': ['/usr/bin/python3', ROOT + '/guard_uma.py']},
                                          {'role': 'server', 'pid': 3902, 'starttime': '99112234',
                                           'args': ['/workspace/mimo-exl3/venv/bin/python',
                                                    ROOT + '/round6-quant-readback/serve_native.py']}]}
        receipt['scopes'] = {'3901': {'uid': 1000, 'ppid': 1, 'pgid': 3901, 'sid': 3901},
                            '3902': {'uid': 1000, 'ppid': 3901, 'pgid': 3902, 'sid': 3902}}
        receipt['launch_sha256'] = FILLER64
        receipt['live_config_sha256'] = 'c' * 64
        store('receipt', receipt)

        auth = load('authorization')
        auth.update({'root': ROOT, 'operator_lock': 'operator.lock', 'uid': 1000,
                     'boot_id': receipt['boot_id'], 'stage': 'release', 'issued_at': 1005.0,
                     'failure_at': 1006.0, 'expires_at': 1300.0,
                     'width': {'status': 'failed', 'error': 'not_run_before_staging',
                               'phase': 'before_width'},
                     'scope': 'ONE bounded native16 width trial on the GB10 host; no deployment'})
        store('authorization', auth)
        # The four document hashes the CLI will be handed (the authorization
        # carries the first three; its own hash is computed by the operator).
        hashes = {name: hashlib.sha256((target / (name + '.json')).read_bytes()).hexdigest()
                  for name in device_docs.DOCUMENTS}
        auth.update({'config_sha256': hashes['config'], 'receipt_sha256': hashes['receipt'],
                     'trial_sha256': hashes['trial']})
        store('authorization', auth)
        hashes = {name: hashlib.sha256((target / (name + '.json')).read_bytes()).hexdigest()
                  for name in device_docs.DOCUMENTS}
        yield target, hashes, docs


def mutate(target, name, change):
    path = target / (name + '.json')
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload, indent=2) + '\n')


class DeviceDocumentGateTests(unittest.TestCase):
    def test_complete_document_set_is_accepted_with_four_hashes(self):
        with filled_documents() as (target, hashes, docs):
            result = device_docs.check(target)
            self.assertEqual(set(result['documents']), set(device_docs.DOCUMENTS))
            self.assertEqual(result['documents'], hashes)
            self.assertEqual(result['stage'], 'release')
            self.assertEqual(result['pins_sha256'], hashlib.sha256(
                (HERE / 'integration-pins.json').read_bytes()).hexdigest())
            self.assertEqual(device_docs.main(['--check', str(target)]), 0)
            self.assertEqual(len(docs), 4)

    def test_the_printed_command_carries_every_document_and_root(self):
        with filled_documents() as (target, hashes, _):
            command = device_docs.check(target)['command']
            self.assertIn('EXL3_WIDTH_TRIAL_AUTHORIZED=YES', command)
            self.assertIn('--authorize-width-trial', command)
            self.assertIn('--protected-roots-complete', command)
            for name in device_docs.DOCUMENTS:
                self.assertIn('--' + name, command)
                self.assertIn(hashes[name], command)
            self.assertEqual(command.count('--protected-root'), 2)

    def test_unfilled_templates_are_refused(self):
        with tempfile.TemporaryDirectory(dir=str(HERE), prefix='device-docs-') as tmp:
            device_docs.write_templates(tmp)
            with self.assertRaises(device_docs.Refused) as caught:
                device_docs.check(tmp)
            self.assertIn('still has placeholders', str(caught.exception))
            self.assertEqual(device_docs.main(['--check', tmp]), 1)

    def test_templates_are_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=str(HERE), prefix='device-docs-') as tmp:
            device_docs.write_templates(tmp)
            with self.assertRaises(device_docs.Refused) as caught:
                device_docs.write_templates(tmp)
            self.assertIn('refusing to overwrite', str(caught.exception))

    def test_a_mismatched_configuration_hash_is_refused(self):
        with filled_documents() as (target, _, __):
            mutate(target, 'config', lambda payload: payload.update({'port': 8096, 'receipt_dir': 'receipts2'}))
            with self.assertRaises(device_docs.Refused) as caught:
                device_docs.check(target)
            self.assertIn('config_sha256 does not match', str(caught.exception))

    def test_a_tampered_trial_hash_is_refused(self):
        with filled_documents() as (target, _, __):
            mutate(target, 'authorization', lambda payload: payload.update({'trial_sha256': 'd' * 64}))
            with self.assertRaises(device_docs.Refused) as caught:
                device_docs.check(target)
            self.assertIn('trial_sha256 does not match', str(caught.exception))

    def test_a_nonproduction_configuration_is_refused_by_the_sealed_contract(self):
        with filled_documents() as (target, _, __):
            mutate(target, 'config', lambda payload: payload.update({'root': '/workspace/other'}))

            def move_trial(payload):
                payload['root'] = '/workspace/other'
                payload['width']['env']['ROUND8_WIDTH_OUT'] = (
                    '/workspace/other/round8-cost-width/width/ndt1')
            mutate(target, 'trial', move_trial)
            with self.assertRaises(Exception) as caught:
                device_docs.check(target)
            self.assertIn('production root/health target mismatch', str(caught.exception))

    def test_a_trial_root_that_is_not_the_configuration_root_is_refused(self):
        with filled_documents() as (target, _, __):
            mutate(target, 'trial', lambda payload: payload.update({'root': '/workspace/other'}))
            with self.assertRaises(Exception) as caught:
                device_docs.check(target)
            self.assertIn('trial root does not match the owned root', str(caught.exception))

    def test_an_inert_fixture_document_is_refused_at_the_device_gate(self):
        with filled_documents() as (target, _, __):
            mutate(target, 'trial', lambda payload: payload.update({'inert_fixture': True}))
            with self.assertRaises(device_docs.Refused) as caught:
                device_docs.check(target)
            self.assertIn('inert_fixture false', str(caught.exception))

    def test_a_phase_mismatch_with_the_stage_is_refused(self):
        with filled_documents() as (target, _, __):
            mutate(target, 'authorization',
                   lambda payload: payload.update({'stage': 'restore'}))
            with self.assertRaises(device_docs.Refused) as caught:
                device_docs.check(target)
            self.assertIn('after_width', str(caught.exception))

    def test_a_failed_outcome_without_its_error_text_is_refused(self):
        with filled_documents() as (target, _, __):
            def blank(payload):
                payload['width']['phase'] = 'after_width'
                payload['stage'] = 'restore'
                payload['width']['error'] = ''
            mutate(target, 'authorization', blank)
            with self.assertRaises(device_docs.Refused) as caught:
                device_docs.check(target)
            self.assertIn('observed error text', str(caught.exception))

    def test_an_extra_trial_field_is_refused_by_the_sealed_contract(self):
        with filled_documents() as (target, _, __):
            mutate(target, 'trial', lambda payload: payload.update({'extra': 1}))
            with self.assertRaises(Exception) as caught:
                device_docs.check(target)
            self.assertIn('trial', str(caught.exception))

    def test_the_shipped_templates_match_the_live_contract_field_sets(self):
        with tempfile.TemporaryDirectory(dir=str(HERE), prefix='device-docs-') as tmp:
            device_docs.write_templates(tmp)
            cfg = json.loads((Path(tmp) / 'config.json').read_text())
            trial = json.loads((Path(tmp) / 'trial.json').read_text())
            auth = json.loads((Path(tmp) / 'authorization.json').read_text())
            receipt = json.loads((Path(tmp) / 'receipt.json').read_text())
            self.assertEqual(set(trial), tc.TRIAL_FIELDS)
            self.assertEqual(set(trial['width']), tc.WIDTH_FIELDS)
            self.assertEqual(set(trial['width']['expect']), set(tc.EXPECTED))
            self.assertEqual(set(auth), tc.AUTH_FIELDS)
            self.assertEqual(set(receipt), {'schema', 'phase', 'captured_at', 'boot_id', 'uid',
                                           'owner', 'scopes', 'launch_sha256', 'live_config_sha256'})
            self.assertEqual(cfg['root'], a.PRODUCTION_ROOT)
            self.assertEqual(set(cfg['retained']), {'memory_policy', 'max_seconds', 'env', 'command'})
            self.assertEqual(trial['width']['drafter_config_sha256'], tc.WIDTH_CONFIG_SHA256)


if __name__ == '__main__':
    unittest.main(verbosity=2)
