"""Candidate tests: the real two-stage width trial on real local Linux processes.

Every process here is an inert child (`inert_width_stack.py`) and every HTTP
endpoint is an ephemeral loopback port (never 8096). The CLI under test is the
real `width_trial.py`, run as `/usr/bin/python3 -I -B`, with the single
documented inert-witness hook (`witness.inert_trial_io`) that the CLI itself
refuses for any production trial document.

Sequence exercised end to end:

  A. the retained native8 serve is live            -> preflight + release
     (identity -> STOP -> both gone -> bounded removal -> headroom; NO launch)
  B. the native16 width run is staged with the sealed diagnostic enabled and a
     probe event stream on disk                     -> preflight + restore
     (the ONE documented sequence plus the independent R1-R12 readback)

Fail-closed controls re-run the same CLI against incomplete evidence, a
contradicted outcome, a wrong live owner, a live previous attempt and a stale
probe directory; each must refuse with a specific message.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import linux_adapter as a  # noqa: E402
import fixture_width_trial as fx  # noqa: E402
import trial_contract as tc  # noqa: E402
import trial_readiness as tr  # noqa: E402

CLEANUP = []


def record_cleanup(evidence):
    CLEANUP.extend(evidence)
    print('INERT_CLEANUP ' + json.dumps(evidence), flush=True)


class WidthTrialIntegration(unittest.TestCase):
    def assert_refused(self, result, needle):
        self.assertEqual(result.get('status'), 'refused', result)
        self.assertIn(needle, result.get('rollback_error') or '', result)

    # ------------------------------------------------------------- happy path
    def test_two_stage_release_then_restore(self):
        with fx.fixture() as f:
            try:
                with f.start_owned(fx.RETAINED_PREFIX) as state_a:
                    docs = f.write_documents(f.receipt_for(state_a), phase='before_width')
                    preflight = f.run_cli('preflight', docs)
                    self.assertEqual(preflight['status'], 'preflighted')
                    checks = preflight['preflight']['checks']
                    self.assertEqual(set(checks), {'P1_profile', 'P2_documents', 'P3_protected_roots',
                                                    'P4_retained_inputs', 'P4L_live_owner',
                                                    'P5_previous_width_attempt', 'P6_probe_state'})
                    self.assertEqual(preflight['preflight']['checks']['P6_probe_state']['probe_absent'],
                                     f.trial['width']['probe_dir'])
                    self.assertEqual(checks['P4L_live_owner']['health']['body']['requests'], 0)
                    self.assertFalse(f.probe_dir.exists())

                    release = f.run_cli('release', docs)
                    self.assertEqual(release['status'], 'released')
                    self.assertEqual(release['rollback'], 'not_run')
                    self.assertNotIn('restore', release)
                    report = release['report']
                    self.assertEqual(report['status'], 'released')
                    self.assertTrue(report['launch_count_unchanged'])
                    self.assertFalse(report['probe_created'])
                    self.assertEqual(report['guard_result']['reason'], 'requested_stop')
                    self.assertEqual(sorted(report['processes_gone']),
                                     sorted([state_a.pids['guard_pid'], state_a.pids['server_pid']]))
                    for row in state_a.owner['processes']:
                        self.assertIsNone(a.proc_snapshot(row['pid']),
                                          'released process still visible: ' + str(row['pid']))
                self.assertEqual((f.root / 'launch-count').read_text(), '',
                                 'the release stage launched a model')

                # B. stage the native16 width run (outside this package on the device)
                with f.start_owned(fx.WIDTH_PREFIX, fx.WIDTH_ENV) as state_b:
                    self.assertTrue((f.probe_dir / 'events.jsonl').is_file())
                    docs = f.write_documents(f.receipt_for(state_b), phase='after_width',
                                             status='passed', error=None)
                    preflight = f.run_cli('preflight', docs)
                    checks = preflight['preflight']['checks']
                    self.assertIn('P4R_live_width_owner', checks)
                    self.assertEqual(checks['P4R_live_width_owner']['width_env'],
                                     {'ROUND8_WIDTH': '1',
                                      'ROUND8_WIDTH_OUT': str(f.probe_dir)})
                    verdict = checks['P6_probe_state']['probe']
                    self.assertEqual(verdict['status'], 'passed', verdict)
                    self.assertEqual(verdict['records'], 15, verdict['counts'])
                    self.assertEqual(preflight['width']['phase'], 'after_width')

                    restore = f.run_cli('restore', docs)
                    self.assertEqual(restore['status'], 'restored', restore)
                    self.assertEqual(restore['rollback'], 'restored')
                    self.assertIsNone(restore['rollback_error'])
                    self.assertIsNone(restore['readback_error'])
                    readiness = restore['readiness']
                    self.assertEqual(readiness['status'], 'passed')
                    self.assertEqual(set(readiness['checks']),
                                     {'R1_active_pointer', 'R2_pid_record', 'R3_identity',
                                      'R4_prior_gone', 'R5_health', 'R6_configuration',
                                      'R7_drafter_configs', 'R8_no_width_env', 'R9_identities',
                                      'R10_process_inventory', 'R11_memory', 'R12_protected_roots'})
                    self.assertEqual(readiness['checks']['R5_health']['status'], 200)
                    self.assertFalse(readiness['checks']['R5_health']['body'].get('width'))
                    self.assertEqual((f.root / 'launch-count').read_text(), '1\n')
                    for row in state_b.owner['processes']:
                        self.assertIsNone(a.proc_snapshot(row['pid']),
                                          'width process survived the restore')
                    # the probe evidence must survive the recovery untouched
                    self.assertTrue((f.probe_dir / 'events.jsonl').is_file())
                    self.assertEqual(a.digest((f.probe_dir / 'events.jsonl').read_bytes()),
                                     verdict['events_sha256'])
                    self.assertTrue(Path(restore['status_path']).is_file())
                    self.assertEqual(json.loads(Path(restore['status_path']).read_text())['probe']['status'],
                                     'passed')
            finally:
                record_cleanup(f.stop_everything())

    # ------------------------------------------------------- fail-closed gates
    def test_restore_refused_when_recorded_outcome_contradicts_the_probe(self):
        with fx.fixture(width_records=fx.aborted_probe_records()) as f:
            try:
                with f.start_owned(fx.WIDTH_PREFIX, fx.WIDTH_ENV) as state_b:
                    docs = f.write_documents(f.receipt_for(state_b), phase='after_width',
                                             status='passed', error=None)
                    result = f.run_cli('restore', docs, expect=1)
                    self.assert_refused(result, 'contradicts the probe evidence on disk')
                    self.assertEqual((f.root / 'launch-count').read_text(), '',
                                     'a refused restore launched a model')
                    self.assertIsNotNone(a.proc_snapshot(state_b.pids['server_pid']),
                                         'a refused restore stopped the width run')
            finally:
                record_cleanup(f.stop_everything())

    def test_failed_outcome_with_incomplete_probe_is_reported_and_still_restored(self):
        """A crashed width attempt must still be recoverable - but its missing
        gates are named exactly, and the trial does not report success."""
        with fx.fixture(width_records=fx.aborted_probe_records()) as f:
            try:
                with f.start_owned(fx.WIDTH_PREFIX, fx.WIDTH_ENV) as state_b:
                    docs = f.write_documents(f.receipt_for(state_b), phase='after_width',
                                             status='failed', error='device refusal: CPU vs CUDA equal')
                    preflight = f.run_cli('preflight', docs)
                    verdict = preflight['preflight']['checks']['P6_probe_state']['probe']
                    self.assertEqual(verdict['status'], 'failed')
                    self.assertIn('input', verdict['missing'])
                    self.assertIn('terminal', verdict['missing'])
                    self.assertTrue(verdict['error'].startswith('native16 probe incomplete'))
                    restore = f.run_cli('restore', docs, expect=1)
                    self.assertEqual(restore['status'], 'restored')
                    self.assertEqual(restore['width']['status'], 'failed')
                    self.assertEqual(restore['rollback'], 'restored')
                    self.assertEqual(restore['readiness']['status'], 'passed')
                    self.assertEqual(restore['restore']['width'], 'failed')
            finally:
                record_cleanup(f.stop_everything())

    def test_restore_refused_when_the_probe_directory_is_gone(self):
        with fx.fixture() as f:
            try:
                with f.start_owned(fx.WIDTH_PREFIX, fx.WIDTH_ENV) as state_b:
                    import shutil
                    shutil.rmtree(f.probe_dir)
                    docs = f.write_documents(f.receipt_for(state_b), phase='after_width',
                                             status='failed', error='aborted')
                    result = f.run_cli('restore', docs, expect=1)
                    self.assert_refused(result, 'left no probe directory')
                    self.assertEqual((f.root / 'launch-count').read_text(), '')
            finally:
                record_cleanup(f.stop_everything())

    def test_missing_event_stream_is_reported_not_silently_accepted(self):
        """A directory with no stream cannot be claimed as passed; with a
        recorded failed outcome the restore still runs and the gap is printed."""
        with fx.fixture() as f:
            try:
                with f.start_owned(fx.WIDTH_PREFIX, fx.WIDTH_ENV) as state_b:
                    (f.probe_dir / 'events.jsonl').unlink()
                    docs = f.write_documents(f.receipt_for(state_b), phase='after_width',
                                             status='failed', error='no stream captured')
                    preflight = f.run_cli('preflight', docs)
                    verdict = preflight['preflight']['checks']['P6_probe_state']['probe']
                    self.assertEqual(verdict['status'], 'failed')
                    self.assertEqual(verdict['error'], 'probe evidence missing: '
                                     'round8-cost-width/width/ndt1/events.jsonl')
                    restore = f.run_cli('restore', docs, expect=1)
                    self.assertEqual(restore['status'], 'restored')
                    self.assertEqual(restore['width']['status'], 'failed')
            finally:
                record_cleanup(f.stop_everything())

    def test_restore_refused_when_the_live_owner_never_had_the_diagnostic(self):
        with fx.fixture() as f:
            try:
                with f.start_owned(fx.WIDTH_PREFIX) as state_b:
                    docs = f.write_documents(f.receipt_for(state_b), phase='after_width',
                                             status='failed', error='aborted')
                    result = f.run_cli('restore', docs, expect=1)
                    self.assert_refused(result, 'never had the sealed diagnostic enabled')
            finally:
                record_cleanup(f.stop_everything())

    def test_release_refused_when_a_previous_probe_directory_exists(self):
        with fx.fixture() as f:
            try:
                f.probe_dir.mkdir(parents=True)
                (f.probe_dir / 'events.jsonl').write_text('')
                with f.start_owned(fx.RETAINED_PREFIX) as state_a:
                    docs = f.write_documents(f.receipt_for(state_a), phase='before_width')
                    result = f.run_cli('release', docs, expect=1)
                    self.assert_refused(result, 'APPENDS')
            finally:
                record_cleanup(f.stop_everything())

    def test_release_refused_when_the_previous_attempt_is_still_live(self):
        with fx.fixture() as f:
            f.trial['width']['failed_pids'] = {'guard_pid': os.getpid(), 'server_pid': os.getpid() + 1}
            try:
                with f.start_owned(fx.RETAINED_PREFIX) as state_a:
                    docs = f.write_documents(f.receipt_for(state_a), phase='before_width')
                    result = f.run_cli('release', docs, expect=1)
                    self.assert_refused(result, 'still live')
            finally:
                record_cleanup(f.stop_everything())

    def test_release_refused_without_the_authorization_token(self):
        with fx.fixture() as f:
            try:
                with f.start_owned(fx.RETAINED_PREFIX) as state_a:
                    docs = f.write_documents(f.receipt_for(state_a), phase='before_width')
                    result = f.run_cli('release', docs, token=False, expect=1)
                    self.assertIn('EXL3_WIDTH_TRIAL_AUTHORIZED', json.dumps(result))
            finally:
                record_cleanup(f.stop_everything())

    def test_stage_must_match_the_authorization(self):
        with fx.fixture() as f:
            try:
                with f.start_owned(fx.RETAINED_PREFIX) as state_a:
                    docs = f.write_documents(f.receipt_for(state_a), phase='before_width')
                    result = f.run_cli('restore', docs, expect=1)
                    self.assertIn('stage does not match', json.dumps(result))
            finally:
                record_cleanup(f.stop_everything())

    def test_a_tampered_document_is_refused_before_anything_acts(self):
        with fx.fixture() as f:
            try:
                with f.start_owned(fx.RETAINED_PREFIX) as state_a:
                    docs = f.write_documents(f.receipt_for(state_a), phase='before_width')
                    trial = json.loads(Path(docs.trial).read_text())
                    trial['scope'] = 'tampered scope'
                    Path(docs.trial).write_text(json.dumps(trial))
                    result = f.run_cli('release', docs, expect=1)
                    self.assertIn('trusted document hash mismatch: trial', json.dumps(result))
                    # Supply the matching document hash but leave the
                    # authorization pinned to the previous trial document.
                    docs.trial_sha256 = a.digest(Path(docs.trial).read_bytes())
                    result = f.run_cli('release', docs, expect=1)
                    self.assertIn('trial-document hash mismatch', json.dumps(result))
                    # Restore the trial and tamper the CONFIGURATION instead.
                    Path(docs.trial).write_text(json.dumps(self_trial := f.trial))
                    docs.trial_sha256 = a.digest(Path(docs.trial).read_bytes())
                    cfg = json.loads(Path(docs.config).read_text())
                    cfg['timeouts']['stop'] = 1
                    Path(docs.config).write_text(json.dumps(cfg))
                    docs.config_sha256 = a.digest(Path(docs.config).read_bytes())
                    result = f.run_cli('release', docs, expect=1)
                    self.assertIn('authorization/config hash mismatch', json.dumps(result))
            finally:
                record_cleanup(f.stop_everything())

    def test_production_profile_still_refuses_an_inert_fixture_tree(self):
        with fx.fixture() as f:
            try:
                with self.assertRaisesRegex(a.Refusal, 'production root/health target mismatch'):
                    a.validate_production(f.cfg)
                self.assertEqual(json.loads((HERE / 'runtime-pins.json').read_text())['linux_adapter.py'],
                                 a.digest((HERE / 'linux_adapter.py').read_bytes()))
            finally:
                record_cleanup(f.stop_everything())


class InProcessRecoveryTests(unittest.TestCase):
    """The same sequence driven in-process, plus the readback negatives."""

    def test_perform_restore_and_readiness_negatives(self):
        import unittest.mock as mock
        import trial_io as ti
        import witness.inert_trial_io as inert
        with fx.fixture() as f:
            try:
                with f.start_owned(fx.WIDTH_PREFIX, fx.WIDTH_ENV) as state_b:
                    docs = f.write_documents(f.receipt_for(state_b), phase='after_width',
                                             status='passed', error=None)
                    cfg = json.loads(Path(docs.config).read_text())
                    receipt = json.loads(Path(docs.receipt).read_text())
                    auth = json.loads(Path(docs.authorization).read_text())
                    trial = json.loads(Path(docs.trial).read_text())
                    snapshot = tr.snapshot_protected_roots([str(f.root / 'protected-a'),
                                                           str(f.root / 'protected-b')])
                    factory = inert.build('restore')  # installs the child subreaper
                    with mock.patch.object(a, 'check_permissions'):
                        with factory(cfg, receipt, auth, docs.config_sha256,
                                     docs.receipt_sha256, trial, docs.trial_sha256,
                                     [str(f.root / 'protected-a'),
                                      str(f.root / 'protected-b')], True,
                                     'restore') as io:
                            io.require_trial_state()
                            code, status = ti.perform_restore(io)
                            self.assertEqual(code, 0, status)
                            readback = tr.HostReadback(io.root, cfg['port'])
                            owner = status['restored']
                            report = tr.readiness_report(readback, trial, cfg, owner, io.expected,
                                                         io.protected_roots, snapshot)
                            self.assertEqual(report['status'], 'passed')
                            # R12 negative: a protected root that gained an entry is refused.
                            (f.root / 'protected-a' / 'new-file').write_text('x')
                            with self.assertRaisesRegex(a.Refusal, 'changed at the entry level'):
                                tr.readiness_report(readback, trial, cfg, owner, io.expected,
                                                    io.protected_roots, snapshot)
                            (f.root / 'protected-a' / 'new-file').unlink()
                            # R5 negative: a busy restored serve is refused.
                            (f.root / 'health.json').write_text(json.dumps(
                                {'status': 200, 'body': {'healthy': True, 'requests': 3}}))
                            with self.assertRaisesRegex(a.Refusal, 'HTTP200/healthy/idle'):
                                tr.readiness_report(readback, trial, cfg, owner, io.expected,
                                                    io.protected_roots, snapshot)
                            (f.root / 'health.json').write_text(json.dumps(
                                {'status': 200, 'body': {'healthy': True, 'requests': 0,
                                                        'max_active_requests': 1}}))
                            # R11 negative: a damaged OOM counter is refused.
                            broken = dict(snapshot)
                            with mock.patch.object(tr.HostReadback, 'oom_and_memory') as fake:
                                fake.return_value = {'meminfo': {}, 'vmstat': {'oom_kill': 2, 'oom': 0},
                                                     'cgroup_events': {'oom': 0, 'oom_kill': 0},
                                                     'mem_available_gib': 40.0, 'mem_free_gib': 30.0}
                                with self.assertRaisesRegex(a.Refusal, 'host OOM kill counter'):
                                    tr.readiness_report(readback, trial, cfg, owner, io.expected,
                                                        io.protected_roots, broken)
                            # R7 negative: a swapped drafter config is refused.
                            snapshot = tr.snapshot_protected_roots([str(f.root / 'protected-a'),
                                                                   str(f.root / 'protected-b')])
                            swapped = Path(trial['width']['retained_drafter_config'])
                            original = swapped.read_bytes()
                            # Install the native16 width config at the retained path.
                            swapped.write_bytes((HERE / 'width/candidate/width_config.json').read_bytes())
                            with self.assertRaisesRegex(a.Refusal, 'retained drafter config hash changed'):
                                tr.readiness_report(readback, trial, cfg, owner, io.expected,
                                                    io.protected_roots, snapshot)
                            swapped.write_bytes(original)
            finally:
                record_cleanup(f.stop_everything())


class DrvfsBypassIsConfinedTests(unittest.TestCase):
    def test_sealed_permission_predicate_still_refuses_a_world_writable_file(self):
        target = HERE / 'verification-tmp-0777.txt'
        target.write_text('x')
        driver = '\n'.join([
            'import sys, os',
            'sys.path.insert(0, sys.argv[1])',
            'import linux_adapter as a',
            'source = open(sys.argv[1] + "/linux_adapter.py").read()',
            'assert "st_mode & 0o022" in source, "sealed predicate changed"',
            'try:',
            '    a.check_permissions(os.stat(sys.argv[2]))',
            'except a.Refusal as exc:',
            '    print("REFUSED", exc)',
            'else:',
            '    raise SystemExit("the sealed predicate accepted a world-writable file")'])
        try:
            result = subprocess.run([sys.executable, '-B', '-c', driver, str(HERE), str(target)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('REFUSED', result.stdout)
        finally:
            target.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
