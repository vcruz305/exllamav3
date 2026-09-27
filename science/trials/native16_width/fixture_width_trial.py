"""LOCAL inert width-trial fixture: a real Linux owned-state tree plus the
documents the CLI demands. No model, torch, CUDA or network.

Two owned states are built and torn down with the same construction:

  state A  the retained native8 serve          -> drives the release stage
  state B  the native16 width run (+ probe)    -> drives the restore stage

Each state is real: a `flock` guard process, a child server process listening on
an EPHEMERAL loopback port (never 8096), a run directory with `launch.json` /
`pid.json`, a live-config pointer and a live PID record captured with the same
`proc_snapshot` the sealed adapter uses.

The probe event stream written for state B is SYNTHESIZED to the sealed
diagnostic's `emit()` contract (same kinds, same field names, native16 values).
It is a contract fixture, not device evidence - see GAPS.md G-FIXTURE-EVIDENCE.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as S

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import linux_adapter as a  # noqa: E402

LAUNCHER_ENV = {'PATH': '/usr/bin:/bin', 'LC_ALL': 'C.UTF-8', 'PYTHONUNBUFFERED': '1'}
STACK = HERE / 'inert_width_stack.py'
WIDTH_PREFIX = 'france-round8-width16-ndt1-'
RETAINED_PREFIX = 'france-round6-quant-readback-'
WIDTH_ENV = {'ROUND8_WIDTH': '1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _probe_records():
    """Synthesized native16 event stream, matched to the sealed `emit()` calls."""
    taps = [0, 11, 23, 35, 47]
    table = list(range(16))
    positions = [[page, 0] for page in range(9)] + [[table[9 + i], 0] for i in range(7)]
    base = [dict(kind='installed', round=0, model_type='DraftModel', block=16, requested=1, reserve=15,
                 mask_shape=[4096], mask_finite=True, tap_shift=0, taps=taps, model_vocab=152576,
                 actual_vocab=151675, window=[[0, 0, 16]]),
            dict(kind='job_init', round=0, identifier='fixture-job', input_tokens=3, cap=32,
                 orig_max_rq=16, requeued=False),
            dict(kind='input_metadata', round=1, anchor_shape=[1, 1], input_shape=[1, 16, 4096],
                 anchor_device='cpu', anchor_dtype='torch.int64', output_device='cuda:0',
                 output_dtype='torch.float16', mask_shape=[4096], mask_device='cuda:0',
                 mask_dtype='torch.float16'),
            dict(kind='input', round=1, anchor_shape=[1, 1], input_shape=[1, 16, 4096],
                 finite=True, mask_equal=True),
            dict(kind='assigned_pages', round=1, start=0, end_exclusive=16, assigned=table,
                 table=table, positions=positions, max_rq=16, requeued=False),
            dict(kind='state', round=1, shape=[1, 16, 4096], finite=True)]
    for layer in (0, 2):
        base.append(dict(kind='qkv', round=1, layer=layer, path='original_eager', q_shape=[1, 16, 64, 128],
                         k_shape=[1, 16, 8, 128], v_shape=[1, 16, 8, 128], finite_q=True,
                         finite_k=True, finite_v=True))
    base.append(dict(kind='native_cache_writes', round=1, layer=0, changed_rows=15, finite_scales=True))
    base.append(dict(kind='pristine_draft_head', round=1, shape=[1, 16, 4096], finite_model_vocab=True))
    base.append(dict(kind='native_samples', round=1, shape=[1, 16], ids=list(range(16)),
                     proposal_count=15, padded_vocab_proposals=0))
    base.append(dict(kind='proposal_truncation', round=1, proposed=15))
    base.append(dict(kind='target_forward', round=1, q=16, prefill=False, counts_enabled=False,
                     batch_verify=False))
    base.append(dict(kind='received', round=1, token=7, before=1, after=[9, 3, 4], result='ok',
                     kv=16))
    base.append(dict(kind='terminal', round=1, values={'identifier': 'fixture-job', 'new_tokens': 9,
                                                       'accepted_draft_tokens': 3,
                                                       'rejected_draft_tokens': 4,
                                                       'eos_reason': None, 'time_generate': 0.1}))
    return base


def aborted_probe_records():
    """What the aborted device attempt actually captured: it died inside the
    input hook, so only the pre-hook gates exist. No `input`/`state`/`qkv`."""
    return [record for record in _probe_records() if record['kind'] in ('installed', 'job_init',
                                                                       'assigned_pages')]


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.probe_dir = self.root / 'round8-cost-width' / 'width' / 'ndt1'
        self.owned = {}
        self.cli_receipts = []

    # ------------------------------------------------------------- construction
    def write_json(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return path

    @contextlib.contextmanager
    def start_owned(self, prefix, width_env=None):
        """Spawn a real guard (+ child server) and capture its trusted identity."""
        if width_env is not None:
            width_env = dict(width_env)
            width_env.setdefault('ROUND8_WIDTH_OUT', str(self.probe_dir))
        run = self.root / 'runs' / (prefix + str(time.time_ns()))
        run.mkdir()
        command = [sys.executable, '-B', str(STACK), 'server', str(self.root), str(self.port)]
        env = dict(LAUNCHER_ENV)
        env.update(width_env or {})
        cfg = dict(self.retained, outdir=str(run), env=dict(width_env or {}))
        raw = json.dumps(cfg).encode()
        (run / 'launch.json').write_bytes(raw)
        (self.root / 'france-uma.json').write_bytes(raw)
        (self.root / 'france-active-run.txt').write_text(str(run))
        (self.root / 'retained-template.json').write_text(json.dumps(self.retained))
        (self.root / 'probe.json').write_text(json.dumps(
            {'mode': 'absent'} if width_env is None else {'mode': 'complete',
                                                          'records': self.probe_records}))
        if width_env is None:
            (self.root / 'health.json').write_text(json.dumps(
                {'status': 200, 'body': {'healthy': True, 'requests': 0, 'max_active_requests': 1}}))
        else:
            (self.root / 'health.json').write_text(json.dumps(
                {'status': 200, 'body': {'healthy': True, 'requests': 0, 'max_active_requests': 1}}))
        # `listening` is rewritten by each server; remove the previous state's
        # marker so readiness cannot be satisfied by a stale file.
        (self.root / 'listening').unlink(missing_ok=True)
        guard_env = dict(env)
        if width_env:
            guard_env.update(width_env)
        guard_log = (self.root / 'guard-fixture.log').open('wb')
        guard = subprocess.Popen([sys.executable, '-B', str(STACK), 'guard', str(self.root / 'france-uma.json')],
                                 env=guard_env, stdin=subprocess.DEVNULL, start_new_session=True,
                                 stdout=guard_log, stderr=guard_log)
        guard_log.close()  # the child holds its own descriptor
        # Reap the inert guard as soon as it exits. Without this the guard is a
        # zombie, and the sealed adapter correctly refuses to call a zombie
        # "proven gone". Exactly what the adapter's own local fixture does.
        reaper = threading.Thread(target=guard.wait, daemon=True)
        reaper.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            ready = (run / 'pid.json').exists() and (self.root / 'listening').exists()
            if ready and width_env:
                ready = (self.probe_dir / 'events.jsonl').exists()
            if ready:
                break
            time.sleep(.02)
        else:
            raise AssertionError('inert fixture state failed to start: '
                                 + (self.root / 'guard-fixture.log').read_text(errors='replace')[-1500:])
        pids = json.loads((run / 'pid.json').read_text())
        # The mapped-DSO identity must be a real mapping of the running server.
        # The sealed adapter's own inert fixture uses the server's libc for this;
        # it is the only honest local stand-in for the CUDA extension.
        libc = next(line.split()[-1] for line in
                    Path('/proc', str(pids['server_pid']), 'maps').read_text().splitlines()
                    if '/libc.so' in line)
        for row in self.identities:
            if row['role'] == 'dso':
                row['path'] = libc
                row['sha256'] = sha(libc)
        self.cfg['identities'] = self.identities
        snapshots = [a.proc_snapshot(pids[key]) for key in ('guard_pid', 'server_pid')]
        owner = {'run': str(run), 'pids': pids, 'processes': [row for row, _ in snapshots]}
        scopes = {str(row['pid']): scope for row, scope in snapshots}
        self.owned[str(run)] = (guard, owner, reaper)
        yield S(run=str(run), pids=pids, owner=owner, scopes=scopes, command=command)

    def receipt_for(self, state):
        raw = (Path(state.run) / 'launch.json').read_bytes()
        return {'schema': 1, 'phase': 'pre_failure', 'captured_at': time.time() - 1,
                'boot_id': self.boot_id, 'uid': os.getuid(), 'owner': state.owner,
                'scopes': state.scopes, 'launch_sha256': a.digest(raw),
                'live_config_sha256': a.digest((self.root / 'france-uma.json').read_bytes())}

    def authorization(self, receipt, config_sha, receipt_sha, trial_sha, phase, status=None, error=None):
        if phase == 'before_width':
            status, error = 'failed', 'not_run_before_staging'
        width = {'status': status, 'error': error, 'phase': phase}
        auth = {'schema': 1, 'intent': 'stop_owned_then_launch_retained_once', 'root': str(self.root),
                'operator_lock': 'operator.lock', 'ingress_blocked': True, 'uid': os.getuid(),
                'boot_id': self.boot_id, 'config_sha256': config_sha, 'receipt_sha256': receipt_sha,
                'trial_sha256': trial_sha, 'issued_at': time.time(), 'failure_at': time.time() - .5,
                'expires_at': time.time() + 300, 'width': width,
                'stage': 'release' if phase == 'before_width' else 'restore',
                'scope': 'LOCAL inert fixture only; no device, no model, no deployment'}
        return auth

    def write_documents(self, receipt, phase, status=None, error=None):
        self.write_json('documents/config.json', self.cfg)
        self.write_json('documents/receipt.json', receipt)
        self.write_json('documents/trial.json', self.trial)
        csha = a.digest(json.dumps(self.cfg).encode())
        rsha = a.digest(json.dumps(receipt).encode())
        tsha = a.digest(json.dumps(self.trial).encode())
        self.write_json('documents/authorization.json',
                        self.authorization(receipt, csha, rsha, tsha, phase, status, error))
        self.document_shas = {'config': csha, 'receipt': rsha, 'trial': tsha}
        return self.documents()

    def documents(self):
        return S(config=str(self.root / 'documents/config.json'),
                 config_sha256=self.document_shas['config'],
                 receipt=str(self.root / 'documents/receipt.json'),
                 receipt_sha256=self.document_shas['receipt'],
                 trial=str(self.root / 'documents/trial.json'),
                 trial_sha256=self.document_shas['trial'],
                 authorization=str(self.root / 'documents/authorization.json'))

    # --------------------------------------------------------------- the CLI
    def run_cli(self, stage, documents, *extra, hook=True, token=True, expect=0):
        command = [sys.executable, '-I', '-B', str(HERE / 'width_trial.py'), '--stage', stage,
                   '--config', documents.config, '--config-sha256', documents.config_sha256,
                   '--receipt', documents.receipt, '--receipt-sha256', documents.receipt_sha256,
                   '--authorization', documents.authorization,
                   '--authorization-sha256', a.digest(Path(documents.authorization).read_bytes()),
                   '--trial', documents.trial, '--trial-sha256', documents.trial_sha256,
                   '--protected-root', str(self.root / 'protected-a'),
                   '--protected-root', str(self.root / 'protected-b'),
                   '--protected-roots-complete', '--authorize-width-trial', *extra]
        environ = dict(os.environ)
        environ.pop('ROUND8_WIDTH', None)
        environ.pop('ROUND8_WIDTH_OUT', None)
        if token:
            environ['EXL3_WIDTH_TRIAL_AUTHORIZED'] = 'YES'
        if hook:
            environ['EXL3_WIDTH_TRIAL_IO_HOOK'] = 'witness.inert_trial_io'
        proc = subprocess.run(command, cwd=str(HERE), env=environ, capture_output=True, text=True,
                              timeout=180)
        self.cli_receipts.append({'stage': stage, 'command': command, 'returncode': proc.returncode,
                                  'stdout': proc.stdout, 'stderr': proc.stderr})
        if expect is not None and proc.returncode != expect:
            try:
                parsed = json.loads(proc.stdout)
            except ValueError:
                parsed = {'raw': proc.stdout[:800]}
            raise AssertionError(json.dumps({'stage': stage, 'returncode': proc.returncode,
                                             'expected': expect, 'status': parsed.get('status'),
                                             'rollback_error': parsed.get('rollback_error'),
                                             'error': parsed.get('error'),
                                             'width': parsed.get('width'),
                                             'stderr': proc.stderr[-1500:]}, indent=2))
        try:
            return json.loads(proc.stdout)
        except ValueError:
            return {'status': 'unparseable', 'raw': proc.stdout}

    # ------------------------------------------------------------- teardown
    def stop_everything(self):
        """STOP every fixture run and prove every fixture PID is gone."""
        evidence = []
        for run in sorted((self.root / 'runs').iterdir()) if (self.root / 'runs').is_dir() else []:
            if not (run / 'pid.json').exists():
                continue
            pids = json.loads((run / 'pid.json').read_text())
            (run / 'STOP').write_text('fixture teardown')
            deadline = time.monotonic() + 10
            left = [pids['guard_pid'], pids['server_pid']]
            while time.monotonic() < deadline:
                # The retained launcher's guard is reparented to this process
                # when the CLI exits, so a STOP leaves a zombie behind unless the
                # reaper runs here. A zombie is not a live process, but the /proc
                # entry survives until it is reaped, so reap explicitly.
                for pid in (pids['guard_pid'], pids['server_pid']):
                    try:
                        os.waitpid(int(pid), os.WNOHANG)
                    except (ChildProcessError, ProcessLookupError, OSError):
                        pass
                left = [pid for pid in (pids['guard_pid'], pids['server_pid'])
                        if Path('/proc', str(pid)).exists()]
                if not left:
                    break
                time.sleep(.05)
            evidence.append({'run': str(run), 'pids': list(pids.values())[:2], 'left': left,
                             'result': json.loads((run / 'result.json').read_text())
                             if (run / 'result.json').exists() else None})
        for handle in list(self.owned.values()):
            guard = handle[0] if isinstance(handle, tuple) else handle
            try:
                guard.wait(timeout=5)
            except Exception:  # noqa: BLE001 - teardown must not raise over a dead child
                pass
        return evidence


@contextlib.contextmanager
def fixture(*, width_records=None):
    import unittest.mock as mock
    with tempfile.TemporaryDirectory(dir=str(HERE), prefix='fixture-') as tmp, \
            mock.patch.object(a, 'check_permissions'):
        root = Path(tmp)
        (root / 'runs').mkdir()
        (root / 'receipts').mkdir()
        (root / 'documents').mkdir()
        (root / 'protected-a').mkdir()
        (root / 'protected-b').mkdir()
        (root / 'operator.lock').write_text('')
        (root / 'server.lock').write_text('')
        (root / 'launch-count').write_text('')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        assert port != 8096, 'the inert fixture must never use the production health port'
        fixture_object = Fixture(root)
        fixture_object.port = port
        fixture_object.probe_records = width_records if width_records is not None else _probe_records()
        fixture_object.retained = {'memory_policy': 'uma', 'max_seconds': 43200, 'env': {},
                                  'command': [sys.executable, '-B', str(STACK), 'server', str(root), str(port)]}
        fixture_object.identities = [
            {'role': 'guard', 'path': str(STACK), 'sha256': sha(STACK)},
            {'role': 'launcher', 'path': str(STACK), 'sha256': sha(STACK)},
            {'role': 'server', 'path': str(STACK), 'sha256': sha(STACK)},
            {'role': 'source', 'path': str(STACK), 'sha256': sha(STACK)},
            {'role': 'dso', 'path': fixture_object.retained['command'][0],
             'sha256': sha(fixture_object.retained['command'][0])},
            {'role': 'interpreter', 'path': str(Path(sys.executable).resolve()),
             'sha256': sha(Path(sys.executable).resolve())},
            {'role': 'target_config', 'path': str(HERE / 'width/candidate/width_config.json'),
             'sha256': sha(HERE / 'width/candidate/width_config.json')},
            {'role': 'drafter_config', 'path': str(HERE / 'width/candidate/width_config.json'),
             'sha256': sha(HERE / 'width/candidate/width_config.json')}]
        fixture_object.cfg = {
            'schema': 1, 'source_pin': a.SOURCE_PIN, 'root': str(root), 'operator_lock': 'operator.lock',
            'active': 'france-active-run.txt', 'live_config': 'france-uma.json',
            'server_lock': 'server.lock',
            'guard_argv': [sys.executable, '-B', str(STACK), 'guard', str(root / 'france-uma.json')],
            'launcher_argv': [sys.executable, '-B', str(STACK), 'launcher', str(root)],
            'launcher_env': dict(LAUNCHER_ENV), 'retained': fixture_object.retained,
            'identities': fixture_object.identities, 'port': port, 'new_run_prefix': RETAINED_PREFIX,
            'receipt_dir': 'receipts',
            'timeouts': {'stop': 10, 'headroom': 10, 'launch': 30, 'ready': 30, 'http': 3}}
        fixture_object.cfg['guard_argv'][4] = str(root / 'france-uma.json')
        fixture_object.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        fixture_object.trial = {
            'schema': 1, 'inert_fixture': True, 'source_pin': a.SOURCE_PIN, 'root': str(root),
            'width': {'native_block': 16, 'retained_block': 8,
                      'drafter_config': str(HERE / 'width/candidate/width_config.json'),
                      'drafter_config_sha256': sha(HERE / 'width/candidate/width_config.json'),
                      'retained_drafter_config': str(HERE / 'width/candidate/width_config.json'),
                      'retained_drafter_config_sha256': sha(HERE / 'width/candidate/width_config.json'),
                      'probe_dir': 'round8-cost-width/width/ndt1', 'events': 'events.jsonl',
                      'run_prefix': WIDTH_PREFIX, 'failed_run': None,
                      'failed_pids': {'guard_pid': 999000, 'server_pid': 999001},
                      'env': {'ROUND8_WIDTH': '1',
                              'ROUND8_WIDTH_OUT': str(root / 'round8-cost-width/width/ndt1')},
                      'expect': {'input_shape': [1, 16, 4096], 'state_shape': [1, 16, 4096],
                                 'q_shape': [1, 16, 64, 128], 'k_shape': [1, 16, 8, 128],
                                 'samples_shape': [1, 16], 'proposal_count': 15, 'assigned_span': 16,
                                 'max_round': 512}},
            'protected': {'mandatory_roots': [str(root / 'protected-a')],
                          'nesting_acknowledged': False},
            'artifacts': {'integration_pins': 'integration-pins.json',
                          'integration_pins_sha256': sha(HERE / 'integration-pins.json')},
            'scope': 'LOCAL inert fixture; no device, model, GPU, network or deployment'}
        # The retained (native8) drafter config must differ from the width config;
        # the fixture uses a block-8 sibling file so both are real pinned files.
        retained_view = root / 'retained-view.json'
        retained_doc = json.loads((HERE / 'width/candidate/width_config.json').read_text())
        retained_doc['block_size'] = 8
        retained_doc['dflash_config']['block_size'] = 8
        retained_view.write_text(json.dumps(retained_doc))
        fixture_object.trial['width']['retained_drafter_config'] = str(retained_view)
        fixture_object.trial['width']['retained_drafter_config_sha256'] = sha(retained_view)
        try:
            yield fixture_object
        finally:
            evidence = fixture_object.stop_everything()
            fixture_object.cleanup_evidence = evidence
