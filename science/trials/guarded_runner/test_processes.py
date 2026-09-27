"""Real LOCAL Linux regression. All fixture work is inert and expires in 12s."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
R = ROOT.parent
CANDIDATE = R / 'mixedk-three-stage-candidate'
GUARD = R / 'guard-host-oom-candidate/guard_uma.py'
GUARD_SHA = 'eef5706b327cf20007dd49087b6d70b97ec178fcdbde616374188a0bc916d24d'
sys.path.insert(0, str(ROOT))
from process_fixture import identity


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sample_for(out, trigger):
    def sample():
        ready = (out / 'ready').exists()
        if ready and trigger == 'monitor_error':
            raise RuntimeError('injected-monitor-error')
        if ready and trigger == 'monitor_interrupt':
            raise KeyboardInterrupt('injected-monitor-interrupt')
        if ready and trigger == 'stop':
            (out / 'guard' / 'STOP').touch()
        if ready and trigger == 'term':
            os.kill(os.getpid(), signal.SIGTERM)
        return dict(available_gib=7 if ready and trigger == 'memory' else 120,
                    free_gib=120, cgroup_headroom_gib=120, psi_full10=0,
                    oom_kill=0, host_oom_kill=0)
    return sample


def worker(out, implementation, trigger):
    guard = load(GUARD, 'exact_guard')
    assert hashlib.sha256(GUARD.read_bytes()).hexdigest() == GUARD_SHA
    env = dict(os.environ, FIXTURE_OUT=str(out),
               EXL3_TRIAL_MODEL_STOPPED='YES', EXL3_THREE_STAGE_AUTHORIZED='YES',
               EXL3_THREE_STAGE_GUARD_ACTIVE='YES', MAX_JOBS='1')
    if implementation == 'old':
        for name in ('bounded_deferred.py', 'build_trial.py'):
            shutil.copyfile(CANDIDATE / name, out / name)
        shutil.copyfile(ROOT / 'process_fixture.py', out / 'deferred_gpu.py')
        command = [sys.executable, '-B', str(out / 'bounded_deferred.py'),
                   '--receipt', str(out / 'nested'), '--seconds', '5', '--']
        result = guard.supervise(command, out / 'guard', env=env,
                                 sample=sample_for(out, trigger), interval=.03,
                                 max_seconds=.7, memory_policy='uma')
        (out / 'worker-result.json').write_text(json.dumps(result, indent=2))
        return 0 if result['reason'] == 'exit' and result['returncode'] == 0 else 125
    runner = load(ROOT / 'guarded_deferred.py', 'repair')
    return runner.run_guarded(guard, [sys.executable, '-B', str(ROOT / 'process_fixture.py')],
                              out / 'guard', env=env, sample=sample_for(out, trigger),
                              interval=.03, seconds=.7, memory_policy='uma')


def live(v):
    if not v:
        return False
    now = identity(v['pid'])
    return now is not None and now['starttime'] == v['starttime'] and now['state'] not in ('Z', 'X')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--implementation', choices=('old', 'repair'), default='repair')
    ap.add_argument('--trigger', default='memory', choices=('memory', 'timeout', 'stop', 'term', 'monitor_error', 'exit', 'wrapperkill', 'guardkill', 'wrapperterm', 'wrapperint', 'monitor_interrupt', 'guardfreeze'))
    ap.add_argument('--mode', default='wait', choices=('wait', 'descendant', 'leader_exit', 'escape', 'exit0', 'exit7'))
    ap.add_argument('--worker', action='store_true')
    a = ap.parse_args()
    assert sys.platform == 'linux'
    assert a.label.replace('-', '').replace('_', '').isalnum()
    out = ROOT / 'evidence' / a.label
    if a.worker:
        return worker(out, a.implementation, a.trigger)
    out.mkdir(parents=True, exist_ok=False)
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    env = dict(os.environ, FIXTURE_MODE=a.mode)
    p = None
    result = dict(scope='LOCAL WSL Linux; real processes; injected samples; inert workload; no GPU/network',
                  args=vars(a), guard_sha256=hashlib.sha256(GUARD.read_bytes()).hexdigest())
    identities = []
    try:
        with (out / 'outer-stdout.txt').open('xb') as stdout, (out / 'outer-stderr.txt').open('xb') as stderr:
            p = subprocess.Popen([sys.executable, '-B', __file__, *sys.argv[1:], '--worker'],
                                 stdout=stdout, stderr=stderr, env=env)
            if a.trigger in ('wrapperkill', 'guardkill', 'wrapperterm', 'wrapperint', 'guardfreeze'):
                deadline = time.monotonic() + 2
                while not (out / 'ready').exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                assert (out / 'ready').exists(), 'HARNESS: no READY before kill'
                victim = p.pid if a.trigger.startswith('wrapper') else json.loads((out / 'guard/pid.json').read_text())['guard_pid']
                result['killed_identity'] = identity(victim)
                fd = os.pidfd_open(victim)
                try:
                    sig = {'wrapperterm': signal.SIGTERM, 'wrapperint': signal.SIGINT, 'guardfreeze': signal.SIGSTOP}.get(a.trigger, signal.SIGKILL)
                    signal.pidfd_send_signal(fd, sig)
                finally:
                    os.close(fd)
            result['outer_returncode'] = p.wait(timeout=5)
        assert (out / 'leader.json').exists(), 'HARNESS: fixture never reached start'
        identities = [json.loads(f.read_text()) for f in out.glob('*.json') if f.stem in ('leader', 'descendant')]
        if a.trigger in ('wrapperkill', 'guardkill', 'wrapperterm', 'wrapperint', 'guardfreeze'):
            deadline = time.monotonic() + 2
            while (any(identity(v['pid']) is not None for v in identities) or
                   not (out / 'guard/custody.json').exists()) and time.monotonic() < deadline:
                time.sleep(.01)
        if a.implementation == 'repair':
            assert (out / 'guard/custody.json').exists(), 'missing crash-safe custody receipt'
            custody = json.loads((out / 'guard/custody.json').read_text())
            result['custody'] = custody
            assert not custody['cleanup']['remaining_children'], 'undrained custodial children'
            assert all(identity(v['pid']) is None for v in identities), 'owned workload PIDs not fully reaped'
        result['identities'] = identities
        result['before_harness_cleanup'] = [identity(v['pid']) for v in identities]
        result['live_before_harness_cleanup'] = [v for v in identities if live(v)]
        log = (out / ('nested/raw.log' if a.implementation == 'old' else 'guard/server.log')).read_text()
        assert 'fixture-stdout-leader' in log and 'fixture-stderr-leader' in log, 'lost stdout/stderr'
        assert not result['live_before_harness_cleanup'], 'guard returned but owned inert descendant is still alive'
        expected = 0 if a.mode in ('exit0', 'leader_exit') else (7 if a.mode == 'exit7' else (124 if a.trigger == 'timeout' else 125))
        if a.trigger == 'wrapperkill':
            expected = -signal.SIGKILL
        assert result['outer_returncode'] == expected, result
        if a.implementation == 'repair':
            pids = json.loads((out / 'guard/pid.json').read_text())
            leader = json.loads((out / 'leader.json').read_text())
            assert pids['server_pid'] == leader['pid'] and pids['guard_pid'] == leader['ppid'], 'guard must own actual workload directly'
            if a.trigger == 'monitor_interrupt':
                status = json.loads((out / 'guard/exit.json').read_text())
                assert 'KeyboardInterrupt' in status['error'], status
            elif a.trigger not in ('guardkill', 'guardfreeze'):
                status = json.loads((out / 'guard/result.json').read_text())
                expected_reason = {'memory': 'host_headroom', 'timeout': 'time_limit', 'stop': 'requested_stop',
                                   'term': 'requested_stop', 'wrapperkill': 'requested_stop', 'wrapperterm': 'requested_stop',
                                   'wrapperint': 'requested_stop', 'exit': 'exit',
                                   'monitor_error': "monitor_error:RuntimeError('injected-monitor-error')"}[a.trigger]
                assert status['reason'] == expected_reason, status
        result['passed'] = True
    except BaseException:
        result['passed'] = False
        result['failure'] = traceback.format_exc()
        raise
    finally:
        if p is not None and p.poll() is None:
            p.kill(); p.wait(timeout=3)
        if not identities:
            identities = [json.loads(f.read_text()) for f in out.glob('*.json') if f.stem in ('leader', 'descendant')]
        result['harness_cleanup_required'] = [v for v in identities if live(v)]
        for v in identities:
            if live(v):
                fd = os.pidfd_open(v['pid'])
                try:
                    if live(v):
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                finally:
                    os.close(fd)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            for v in identities:
                try:
                    os.waitpid(v['pid'], os.WNOHANG)
                except ChildProcessError:
                    pass
            if all(identity(v['pid']) is None for v in identities):
                break
            time.sleep(.01)
        custody_path = out / 'guard/custody-pids.json'
        if custody_path.exists():
            custody_ids = json.loads(custody_path.read_text())
            for key in ('custodian_pid', 'guard_pid'):
                try:
                    os.waitpid(custody_ids[key], os.WNOHANG)
                except ChildProcessError:
                    pass
            result['supervisors_after_reap'] = {key: identity(custody_ids[key]) for key in ('custodian_pid', 'guard_pid')}
        result['after_harness_cleanup'] = [identity(v['pid']) for v in identities]
        result['cleanup_verified_gone'] = all(v is None for v in result['after_harness_cleanup'])
        with (out / 'receipt.json').open('x') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
