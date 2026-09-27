"""Standalone orchestration overlay. The unchanged guard owns the actual workload."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import select
from pathlib import Path
import signal
import time
import traceback


def child_ids():
    # This runner is single-threaded and owns no unrelated children.
    return [int(p) for p in Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()]


def starttime(pid):
    try:
        return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError:
        return None


def cleanup_children():
    """Drain this dedicated subreaper's children, never global names/PGIDs.

    Killing an adopted parent reparents its children here; repeat until empty.
    pidfd plus starttime pins each target; waitpid reaps only our exact children.
    """
    actions = []
    deadline = time.monotonic() + 3
    while child_ids() and time.monotonic() < deadline:
        for pid in child_ids():
            born = starttime(pid)
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                if born is not None and starttime(pid) == born:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                    actions.append(dict(pid=pid, starttime=born, signal='SIGKILL'))
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        time.sleep(.01)
    return dict(actions=actions, remaining_children=child_ids())


def enable_subreaper():
    if child_ids():
        raise RuntimeError('Use a dedicated process: pre-existing children refused')
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')


def exit_status(result):
    if result['reason'] == 'time_limit':
        return 124
    if result['reason'] != 'exit':
        return 125
    rc = result['returncode']
    return rc if rc >= 0 else 128 - rc


def _guard_worker(guard, command, out, *, env, sample=None, interval=.25,
                  seconds=1800, memory_policy='physical'):
    enable_subreaper()
    kwargs = {} if sample is None else {'sample': sample}
    result = None
    error = None
    rc = 125
    try:
        result = guard.supervise(command, out, env=env, interval=interval,
                                 max_seconds=seconds, memory_policy=memory_policy, **kwargs)
        rc = exit_status(result)
    except BaseException:
        error = traceback.format_exc()
        Path(out, 'failure.txt').write_text(error)
    finally:
        cleanup = cleanup_children()
        if cleanup['remaining_children']:
            rc = 125
        Path(out, 'exit.json').write_text(json.dumps(
            dict(exit_code=rc, guard_result=result, error=error, cleanup=cleanup), indent=2))
    return rc


def run_guarded(guard, command, out, *, env, sample=None, interval=.25,
                seconds=1800, memory_policy='physical'):
    """Dedicated CLI -> custodian -> exact guard -> direct workload.

    The custodian is not a timeout wrapper around the workload. It handles death
    of the CLI/guard and reaps adopted descendants; guard enforces the walltime
    and memory policy. Only a stalled/dead guard needs custody escalation.
    """
    if child_ids():
        raise RuntimeError('Dedicated process required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    parent_fd = os.pidfd_open(os.getpid())
    stopped = []

    def stop(sig, frame):
        stopped.append(sig)
        (out / 'STOP').touch()

    old = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    custody = os.fork()
    if custody == 0:
        rc = 125
        reason = 'guard_finished'
        guard_pid = None
        guard_status = None
        failure = None
        try:
            enable_subreaper()
            guard_pid = os.fork()
            if guard_pid == 0:
                os.close(parent_fd)
                # Guard-worker death on custodian death bounds the direct child.
                parent = os.getppid()
                if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
                    os._exit(125)
                with (out / 'guard-worker.log').open('x') as log:
                    os.dup2(log.fileno(), 1)
                    os.dup2(log.fileno(), 2)
                    status = _guard_worker(guard, command, out, env=env, sample=sample,
                                           interval=interval, seconds=seconds,
                                           memory_policy=memory_policy)
                os._exit(status)
            (out / 'custody-pids.json').write_text(json.dumps(
                dict(custodian_pid=os.getpid(), custodian_starttime=starttime(os.getpid()),
                     guard_pid=guard_pid, guard_starttime=starttime(guard_pid))))
            poller = select.poll()
            poller.register(parent_fd, select.POLLIN)
            started = time.monotonic()
            stop_at = None
            while True:
                pid, status = os.waitpid(guard_pid, os.WNOHANG)
                if pid:
                    guard_status = os.waitstatus_to_exitcode(status)
                    rc = guard_status if guard_status >= 0 else 128 - guard_status
                    if guard_status < 0:
                        reason = 'guard_died'
                    break
                if poller.poll(0) and stop_at is None:
                    reason = 'entrypoint_died'
                    stop_at = time.monotonic()
                    (out / 'STOP').touch()
                if stopped and stop_at is None:
                    reason = 'custodian_signalled'
                    stop_at = time.monotonic()
                if time.monotonic() - started > seconds + 3 or (stop_at is not None and time.monotonic() - stop_at > 2):
                    reason += ':guard_unresponsive'
                    break
                time.sleep(.02)
        except BaseException:
            failure = traceback.format_exc()
            (out / 'custody-failure.txt').write_text(failure)
        finally:
            cleanup = cleanup_children()
            if reason != 'guard_finished' or cleanup['remaining_children']:
                rc = 125
            (out / 'custody.json').write_text(json.dumps(dict(
                exit_code=rc, reason=reason, guard_status=guard_status,
                cleanup=cleanup, error=failure), indent=2))
            os.close(parent_fd)
        os._exit(rc)
    os.close(parent_fd)
    try:
        _, status = os.waitpid(custody, 0)
        rc = os.waitstatus_to_exitcode(status)
        return rc if rc >= 0 else 128 - rc
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--guard', type=Path, required=True)
    ap.add_argument('--guard-sha256', required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--deferred-sha256', required=True, choices=('29a870a4e11c55d185066df788402c283a46409ea08601758d721bb3eeeb1500',))
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--owner-lock', type=Path, required=True)
    ap.add_argument('--seconds', type=int, default=1800)
    ap.add_argument('--memory-policy', choices=('physical', 'uma'), default='uma')
    ap.add_argument('--sole-gpu-owner', action='store_true')
    ap.add_argument('--model-stopped', action='store_true')
    ap.add_argument('--headroom-verified', action='store_true')
    ap.add_argument('trial_args', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if os.environ.get('EXL3_TRIAL_MODEL_STOPPED') != 'YES' or os.environ.get('EXL3_THREE_STAGE_AUTHORIZED') != 'YES':
        raise SystemExit('REFUSE: authorization: explicit environment authorization required')
    import sys
    if sys.platform != 'linux':
        raise SystemExit('REFUSE: platform: Linux with procfs, fork and pidfd required')
    if not (a.sole_gpu_owner and a.model_stopped and a.headroom_verified):
        raise SystemExit('REFUSE: owner: sole GPU owner, stopped model and headroom confirmations required')
    expected_guard = 'eef5706b327cf20007dd49087b6d70b97ec178fcdbde616374188a0bc916d24d'
    if a.guard_sha256 != expected_guard or hashlib.sha256(a.guard.read_bytes()).hexdigest() != expected_guard:
        raise SystemExit('REFUSE: hash: guard must match the exact reviewed source')
    spec = importlib.util.spec_from_file_location('pinned_guard', a.guard)
    guard = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(guard)
    try:
        first = guard.memory_sample()
        guard.require_load_headroom(first, a.memory_policy)
    except Exception as exc:
        raise SystemExit('REFUSE: headroom: ' + str(exc))
    if not 1 <= a.seconds <= 1800:
        ap.error('seconds must be 1..1800')
    pinned = {
        'deferred_gpu.py': a.deferred_sha256,
        'build_trial.py': 'a4f10a7fc85596a57bccb9d764f573531ab735844ed23a4d160cfe3540eced0f',
        'gpu_reference.py': '0cde9ed13e945cc4f79a5e4a8366a310c3820643fb44cfbb94fb85a235dd48eb',
        'source-manifest.json': 'cf6a7cdb9392c50e771d651df94c9c13e7bf8cbbc27dbcf0fdd6900801bd6086',
        'scalar_reference.py': 'c503fc9c3ef82f056b553c359e95bc6928c24dfb689fd057081d78c50823c7d5',
    }
    a.candidate = a.candidate.resolve()
    for name, digest in pinned.items():
        if hashlib.sha256((a.candidate / name).read_bytes()).hexdigest() != digest:
            raise SystemExit('REFUSE: hash: candidate differs: ' + name)
    spec = importlib.util.spec_from_file_location('pinned_trial', a.candidate / 'deferred_gpu.py')
    trial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial)  # Defines functions only; no Torch import.
    args = a.trial_args[1:] if a.trial_args[:1] == ['--'] else a.trial_args
    original_argv = sys.argv
    try:
        sys.argv = [str(a.candidate / 'deferred_gpu.py'), *args]
        trial_args = trial.arguments()
    finally:
        sys.argv = original_argv
    if trial_args.plan or not trial_args.output:
        raise SystemExit('REFUSE: trial requires non-plan mode and --output NEW_DIRECTORY')
    for path in (a.receipt, a.owner_lock, trial_args.output, trial_args.cache):
        if path.resolve() == a.candidate or a.candidate in path.resolve().parents:
            raise SystemExit('REFUSE: all writable paths including --cache must be outside sealed candidate')
    spec = importlib.util.spec_from_file_location('pinned_build', a.candidate / 'build_trial.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.require_authorization()  # Preserve original MAX_JOBS/RTLD/platform checks.
    admitted_paths = build.preflight_trial_paths(
        a.candidate, trial_args.cache, trial_args.output,
        [*trial_args.protected_root, a.guard.resolve(), Path(__file__).resolve()],
        inventory_complete=trial_args.protected_roots_complete,
        extra_writable=[a.receipt], shared_lock=a.owner_lock,
    )
    import fcntl
    with a.owner_lock.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('REFUSE: owner: shared owner lock is already held')
        a.receipt.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-B', str(a.candidate / 'deferred_gpu.py'), *args]
        provenance = dict(command=command, guard_path=str(a.guard.resolve()), guard_sha256=expected_guard,
                          candidate_path=str(a.candidate), candidate_hashes=pinned,
                          owner_lock=str(a.owner_lock.resolve()), seconds=a.seconds,
                          memory_policy=a.memory_policy,
                          protected_roots=admitted_paths['protected'],
                          protected_roots_complete=trial_args.protected_roots_complete,
                          confirmations=dict(sole_gpu_owner=True, model_stopped=True, headroom_verified=True),
                          note='Operator assertions are not proof of ownership; guard uses actual memory samples.')
        (a.receipt / 'invocation.json').write_text(json.dumps(provenance, indent=2))
        try:
            # Recheck under the shared owner lock immediately before supervision.
            first = guard.memory_sample()
            guard.require_load_headroom(first, a.memory_policy)
            (a.receipt / 'headroom.json').write_text(json.dumps(first, indent=2))
            return run_guarded(guard, command, a.receipt / 'guard', env=dict(os.environ),
                               seconds=a.seconds, memory_policy=a.memory_policy)
        except BaseException:
            (a.receipt / 'failure.txt').write_text(traceback.format_exc())
            raise



if __name__ == '__main__':
    raise SystemExit(main())
