"""CLI integration with explicit inert command / healthy sample replacements.

Real guard source executes. Only unavailable memory samples and the prohibited
GPU workload are replaced, never process creation, signals, waits or receipts.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from test_preflight import NoGPU, GUARD, CANDIDATE, SHA
import guarded_deferred as runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--timing-overlay', action='store_true')
    a = ap.parse_args()
    candidate = ROOT.parent / 'mixedk-three-stage-timing-coverage/package' if a.timing_overlay else CANDIDATE
    out = ROOT / 'evidence' / a.label
    out.mkdir(parents=True, exist_ok=False)
    sys.meta_path.insert(0, NoGPU())
    spec_original = importlib.util.spec_from_file_location

    def spec_for(name, path, *args, **kwargs):
        spec = spec_original(name, path, *args, **kwargs)
        if Path(path).resolve() == GUARD.resolve():
            execute = spec.loader.exec_module
            def injected(module):
                execute(module)
                module.memory_sample = lambda: dict(available_gib=120, free_gib=120,
                    cgroup_headroom_gib=120, psi_full10=0, oom_kill=0, host_oom_kill=0)
            spec.loader.exec_module = injected
        return spec

    importlib.util.spec_from_file_location = spec_for
    actual_run = runner.run_guarded
    observed = {}

    def inert_run(guard, command, folder, **kwargs):
        assert command[:3] == [sys.executable, '-B', str(candidate / 'deferred_gpu.py')], command
        assert not any('bounded_deferred' in s for s in command)
        observed['production_command_built_but_NOT_executed'] = command
        observed['replacement_command'] = [sys.executable, '-B', str(ROOT / 'process_fixture.py')]
        kwargs['env'] = dict(kwargs['env'], FIXTURE_OUT=str(out), FIXTURE_MODE='exit7')
        kwargs['sample'] = guard.memory_sample
        return actual_run(guard, observed['replacement_command'], folder, **kwargs)

    runner.run_guarded = inert_run
    os.environ.update(EXL3_TRIAL_MODEL_STOPPED='YES', EXL3_THREE_STAGE_AUTHORIZED='YES', MAX_JOBS='1')
    sys.argv = [str(ROOT / 'guarded_deferred.py'), '--guard', str(GUARD), '--guard-sha256', SHA,
                '--candidate', str(candidate), '--deferred-sha256',
                '00500630f3f875610d9e9f70d0a80d8f4237f2cf42cb626f6152f08c3d7ad83e' if a.timing_overlay else '6eedae4b8c4afe86f340d58b63a4a9178d3c76cb7bee36cf4e02cb55eaa71b31',
                '--receipt', str(out / 'run'),
                '--owner-lock', str(out / 'owner.lock'), '--seconds', '2',
                '--sole-gpu-owner', '--model-stopped', '--headroom-verified', '--',
                '--mode', 'numeric', '--cache', str(out / 'never-built-cache'),
                '--output', str(out / 'never-gpu-output')]
    code = 125
    try:
        code = runner.main()
    except SystemExit as exc:
        observed['system_exit'] = str(exc)
        code = exc.code
    finally:
        observed['returncode'] = code
        observed['passed'] = code == 7 and (out / 'run/guard/custody.json').exists()
        (out / 'receipt.json').write_text(json.dumps(observed, indent=2))
        print(json.dumps(observed, indent=2))
    assert observed['passed'], 'valid CLI must directly guard workload and propagate exit7'


if __name__ == '__main__':
    main()
