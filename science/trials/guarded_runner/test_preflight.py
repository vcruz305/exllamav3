"""CPU-only entrypoint refusal proof; never permits Torch import."""
import argparse
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GUARD = ROOT.parent / 'guard-host-oom-candidate/guard_uma.py'
CANDIDATE = ROOT.parent / 'mixedk-three-stage-candidate'
SHA = 'eef5706b327cf20007dd49087b6d70b97ec178fcdbde616374188a0bc916d24d'


class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ('torch', 'exllamav3'):
            raise RuntimeError('FORBIDDEN_GPU_IMPORT:' + fullname)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--case', choices=('authorization', 'platform', 'hash', 'owner', 'headroom'), default='authorization')
    ap.add_argument('--worker', action='store_true')
    a = ap.parse_args()
    out = ROOT / 'evidence' / a.label
    if a.worker:
        sys.meta_path.insert(0, NoGPU())
        if a.case == 'platform':
            # Explicit injected platform refusal; real Windows run also provided.
            sys.platform = 'win32'
        if a.case != 'authorization':
            os.environ.update(EXL3_TRIAL_MODEL_STOPPED='YES', EXL3_THREE_STAGE_AUTHORIZED='YES')
        sys.argv = [str(ROOT / 'guarded_deferred.py'), '--guard', str(GUARD),
                    '--guard-sha256', '0' * 64 if a.case == 'hash' else SHA,
                    '--candidate', str(CANDIDATE), '--deferred-sha256', '6eedae4b8c4afe86f340d58b63a4a9178d3c76cb7bee36cf4e02cb55eaa71b31',
                    '--receipt', str(out / 'run'),
                    '--owner-lock', str(out / 'owner.lock'), '--seconds', '2',
                    '--model-stopped', '--headroom-verified']
        if a.case != 'owner':
            sys.argv.append('--sole-gpu-owner')
        sys.argv += ['--', '--mode', 'numeric', '--output', str(out / 'never-created')]
        runpy.run_path(str(ROOT / 'guarded_deferred.py'), run_name='__main__')
        return 0
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop('EXL3_TRIAL_MODEL_STOPPED', None)
    env.pop('EXL3_THREE_STAGE_AUTHORIZED', None)
    p = subprocess.run([sys.executable, '-B', __file__, *sys.argv[1:], '--worker'],
                       env=env, capture_output=True, text=True, timeout=5)
    (out / 'stdout.txt').write_text(p.stdout)
    (out / 'stderr.txt').write_text(p.stderr)
    expected = 'REFUSE: ' + a.case
    result = dict(returncode=p.returncode, expected=expected, stdout=p.stdout, stderr=p.stderr,
                  passed=p.returncode != 0 and expected in p.stderr and 'FORBIDDEN_GPU_IMPORT' not in p.stderr)
    (out / 'receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert result['passed'], 'entrypoint must refuse before any GPU import: ' + a.case
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
