"""Seal this overlay's hashes and verify source immutability; no remote actions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
R = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--suite', required=True)
    a = ap.parse_args()
    assert a.label.replace('-', '').replace('_', '').isalnum()
    target = ROOT / ('manifest-' + a.label + '.json')
    assert not target.exists(), 'Never overwrite a manifest label'
    before = {k.replace('\\', '/'): v for k, v in json.loads((ROOT / 'inputs-before.json').read_text()).items()}
    after = {p.relative_to(R).as_posix(): sha(p)
             for name in ('mixedk-three-stage-candidate', 'guard-host-oom-candidate')
             for p in (R / name).rglob('*') if p.is_file() and '.git' not in p.parts}
    preservation = dict(before_count=len(before), after_count=len(after),
                        changed=[k for k in before if k in after and before[k] != after[k]],
                        removed=sorted(before.keys() - after.keys()),
                        added=sorted(after.keys() - before.keys()))
    summary_path = ROOT / ('suite-' + a.suite) / 'summary.json'
    summary = json.loads(summary_path.read_text())
    files = sorted(p for p in ROOT.rglob('*') if p.is_file())
    outputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    timing = R / 'mixedk-three-stage-timing-coverage/package'
    binding = {p.relative_to(R).as_posix(): sha(p) for p in
               [timing / n for n in ('deferred_gpu.py', 'build_trial.py', 'gpu_reference.py')]}
    passed = (not any(preservation[k] for k in ('changed', 'removed', 'added')) and
              summary['case_count'] == summary['passed'] and
              summary['overlay_hashes']['guarded_deferred.py'] == sha(ROOT / 'guarded_deferred.py'))
    manifest = dict(status='LOCAL_CPU_PROCESS_PROOF_ONLY' if passed else 'FAILED_VERIFICATION',
                    preservation=preservation, original_inputs=before,
                    timing_package_binding=binding, output_hashes=outputs,
                    summary_path=summary_path.relative_to(ROOT).as_posix(),
                    suite_case_count=summary['case_count'], suite_passed=summary['passed'],
                    suite_expected_red=summary['expected_red_count'],
                    runner_sha256=sha(ROOT / 'guarded_deferred.py'),
                    input_roots='Original candidate and guard roots; .git internals excluded',
                    manifest_excludes='This manifest itself; later files require a new seal',
                    passed=passed)
    with target.open('x') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(manifest=str(target), manifest_sha256=sha(target), passed=passed,
                          output_file_count=len(outputs), runner_sha256=manifest['runner_sha256'],
                          preservation=preservation, suite_case_count=summary['case_count'],
                          suite_passed=summary['passed'], suite_expected_red=summary['expected_red_count']), indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
