"""Re-run all CPU/local-Linux proofs; each run uses a NEW label and raw logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    a = ap.parse_args()
    assert sys.platform == 'linux', 'Run in LOCAL WSL Linux, not an emulated process API'
    assert a.label.replace('-', '').replace('_', '').isalnum()
    suite = ROOT / ('suite-' + a.label)
    suite.mkdir(exist_ok=False)
    cases = [('original-nested-red', 'test_processes.py', ['--implementation', 'old', '--trigger', 'memory'], 1)]
    for trigger, mode in [('memory', 'wait'), ('timeout', 'wait'), ('stop', 'wait'), ('term', 'wait'),
                          ('monitor_error', 'wait'), ('exit', 'exit0'), ('exit', 'exit7'),
                          ('memory', 'descendant'), ('timeout', 'descendant'), ('stop', 'descendant'),
                          ('term', 'descendant'), ('monitor_error', 'descendant'),
                          ('exit', 'leader_exit'), ('memory', 'escape'),
                          ('wrapperkill', 'descendant'), ('wrapperkill', 'escape'), ('guardkill', 'escape'),
                          ('wrapperterm', 'descendant'), ('wrapperint', 'descendant'), ('monitor_interrupt', 'descendant'),
                          ('guardfreeze', 'descendant')]:
        cases.append((trigger + '-' + mode, 'test_processes.py', ['--trigger', trigger, '--mode', mode], 0))
    for case in ('authorization', 'platform', 'owner', 'hash', 'headroom'):
        cases.append(('preflight-' + case, 'test_preflight.py', ['--case', case], 0))
    for case in ('admitted','receipt_protected','lock_protected','cache_protected','output_protected',
                 'alias_receipt','missing_inventory','reused_cache','receipt_parent','old_pin',
                 'changed_build','changed_deferred','changed_gpu','changed_scalar','changed_manifest'):
        cases.append(('integration-'+case,'test_integration.py',['--case',case],0))
    results = []
    for name, script, extra, expected in cases:
        cmd = [sys.executable, '-B', str(ROOT / script), '--label', a.label + '-' + name, *extra]
        r = subprocess.run(cmd, capture_output=True, timeout=20)
        (suite / (name + '.stdout.txt')).write_bytes(r.stdout)
        (suite / (name + '.stderr.txt')).write_bytes(r.stderr)
        result = dict(name=name, command=cmd, returncode=r.returncode, expected=expected,
                      passed=r.returncode == expected)
        receipt = ROOT / 'evidence' / (a.label + '-' + name) / 'receipt.json'
        if receipt.exists():
            detail = json.loads(receipt.read_text())
            if name == 'original-nested-red':
                result['passed'] &= bool(detail.get('live_before_harness_cleanup')) and detail.get('cleanup_verified_gone', False)
            elif script == 'test_processes.py':
                result['passed'] &= detail.get('passed', False) and not detail.get('harness_cleanup_required', ['missing'])
                result['passed'] &= detail.get('cleanup_verified_gone', False)
                result['passed'] &= all(v is None for v in detail.get('supervisors_after_reap', {}).values())
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = dict(python=sys.version, platform=sys.platform, cases=results,
                   case_count=len(results), passed=sum(x['passed'] for x in results),
                   expected_red_count=sum(x['expected'] != 0 for x in results),
                   overlay_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')})
    (suite / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases', 'overlay_hashes')}, indent=2))
    return 0 if all(x['passed'] for x in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
