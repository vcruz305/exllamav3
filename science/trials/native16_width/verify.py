#!/usr/bin/env python3
"""LOCAL end-to-end verifier for the native16 width-trial integration package.

Run it once per fresh label (Linux / WSL is required for the real inert process
and ephemeral-loopback steps):

    /usr/bin/python3 verify.py --label <fresh-label>

It writes only `verification/<label>/` inside this package, prints a
machine-readable PASS/FAIL summary, and refuses to claim a pass unless:

  * every composed artifact still matches `integration-pins.json`, including the
    upstream origin it was copied from (byte identity);
  * every upstream file this package read or composed from still matches
    `input-pins.json` (originals untouched);
  * the sealed runtime pins (adapter hash, CORE_HASHES, witness pins, source
    provenance) still verify;
  * the candidate suite is green (contract + source-executed device contract +
    the real two-stage Linux integration);
  * the SEALED harness suite (37) and the SEALED adapter suite (36) are green
    when run in place against the composed copies, and neither original tree
    changed;
  * a FRESH mutation of this package's own consistency seam reproduces the
    intended failing test;
  * the CLI fails closed without authorization, and refuses tampered artifacts;
  * the inert integration left no process behind.

NO GPU, no network service, no SSH, no torch/exllamav3/model import, no
deployment. The device gates this verifier cannot cover are listed in GAPS.md
and are repeated in the summary under `limitations`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MIMO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

HARNESS = MIMO / 'round8-width-harness-repair'
ADAPTER = MIMO / 'round8-width-recovery-adapter'
CANDIDATE_SUITE = ['test_trial_contract', 'test_device_contract', 'test_device_docs',
                   'test_trial_linux']
SEALED_RED_TEST = 'test_trial_linux.WidthTrialIntegration.test_restore_refused_when_recorded_outcome_contradicts_the_probe'
MUTATION_BODY = '    def require_trial_outcome_consistent(self, verdict):'
MUTATION_SOURCE = ('    def require_trial_outcome_consistent(self, verdict):\n'
                   '        # DELIBERATE UNSAFE MUTATION FOR RED ONLY: the probe/authorization\n'
                   '        # consistency seam is removed, so this build must accept a\n'
                   '        # restore the unmutated package refuses.\n'
                   '        return verdict')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def tree_hashes(root, skip=('.git',)):
    """Every file under `root`, as {relative path: sha256}."""
    records = {}
    for path in sorted(Path(root).rglob('*')):
        if not path.is_file() or any(part in skip for part in path.parts):
            continue
        records[str(path.relative_to(root))] = sha(path)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.label), 'unsafe verification label')
    require(sys.platform == 'linux', 'verification requires LOCAL Linux/WSL inert process support')
    out = HERE / 'verification' / args.label
    out.mkdir(parents=True, exist_ok=False)
    runs = []
    summary = {}

    def run(label, command, cwd, timeout=1200, expect=None, env=None):
        environ = dict(os.environ)
        environ['PYTHONDONTWRITEBYTECODE'] = '1'
        environ.pop('ROUND8_WIDTH', None)
        environ.pop('ROUND8_WIDTH_OUT', None)
        if env:
            environ.update(env)
        proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True,
                              timeout=timeout, env=environ)
        (out / (label + '.stdout.txt')).write_text(proc.stdout)
        (out / (label + '.stderr.txt')).write_text(proc.stderr)
        match = re.search(r'Ran (\d+) test', proc.stderr) or re.search(r'Ran (\d+) test', proc.stdout)
        record = {'label': label, 'command': command, 'returncode': proc.returncode,
                  'tests': int(match.group(1)) if match else None}
        runs.append(record)
        print(label, 'exit', proc.returncode, 'tests', record['tests'], flush=True)
        if expect is not None:
            require(proc.returncode == expect,
                    label + ' exited ' + str(proc.returncode) + ', expected ' + str(expect))
        return proc

    # ------------------------------------------------------------------ pins
    inputs = json.loads((HERE / 'input-pins.json').read_text())
    require(inputs['schema'] == 1 and inputs['inputs'], 'input-pins.json incomplete')
    originals = {name: inputs['inputs'][name] for name in sorted(inputs['inputs'])}
    for relative, expected in originals.items():
        path = MIMO / relative
        require(path.is_file(), 'upstream input missing: ' + relative)
        require(sha(path) == expected, 'upstream input changed: ' + relative)

    pins = json.loads((HERE / 'integration-pins.json').read_text())
    require(pins['schema'] == 1, 'integration-pins schema mismatch')
    for name, record in sorted(pins['artifacts'].items()):
        path = HERE / name
        require(path.is_file(), 'artifact missing: ' + name)
        require(sha(path) == record['sha256'], 'artifact hash mismatch: ' + name)
        origin = Path(record['origin'])
        require(origin.is_file(), 'artifact origin missing: ' + str(origin))
        require(sha(origin) == record['origin_sha256'], 'artifact origin changed: ' + str(origin))
        require(sha(path) == sha(origin), 'artifact differs from its origin: ' + name)
    adapter_pin = json.loads((HERE / 'runtime-pins.json').read_text())['linux_adapter.py']
    require(adapter_pin == sha(HERE / 'linux_adapter.py'), 'sealed adapter load pin mismatch')
    import linux_adapter as a
    for name, expected in sorted(a.CORE_HASHES.items()):
        require(sha(HERE / 'core' / name) == expected, 'sealed core pin mismatch: ' + name)
        require((HERE / 'core' / name).read_bytes() == (ADAPTER / 'core' / name).read_bytes(),
                'sealed core bytes changed: ' + name)
    for role, (path, digest) in sorted(a.PRODUCTION_PINS.items()):
        witness = {'guard': 'guard_uma.py', 'launcher': 'retained_launcher.py',
                   'server': 'retained_server.py'}.get(role)
        if witness:
            require(sha(HERE / 'witness' / witness) == digest, 'witness pin mismatch: ' + role)
    source_witness = json.loads((HERE / 'witness/source-provenance.json').read_text())
    require(re.fullmatch(r'[0-9a-f]{40}', source_witness['pin']) and source_witness['pin'] == a.SOURCE_PIN,
            'source provenance pin mismatch')
    for name, row in sorted(source_witness['files'].items()):
        require(sha(HARNESS / 'source' / name) == row['blob_sha256'],
                'pinned source witness differs: ' + name)
    src_text = (HERE / 'linux_adapter.py').read_text()
    require('st_mode & 0o022' in src_text, 'sealed permission predicate changed')
    summary['pins'] = {'artifacts_verified': len(pins['artifacts']),
                       'upstream_originals_verified': len(originals),
                       'sealed_core_pins': len(a.CORE_HASHES),
                       'source_witnesses': len(source_witness['files']),
                       'source_pin': a.SOURCE_PIN}

    # ---------------------------------------------- sealed suites, in place
    harness_before = tree_hashes(HARNESS)
    adapter_before = tree_hashes(ADAPTER)
    sealed = run('sealed-harness-37', ['/usr/bin/python3', '-B', 'safe_test_runner.py'], HARNESS,
                 env={'DIAG_SOURCE': str(HERE / 'width/candidate/width_diag.py')}, expect=0)
    require(sealed.stderr.strip().endswith('OK'), 'sealed harness suite is not green')
    sealed_adapter = run('sealed-adapter-36',
                         ['/usr/bin/python3', '-B', '-m', 'unittest', 'test_adapter',
                          'test_integration', 'test_negative', 'test_controls'], ADAPTER, expect=0)
    require(sealed_adapter.stderr.strip().endswith('OK'), 'sealed adapter suite is not green')
    require(tree_hashes(HARNESS) == harness_before, 'the harness-repair tree changed under the run')
    require(tree_hashes(ADAPTER) == adapter_before, 'the recovery-adapter tree changed under the run')
    summary['sealed_suites'] = {'harness_repair_tests': runs[-2]['tests'],
                               'recovery_adapter_tests': runs[-1]['tests'],
                               'harness_files_rehashed_unchanged': len(harness_before),
                               'adapter_files_rehashed_unchanged': len(adapter_before)}

    # ------------------------------------------------------- candidate suite
    candidate = run('candidate-suite', ['/usr/bin/python3', '-B', '-m', 'unittest', '-v']
                    + CANDIDATE_SUITE, HERE, expect=0)
    require(candidate.stderr.strip().endswith('OK'), 'candidate suite is not green')
    cleanup = [json.loads(line[len('INERT_CLEANUP '):])
               for line in candidate.stdout.splitlines() if line.startswith('INERT_CLEANUP ')]
    leaked = [entry for group in cleanup for entry in group if entry['left']]
    require(cleanup, 'no inert cleanup evidence was produced')
    require(not leaked, 'inert integration leaked processes: ' + repr(leaked))
    summary['candidate_suite'] = {
        'tests': runs[-1]['tests'], 'modules': CANDIDATE_SUITE,
        'inert_fixture_runs': len([entry for group in cleanup for entry in group]),
        'inert_pids_still_alive': len(leaked),
        'inert_guard_results': sorted({entry['result']['reason'] for group in cleanup
                                      for entry in group if entry['result']})}
    summary['preservation_controls'] = {
        'originals_rehashed_after_run': len(originals),
        'harness_tree_files': len(harness_before), 'adapter_tree_files': len(adapter_before)}

    # ------------------------------------ artifact tamper controls (in place)
    sealed_control = out / 'mutation-pinned'
    shutil.copytree(HERE, sealed_control, ignore=shutil.ignore_patterns('verification', 'fixture-*',
                                                                       'verification-tmp-*',
                                                                       '__pycache__'))
    pin_args = ['/usr/bin/python3', '-I', '-B', 'width_trial.py', '--stage', 'release',
                '--authorize-width-trial']
    (sealed_control / 'linux_adapter.py').write_bytes(
        (sealed_control / 'linux_adapter.py').read_bytes() + b'\n')
    tampered_adapter = run('cli-tampered-adapter', pin_args, sealed_control, expect=1)
    require('pinned artifact hash mismatch: linux_adapter.py' in tampered_adapter.stdout,
            'the CLI accepted a tampered sealed adapter')
    (sealed_control / 'linux_adapter.py').write_bytes((HERE / 'linux_adapter.py').read_bytes())
    (sealed_control / 'runtime-pins.json').write_text(
        (sealed_control / 'runtime-pins.json').read_text().replace('a', 'b', 1))
    tampered_runtime = run('cli-tampered-runtime-pins', pin_args, sealed_control, expect=1)
    require('pinned artifact hash mismatch: runtime-pins.json' in tampered_runtime.stdout,
            'the CLI accepted tampered runtime pins')

    # ------------------------------------------------- fresh mutation, RED
    mutant = out / 'mutation-mutant'
    shutil.copytree(HERE, mutant, ignore=shutil.ignore_patterns('verification', 'fixture-*',
                                                                'verification-tmp-*', '__pycache__'))
    text = (mutant / 'trial_io.py').read_text()
    require(MUTATION_BODY in text, 'mutation target absent from trial_io.py')
    mutated, applied = re.subn(r'(?ms)^' + re.escape(MUTATION_BODY) + r'.*?^        return verdict$',
                               MUTATION_SOURCE.replace('\\', '\\\\'), text)
    require(applied == 1 and mutated != text,
            'the mutation did not replace exactly the probe-consistency seam')
    compile(mutated, 'trial_io.py', 'exec')  # a syntactically broken mutation proves nothing
    (mutant / 'trial_io.py').write_text(mutated)
    # Regenerate the mutant's own pins so the behavioral seam, not the pin gate,
    # is the thing under test here (the pin gate has its own controls above).
    repin = run('mutation-repin', ['/usr/bin/python3', '-B', 'make_pins.py'], mutant, expect=0,
                timeout=120, env={'MIMO_TUNE_ROOT': str(MIMO)})
    require('artifacts' in (mutant / 'integration-pins.json').read_text(),
            'the mutant pins were not regenerated')
    red = run('fresh-mutation-red', ['/usr/bin/python3', '-B', '-m', 'unittest', '-v', SEALED_RED_TEST],
              mutant, expect=1)
    require('FAILED (failures=1)' in red.stderr and SEALED_RED_TEST.split('.')[-1] in red.stderr,
            'the fresh mutation did not reproduce the intended failing test')
    observed = red.stderr.split('AssertionError: ', 1)[-1]
    try:
        parsed = json.loads(observed[observed.index('{'):observed.rindex('}') + 1])
        observed = {key: parsed.get(key) for key in ('stage', 'returncode', 'expected', 'status')}
    except ValueError:
        observed = observed.splitlines()[0][:200]
    summary['fresh_mutation'] = {'test': SEALED_RED_TEST, 'files_mutated': ['trial_io.py'],
                                 'mutation': 'require_trial_outcome_consistent() body removed',
                                 'observed_failure': observed,
                                 'reproof': 'the unmutated package passes the same test in the candidate run'}
    summary['artifact_tamper_controls'] = {'sealed_adapter': 'pinned artifact hash mismatch',
                                           'runtime_pins': 'pinned artifact hash mismatch'}

    # ------------------------------------------------------------ CLI gates
    help_result = run('cli-help', ['/usr/bin/python3', '-I', '-B', 'width_trial.py', '--help'], HERE,
                      expect=0)
    require('--authorize-width-trial' in help_result.stdout, 'CLI help is incomplete')
    unauth = run('cli-no-authorization', ['/usr/bin/python3', '-I', '-B', 'width_trial.py',
                                          '--stage', 'release', '--config', '/dev/null',
                                          '--config-sha256', '0' * 64, '--receipt', '/dev/null',
                                          '--receipt-sha256', '0' * 64, '--authorization', '/dev/null',
                                          '--authorization-sha256', '0' * 64, '--trial', '/dev/null',
                                          '--trial-sha256', '0' * 64, '--protected-root', '/tmp',
                                          '--protected-roots-complete'], HERE, expect=1)
    require('EXL3_WIDTH_TRIAL_AUTHORIZED=YES is required' in unauth.stdout,
            'the CLI did not fail closed without authorization')
    no_switch = run('cli-no-switch', ['/usr/bin/python3', '-I', '-B', 'width_trial.py',
                                     '--stage', 'release'], HERE, expect=1,
                    env={'EXL3_WIDTH_TRIAL_AUTHORIZED': 'YES'})
    require('--authorize-width-trial is required' in no_switch.stdout,
            'the CLI did not require its explicit switch')
    bad_stage = run('cli-bad-stage', ['/usr/bin/python3', '-I', '-B', 'width_trial.py',
                                     '--stage', 'deploy'], HERE, expect=2)
    require('invalid choice' in bad_stage.stderr, 'the CLI accepted an unknown stage')
    summary['cli_gates'] = {'help': 0, 'no_authorization_env': 1, 'no_authorization_switch': 1,
                            'unknown_stage_argparse': 2}

    # -------------------------------------------------- final preservation
    for relative, expected in sorted(originals.items()):
        require(sha(MIMO / relative) == expected, 'upstream input changed under the run: ' + relative)
    for name, record in sorted(pins['artifacts'].items()):
        require(sha(HERE / name) == record['sha256'], 'artifact modified by tests: ' + name)
    require(tree_hashes(HARNESS) == harness_before, 'the harness-repair tree changed overall')
    require(tree_hashes(ADAPTER) == adapter_before, 'the recovery-adapter tree changed overall')

    summary.update({
        'result': 'PASS',
        'scope': ('LOCAL only: real Linux inert guard/server processes, ephemeral loopback HTTP '
                  '(never 8096), /proc + pidfd + flock identity, filesystem boundaries. '
                  'NO GPU, model, torch, exllamav3, network service, SSH or deployment.'),
        'python': sys.version.split()[0],
        'label': args.label,
        'runs': runs,
        'output_sha256': {str(path.relative_to(out)): sha(path)
                          for path in sorted(out.rglob('*'))
                          if path.is_file() and 'mutation-mutant' not in str(path)},
        'limitations': [
            'The real /workspace/mimo-tune runtime root, the 8096 serve, CUDA and the EXL3 model are untouched.',
            'Inert substitutions: memory() headroom sample and guard_parent_pid(); both documented in INTEGRATION.md.',
            'The DrvFS mode predicate is rebound inside the inert witness process only (production rejects 0777).',
            'No probe event stream from a real device run exists yet: the fixture stream is synthesized to the sealed emit() contract.',
            'No performance, acceptance-rate or 60 TPS claim of any kind.',
        ]})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'output_sha256'}, indent=2))
    print('SUMMARY', str(out / 'summary.json'))
    return 0


if __name__ == '__main__':
    try:
        CODE = main()
    except BaseException as exc:  # noqa: BLE001 - one refusal, one printed record
        print(json.dumps({'result': 'FAIL', 'error': repr(exc)}, indent=2))
        raise SystemExit(1) from exc
    raise SystemExit(CODE)
