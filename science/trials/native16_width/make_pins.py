#!/usr/bin/env python3
"""Generate integration-pins.json and input-pins.json for this package.

Run from the package directory with any Python 3:

    python3 make_pins.py

* `integration-pins.json` pins every artifact the trial CLI loads (the runtime
  closure) plus every sealed source copied into this tree, each with the
  upstream origin path and the origin's hash at pin time.
* `input-pins.json` is the preservation record: every upstream file this
  package read or copied, with its hash, so the verifier can prove the originals
  were not modified by this work.

Writes only inside this package directory.
"""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The upstream root. Overridable so a RELOCATED COPY of this package (used for
# a fresh-mutation RED run) can still pin against the real upstream trees.
MIMO = Path(os.environ.get('MIMO_TUNE_ROOT', HERE.parent)).resolve()
RUNTIME = [
    'linux_adapter.py', 'recover_linux.py', 'production-template.json', 'runtime-pins.json',
    'trust.schema.json', 'core/recover.py', 'core/width_controller.py',
    'witness/inert_trial_io.py', 'witness/guard_uma.py', 'witness/retained_launcher.py',
    'witness/retained_server.py', 'witness/retained_launch.json', 'witness/failed_launch.json',
    'witness/source-provenance.json', 'witness/supplementary-provenance.json',
    'trial_contract.py', 'trial_io.py', 'trial_preflight.py', 'trial_readiness.py',
    'device_contract.py', 'width_trial.py', 'inert_width_stack.py', 'device_docs.py',
    'device-docs/config.json', 'device-docs/trial.json', 'device-docs/receipt.json',
    'device-docs/authorization.json',
    'width/device_preflight.py', 'width/candidate/width_diag.py',
    'width/candidate/width_contract.py', 'width/candidate/width_config.json',
    'width/candidate/width_expected.json', 'width/baseline/width_diag.py',
    'width/baseline/width_controller.py', 'width/baseline/recover.py',
    'width/baseline/failed-server.log', 'width/retained/retained_launcher.py',
    'width/retained/retained_launch.json', 'width/retained/failed_launch.json',
]
TESTS = ['fixture_width_trial.py', 'test_trial_contract.py', 'test_device_contract.py',
         'test_trial_linux.py', 'test_device_docs.py']
ORIGINS = {
    'linux_adapter.py': 'round8-width-recovery-adapter/linux_adapter.py',
    'recover_linux.py': 'round8-width-recovery-adapter/recover_linux.py',
    'production-template.json': 'round8-width-recovery-adapter/production-template.json',
    'runtime-pins.json': 'round8-width-recovery-adapter/runtime-pins.json',
    'trust.schema.json': 'round8-width-recovery-adapter/trust.schema.json',
    'core/recover.py': 'round8-width-recovery-adapter/core/recover.py',
    'core/width_controller.py': 'round8-width-recovery-adapter/core/width_controller.py',
    'witness/guard_uma.py': 'round8-width-recovery-adapter/witness/guard_uma.py',
    'witness/retained_launcher.py': 'round8-width-recovery-adapter/witness/retained_launcher.py',
    'witness/retained_server.py': 'round8-width-recovery-adapter/witness/retained_server.py',
    'witness/retained_launch.json': 'round8-width-recovery-adapter/witness/retained_launch.json',
    'witness/failed_launch.json': 'round8-width-recovery-adapter/witness/failed_launch.json',
    'witness/source-provenance.json': 'round8-width-recovery-adapter/witness/source-provenance.json',
    'witness/supplementary-provenance.json': 'round8-width-recovery-adapter/witness/supplementary-provenance.json',
    'width/device_preflight.py': 'round8-width-harness-repair/deferred_device_preflight.py',
    'width/candidate/width_diag.py': 'round8-width-harness-repair/candidate/width_diag.py',
    'width/candidate/width_contract.py': 'round8-width-harness-repair/candidate/width_contract.py',
    'width/candidate/width_config.json': 'round8-width-harness-repair/candidate/width_config.json',
    'width/candidate/width_expected.json': 'round8-width-harness-repair/candidate/width_expected.json',
    'width/baseline/width_diag.py': 'round8-width-harness-repair/baseline/width_diag.py',
    'width/baseline/width_controller.py': 'round8-width-harness-repair/baseline/width_controller.py',
    'width/baseline/recover.py': 'round8-width-harness-repair/baseline/recover.py',
    'width/baseline/failed-server.log': 'round8-width-harness-repair/baseline/failed-server.log',
    'width/retained/retained_launcher.py': 'round8-width-harness-repair/source/retained_launcher.py',
    'width/retained/retained_launch.json': 'round8-width-harness-repair/source/retained_launch.json',
    'width/retained/failed_launch.json': 'round8-width-harness-repair/source/failed_launch.json',
}
# Every upstream file this package read or composed from, for the preservation
# record. Copied files must stay byte-identical; read-only inputs must not move.
READ_INPUTS = [
    'round8-width-harness-repair/README.md', 'round8-width-harness-repair/AUDIT.md',
    'round8-width-harness-repair/WORKFLOW.md', 'round8-width-harness-repair/verify.py',
    'round8-width-harness-repair/safe_test_runner.py', 'round8-width-harness-repair/run_evidence.py',
    'round8-width-harness-repair/test_diagnostic.py', 'round8-width-harness-repair/test_recovery.py',
    'round8-width-harness-repair/test_interfaces.py',
    'round8-width-harness-repair/candidate/test_width_contract.py',
    'round8-width-harness-repair/baseline/preflight_width.py',
    'round8-width-harness-repair/baseline/stage_width.py',
    'round8-width-harness-repair/baseline/stop_launch.py',
    'round8-width-harness-repair/baseline/closeout_v2.py',
    'round8-width-harness-repair/baseline/width_contract.py',
    'round8-width-harness-repair/baseline/width_config.json',
    'round8-width-harness-repair/baseline/width_expected.json',
    'round8-width-harness-repair/baseline/guard_uma.py',
    'round8-width-harness-repair/baseline/test_width_contract.py',
    'round8-width-harness-repair/source/exllamav3/generator/generator.py',
    'round8-width-harness-repair/source/exllamav3/generator/job.py',
    'round8-width-harness-repair/source/exllamav3/generator/pagetable.py',
    'round8-width-harness-repair/source/exllamav3/model/config.py',
    'round8-width-harness-repair/source/exllamav3/modules/architecture/dflash.py',
    'round8-width-harness-repair/source/provenance.json',
    'round8-width-recovery-adapter/README.md', 'round8-width-recovery-adapter/AUDIT.md',
    'round8-width-recovery-adapter/WORKFLOW.md', 'round8-width-recovery-adapter/OPERATOR.md',
    'round8-width-recovery-adapter/verify.py', 'round8-width-recovery-adapter/run_tests.py',
    'round8-width-recovery-adapter/test_adapter.py',
    'round8-width-recovery-adapter/test_integration.py',
    'round8-width-recovery-adapter/test_negative.py', 'round8-width-recovery-adapter/test_controls.py',
    'round8-width-recovery-adapter/inert_stack.py', 'round8-width-recovery-adapter/inert_http.py',
    'round8-width-recovery-adapter/artifact-manifest.json',
    'round8-width-recovery-adapter/verification/final01/summary.json',
    'round8-width-parent-audit/REVIEW.md',
    'round8-width-parent-audit/REVIEW-adapter-deleg-29c4bfee.md',
    'round8-cost-width/mirror/events.jsonl', 'round8-cost-width/mirror/frozen.json',
    'round8-cost-width/mirror/cost-before.json',
    'dflash-native-width-study/receipt.json',
    'round9-three-stage-gpu/readiness.py', 'parent_verify_round9b_live.py',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    artifacts = {}
    missing = []
    for name in RUNTIME + TESTS:
        path = HERE / name
        if not path.is_file():
            missing.append(name)
            continue
        record = {'sha256': sha(path), 'bytes': path.stat().st_size}
        origin = ORIGINS.get(name)
        if origin:
            origin_path = MIMO / origin
            if not origin_path.is_file():
                missing.append(origin)
                continue
            record.update({'origin': str(origin_path), 'origin_sha256': sha(origin_path)})
            if sha(origin_path) != record['sha256']:
                raise SystemExit('copied artifact differs from its origin: ' + name)
        else:
            record.update({'origin': name, 'origin_sha256': record['sha256']})
        artifacts[name] = record
    if missing:
        raise SystemExit('missing artifacts: ' + repr(missing))

    pins = {'schema': 1, 'source_pin': 'ca4a880e8918e1985fd25e06c6aff561666d3f14',
            'composed_from': {
                'harness_repair': str(MIMO / 'round8-width-harness-repair'),
                'recovery_adapter': str(MIMO / 'round8-width-recovery-adapter'),
                'native_width_study': str(MIMO / 'dflash-native-width-study'),
                'parent_audit': str(MIMO / 'round8-width-parent-audit'),
                'read_only_context': [str(MIMO / 'round9-three-stage-gpu'),
                                      str(MIMO / 'round9b-three-stage-timing'),
                                      str(MIMO / 'round8-cost-width'),
                                      str(MIMO / 'mixedk-three-stage-trial-integration')]},
            'artifacts': artifacts}
    (HERE / 'integration-pins.json').write_text(json.dumps(pins, indent=2) + '\n')

    inputs = {}
    for relative in READ_INPUTS:
        path = MIMO / relative
        if not path.is_file():
            continue
        inputs[relative] = sha(path)
    for name, record in sorted(artifacts.items()):
        origin = record['origin']
        if origin.startswith(str(MIMO)):
            inputs[str(Path(origin).relative_to(MIMO))] = record['origin_sha256']
    (HERE / 'input-pins.json').write_text(json.dumps(
        {'schema': 1, 'note': 'Preservation record: these upstream files must stay byte-identical.',
         'inputs': dict(sorted(inputs.items()))}, indent=2) + '\n')
    print('artifacts', len(artifacts), 'inputs', len(inputs))


if __name__ == '__main__':
    main()
