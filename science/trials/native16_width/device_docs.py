#!/usr/bin/env python3
"""Host-side gate for the FOUR documents a width-trial stage needs.

    /usr/bin/python3 -I -B device_docs.py --write-templates device-docs
    /usr/bin/python3 -I -B device_docs.py --check device-docs-filled

`--write-templates` generates the four templates from the SEALED contract's own
field lists, so a template cannot drift from what the trial CLI validates.

`--check` validates a filled document set WITHOUT touching the device: strict
JSON, no leftover placeholder, the sealed contract checks that need no runtime
(`trial_contract.validate_trial`, `validate_receipt_document`,
`linux_adapter.validate_production`), the pins link, and the four SHA256 values
the CLI will be given. On success it prints the exact stage commands; on any
failure it prints a refusal and exits 1. It never starts, stops or inspects a
process, and it never opens a socket.

What this tool CANNOT check is listed in GAPS.md (G-AUTHORIZATION-DEVICE-GATE):
the authorization's boot_id, uid, lease, time window and monotonic ordering are
validated on the host, inside the CLI, against the live machine.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import linux_adapter as a  # noqa: E402 - sealed pins, stdlib only
import trial_contract as tc  # noqa: E402

PLACEHOLDER = '<fill-me>'
DOCUMENTS = ('config', 'trial', 'receipt', 'authorization')


class Refused(Exception):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest_file(path):
    return sha(Path(path).read_bytes())


def load(path):
    raw = Path(path).read_bytes()
    try:
        return json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Refused('document is not strict UTF-8 JSON: ' + str(path) + ': ' + repr(exc))


def placeholders(value, trail=''):
    """Every key path whose value is still a placeholder."""
    found = []
    if isinstance(value, dict):
        for key in value:
            found += placeholders(value[key], trail + '/' + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += placeholders(item, trail + '/' + str(index))
    elif isinstance(value, str) and value.startswith('<') and value.endswith('>'):
        found.append(trail)
    return found


# --------------------------------------------------------------------- templates
def _config_template():
    """Derived from the sealed witness + pins, so it cannot drift from
    `linux_adapter.validate_production`."""
    retained = json.loads((HERE / 'witness/retained_launch.json').read_bytes())
    retained.pop('outdir')
    command = retained['command']
    identities = [{'role': role, 'path': path, 'sha256': digest}
                  for role, (path, digest) in a.PRODUCTION_PINS.items()]
    identities += [
        {'role': 'interpreter', 'path': '<abs path to the retained server interpreter>',
         'sha256': PLACEHOLDER},
        {'role': 'target_config', 'path': command[command.index('-m') + 1] + '/config.json',
         'sha256': PLACEHOLDER},
        {'role': 'drafter_config', 'path': command[command.index('-dm') + 1] + '/config.json',
         'sha256': PLACEHOLDER},
        {'role': 'source_manifest', 'path': '<abs path to the pinned source manifest>',
         'sha256': PLACEHOLDER}]
    return {
        'schema': 1, 'source_pin': a.SOURCE_PIN,
        'root': '/workspace/mimo-tune', 'operator_lock': 'operator.lock',
        'active': 'france-active-run.txt', 'live_config': 'france-uma.json',
        'server_lock': 'server.lock',
        'guard_argv': ['/usr/bin/python3', a.PRODUCTION_PINS['guard'][0],
                       a.PRODUCTION_ROOT + '/france-uma.json'],
        'launcher_argv': ['/usr/bin/python3', a.PRODUCTION_PINS['launcher'][0]],
        'launcher_env': {'PATH': '/usr/bin:/bin', 'HOME': '/root', 'LC_ALL': 'C.UTF-8',
                         'PYTHONUNBUFFERED': '1'},
        'retained': retained,
        'identities': identities,
        'port': 8096, 'new_run_prefix': 'france-round6-quant-readback-', 'receipt_dir': 'receipts',
        'timeouts': {'stop': 30, 'headroom': 30, 'launch': 120, 'ready': 300, 'http': 10},
    }


def _trial_template():
    return {
        'schema': 1, 'inert_fixture': False, 'source_pin': a.SOURCE_PIN,
        'root': a.PRODUCTION_ROOT,
        'width': {'native_block': 16, 'retained_block': 8,
                  'drafter_config': '<abs path to the installed native16 dflash config.json>',
                  'drafter_config_sha256': tc.WIDTH_CONFIG_SHA256,
                  'retained_drafter_config': '<abs path to the installed native8 dflash config.json>',
                  'retained_drafter_config_sha256': PLACEHOLDER,
                  'probe_dir': 'round8-cost-width/width/ndt1', 'events': 'events.jsonl',
                  'run_prefix': 'france-round8-width16-ndt1-', 'failed_run': None,
                  'failed_pids': {'guard_pid': 0, 'server_pid': 0},
                  'env': {'ROUND8_WIDTH': '1',
                          'ROUND8_WIDTH_OUT': '/workspace/mimo-tune/round8-cost-width/width/ndt1'},
                  'expect': {'input_shape': [1, 16, 4096], 'state_shape': [1, 16, 4096],
                             'q_shape': [1, 16, 64, 128], 'k_shape': [1, 16, 8, 128],
                             'samples_shape': [1, 16], 'proposal_count': 15,
                             'assigned_span': 16, 'max_round': 512}},
        'protected': {'mandatory_roots': ['<abs protected root 1>', '<abs protected root 2>'],
                      'nesting_acknowledged': False},
        'artifacts': {'integration_pins': 'integration-pins.json',
                      'integration_pins_sha256': PLACEHOLDER},
        'scope': 'ONE bounded native16 width trial on the GB10 host; no deployment, no tuning loop',
    }


def _receipt_template():
    return {'schema': 1, 'phase': 'pre_failure', 'captured_at': 0,
            'boot_id': PLACEHOLDER, 'uid': 0,
            'owner': {'run': '<abs path of the LIVE retained run>',
                      'pids': {'guard_pid': 0, 'server_pid': 0,
                               'command': ['<the exact server argv of the live run>']},
                      'processes': [{'role': role, 'pid': 0, 'starttime': PLACEHOLDER,
                                     'args': ['<exact argv>']} for role in ('guard', 'server')]},
            'scopes': {'<guard_pid>': {'uid': 0, 'ppid': 0, 'pgid': 0, 'sid': 0},
                       '<server_pid>': {'uid': 0, 'ppid': 0, 'pgid': 0, 'sid': 0}},
            'launch_sha256': PLACEHOLDER, 'live_config_sha256': PLACEHOLDER}


def _authorization_template():
    return {'schema': 1, 'intent': 'stop_owned_then_launch_retained_once',
            'root': '/workspace/mimo-tune', 'operator_lock': 'operator.lock',
            'ingress_blocked': True, 'uid': 0, 'boot_id': PLACEHOLDER,
            'config_sha256': PLACEHOLDER, 'receipt_sha256': PLACEHOLDER,
            'trial_sha256': PLACEHOLDER, 'issued_at': 0, 'failure_at': 0, 'expires_at': 0,
            'width': {'status': '<passed or failed>', 'phase': 'after_width',
                      'error': '<the observed error text when status is failed>'},
            'stage': '<release or restore>',
            'scope': 'ONE bounded native16 width trial on the GB10 host; no deployment'}


def write_templates(directory):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    builders = {'config': _config_template, 'trial': _trial_template,
                'receipt': _receipt_template, 'authorization': _authorization_template}
    written = []
    for name in DOCUMENTS:
        path = target / (name + '.json')
        if path.exists():
            raise Refused('refusing to overwrite an existing document: ' + str(path))
        path.write_text(json.dumps(builders[name](), indent=2) + '\n')
        written.append(str(path))
    return written


# ------------------------------------------------------------------------- check
def check(directory):
    directory = Path(directory)
    docs = {}
    for name in DOCUMENTS:
        path = directory / (name + '.json')
        if not path.is_file():
            raise Refused('missing document: ' + str(path))
        docs[name] = load(path)
        stale = placeholders(docs[name])
        if stale:
            raise Refused('document ' + name + ' still has placeholders at: ' + ', '.join(stale))

    cfg, trial, receipt, auth = (docs[n] for n in DOCUMENTS)

    # 0. This is the DEVICE gate: an inert fixture document must never get here.
    if trial['inert_fixture'] is not False:
        raise Refused('the device gate requires a production trial document (inert_fixture false)')

    # 1. Sealed contract checks that need no live machine.
    tc.validate_trial(trial, Path(cfg['root']))
    tc.validate_receipt_document(receipt, auth)
    import linux_adapter as a
    a.validate_production(cfg)

    # 2. The pins link: the trial must name THIS package's pins file and hash.
    pins_path = HERE / trial['artifacts']['integration_pins']
    if not pins_path.is_file():
        raise Refused('the trial names a pins file that is not in this package: '
                      + trial['artifacts']['integration_pins'])
    if digest_file(pins_path) != trial['artifacts']['integration_pins_sha256']:
        raise Refused('the trial pins a different ' + pins_path.name)

    # 3. Cross-document hashes: exactly what the CLI will verify.
    hashes = {name: digest_file(directory / (name + '.json')) for name in DOCUMENTS}
    for name in ('config', 'receipt', 'trial'):
        if auth[name + '_sha256'] != hashes[name]:
            raise Refused('authorization.' + name + '_sha256 does not match the '
                          + name + ' document bytes')
    if auth['root'] != cfg['root'] or auth['root'] != trial['root']:
        raise Refused('root mismatch between authorization/config/trial')
    if auth['operator_lock'] != cfg['operator_lock']:
        raise Refused('operator_lock mismatch between authorization and config')
    if receipt['owner']['run'] != cfg['root'] + '/' + tc.RUNS_DIR + '/' + \
            Path(receipt['owner']['run']).name:
        raise Refused('receipt owner run is not a run of the owned root')
    if auth['stage'] == 'release' and auth['width']['phase'] != 'before_width':
        raise Refused('a release authorization must carry the before_width phase')
    if auth['stage'] == 'restore' and auth['width']['phase'] != 'after_width':
        raise Refused('a restore authorization must carry the after_width phase')
    if auth['width']['status'] == 'failed' and not auth['width']['error']:
        raise Refused('a failed width outcome requires the observed error text')

    command = ['EXL3_WIDTH_TRIAL_AUTHORIZED=YES', '/usr/bin/python3', '-I', '-B', 'width_trial.py',
               '--stage', '<' + auth['stage'] + '|preflight>', '--authorize-width-trial']
    for name in DOCUMENTS:
        command += ['--' + name, str(directory / (name + '.json')), '--' + name + '-sha256', hashes[name]]
    for root in trial['protected']['mandatory_roots']:
        command += ['--protected-root', root]
    command += ['--protected-roots-complete']
    return {'documents': hashes, 'stage': auth['stage'], 'command': command,
            'pins_sha256': trial['artifacts']['integration_pins_sha256'],
            'protected_roots': trial['protected']['mandatory_roots']}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-templates', metavar='DIR')
    parser.add_argument('--check', metavar='DIR')
    args = parser.parse_args(argv)
    try:
        if args.write_templates:
            for path in write_templates(args.write_templates):
                print('TEMPLATE', path)
        if args.check:
            result = check(args.check)
            print(json.dumps(result, indent=2))
            print('COMMAND', ' '.join(result['command']))
        if not args.write_templates and not args.check:
            raise Refused('nothing to do: pass --write-templates or --check')
    except Refused as exc:
        print(json.dumps({'status': 'refused', 'error': repr(exc)}, indent=2))
        return 1
    except Exception as exc:  # includes the sealed adapter's Refusal: fail closed either way
        print(json.dumps({'status': 'refused-by-contract', 'error': repr(exc)}, indent=2))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
