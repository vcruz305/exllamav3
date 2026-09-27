#!/usr/bin/env python3
"""Width-trial driver: the ONLY authorized entry point. Run as:

    /usr/bin/python3 -I -B width_trial.py --stage <preflight|release|restore> ...

Three stages, one invocation each:

  preflight  read-only: validates every document, the protected-root
             inventory, the pinned retained inputs, the death of the previous
             width attempt and the probe state for the stage. No STOP, no
             launch, no write.
  release    the preflight plus the first half of the sealed recovery sequence:
             verify identity -> STOP the exact guard -> prove both owned
             processes gone -> bounded removal -> headroom check. It CANNOT
             launch: `ReleaseIO.launch_retained` raises.
  restore    the preflight plus the ONE documented recovery sequence and the
             independent post-restore readiness readback.

Fail-closed rules, all enforced before anything acts:
  * `EXL3_WIDTH_TRIAL_AUTHORIZED=YES` and `--authorize-width-trial` together;
  * an explicitly supplied configuration, PRE-FAILURE receipt, authorization
    and trial document, each with its own SHA256;
  * every composed artifact re-hashed against `integration-pins.json` BEFORE
    any module is imported;
  * the production profile unless the trial document is an explicitly labelled
    inert fixture (reachable from tests only);
  * no implicit default, no silent fallback, no retry, no second launch.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PINS_NAME = 'integration-pins.json'
# The single inert-witness hook. It is refused unless the trial document is an
# explicitly labelled inert fixture, and the witness module is itself pinned in
# integration-pins.json. Production trial documents cannot reach it.
IO_HOOK_ALLOWLIST = {'witness.inert_trial_io'}
# `-I` removes the script directory from sys.path. The sealed pins are verified
# BEFORE any import below, so putting this package first on the path is bounded.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def fail(message):
    print(json.dumps({'status': 'fail', 'error': message}, indent=2))
    return 1


def verify_pins():
    """Re-hash every composed artifact before importing anything."""
    raw = (HERE / PINS_NAME).read_bytes()
    pins = json.loads(raw)
    if set(pins) != {'schema', 'source_pin', 'artifacts', 'composed_from'}:
        raise RuntimeError('integration-pins.json field set mismatch')
    if pins['schema'] != 1 or not pins['artifacts']:
        raise RuntimeError('integration-pins.json schema/coverage missing')
    for name, record in sorted(pins['artifacts'].items()):
        path = HERE / name
        if not path.is_file():
            raise RuntimeError('pinned artifact missing: ' + name)
        if set(record) != {'sha256', 'bytes', 'origin', 'origin_sha256'}:
            raise RuntimeError('pinned artifact record malformed: ' + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record['sha256']:
            raise RuntimeError('pinned artifact hash mismatch: ' + name)
        origin = Path(record['origin'])
        if origin.is_file() and hashlib.sha256(origin.read_bytes()).hexdigest() != record['origin_sha256']:
            raise RuntimeError('upstream origin changed under the pin: ' + record['origin'])
    return {'pins_sha256': hashlib.sha256(raw).hexdigest(), 'artifacts': len(pins['artifacts']),
            'source_pin': pins['source_pin']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--stage', choices=['preflight', 'release', 'restore'], required=True)
    for name in ('config', 'receipt', 'authorization', 'trial'):
        parser.add_argument('--' + name)
        parser.add_argument('--' + name + '-sha256')
    parser.add_argument('--protected-root', action='append', default=None)
    parser.add_argument('--protected-roots-complete', action='store_true')
    parser.add_argument('--preflight-out')
    parser.add_argument('--authorize-width-trial', action='store_true')
    args = parser.parse_args(argv)

    status = {'stage': args.stage, 'status': 'not_run', 'error': None,
              'width': {'status': 'not_run', 'error': None, 'phase': None},
              'rollback': 'not_run', 'rollback_error': None, 'readback_error': None}
    try:
        pins = verify_pins()
        status['pins'] = pins
    except BaseException as exc:  # noqa: BLE001 - pinned inputs are the first gate
        return fail('artifact pin verification failed: ' + repr(exc))

    import linux_adapter as a
    import trial_contract as tc
    import trial_io as ti
    import trial_preflight as tp
    import trial_readiness as tr

    try:
        require = a.require
        require(sys.platform == 'linux', 'Linux /usr/bin/python3 is required')
        require(sys.flags.isolated, '/usr/bin/python3 -I -B is required (isolated mode)')
        tc.require_authorization_token(os.environ, args)
        ti.guarded_environment(os.environ)
        # The inert witness must be loaded BEFORE the first filesystem boundary
        # is opened, because it is the module that rebinds the DrvFS mode
        # predicate. Loading it is not the same as using it: the factory is only
        # selected below, after the trial document has been validated.
        hook = os.environ.get('EXL3_WIDTH_TRIAL_IO_HOOK')
        if hook:
            require(hook in IO_HOOK_ALLOWLIST,
                    'unknown inert IO hook (allowlist: ' + ','.join(sorted(IO_HOOK_ALLOWLIST)) + ')')
            importlib.import_module(hook)
            status['io_hook'] = hook
        require(args.protected_root, '--protected-root is required at least once')
        require(args.protected_roots_complete, '--protected-roots-complete must be declared explicitly')

        documents, hashes = {}, {}
        for name in ('config', 'receipt', 'authorization', 'trial'):
            path, expected = getattr(args, name), getattr(args, name + '_sha256')
            require(path and tc.valid_sha(expected),
                    'explicit trusted ' + name + ' document and its SHA256 are required')
            raw = a.read_absolute(path)
            require(a.digest(raw) == expected, 'explicit trusted document hash mismatch: ' + name)
            documents[name] = a.strict_json(raw)
            hashes[name] = expected
        cfg, receipt, auth, trial = (documents[n] for n in ('config', 'receipt', 'authorization', 'trial'))
        trial_sha = hashes['trial']
        require(trial['artifacts']['integration_pins'] == PINS_NAME,
                'the trial document must pin ' + PINS_NAME)
        require(trial['artifacts']['integration_pins_sha256'] == pins['pins_sha256'],
                'the trial document pins a different integration-pins.json')
        require(trial['source_pin'] == pins['source_pin'], 'trial/source pin mismatch')

        tc.validate_trial(trial, Path(cfg['root']))
        tc.validate_receipt_document(receipt, auth)
        # The authorization declares the stage it belongs to; `--stage preflight`
        # is a read-only action that validates against that declared stage. Any
        # other value is refused by validate_authorization itself.
        auth_stage = auth.get('stage')
        require(auth_stage in tc.STAGES, 'authorization stage must be one of ' + ','.join(tc.STAGES))
        tc.validate_authorization(auth, trial, Path(cfg['root']), hashes['config'], hashes['receipt'],
                                  trial_sha, Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                  os.getuid(), time.time(), auth_stage)
        if args.stage in tc.STAGES:
            require(args.stage == auth_stage,
                    'the requested stage does not match the authorization stage: '
                    + args.stage + ' != ' + auth_stage)
        status['width'] = {'status': auth['width']['status'], 'error': auth['width']['error'],
                           'phase': auth['width']['phase']}

        if hook:
            require(trial['inert_fixture'] is True,
                    'the inert IO hook is refused for a production trial document')
            factory = importlib.import_module(hook).build(args.stage)
        else:
            factory = ti.ReleaseIO if args.stage == 'release' or auth_stage == 'release' \
            else ti.WidthTrialIO
        with factory(cfg, receipt, auth, hashes['config'], hashes['receipt'], trial, trial_sha,
                     args.protected_root, args.protected_roots_complete, auth_stage) as io:
            preflight = tp.run_preflight(io)
            status['preflight'] = preflight
            if args.preflight_out:
                tc.canonical_relative(args.preflight_out, '--preflight-out')
                status['preflight_path'] = ti.write_receipt(io, args.preflight_out, preflight)
            if args.stage == 'preflight':
                status['status'] = 'preflighted'
                print(json.dumps(status, indent=2))
                return 0

            snapshot = tr.snapshot_protected_roots(io.protected_roots)
            before_runs = sorted(os.listdir(Path(io.root) / tc.RUNS_DIR))
            if args.stage == 'release':
                status['release'] = ti.release_owned(io.expected, io, io.cfg['timeouts']['stop'],
                                                     io.cfg['timeouts']['headroom'])
                status['report'] = ti.release_report(io, snapshot, before_runs)
                status['status'] = 'released'
            else:
                code, restore = ti.perform_restore(io)
                status['restore'] = restore
                status['rollback'] = restore['rollback']
                status['rollback_error'] = restore['rollback_error']
                status['readback_error'] = restore['readback_error']
                status['status_path'] = restore['status_path']
                status['restore_exit_code'] = code
                if restore['rollback'] == 'restored' and restore.get('restored'):
                    restored = restore['restored']
                    receipt_owner = restore['recovery']['retained']['receipt']['owner']
                    status['readiness'] = tr.readiness_report(
                        tr.HostReadback(io.root, io.cfg['port']), trial, cfg,
                        {'run': restored['run'], 'pids': restored['pids'],
                         'starttimes': restored['starttimes']},
                        io.expected, io.protected_roots, snapshot)
                    status['status'] = 'restored'
                    status['retained_owner'] = receipt_owner
                else:
                    status['status'] = 'restore_failed'

        print(json.dumps(status, indent=2))
        if args.stage == 'restore':
            # A successful restore of a FAILED width trial is still a failed trial.
            return int(status.get('restore_exit_code', 1))
        return 0 if status['status'] in ('preflighted', 'released') else 1
    except BaseException as exc:  # noqa: BLE001 - one refusal, one printed record, no retry
        status['status'] = 'refused'
        status['rollback'] = 'failed' if args.stage == 'restore' else status['rollback']
        status['rollback_error'] = repr(exc)
        print(json.dumps(status, indent=2))
        return 1


if __name__ == '__main__':
    sys.exit(main())
