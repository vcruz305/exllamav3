"""Fail-closed width-trial preflight, split by stage.

Nothing here is optional and nothing has a default. Every check either passes
with recorded evidence or raises a refusal naming the exact gap.

Common to both stages
  P1 explicit authorization token (env switch AND CLI switch) plus a hashed,
     stage-bound authorization document (intent, boot id, uid, ingress, expiry)
  P2 an explicitly supplied configuration document whose bytes match the
     supplied hash, a trusted PRE-FAILURE receipt with its own hash, and a
     hash-pinned width-trial document
  P3 a complete, explicitly-declared protected-root inventory containing the
     mandatory roots and excluding every writable destination of the trial
  P4 the retained launch inputs are pinned and hash-verified on disk (identity
     files, native8 retained drafter config, native16 width config)
  P5 the previous width attempt is provably dead (trusted PIDs absent, no live
     process holding the diagnostic env or probe path, no stray aligned runs)

Release stage (before the width run exists)
  P6a the probe directory does not exist, so the diagnostic cannot APPEND to a
      previous attempt's event stream
  P4L the live owner is the retained native8 serve and it is healthy and idle

Restore stage (after the width run)
  P6b the probe event stream exists, is complete for every required gate, and
      is CONSISTENT with the operator's recorded width outcome
  P4R the live owner is the width run, it really had the diagnostic installed
      (ROUND8_WIDTH/ROUND8_WIDTH_OUT in its own environment), and it is healthy
      and idle, so the evidence can be attributed to the process being stopped

The preflight performs no STOP, no launch and no write into the runtime root.
"""
import os
from pathlib import Path

import linux_adapter as a
import trial_contract as tc
import trial_io as ti

require = a.require
Refusal = a.Refusal
digest = a.digest
strict_json = a.strict_json


def retained_launch_inputs(io):
    """P4: everything the single retained launch depends on is pinned on disk."""
    trial = io.trial
    width = trial['width']
    io.verify_identities()
    retained_raw = a.read_absolute(width['retained_drafter_config'])
    require(digest(retained_raw) == width['retained_drafter_config_sha256'],
            'retained drafter config hash changed')
    retained_blocks = tc.validate_width_config(retained_raw, 'retained drafter config', tc.RETAINED_BLOCK)
    width_raw = a.read_absolute(width['drafter_config'])
    require(digest(width_raw) == width['drafter_config_sha256'], 'width drafter config hash changed')
    width_blocks = tc.validate_width_config(width_raw, 'width drafter config', tc.NATIVE_BLOCK)
    return {'identities': {row['role']: row['sha256'] for row in io.cfg['identities']},
            'retained_drafter_config': {'path': width['retained_drafter_config'],
                                        'block_size': retained_blocks,
                                        'sha256': digest(retained_raw)},
            'width_drafter_config': {'path': width['drafter_config'], 'block_size': width_blocks,
                                     'sha256': digest(width_raw)}}


def live_owner_identity(io):
    """P4L: the release stage's live owner is the exact, healthy, idle serve."""
    observed = {}
    require(io.active() == io.expected['run'], 'active pointer does not name the retained run')
    require(io.pids(io.expected['run']) == io.expected['pids'], 'retained PID record changed')
    for row in io.expected['processes']:
        require(io.process(row['pid']) == row, 'retained process identity changed: ' + str(row['pid']))
    health = io.core.recover.health(io)
    require(health['status'] == 200 and health['body'].get('healthy') is True,
            'retained serve is not healthy')
    server_env = io.receipt['scopes'][str(io.expected['pids']['server_pid'])]['environment']
    for key in ('ROUND8_WIDTH', 'ROUND8_WIDTH_OUT'):
        require(not server_env.get(key),
                'the retained serve already carries the width diagnostic environment: ' + key)
    inode = a.listener_inode(io.expected['processes'][1], io.cfg['port'])
    io.assert_server_lock(io.expected['pids']['guard_pid'])
    require(io.active() == io.expected['run'], 'active pointer changed during identity readback')
    observed.update({'run': io.expected['run'], 'pids': io.expected['pids'], 'listener_inode': inode,
                     'health': health, 'roles': sorted(row['role'] for row in io.cfg['identities'])})
    return observed


def live_width_owner(io):
    """P4R: the restore stage's live owner is the width run with the diagnostic
    actually installed in its own environment."""
    width = io.trial['width']
    run_name = io.require_width_run_state()
    require(io.active() == io.expected['run'], 'active pointer does not name the width run')
    require(io.pids(io.expected['run']) == io.expected['pids'], 'width PID record changed')
    for row in io.expected['processes']:
        require(io.process(row['pid']) == row, 'width process identity changed: ' + str(row['pid']))
    server_env = io.receipt['scopes'][str(io.expected['pids']['server_pid'])]['environment']
    require(server_env.get('ROUND8_WIDTH') == width['env']['ROUND8_WIDTH'],
            'the process being stopped never had the sealed diagnostic enabled')
    require(server_env.get('ROUND8_WIDTH_OUT') == width['env']['ROUND8_WIDTH_OUT'],
            'the process being stopped did not write to the configured probe directory')
    require(io.process(io.expected['processes'][1]['pid']) == io.expected['processes'][1],
            'width server identity changed during environment readback')
    health = io.core.recover.health(io)
    require(health['status'] == 200 and health['body'].get('healthy') is True,
            'width serve is not healthy')
    inode = a.listener_inode(io.expected['processes'][1], io.cfg['port'])
    io.assert_server_lock(io.expected['pids']['guard_pid'])
    return {'run': io.expected['run'], 'run_name': run_name, 'pids': io.expected['pids'],
            'listener_inode': inode, 'health': health,
            'width_env': {'ROUND8_WIDTH': server_env['ROUND8_WIDTH'],
                          'ROUND8_WIDTH_OUT': server_env['ROUND8_WIDTH_OUT']}}


def previous_width_attempt_dead(io):
    """P5: no part of a previous native16 width attempt is live or half-present."""
    width = io.trial['width']
    observed = {}
    failed_pids = width['failed_pids']
    failed_run = width['failed_run']
    if failed_run is not None and io.exists(failed_run + '/pid.json'):
        recorded = strict_json(io.fs.read(failed_run + '/pid.json'))
        require(set(recorded) == {'guard_pid', 'server_pid', 'command'},
                'the recorded failed width run has a malformed PID record')
        require({recorded['guard_pid'], recorded['server_pid']} ==
                {failed_pids['guard_pid'], failed_pids['server_pid']},
                'the recorded failed width run disagrees with the trusted failed PIDs')
        require(io.exists(failed_run + '/result.json'),
                'the recorded failed width run has no owned guard result')
        result = strict_json(io.fs.read(failed_run + '/result.json'))
        require(result.get('reason') == 'requested_stop' and type(result.get('returncode')) is int,
                'the failed width run does not carry an owned requested-stop guard result')
        require(result.get('oom_kill_delta') == 0 and result.get('host_oom_kill_delta') == 0,
                'the failed width run reported OOM deltas')
        require(io.active() != io.absolute(failed_run), 'the active pointer still names the failed run')
        observed['failed_run'] = failed_run
        observed['failed_run_result'] = result
    elif failed_run is not None:
        require(not io.exists(failed_run),
                'the recorded failed width run is partially present: ' + failed_run)
        observed['failed_run'] = failed_run
        observed['failed_run_absent'] = True
    else:
        require(None is failed_run, 'failed_run must be null when the attempt was already cleaned up')
        observed['failed_run'] = None
    for role, pid in sorted(failed_pids.items()):
        require(a.proc_snapshot(pid) is None,
                'the previous width attempt is still live: ' + role + '=' + str(pid))
    observed['failed_pids_absent'] = dict(failed_pids)

    owned = {int(io.expected['pids'][key]) for key in ('guard_pid', 'server_pid')}
    live = ti.scan_live_width_probe(io.trial, owned | {os.getpid()})
    require(not live, 'a live process holds the width diagnostic environment or probe path: ' + repr(live))
    observed['live_probe_processes'] = live

    runs = Path(io.root) / tc.RUNS_DIR
    require(runs.is_dir(), 'the runtime runs directory is missing')
    stray = [entry for entry in sorted(os.listdir(runs))
             if entry.startswith(width['run_prefix'])
             and str(Path(tc.RUNS_DIR) / entry) not in (failed_run, io.relative_run(io.expected['run']))]
    require(not stray, 'stray width-prefixed run directories: ' + repr(stray))
    observed['stray_runs'] = stray
    return observed


def probe_state(io):
    """P6: the probe directory is either provably absent or provably complete."""
    width = io.trial['width']
    if io.stage == 'release':
        require(not io.dir_exists(width['probe_dir']),
                'a width probe path already exists: ' + width['probe_dir']
                + '; the sealed diagnostic APPENDS to events.jsonl, so a previous attempt must be '
                  'proven absent before staging')
        return {'stage': 'release', 'probe_absent': width['probe_dir']}
    require(io.dir_exists(width['probe_dir']),
            'the staged width attempt left no probe directory: ' + width['probe_dir']
            + '; a restore cannot be justified without the evidence of the run being stopped')
    verdict = io.require_trial_state()
    require(verdict['status'] in ('passed', 'failed'), 'unknown probe verdict')
    return {'stage': 'restore', 'probe': verdict}


def run_preflight(io):
    """Run P1-P6 for the current stage and return a machine-readable receipt."""
    require(io.launched is False, 'preflight must run before any retained launch exists')
    trial = io.trial
    receipt = {'stage': io.stage, 'root': str(io.root), 'checks': {}, 'status': 'passed'}
    receipt['checks']['P1_profile'] = {
        'ok': True, 'validator': 'inert_fixture' if trial['inert_fixture'] else 'production',
        'source_pin': io.cfg['source_pin'], 'health_port': io.cfg['port'],
        'health_port_is_production_8096': io.cfg['port'] == 8096}
    receipt['checks']['P2_documents'] = {
        'ok': True, 'config_sha256': io.config_sha, 'receipt_sha256': io.receipt_sha,
        'trial_sha256': io.trial_doc_sha256, 'authorization_stage': io.auth['stage'],
        'width_phase': io.auth['width']['phase'],
        'width_config_sha256': trial['width']['drafter_config_sha256'],
        'retained_config_sha256': trial['width']['retained_drafter_config_sha256']}
    receipt['checks']['P3_protected_roots'] = dict(io.check_protected_roots(), ok=True)
    receipt['checks']['P4_retained_inputs'] = dict(retained_launch_inputs(io), ok=True)
    if io.stage == 'release':
        receipt['checks']['P4L_live_owner'] = dict(live_owner_identity(io), ok=True)
    else:
        receipt['checks']['P4R_live_width_owner'] = dict(live_width_owner(io), ok=True)
    receipt['checks']['P5_previous_width_attempt'] = dict(previous_width_attempt_dead(io), ok=True)
    receipt['checks']['P6_probe_state'] = dict(probe_state(io), ok=True)
    receipt['scope'] = ('Read-only preflight. No STOP, no retained launch, no probe creation, no runtime '
                        'write. Activating the device probe and running inference remain separate, '
                        'labelled device gates (see GAPS.md).')
    return receipt
