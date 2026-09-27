"""The width trial's own recovery/readback boundary.

`WidthTrialIO` is a `LinuxIO` subclass that keeps the sealed adapter's real
filesystem, /proc, pidfd, flock, urllib and subprocess boundaries and adds the
width-trial-specific fail-closed gates:

* the trial document is validated and hash-pinned before any process action;
* the protected-root inventory is validated and must not contain any of the
  trial's own writable destinations;
* the ONE retained launch is refused unless the width probe evidence on disk is
  present, complete and CONSISTENT with the operator's recorded width outcome;
* the released/restored run paths are re-checked against the protected roots.

Production mode calls the sealed `validate_production` unchanged. The
`inert_fixture` mode is only reachable from tests: the CLI hard-requires the
production profile, and a dedicated control proves the fixture tree is still
refused by `validate_production`.
"""
import json
import os
from pathlib import Path

import trial_contract as tc

import linux_adapter as a
import trial_contract as tc

require = a.require
Refusal = a.Refusal

PRODUCTION_FIELDS = {'schema', 'source_pin', 'root', 'operator_lock', 'active', 'live_config',
                     'server_lock', 'guard_argv', 'launcher_argv', 'launcher_env', 'retained',
                     'identities', 'port', 'new_run_prefix', 'receipt_dir', 'timeouts'}
REQUIRED_ROLES = {'guard', 'launcher', 'server', 'source', 'dso', 'interpreter',
                  'target_config', 'drafter_config'}
CLEAN_ENV_KEYS = {'PATH', 'HOME', 'LC_ALL', 'LANG', 'PYTHONUNBUFFERED', 'TMPDIR'}


def validate_fixture_profile(cfg, trial):
    """Structural validation of the explicitly-labelled inert fixture profile.

    This never accepts production values that differ: it demands the same shape
    and cleanliness the sealed production validator demands, while permitting a
    private temporary root and an ephemeral port. It is unreachable from the CLI.
    """
    require(trial['inert_fixture'] is True, 'fixture profile requires trial.inert_fixture=true')
    try:
        require(set(cfg) == PRODUCTION_FIELDS, 'fixture profile field set mismatch')
        require(cfg['schema'] == 1 and cfg['source_pin'] == a.SOURCE_PIN, 'fixture schema/source pin mismatch')
        root = Path(cfg['root'])
        require(root.is_absolute() and str(root) == cfg['root'] and '..' not in root.parts,
                'fixture root must be a canonical absolute path')
        require(type(cfg['port']) is int and 0 < cfg['port'] < 65536 and cfg['port'] != 8096,
                'inert fixture must never use the production health port')
        for key in ('active', 'live_config', 'server_lock', 'operator_lock', 'receipt_dir'):
            tc.canonical_relative(cfg[key], 'fixture.' + key)
        require(cfg['operator_lock'] != cfg['server_lock'], 'fixture locks must be distinct')
        require(cfg['receipt_dir'] not in ('runs', '.'), 'fixture receipt dir must be explicit')
        require(isinstance(cfg['new_run_prefix'], str) and cfg['new_run_prefix'].endswith('-'),
                'fixture run prefix must be explicit')
        for key in ('guard_argv', 'launcher_argv'):
            require(isinstance(cfg[key], list) and cfg[key] and
                    all(isinstance(v, str) and v and '\0' not in v for v in cfg[key]),
                    'fixture ' + key + ' must be a non-empty argv')
        env = cfg['launcher_env']
        require(isinstance(env, dict) and {'PATH', 'LC_ALL', 'PYTHONUNBUFFERED'} <= set(env) and
                set(env) <= CLEAN_ENV_KEYS and
                all(isinstance(v, str) and v and '\0' not in v and 'REQUIRED_' not in v for v in env.values()),
                'fixture launcher environment must be explicit and clean')
        retained = cfg['retained']
        require(isinstance(retained, dict) and {'command', 'env'} <= set(retained), 'fixture retained shape')
        identities = {r['role']: r for r in cfg['identities']}
        require(len(identities) == len(cfg['identities']) and REQUIRED_ROLES <= set(identities),
                'fixture identity roles missing/duplicated')
        for row in identities.values():
            require(set(row) == {'role', 'path', 'sha256'} and a.valid_sha(row['sha256']),
                    'fixture identity needs an explicit trusted hash')
            tc.canonical_absolute(row['path'], 'fixture identity path')
        for key, maximum in (('stop', 120), ('headroom', 120), ('launch', 120), ('ready', 500), ('http', 5)):
            value = cfg['timeouts'][key]
            require(type(value) in (int, float) and 0 < value <= maximum, 'fixture timeout out of range: ' + key)
        return cfg
    except (KeyError, TypeError, ValueError) as exc:
        raise Refusal('incomplete inert fixture profile: ' + str(exc)) from exc


class WidthTrialIO(a.LinuxIO):
    """Recovery/Ownership boundary for the native16 width trial."""

    def __init__(self, cfg, receipt, auth, config_sha, receipt_sha, trial, trial_sha,
                 protected_roots, protected_complete, stage):
        if trial.get('inert_fixture'):
            validate_fixture_profile(cfg, trial)
        else:
            a.validate_production(cfg)
            require(trial['width']['drafter_config_sha256'] == tc.WIDTH_CONFIG_SHA256,
                    'production width drafter config must be the audited native16 config')
        super().__init__(cfg, receipt, auth, config_sha, receipt_sha)
        tc.validate_trial(trial, self.root)
        require(stage in tc.STAGES, 'unknown trial stage: ' + repr(stage))
        require(auth['stage'] == stage, 'authorization stage mismatch')
        self.trial = trial
        self.trial_doc_sha256 = trial_sha
        self.protected_roots = list(protected_roots)
        self.protected_complete = protected_complete
        self.stage = stage
        self.probe_verdict = None
        self.prefailure = None

    # ------------------------------------------------------------------ helpers
    def absolute(self, relative):
        tc.canonical_relative(relative, 'relative path')
        return str(self.root / relative)

    def dir_exists(self, relative):
        """Does any entry (directory OR file) exist at this relative path?

        `exists()` reads a single-link regular file and therefore cannot answer
        this for a directory. The sealed diagnostic appends to
        `<probe>/events.jsonl`, so a probe directory - not just the file - must
        be proven absent before staging.
        """
        return os.path.lexists(self.absolute(relative))

    def trial_destinations(self):
        width = self.trial['width']
        return [self.absolute(width['probe_dir']), self.absolute(tc.RUNS_DIR),
                self.absolute(self.cfg['receipt_dir'])]

    def check_protected_roots(self):
        return tc.validate_protected_roots(self.protected_roots, self.protected_complete,
                                          self.trial, self.trial_destinations())

    # -------------------------------------------------- width-trial state gates
    def require_trial_outcome_consistent(self, verdict):
        """The restore stage may only run for a REAL, probe-consistent width outcome."""
        recorded = self.auth['width']
        require(recorded['phase'] == 'after_width',
                'the ONE retained restore requires an after_width authorization')
        require(self.stage == 'restore', 'probe-consistent outcomes are restore-stage only')
        require(verdict['status'] == recorded['status'],
                'operator-recorded width outcome (' + recorded['status'] + ') contradicts the probe '
                'evidence on disk (' + verdict['status'] + ')')
        if recorded['status'] == 'failed':
            require(isinstance(recorded['error'], str) and recorded['error'],
                    'a failed width outcome requires the observed error text')
        return verdict

    def require_trial_state(self):
        """The width probe evidence is a hard precondition of the ONE restore.

        This is the seam that ties the recovery core to the trial's real state
        and pointers instead of a fake boundary: without a complete, consistent
        native16 probe capture on disk there is no retained launch at all.
        """
        verdict = tc.analyse_probe(self.fs.read, self.trial)
        self.probe_verdict = self.require_trial_outcome_consistent(verdict)
        require(self.probe_verdict['status'] in ('passed', 'failed'), 'unknown probe verdict')
        return self.probe_verdict

    def require_width_run_state(self):
        """The run being restored from must be the trial's own width run."""
        width = self.trial['width']
        name = Path(self.expected['run']).name
        require(name.startswith(width['run_prefix']),
                'the run being stopped is not a width-trial run: ' + name)
        require(self.exists(self.relative_run(self.expected['run']) + '/launch.json'),
                'width run has no recorded launch configuration')
        require(not self.exists(self.relative_run(self.expected['run']) + '/STOP'),
                'width run already has a STOP record')
        return name

    # ------------------------------------------------------------ the ONE launch
    def launch_retained(self):
        # Trial-specific preconditions, evaluated immediately before the sealed
        # adapter's single retained-launch invocation. The width-run/STOP
        # precondition is checked during the preflight, BEFORE the STOP is
        # written; here the probe evidence must still be present on disk.
        self.require_trial_state()
        roots = self.check_protected_roots()
        self.protected_receipt = roots
        receipt = super().launch_retained()
        # Path-level containment: the *new* run must not sit inside a protected root.
        new_absolute = str(self.root / self.relative_run(receipt['run']))
        tc.validate_protected_roots(self.protected_roots, self.protected_complete, self.trial,
                                    self.trial_destinations() + [new_absolute])
        return receipt


class ReleaseIO(WidthTrialIO):
    """GPU-release stage: the SAME identity/STOP/removal sequence, no launch.

    `launch_retained` is not merely unused here, it is a hard failure, so a
    release run that somehow reached a load would abort instead of silently
    starting a second model.
    """

    def launch_retained(self):
        raise Refusal('release stage must never launch a model; use the restore stage')

    def wait_retained(self, receipt):
        raise Refusal('release stage must never wait for a restored model')


def release_owned(expected, io, stop_seconds, headroom_seconds):
    """Identity -> STOP -> both owned processes gone + guard result -> headroom.

    Deliberately the first half of the sealed core's `launch()` and nothing
    more: no launch, no retry, no candidate.
    """
    before = io.core.recover.identity(expected, io)
    io.before_stop()
    memory = io.core.recover.stop_and_release(expected, io, stop_seconds=stop_seconds,
                                              headroom_seconds=headroom_seconds)
    for row in expected['processes']:
        require(io.process(row['pid']) is None, 'released process still present')
    require(io.active() == expected['run'], 'active pointer changed during release')
    require(io.result(expected['run']) is not None, 'released owner left no guard result')
    return {'before': before, 'memory': memory, 'released_run': expected['run'],
            'processes_gone': [row['pid'] for row in expected['processes']]}


def perform_restore(io):
    """The ONE documented recovery sequence, via the sealed controller.

    verify identity -> STOP the exact guard -> prove old identities gone ->
    bounded removal -> headroom check -> ONE retained launch -> authenticated
    health 200 readiness.  The width outcome is preserved independently of the
    rollback outcome and of the status readback.
    """
    width = io.auth['width']
    status = {'width': width['status'], 'width_error': width['error'], 'rollback': 'not_run',
              'rollback_error': None, 'readback_error': None,
              'trial_sha256': io.trial_doc_sha256, 'probe': None}
    name = io.cfg['receipt_dir'] + '/width-trial-status-' + io.receipt_sha + '.json'
    status['status_path'] = str(io.root / name)
    try:
        status['recovery'] = io.core.launch(io.expected, io, stop_seconds=io.cfg['timeouts']['stop'],
                                            headroom_seconds=io.cfg['timeouts']['headroom'])
        status['rollback'] = 'restored'
    except BaseException as exc:  # noqa: BLE001 - the outcome must be preserved verbatim
        status['rollback'] = 'failed'
        status['rollback_error'] = repr(exc)
    status['probe'] = io.probe_verdict
    recovery = status.get('recovery') or {}
    retained = recovery.get('retained') or {}
    owner = (retained.get('receipt') or {}).get('owner') or {}
    rows = owner.get('processes') or []
    # new_snapshot records processes as [guard, server].
    status['restored'] = ({'run': owner.get('run'), 'pids': owner.get('pids'),
                           'starttimes': ({'guard_pid': rows[0]['starttime'], 'server_pid': rows[1]['starttime']}
                                          if len(rows) == 2 else {}),
                           'readback_path': retained.get('readback_path')} if owner else None)
    try:
        io.fs.create(name, json.dumps(status, indent=2).encode())
        require(a.strict_json(io.fs.read(name)) == status, 'width-trial status readback mismatch')
    except BaseException as exc:  # noqa: BLE001
        status['readback_error'] = repr(exc)
    code = 0 if (status['width'] == 'passed' and status['rollback'] == 'restored'
                 and status['readback_error'] is None) else 1
    return code, status


def release_report(io, snapshot, before_runs):
    """Read-only proof that the release stage released and launched nothing."""
    require(io.launched is False, 'the release stage must never launch')
    for row in io.expected['processes']:
        require(a.proc_snapshot(row['pid']) is None,
                'released process still present: ' + str(row['pid']))
    require(io.active() == io.expected['run'], 'the active pointer changed during release')
    require(io.result(io.expected['run']) is not None, 'the released owner left no guard result')
    require(not io.dir_exists(io.trial['width']['probe_dir']),
            'the release stage created a probe directory')
    after_runs = sorted(os.listdir(Path(io.root) / tc.RUNS_DIR))
    require(after_runs == before_runs, 'the release stage created or removed a run directory')
    roots = {}
    for root in io.protected_roots:
        before = snapshot.get(root)
        require(before is not None, 'protected root was not snapshotted: ' + root)
        path = Path(root)
        require(path.is_dir(), 'protected root vanished: ' + root)
        after = {'mtime_ns': path.stat().st_mtime_ns, 'entries': len(list(path.iterdir()))}
        require(after == before, 'protected root changed during release: ' + root)
        roots[root] = after
    return {'status': 'released', 'released_run': io.expected['run'],
            'processes_gone': [row['pid'] for row in io.expected['processes']],
            'guard_result': io.result(io.expected['run']),
            'launch_count_unchanged': True, 'runs': after_runs,
            'protected_roots': roots, 'probe_created': False,
            'scope': ('GPU released and nothing launched. Staging the native16 width run and running '
                      'bounded inference are separate, explicitly-authorized device steps (GAPS.md).')}


def write_receipt(io, name, payload):
    raw = json.dumps(payload, indent=2).encode()
    io.fs.create(name, raw)
    require(a.strict_json(io.fs.read(name)) == payload, 'receipt readback mismatch: ' + name)
    return str(io.root / name)


def guarded_environment(environ):
    """Refuse to run the trial CLI from an environment already carrying the probe."""
    require('ROUND8_WIDTH' not in environ,
            'ROUND8_WIDTH must not be exported into the trial CLI; the width-run launcher owns it')
    require('ROUND8_WIDTH_OUT' not in environ,
            'ROUND8_WIDTH_OUT must not be exported into the trial CLI; the width-run launcher owns it')
    return True


def scan_live_width_probe(trial, skip_pids):
    """Any OTHER live process holding the probe dir or its diagnostic env.

    `skip_pids` must include the caller and the PIDs the trusted pre-failure
    receipt already owns: the current owner is *supposed* to carry the
    diagnostic environment during the restore stage, and is accounted for by
    identity checks instead of this scan.
    """
    skip_pids = {int(pid) for pid in skip_pids}
    hits = []
    probe = trial['width']['env']['ROUND8_WIDTH_OUT']
    prefix = trial['width']['run_prefix']
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) in skip_pids:
            continue
        try:
            args = entry.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        except OSError:
            continue
        try:
            raw_env = entry.joinpath('environ').read_bytes()
        except OSError:
            raw_env = b''
        env = dict(item.split('=', 1) for item in raw_env.decode(errors='replace').split('\0') if '=' in item)
        if env.get('ROUND8_WIDTH') == '1' or probe in args or any(prefix in item for item in args):
            hits.append({'pid': int(entry.name), 'args': args[:4],
                         'round8_width': env.get('ROUND8_WIDTH')})
    return hits
