"""Width-trial policy contract: documents, protected roots, tokens, probe gates.

Nothing here touches a process, socket or model. Every function is fail-closed:
a missing, unknown, duplicated, non-canonical or non-finite field is a refusal,
never a default. The native8->native16 width expectations are pinned here so
that an operator cannot silently widen, narrow or reinterpret the trial.
"""
import hashlib
from pathlib import Path

import linux_adapter as a

require = a.require
Refusal = a.Refusal
digest = a.digest
strict_json = a.strict_json
valid_sha = a.valid_sha

SOURCE_PIN = a.SOURCE_PIN

# native16 width, exactly as the sealed study/repair configs declare it.
NATIVE_BLOCK = 16
RETAINED_BLOCK = 8
EXPECTED_TAPS = [0, 11, 23, 35, 47]
TAP_SHIFT = 0
DRAFT_RESERVE = 15
MASK_SHAPE = [4096]
MODEL_VOCAB = 152576
WIDTH_CONFIG_SHA256 = '259759b66599137681e9ef2c62b4047a2331f21466ca5dd33276b0e2ce8efd52'
DIAG_ENV = {'ROUND8_WIDTH': '1'}
PROBE_EVENTS = 'events.jsonl'
RUNS_DIR = 'runs'

EXPECTED = {
    'input_shape': [1, 16, 4096],
    'state_shape': [1, 16, 4096],
    'q_shape': [1, 16, 64, 128],
    'k_shape': [1, 16, 8, 128],
    'samples_shape': [1, 16],
    'proposal_count': 15,
    'assigned_span': 16,
    'max_round': 512,
}

# The sealed diagnostic's complete emit() vocabulary. An unknown kind means a
# DIFFERENT diagnostic was installed, which must never be accepted silently.
KNOWN_KINDS = frozenset((
    'installed', 'job_init', 'assigned_pages', 'input_metadata', 'input', 'state',
    'qkv', 'native_cache_writes', 'pristine_draft_head', 'native_samples',
    'proposal_truncation', 'target_forward', 'received', 'terminal',
))
# Records that only exist once the neural path actually ran to a proposal.
REQUIRED_KINDS = (
    'installed', 'assigned_pages', 'input_metadata', 'input', 'state', 'qkv',
    'native_cache_writes', 'pristine_draft_head', 'native_samples',
    'proposal_truncation', 'received', 'terminal',
)

TRIAL_FIELDS = {'schema', 'inert_fixture', 'source_pin', 'root', 'width', 'protected', 'artifacts', 'scope'}
WIDTH_FIELDS = {'native_block', 'retained_block', 'drafter_config', 'drafter_config_sha256',
                'retained_drafter_config', 'retained_drafter_config_sha256', 'probe_dir', 'events',
                'run_prefix', 'failed_run', 'failed_pids', 'env', 'expect'}
# The sealed adapter reads these keys from the authorization document; the trial
# layer adds 'trial_sha256' and 'stage'. The adapter does not exact-check the
# authorization field set, so both layers can share ONE document.
ADAPTER_INTENT = 'stop_owned_then_launch_retained_once'
TRIAL_INTENT = ADAPTER_INTENT
AUTH_FIELDS = {'schema', 'intent', 'root', 'operator_lock', 'ingress_blocked', 'uid', 'boot_id',
               'config_sha256', 'receipt_sha256', 'issued_at', 'failure_at', 'expires_at',
               'width', 'trial_sha256', 'stage', 'scope'}
STAGES = ('release', 'restore')
# The sealed adapter's authorization schema can express only a binary width
# outcome ('passed'/'failed'), yet the GPU-release stage runs BEFORE the width
# trial exists. The trial layer therefore requires an explicit phase field and
# encodes "not run yet" as a fixed, non-empty error string. The release stage
# never reports this value as a width outcome: its status document reports
# width='not_run'. See GAPS.md G-WIDTH-SCHEMA-BINARY.
WIDTH_OUTCOME_FIELDS = {'status', 'error', 'phase'}
WIDTH_PHASES = ('before_width', 'after_width')
NOT_RUN_ERROR = 'not_run_before_staging'


def canonical_relative(value, where):
    require(isinstance(value, str) and value, where + ': relative path required')
    path = Path(value)
    require(not path.is_absolute() and str(path) == value and '..' not in path.parts
            and '.' not in path.parts and bool(path.parts), where + ': unsafe relative path')
    return value


def canonical_absolute(value, where):
    require(isinstance(value, str) and value, where + ': absolute path required')
    path = Path(value)
    require(path.is_absolute() and str(path) == value and '..' not in path.parts,
            where + ': noncanonical absolute path')
    return value


def exact_fields(mapping, fields, where):
    require(isinstance(mapping, dict), where + ': object required')
    require(set(mapping) == fields, where + ': fields must be exactly ' + ','.join(sorted(fields))
            + ' (missing=' + ','.join(sorted(fields - set(mapping)))
            + '; unknown=' + ','.join(sorted(set(mapping) - fields)) + ')')


def validate_expect(value):
    """Pin the diagnostic's expected geometry. No tolerance may be relaxed here."""
    exact_fields(value, set(EXPECTED), 'width.expect')
    for key, expected in EXPECTED.items():
        require(value[key] == expected, 'width.expect.' + key + ' must be ' + repr(expected))
    require(value['proposal_count'] == NATIVE_BLOCK - 1, 'width.expect.proposal_count must be native-1')
    require(value['assigned_span'] == NATIVE_BLOCK, 'width.expect.assigned_span must be native block')


def validate_trial(trial, root):
    """Validate the width-trial document. `root` is the owned runtime root."""
    exact_fields(trial, TRIAL_FIELDS, 'trial')
    require(trial['schema'] == 1, 'trial schema must be 1')
    require(type(trial['inert_fixture']) is bool, 'trial.inert_fixture must be a boolean')
    require(trial['source_pin'] == SOURCE_PIN, 'trial source pin mismatch')
    canonical_absolute(trial['root'], 'trial.root')
    require(trial['root'] == str(root), 'trial root does not match the owned root')
    require(isinstance(trial['scope'], str) and trial['scope'], 'trial.scope required')

    width = trial['width']
    exact_fields(width, WIDTH_FIELDS, 'trial.width')
    require(width['native_block'] == NATIVE_BLOCK, 'trial must be the native16 width trial')
    require(width['retained_block'] == RETAINED_BLOCK, 'retained block must stay native8')
    canonical_absolute(width['drafter_config'], 'width.drafter_config')
    canonical_absolute(width['retained_drafter_config'], 'width.retained_drafter_config')
    require(width['drafter_config'] != width['retained_drafter_config'],
            'width and retained drafter config paths must differ')
    require(valid_sha(width['drafter_config_sha256']), 'width drafter config hash required')
    require(valid_sha(width['retained_drafter_config_sha256']), 'retained drafter config hash required')
    require(width['drafter_config_sha256'] != width['retained_drafter_config_sha256'],
            'width and retained drafter configs must differ')
    canonical_relative(width['probe_dir'], 'width.probe_dir')
    require(width['events'] == PROBE_EVENTS, 'width.events must be ' + PROBE_EVENTS)
    require(isinstance(width['run_prefix'], str) and width['run_prefix'].endswith('-'),
            'width.run_prefix must be an explicit prefix ending in "-"')
    if width['failed_run'] is not None:
        canonical_relative(width['failed_run'], 'width.failed_run')
        require(Path(width['failed_run']).parts[:1] == (RUNS_DIR,),
                'width.failed_run must live under runs/')
    failed = width['failed_pids']
    exact_fields(failed, {'guard_pid', 'server_pid'}, 'width.failed_pids')
    for key in ('guard_pid', 'server_pid'):
        require(type(failed[key]) is int and failed[key] > 0,
                'width.failed_pids.' + key + ' must be the trusted PID of the aborted width attempt')
    require(failed['guard_pid'] != failed['server_pid'], 'failed width PIDs must be distinct')
    validate_expect(width['expect'])

    env = width['env']
    exact_fields(env, set(DIAG_ENV) | {'ROUND8_WIDTH_OUT'}, 'width.env')
    require(env['ROUND8_WIDTH'] == '1', 'width.env must enable the sealed diagnostic explicitly')
    canonical_absolute(env['ROUND8_WIDTH_OUT'], 'width.env.ROUND8_WIDTH_OUT')
    require(env['ROUND8_WIDTH_OUT'] == str(Path(trial['root']) / width['probe_dir']),
            'width env probe output must equal root/probe_dir')

    protected = trial['protected']
    exact_fields(protected, {'mandatory_roots', 'nesting_acknowledged'}, 'trial.protected')
    require(isinstance(protected['mandatory_roots'], list) and protected['mandatory_roots'],
            'trial.protected.mandatory_roots must be a non-empty list')
    for entry in protected['mandatory_roots']:
        canonical_absolute(entry, 'protected.mandatory_roots entry')
    require(type(protected['nesting_acknowledged']) is bool, 'protected.nesting_acknowledged must be boolean')

    artifacts = trial['artifacts']
    exact_fields(artifacts, {'integration_pins', 'integration_pins_sha256'}, 'trial.artifacts')
    canonical_relative(artifacts['integration_pins'], 'artifacts.integration_pins')
    require(valid_sha(artifacts['integration_pins_sha256']), 'artifacts.integration_pins hash required')
    return trial


def validate_width_config(raw, where, block):
    """Return the (top-level, nested) block sizes after proving they agree."""
    config = strict_json(raw)
    top = config['block_size']
    nested = config['dflash_config']['block_size']
    require(top == nested, where + ': block_size disagree between top level and dflash_config')
    require(top == block, where + ': expected block_size ' + str(block) + ', found ' + str(top))
    require(config['dflash_config']['tap_shift'] == TAP_SHIFT, where + ': tap_shift must be 0')
    require(config['dflash_config']['target_layer_ids'] == EXPECTED_TAPS, where + ': tap set changed')
    return top, nested


def validate_protected_roots(roots, complete, trial, destinations):
    """Fail-closed protected-root inventory plus destination containment.

    `destinations` are the trial's own writable absolute paths (probe dir,
    receipt dir, launcher log dir, new run dir). Every one of them must sit
    OUTSIDE every protected root, otherwise the trial would write into a tree it
    promised not to touch.
    """
    require(complete is True, 'protected-root inventory must be explicitly declared complete')
    require(isinstance(roots, list) and roots, 'at least one protected root is required')
    for entry in roots:
        canonical_absolute(entry, 'protected root')
    require(len(set(roots)) == len(roots), 'duplicate protected root')
    mandatory = set(trial['protected']['mandatory_roots'])
    require(mandatory <= set(roots), 'protected-root inventory is incomplete; missing: '
            + ','.join(sorted(mandatory - set(roots))))
    nested = []
    for outer in roots:
        for inner in roots:
            if outer != inner and _under(inner, outer):
                nested.append((inner, outer))
    if nested and trial['protected']['nesting_acknowledged'] is not True:
        raise Refusal('nested protected roots require protected.nesting_acknowledged: ' + repr(sorted(nested)))
    for destination in destinations:
        canonical_absolute(destination, 'trial destination')
        for outer in roots:
            require(not _under(destination, outer) and destination != outer,
                    'trial destination is inside a protected root: ' + destination + ' in ' + outer)
    return {'roots': list(roots), 'count': len(roots), 'nested': sorted(nested),
            'declared_complete': True}


def _under(path, root):
    path = Path(str(path)).parts
    root = Path(str(root)).parts
    return len(path) > len(root) and path[:len(root)] == root


def require_authorization_token(environ, args):
    """Two explicit, independent switches plus the hashed document later."""
    token = environ.get('EXL3_WIDTH_TRIAL_AUTHORIZED')
    require(token == 'YES', 'EXL3_WIDTH_TRIAL_AUTHORIZED=YES is required for any width-trial stage')
    require(getattr(args, 'authorize_width_trial', False) is True,
            '--authorize-width-trial is required for any width-trial stage')
    return True


def validate_authorization(auth, trial, root, config_sha, receipt_sha, trial_sha, boot_id, uid, now, stage):
    """Trusted-document gate for the trial, mirroring the adapter's discipline."""
    exact_fields(auth, AUTH_FIELDS, 'authorization')
    require(auth['schema'] == 1, 'authorization schema must be 1')
    require(auth['intent'] == TRIAL_INTENT, 'explicit width-trial recovery intent required')
    require(auth['stage'] in STAGES, 'authorization stage must be one of ' + ','.join(STAGES))
    require(auth['stage'] == stage, 'authorization stage does not match the requested stage: '
            + auth['stage'] + ' != ' + stage)
    require(auth['root'] == str(root), 'authorization root mismatch')
    require(auth['boot_id'] == boot_id, 'authorization boot identity mismatch')
    require(auth['uid'] == uid, 'authorization UID mismatch')
    require(auth['ingress_blocked'] is True, 'exclusive request quiescence must be authorized')
    require(auth['config_sha256'] == config_sha, 'authorization/config hash mismatch')
    require(auth['receipt_sha256'] == receipt_sha, 'authorization/receipt hash mismatch')
    require(auth['trial_sha256'] == trial_sha, 'authorization/trial-document hash mismatch')
    for value in (auth['issued_at'], auth['failure_at'], auth['expires_at']):
        require(type(value) in (int, float), 'authorization timestamps must be numeric')
        require(value == value and value not in (float('inf'), float('-inf')), 'nonfinite authorization timestamp')
    require(auth['failure_at'] <= auth['issued_at'] <= now < auth['expires_at'],
            'expired or future-dated width-trial authorization')
    width = auth['width']
    exact_fields(width, WIDTH_OUTCOME_FIELDS, 'authorization.width')
    require(width['phase'] in WIDTH_PHASES, 'unknown width phase')
    require(width['status'] in ('passed', 'failed'), 'unknown width outcome')
    require((width['status'] == 'passed' and width['error'] is None) or
            (width['status'] == 'failed' and isinstance(width['error'], str) and width['error']),
            'width error must accompany a failed width outcome')
    if width['phase'] == 'before_width':
        require(stage == 'release', 'a before_width authorization may only authorize the release stage')
        require(width['status'] == 'failed' and width['error'] == NOT_RUN_ERROR,
                'a before_width authorization must encode the sealed adapter\'s binary schema as '
                'failed/' + NOT_RUN_ERROR)
    else:
        require(stage == 'restore', 'an after_width authorization may only authorize the restore stage')
    require(isinstance(auth['scope'], str) and auth['scope'], 'authorization scope required')
    return auth


def events_records(raw, where):
    """Strict JSONL parse: duplicate keys and non-finite numbers are refusals."""
    records = []
    for number, line in enumerate(raw.decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = strict_json(line)
        except (ValueError, UnicodeError) as exc:
            raise Refusal(where + ': line ' + str(number) + ' is not strict JSON: ' + repr(exc)) from exc
        require(isinstance(record, dict), where + ': line ' + str(number) + ' is not an object')
        require('kind' in record, where + ': line ' + str(number) + ' has no kind')
        require(record['kind'] in KNOWN_KINDS,
                where + ': unknown event kind ' + repr(record['kind']) + ' (a different diagnostic is installed)')
        require(type(record.get('round')) is int and 0 <= record['round'] <= EXPECTED['max_round'],
                where + ': line ' + str(number) + ' has an out-of-range round')
        records.append(record)
    require(records, where + ': no diagnostic records were emitted')
    rounds = [record['round'] for record in records]
    require(rounds == sorted(rounds), where + ': event rounds are not monotone')
    return records


def analyse_probe(read, trial, where='width probe'):
    """Validate the sealed diagnostic's captured event stream for a native16 run.

    `read` is a callable taking a path relative to the owned root. Returns
    {'status', 'error', 'counts', 'missing', 'events_sha256', 'records'}.

    'passed' requires EVERY required kind to be present and self-consistent AND
    the observed devices to show the repaired comparison alignment. Anything
    short of that is 'failed' with a specific, quotable error string.
    """
    width = trial['width']
    relative = str(Path(width['probe_dir']) / width['events'])
    try:
        raw = read(relative)
    except FileNotFoundError:
        return _verdict('failed', 'probe evidence missing: ' + relative, {}, list(REQUIRED_KINDS), None, 0)
    records = events_records(raw, where + ' ' + relative)
    counts = {}
    for record in records:
        counts[record['kind']] = counts.get(record['kind'], 0) + 1
    missing = [kind for kind in REQUIRED_KINDS if kind not in counts]
    if missing:
        return _verdict('failed', 'native16 probe incomplete; missing gates: ' + ','.join(missing),
                        counts, missing, hashlib.sha256(raw).hexdigest(), len(records))
    expect = width['expect']
    try:
        _check_records(records, expect)
    except Refusal as exc:
        return _verdict('failed', str(exc), counts, [], hashlib.sha256(raw).hexdigest(), len(records))
    return _verdict('passed', None, counts, [], hashlib.sha256(raw).hexdigest(), len(records))


def _verdict(status, error, counts, missing, events_sha256, records):
    return {'status': status, 'error': error, 'counts': counts, 'missing': missing,
            'events_sha256': events_sha256, 'records': records, 'events': PROBE_EVENTS}


def _check_records(records, expect):
    installed = [r for r in records if r['kind'] == 'installed']
    require(len(installed) == 1, 'exactly one installed record is required')
    row = installed[0]
    require(row.get('block') == NATIVE_BLOCK, 'installed block must be 16')
    require(row.get('reserve') == DRAFT_RESERVE, 'installed draft reserve must be 15')
    require(row.get('mask_shape') == MASK_SHAPE, 'installed mask shape must be [4096]')
    require(row.get('mask_finite') is True, 'installed mask must be finite')
    require(row.get('tap_shift') == TAP_SHIFT, 'installed tap_shift must be 0')
    require(row.get('taps') == EXPECTED_TAPS, 'installed taps must be the audited five')
    require(row.get('model_vocab') == MODEL_VOCAB, 'installed model vocab mismatch')

    metadata = [r for r in records if r['kind'] == 'input_metadata']
    inputs = [r for r in records if r['kind'] == 'input']
    require(records.index(metadata[0]) < records.index(inputs[0]),
            'input_metadata must be emitted before the first input verdict')
    require(metadata[0].get('input_shape') == expect['input_shape'], 'input_metadata shape mismatch')
    require(metadata[0].get('mask_device') == metadata[0].get('output_device'),
            'repaired comparison operand must be aligned to the producer device')

    for row in inputs:
        require(row.get('anchor_shape') == [1, 1], 'input anchor shape must be [1,1]')
        require(row.get('input_shape') == expect['input_shape'], 'input shape must be [1,16,4096]')
        require(row.get('finite') is True, 'input output must be finite')
        require(row.get('mask_equal') is True, 'learned-mask comparison must be equal')

    states = [r for r in records if r['kind'] == 'state']
    for row in states:
        require(row.get('shape') == expect['state_shape'], 'draft state shape must be [1,16,4096]')
        require(row.get('finite') is True, 'draft state must be finite')

    for row in records:
        if row['kind'] == 'qkv':
            require(row.get('q_shape') == expect['q_shape'], 'qkv q shape mismatch')
            require(row.get('k_shape') == expect['k_shape'], 'qkv k shape mismatch')
            require(row.get('v_shape') == expect['k_shape'], 'qkv v shape mismatch')
            require(all(row.get(key) is True for key in ('finite_q', 'finite_k', 'finite_v')),
                    'qkv buffers must be finite')
        elif row['kind'] == 'native_cache_writes':
            require(row.get('finite_scales') is True, 'native cache scales must be finite')
        elif row['kind'] == 'pristine_draft_head':
            require(row.get('finite_model_vocab') is True, 'pristine draft head must be finite')
        elif row['kind'] == 'native_samples':
            require(row.get('shape') == expect['samples_shape'], 'native sample shape must be [1,16]')
            require(row.get('proposal_count') == expect['proposal_count'],
                    'native sample must expose 15 proposals after the anchor')
            require(row.get('padded_vocab_proposals') == 0,
                    'padded-vocabulary proposals must be reported and must be zero here')
        elif row['kind'] == 'assigned_pages':
            require(row.get('end_exclusive', 0) - row.get('start', 0) == expect['assigned_span'],
                    'assigned-page window must span the native block')
            require(len(row.get('positions') or []) == expect['assigned_span'],
                    'assigned positions must cover the native block')
    require(any(r['kind'] == 'target_forward' for r in records),
            'target verification must have run at least once')
    require(any(r['kind'] == 'terminal' for r in records),
            'the run must terminate normally, not on a diagnostic exception')


def validate_receipt_document(receipt, auth):
    """Shape-only check of the pre-failure receipt (the adapter validates scopes)."""
    exact_fields(receipt, {'schema', 'phase', 'captured_at', 'boot_id', 'uid', 'owner', 'scopes',
                           'launch_sha256', 'live_config_sha256'}, 'pre-failure receipt')
    require(receipt['schema'] == 1, 'receipt schema must be 1')
    require(receipt['phase'] == 'pre_failure', 'a trusted PRE-FAILURE receipt is required')
    require(receipt['boot_id'] == auth['boot_id'], 'receipt boot identity mismatch')
    require(receipt['uid'] == auth['uid'], 'receipt UID mismatch')
    require(type(receipt['captured_at']) in (int, float) and receipt['captured_at'] < auth['failure_at'],
            'receipt must predate the recorded failure')
    return receipt
