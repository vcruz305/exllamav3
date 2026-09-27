"""Post-restore readiness/readback in the shape used by the round9 live check.

Read-only. No CUDA initialisation, no inference, no writes.

The same code path runs against the real host and against the local inert
fixture: `HostReadback` talks to /proc, the filesystem and one loopback HTTP
endpoint, and every expectation is a parameter. Checks are ordered cheapest
first and the first failure is a refusal with a specific message, so a partial
"looks restored" claim is not representable.
"""
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

import linux_adapter as a
import trial_contract as tc

require = a.require
Refusal = a.Refusal
strict_json = a.strict_json


class HostReadback:
    """Read-only accessors over the restored host state."""

    def __init__(self, root, port, health_path='/health'):
        self.root = Path(root)
        self.port = port
        self.health_path = health_path

    def relative(self, name):
        return str(self.root / name)

    def read_bytes(self, path):
        return Path(path).read_bytes()

    def read_text(self, path):
        return Path(path).read_text()

    def read_json(self, path):
        return strict_json(self.read_bytes(path))

    def exists(self, path):
        return Path(path).exists()

    def sha256(self, path):
        handle = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                handle.update(chunk)
        return handle.hexdigest()

    def starttime(self, pid):
        return Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()[19]

    def argv(self, pid):
        raw = Path('/proc', str(pid), 'cmdline').read_bytes()
        return raw.decode().strip('\0').split('\0')

    def environ(self, pid):
        raw = Path('/proc', str(pid), 'environ').read_bytes()
        return dict(item.split('=', 1) for item in raw.decode().split('\0') if '=' in item)

    def alive(self, pid):
        return Path('/proc', str(pid), 'cmdline').exists() and \
            bool(Path('/proc', str(pid), 'cmdline').read_bytes())

    def health(self):
        url = 'http://127.0.0.1:' + str(self.port) + self.health_path
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                return response.status, strict_json(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, strict_json(exc.read())

    def proc_inventory(self, root=None):
        """Processes whose argv names `root` itself or a path under it.

        Matching is on exact argument equality or a `root + '/'` directory
        prefix, never a bare substring, so a sibling directory with a shared
        prefix cannot be mistaken for a process inside the runtime root.
        """
        root = str(self.root if root is None else root)
        prefix = root.rstrip('/') + '/'
        found = []
        for entry in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = entry.read_bytes().decode(errors='replace').strip('\0').split('\0')
            except OSError:
                continue
            if int(entry.parent.name) == os.getpid():
                continue
            if any(item == root or item.startswith(prefix) for item in args):
                found.append({'pid': int(entry.parent.name), 'args': args})
        return found

    def oom_and_memory(self):
        mem = {line.split(':')[0]: int(line.split()[1])
               for line in Path('/proc/meminfo').read_text().splitlines()
               if ':' in line and line.split()[1].isdigit()}
        vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines() if len(line.split()) == 2)
        events = {}
        cgroup = Path('/sys/fs/cgroup/memory.events')
        if cgroup.exists():
            events = {key: int(value) for key, value in
                      (line.split() for line in cgroup.read_text().splitlines() if len(line.split()) == 2)}
        return {'meminfo': mem, 'vmstat': vm, 'cgroup_events': events,
                'mem_available_gib': mem.get('MemAvailable', 0) / 2**20,
                'mem_free_gib': mem.get('MemFree', 0) / 2**20}


def readiness_report(readback, trial, cfg, restored, prior, protected_roots,
                     protected_snapshot, min_available_gib=8.0):
    """Verify the restored retained serve and return a readback receipt.

    `restored`  = {'run': str, 'pids': {...}} as recorded by the retained launch.
    `prior`     = {'run': str, 'pids': {...}} the width run that was stopped.
    `protected_snapshot` = {root: {'mtime_ns':..., 'entries': n}} from the preflight.
    """
    report = {'status': 'passed', 'scope': ('Read-only process/filesystem/health readback; no CUDA '
                                           'initialisation, no inference, no writes.'),
              'checks': {}}
    width = trial['width']

    # R1 active pointer
    pointer = readback.read_text(readback.relative(cfg['active'])).strip()
    require(pointer == restored['run'],
            'R1 active pointer is ' + pointer + ', expected the restored run ' + restored['run'])
    report['checks']['R1_active_pointer'] = pointer

    # R2 PID record
    record = readback.read_json(readback.relative(restored['run'] + '/pid.json'))
    require(record == restored['pids'], 'R2 restored PID record changed: ' + repr(record))
    report['checks']['R2_pid_record'] = record

    # R3 exact identity of both restored processes
    for role in ('guard_pid', 'server_pid'):
        pid = restored['pids'][role]
        recorded = restored.get('starttimes', {}).get(role)
        if recorded is not None:
            require(readback.starttime(pid) == recorded, 'R3 ' + role + ' starttime changed')
    require(readback.argv(restored['pids']['server_pid']) == restored['pids']['command'],
            'R3 restored server argv changed')
    report['checks']['R3_identity'] = {'guard_pid': restored['pids']['guard_pid'],
                                       'server_pid': restored['pids']['server_pid']}

    # R4 the prior (width) run identities must be gone
    for role in ('guard_pid', 'server_pid'):
        pid = prior['pids'][role]
        require(not readback.alive(pid), 'R4 pre-restore ' + role + ' is still alive: ' + str(pid))
    report['checks']['R4_prior_gone'] = prior['run']

    # R5 authenticated health
    status, body = readback.health()
    require(status == 200 and body.get('healthy') is True and body.get('requests') == 0,
            'R5 restored health is not HTTP200/healthy/idle: ' + repr((status, body)))
    if 'max_active_requests' in body:
        require(body['max_active_requests'] == 1, 'R5 max_active_requests changed: ' + repr(body))
    report['checks']['R5_health'] = {'status': status, 'body': body}

    # R6 retained configuration binding
    launch = readback.read_json(readback.relative(restored['run'] + '/launch.json'))
    live = readback.read_json(readback.relative(cfg['live_config']))
    require(launch == live, 'R6 launch.json diverged from the live configuration')
    require(launch.get('memory_policy') == 'uma' and launch.get('max_seconds') == 43200,
            'R6 guard policy changed on the restored run')
    require(launch.get('outdir') == restored['run'], 'R6 restored outdir mismatch')
    report['checks']['R6_configuration'] = {'memory_policy': launch['memory_policy'],
                                            'max_seconds': launch['max_seconds']}

    # R7 the restored serve is native8 again, and the native16 config is untouched
    retained_raw = readback.read_bytes(width['retained_drafter_config'])
    require(readback.sha256(width['retained_drafter_config']) == width['retained_drafter_config_sha256'],
            'R7 retained drafter config hash changed')
    tc.validate_width_config(retained_raw, 'R7 retained drafter config', tc.RETAINED_BLOCK)
    width_raw = readback.read_bytes(width['drafter_config'])
    require(readback.sha256(width['drafter_config']) == width['drafter_config_sha256'],
            'R7 width drafter config hash changed')
    tc.validate_width_config(width_raw, 'R7 width drafter config', tc.NATIVE_BLOCK)
    report['checks']['R7_drafter_configs'] = {'retained_block': tc.RETAINED_BLOCK,
                                              'width_block': tc.NATIVE_BLOCK}

    # R8 the width diagnostic must not be installed in the restored serve
    server_env = readback.environ(restored['pids']['server_pid'])
    for key in ('ROUND8_WIDTH', 'ROUND8_WIDTH_OUT'):
        require(not server_env.get(key), 'R8 width diagnostic env leaked into the restored serve: ' + key)
    for key, value in (cfg['retained'].get('env') or {}).items():
        require(server_env.get(key) == value, 'R8 restored environment mismatch: ' + key)
    report['checks']['R8_no_width_env'] = {'inherited_keys': sorted(cfg['retained'].get('env') or {})}

    # R9 pinned runtime identities
    pins = {}
    for row in cfg['identities']:
        require(readback.sha256(row['path']) == row['sha256'],
                'R9 identity hash changed after restore: ' + row['role'])
        pins[row['role']] = row['sha256']
    report['checks']['R9_identities'] = pins

    # R10 process inventory: only the owned pair under the runtime root
    inventory = readback.proc_inventory(str(readback.root))
    owned = {restored['pids']['guard_pid'], restored['pids']['server_pid']}
    if {row['pid'] for row in inventory} != owned:
        detail = []
        for row in inventory:
            extra = {'pid': row['pid'], 'args': row['args']}
            try:
                stat = Path('/proc', str(row['pid']), 'stat').read_text().rsplit(') ', 1)[1].split()
                extra.update({'state': stat[0], 'ppid': int(stat[1]), 'starttime': stat[19]})
            except OSError as exc:
                extra['stat_error'] = repr(exc)
            detail.append(extra)
        raise Refusal('R10 unexpected process inventory under the runtime root (owned '
                      + repr(sorted(owned)) + '): ' + repr(detail))
    report['checks']['R10_process_inventory'] = {
        'owned_pids': sorted(owned), 'observed': sorted(row['pid'] for row in inventory),
        'root': str(readback.root)}

    # R11 OOM counters and headroom
    memory = readback.oom_and_memory()
    for key in ('host_oom_kill', 'cgroup_oom', 'cgroup_oom_kill'):
        if key == 'host_oom_kill':
            require(int(memory['vmstat'].get('oom_kill', 0)) == 0, 'R11 host OOM kill counter is non-zero')
        else:
            require(int(memory['cgroup_events'].get(key.replace('cgroup_', ''), 0)) == 0,
                    'R11 cgroup OOM counter is non-zero: ' + key)
    require(memory['mem_available_gib'] >= min_available_gib,
            'R11 MemAvailable below the retention floor: ' + repr(memory['mem_available_gib']))
    report['checks']['R11_memory'] = {'mem_available_gib': memory['mem_available_gib'],
                                      'mem_free_gib': memory['mem_free_gib'],
                                      'cgroup_events': memory['cgroup_events'],
                                      'vmstat_oom_kill': int(memory['vmstat'].get('oom_kill', 0))}

    # R12 protected roots still present, untouched at the entry/mtime level
    roots = {}
    for root in protected_roots:
        before = protected_snapshot.get(root)
        require(before is not None, 'R12 protected root was not snapshotted during preflight: ' + root)
        path = Path(root)
        require(path.is_dir(), 'R12 protected root vanished: ' + root)
        after = {'mtime_ns': path.stat().st_mtime_ns, 'entries': len(list(path.iterdir()))}
        require(after == before, 'R12 protected root changed at the entry level: ' + root
                + ' before=' + repr(before) + ' after=' + repr(after))
        roots[root] = after
    report['checks']['R12_protected_roots'] = roots
    return report


def snapshot_protected_roots(protected_roots):
    """Preflight-side entry/mtime snapshot used by R12.

    This is a shallow containment check (existence, entry count, directory
    mtime), NOT a recursive content hash. Full content verification of the
    runtime roots stays a device gate: see GAPS.md G-PROTECTED-CONTENT-HASH.
    """
    snapshot = {}
    for root in protected_roots:
        path = Path(root)
        require(path.is_dir(), 'protected root must be an existing directory: ' + root)
        snapshot[root] = {'mtime_ns': path.stat().st_mtime_ns, 'entries': len(list(path.iterdir()))}
    return snapshot
