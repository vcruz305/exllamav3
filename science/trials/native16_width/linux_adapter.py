"""Fail-closed Linux recovery boundaries; no import-time I/O or model imports."""
import os
from pathlib import Path
import contextlib
import fcntl
import stat
import io
import signal
import socket
import time
import urllib.request
import urllib.error
import hashlib
import json
import math
import types
import builtins

CORE_HASHES = {
    'recover.py': 'ab2bd9a4e646c0852bc3aa3507014db516356ba0da49beec0a47ca4ab0420645',
    'width_controller.py': 'c80b438e78b6a917580dee55841d300cb06b209e413f5e246a919de9241b3d7f',
}
SOURCE_PIN = 'ca4a880e8918e1985fd25e06c6aff561666d3f14'
PRODUCTION_ROOT = '/workspace/mimo-tune'
PRODUCTION_PINS = {
    'guard': ('/workspace/mimo-tune/guard_uma.py', 'eef5706b327cf20007dd49087b6d70b97ec178fcdbde616374188a0bc916d24d'),
    'launcher': ('/workspace/mimo-tune/round6-quant-readback/start_quant_readback.py', '7441d4e0de981462c03f05cca38be81ea3bb7c5965d4b83cd96e582d9147b515'),
    'server': ('/workspace/mimo-tune/round6-quant-readback/serve_native.py', '205ee1c08d7a7763522aa7b269eddeb6a8ebf5ae2674532d0c6d62ce89cb2d83'),
    'source': ('/workspace/mimo-tune/60tps-round3/python-shadow/exllamav3/modules/block_sparse_mlp.py', '61653605b432350db3cccddbaa00b12f121e38fa2b072fc9de2032adf9e2a706'),
    'dso': ('/workspace/mimo-tune/60tps-round3/python-shadow/exllamav3_ext.cpython-312-aarch64-linux-gnu.so', '02b0ae5bc8414d335facca41083f561cf24f41d56ef6e1e73a8b41fb16f80207'),
}


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_production(cfg):
    """No local/inert switches exposed by the CLI; no candidate launcher option."""
    try:
        fields = {'schema', 'source_pin', 'root', 'operator_lock', 'active', 'live_config', 'server_lock', 'guard_argv',
                  'launcher_argv', 'launcher_env', 'retained', 'identities', 'port', 'new_run_prefix', 'receipt_dir', 'timeouts'}
        require(set(cfg) == fields and cfg['schema'] == 1 and cfg['source_pin'] == SOURCE_PIN, 'production schema/source pin mismatch')
        require(cfg['root'] == PRODUCTION_ROOT and cfg['port'] == 8096, 'production root/health target mismatch')
        for key, value in [('active', 'france-active-run.txt'), ('live_config', 'france-uma.json'), ('server_lock', 'server.lock'),
                           ('new_run_prefix', 'france-round6-quant-readback-')]:
            require(cfg[key] == value, 'production path binding mismatch: ' + key)
        require(cfg['launcher_argv'] == ['/usr/bin/python3', PRODUCTION_PINS['launcher'][0]], 'retained launcher argv mismatch')
        require(cfg['guard_argv'] == ['/usr/bin/python3', PRODUCTION_PINS['guard'][0], PRODUCTION_ROOT + '/france-uma.json'], 'guard argv mismatch')
        witness = Path(__file__).resolve().parent / 'witness/retained_launch.json'
        raw = witness.read_bytes()
        require(digest(raw) == 'a9bbcf2ff1cefd796abb241a876b560a63b35e5304f12078423bb80d711da32a', 'retained config witness changed')
        expected = strict_json(raw); expected.pop('outdir')
        require(cfg['retained'] == expected, 'retained argv/environment/guard policy changed')
        identities = {r['role']: r for r in cfg['identities']}
        require(len(identities) == len(cfg['identities']) and
                set(identities) == set(PRODUCTION_PINS) | {'interpreter', 'target_config', 'drafter_config', 'source_manifest'}, 'production identity roles missing/unknown')
        for row in identities.values():
            require(set(row) == {'role', 'path', 'sha256'} and valid_sha(row['sha256']), 'explicit trusted identity hash required')
            require(Path(row['path']).is_absolute() and str(Path(row['path'])) == row['path'] and '..' not in Path(row['path']).parts,
                    'noncanonical identity path')
        for role, (path, sha) in PRODUCTION_PINS.items():
            require(identities[role]['path'] == path and identities[role]['sha256'] == sha, 'production pinned identity mismatch: ' + role)
        command = expected['command']
        for role, flag in [('target_config', '-m'), ('drafter_config', '-dm')]:
            require(identities[role]['path'] == command[command.index(flag)+1] + '/config.json', 'model config path mismatch')
        env = cfg['launcher_env']
        require(isinstance(env, dict) and {'PATH', 'HOME', 'LC_ALL', 'PYTHONUNBUFFERED'} <= env.keys() and
                set(env) <= {'PATH', 'HOME', 'LC_ALL', 'LANG', 'PYTHONUNBUFFERED', 'TMPDIR'}, 'unsafe/unbounded inherited launcher environment')
        require(env['PATH'] == '/usr/bin:/bin' and env['LC_ALL'] == 'C.UTF-8' and env['PYTHONUNBUFFERED'] == '1' and
                all(isinstance(v, str) and v and '\0' not in v and 'REQUIRED_' not in v for v in env.values()), 'explicit clean launcher environment required')
        require(Path(env['HOME']).is_absolute(), 'explicit operator HOME required')
        for key in ('operator_lock', 'receipt_dir'):
            p = Path(cfg[key])
            require(not p.is_absolute() and str(p) == cfg[key] and '..' not in p.parts and p.parts and 'REQUIRED_' not in str(p),
                    'explicit existing owner path required: ' + key)
        require(cfg['operator_lock'] not in (cfg['server_lock'], cfg['active'], cfg['live_config']) and
                cfg['receipt_dir'] not in ('runs', '.'), 'unsafe ownership/output path')
        return cfg
    except (KeyError, TypeError, ValueError) as e:
        raise Refusal('incomplete production configuration: ' + str(e)) from e



def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise Refusal('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def load_core():
    """Validate exact source bytes before compile; no path-based recover import."""
    directory = Path(__file__).resolve().parent / 'core'
    modules = {}
    original_import = builtins.__import__
    def pinned_import(name, *args, **kwargs):
        if name == 'recover':
            return modules['recover']
        return original_import(name, *args, **kwargs)
    for name, expected in CORE_HASHES.items():
        raw = (directory / name).read_bytes()
        require(digest(raw) == expected, 'core source pin mismatch: ' + name)
        module = types.ModuleType(name[:-3])
        module.__dict__['__builtins__'] = dict(vars(builtins), __import__=pinned_import)
        exec(compile(raw, str(directory / name), 'exec'), module.__dict__)
        modules[name[:-3]] = module
    return modules['width_controller']


def authorize(cfg, receipt, auth, config_sha, receipt_sha):
    """Trust is supplied by the operator, not reconstructed from current PIDs."""
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        require(receipt['schema'] == auth['schema'] == 1, 'unknown receipt schema')
        require(receipt['phase'] == 'pre_failure', 'missing trusted PRE-FAILURE receipt')
        require(receipt['boot_id'] == auth['boot_id'] == boot, 'boot identity mismatch')
        require(type(receipt['uid']) is int and receipt['uid'] == auth['uid'] == os.getuid(), 'UID mismatch')
        require(auth['intent'] == 'stop_owned_then_launch_retained_once', 'explicit recovery authorization required')
        require(auth['root'] == cfg['root'] and auth['operator_lock'] == cfg['operator_lock'], 'ownership scope mismatch')
        require(auth['ingress_blocked'] is True, 'exclusive request quiescence not authorized')
        require(auth['config_sha256'] == config_sha and auth['receipt_sha256'] == receipt_sha, 'trusted receipt/config hash mismatch')
        for value in (receipt['captured_at'], auth['failure_at'], auth['issued_at'], auth['expires_at']):
            require(type(value) in (int, float) and math.isfinite(value), 'invalid authorization timestamp')
        require(receipt['captured_at'] < auth['failure_at'] <= auth['issued_at'] <= time.time() < auth['expires_at'],
                'expired, future or post-failure receipt')
        width = auth['width']
        require(width['status'] in ('passed', 'failed'), 'unknown width outcome')
        require((width['status'] == 'passed' and width['error'] is None) or
                (width['status'] == 'failed' and isinstance(width['error'], str) and width['error']), 'width error missing')
        owner = receipt['owner']
        require(set(owner) == {'run', 'pids', 'processes'}, 'invalid owner fields')
        root = Path(cfg['root'])
        run = Path(owner['run'])
        require(run.parent == root / 'runs' and str(run) == owner['run'] and run.name not in ('.', '..'), 'run outside owned root')
        pids = owner['pids']
        require(set(pids) == {'guard_pid', 'server_pid', 'command'}, 'invalid PID record')
        require(len(owner['processes']) == 2 and pids['guard_pid'] != pids['server_pid'], 'invalid owned process count')
        for row, pid in zip(owner['processes'], (pids['guard_pid'], pids['server_pid'])):
            require(set(row) == {'pid', 'starttime', 'args'} and type(pid) is int and pid > 0 and row['pid'] == pid,
                    'invalid owned process identity')
            require(isinstance(row['starttime'], str) and row['starttime'].isdigit() and int(row['starttime']) > 0,
                    'invalid trusted starttime')
            require(isinstance(row['args'], list) and row['args'] and all(isinstance(v, str) and v and '\0' not in v for v in row['args']),
                    'invalid trusted argv')
            require(receipt['scopes'][str(pid)]['uid'] == os.getuid(), 'untrusted process scope')
        require(owner['processes'][1]['args'] == pids['command'], 'receipt server argv mismatch')
        for name in ('launch_sha256', 'live_config_sha256'):
            require(isinstance(receipt[name], str) and len(receipt[name]) == 64 and all(c in '0123456789abcdef' for c in receipt[name]),
                    'invalid config hash')
        return owner
    except (KeyError, TypeError, ValueError) as e:
        raise Refusal('incomplete trusted authorization/receipt: ' + str(e)) from e



def listener_inode(row, port):
    require(type(port) is int and 0 < port < 65536, 'invalid listener port')
    current = proc_snapshot(row['pid'])
    require(current is not None and current[0] == row, 'listener process identity changed')
    require(current[1]['netns'] == os.readlink('/proc/self/ns/net'), 'listener network namespace mismatch')
    matches = []
    for name in ('tcp', 'tcp6'):
        for line in Path('/proc/net', name).read_text().splitlines()[1:]:
            fields = line.split()
            address, raw_port = fields[1].split(':')
            if int(raw_port, 16) == port and fields[3] == '0A':
                # Reject wildcard/IPv6 listeners and reuse-port multiple sockets.
                require(name == 'tcp' and address == '0100007F', 'non-loopback/ambiguous listener')
                matches.append(fields[9])
    require(len(matches) == 1, 'missing or ambiguous listener')
    sockets = set()
    try:
        for entry in Path('/proc', str(row['pid']), 'fd').iterdir():
            try:
                sockets.add(os.readlink(entry))
            except FileNotFoundError:
                continue  # unrelated fd closed; required listener must still be found
    except OSError as e:
        raise Refusal('listener ownership unreadable') from e
    require('socket:[' + matches[0] + ']' in sockets, 'listener not owned by exact server PID')
    require(proc_snapshot(row['pid'])[0] == row, 'listener identity raced')
    return matches[0]


@contextlib.contextmanager
def wall_deadline(seconds):
    """Main-thread total deadline, including HTTP headers/slow-drip bodies."""
    require(0 < seconds <= 600, 'invalid deadline')
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'existing alarm: cannot install bounded read')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(*_):
        raise TimeoutError('bounded operation deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch_health(row, port, timeout):
    """Return buffered urllib response/HTTPError only from exact owned listener."""
    url = f'http://127.0.0.1:{port}/health'
    with wall_deadline(timeout):
        inode = listener_inode(row, port)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            response = opener.open(url, timeout=timeout)
        except urllib.error.HTTPError as e:
            response = e
        with response:
            status = response.code
            raw = response.read(65537)
            require(len(raw) <= 65536, 'health body oversized')
        require(listener_inode(row, port) == inode, 'listener changed during HTTP read')
        # Preserve core malformed-body diagnostics, but reject duplicate keys
        # and nonfinite JSON before its intentionally small json.loads boundary.
        try:
            strict_json(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    buffered = io.BytesIO(raw)
    if status != 200:
        raise urllib.error.HTTPError(url, status, 'owned health response', {}, buffered)
    buffered.status = status
    return buffered




def check_permissions(st):
    require(st.st_uid in (0, os.getuid()), 'untrusted file owner')
    require(not (st.st_mode & 0o022), 'group/other writable path')


def fingerprint(st):
    return (st.st_dev, st.st_ino)


def absolute_dir(path):
    path = str(path)
    require(path.startswith('/') and str(Path(path)) == path and '..' not in Path(path).parts,
            'noncanonical absolute directory')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        check_permissions(os.fstat(fd))
        for part in Path(path).parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
            check_permissions(os.fstat(fd))
        return fd
    except BaseException:
        os.close(fd)
        raise


class OwnedRoot:
    """FD-relative I/O, O_NOFOLLOW at every component, exclusive new writes.

    The operator owns the root and trusts other same-UID code. No untrusted
    group/other writers are allowed. The filesystem itself must be trusted.
    """
    def __init__(self, root):
        self.root = str(root)
        self.fd = absolute_dir(root)
        self.identity = fingerprint(os.fstat(self.fd))
        self.pins = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.close(self.fd)

    def parts(self, name):
        p = Path(name)
        require(not p.is_absolute() and str(p) == name and all(x not in ('..', '.') for x in p.parts)
                and bool(p.parts), 'unsafe relative path')
        return p.parts

    def directory(self, parts):
        live = absolute_dir(self.root)
        try:
            require(fingerprint(os.fstat(live)) == self.identity, 'root directory identity changed')
        finally:
            os.close(live)
        fd = os.dup(self.fd)
        try:
            for i, name in enumerate(parts):
                new = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = new
                check_permissions(os.fstat(fd))
                key = '/'.join(parts[:i + 1])
                if key in self.pins:
                    require(fingerprint(os.fstat(fd)) == self.pins[key], 'directory identity changed: ' + key)
            return fd
        except BaseException:
            os.close(fd)
            raise

    def pin(self, name):
        fd = self.directory(self.parts(name))
        try:
            self.pins[name] = fingerprint(os.fstat(fd))
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def open(self, name, flags=os.O_RDONLY):
        parts = self.parts(name)
        parent = self.directory(parts[:-1])
        try:
            fd = os.open(parts[-1], flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent)
            try:
                st = os.fstat(fd)
                require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'not a single-link regular file')
                check_permissions(st)
                yield fd
                require(fingerprint(os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)) == fingerprint(st),
                        'file identity changed')
            finally:
                os.close(fd)
        finally:
            os.close(parent)

    def read(self, name, limit=4 * 1024 * 1024):
        with self.open(name) as fd:
            before = os.fstat(fd)
            chunks = []
            size = 0
            while True:
                data = os.read(fd, min(65536, limit + 1 - size))
                if not data:
                    break
                chunks.append(data)
                size += len(data)
                require(size <= limit, 'file oversized')
            after = os.fstat(fd)
            require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'file changed while reading')
            return b''.join(chunks)

    def create(self, name, raw):
        with self.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL) as fd:
            view = memoryview(raw)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        require(self.read(name) == raw, 'write/readback mismatch')

    @contextlib.contextmanager
    def lock(self, name):
        with self.open(name, os.O_RDWR | os.O_CREAT) as fd:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield fd
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)


class Refusal(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise Refusal(message)


def proc_snapshot(pid):
    """None means ENOENT on the PID directory, never zombie/permission failure.

    pidfd + double stat bind cmdline/environ/scope reads to one incarnation.
    No signals are sent. Unknown/transition states fail closed.
    """
    require(type(pid) is int and pid > 0, 'invalid PID')
    p = Path('/proc') / str(pid)
    try:
        directory = os.open(p, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except FileNotFoundError:
        require(Path('/proc/self/stat').is_file(), 'procfs unavailable')
        # hidepid or another visibility boundary can return ENOENT for a live
        # PID. Kernel ESRCH from pidfd_open is required as an independent proof.
        try:
            probe = os.pidfd_open(pid)
        except ProcessLookupError:
            return None
        except OSError as e:
            raise Refusal('process absence unproved: ' + str(e)) from e
        else:
            os.close(probe)
            raise Refusal('process absence unproved: hidden or raced live PID')
    try:
        pidfd = os.pidfd_open(pid)
        try:
            def read(name):
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
                with os.fdopen(fd, 'rb') as f:
                    raw = f.read(4 * 1024 * 1024 + 1)
                require(len(raw) <= 4 * 1024 * 1024, 'proc record oversized')
                return raw
            def stat():
                s = read('stat').decode().rsplit(') ', 1)[1].split()
                require(s[0] != 'Z', 'zombie is not proven gone')
                require(s[0] in ('R', 'S', 'D', 'I'), 'unknown/stopped process state')
                return s
            before = stat()
            argv = read('cmdline')
            require(argv.endswith(b'\0') and len(argv) > 1, 'empty/unknown argv')
            args = argv[:-1].decode().split('\0')
            raw_env = read('environ')
            env = {}
            for item in raw_env.rstrip(b'\0').split(b'\0'):
                if not item:
                    continue
                k, v = item.decode().split('=', 1)
                require(k not in env, 'duplicate environment key')
                env[k] = v
            status = read('status').decode()
            uids = next(l.split()[1:] for l in status.splitlines() if l.startswith('Uid:'))
            require(len(set(uids)) == 1, 'mixed process UIDs')
            scope = {'uid': int(uids[0]), 'ppid': int(before[1]),
                     'pgid': int(before[2]), 'sid': int(before[3]),
                     'cgroup': read('cgroup').decode(),
                     'cgroupns': os.readlink(p / 'ns/cgroup'),
                     'netns': os.readlink(p / 'ns/net'),
                     'mntns': os.readlink(p / 'ns/mnt'),
                     'exe': os.readlink(p / 'exe'), 'environment': env}
            after = stat()
            require(before[19] == after[19] and before[1:4] == after[1:4], 'process identity raced')
            return {'pid': pid, 'starttime': before[19], 'args': args}, scope
        finally:
            os.close(pidfd)
    except (OSError, ValueError, IndexError, StopIteration) as e:
        raise Refusal('process unreadable/ambiguous: ' + str(e)) from e
    finally:
        os.close(directory)


def read_absolute(path, limit=4 * 1024 * 1024):
    path = Path(path)
    with OwnedRoot(str(path.parent)) as fs:
        return fs.read(path.name, limit)


def file_digest(path):
    path = Path(path)
    with OwnedRoot(str(path.parent)) as fs, fs.open(path.name) as fd:
        before = os.fstat(fd)
        h = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        after = os.fstat(fd)
        require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'identity file changed during hash')
        return h.hexdigest()


def assert_no_unowned_members(pids):
    owned = {pids['guard_pid'], pids['server_pid']}
    with wall_deadline(5):
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
            except FileNotFoundError:
                continue
            parent, group, session = map(int, fields[1:4])
            if parent in owned or group in owned or session in owned:
                require(int(entry.name) in owned, 'unowned member in guarded process scope; refuse group cleanup')


class LinuxIO:
    """Concrete core boundary. CLI additionally enforces the fixed production profile.

    One operator flock is held across the entire transaction. The unchanged
    guard separately owns server.lock. No signals or process groups are created
    by this adapter; only the exact retained launcher may spawn a new guard.
    """
    def __init__(self, cfg, receipt, auth, config_sha, receipt_sha):
        self.cfg, self.receipt, self.auth = cfg, receipt, auth
        self.config_sha, self.receipt_sha = config_sha, receipt_sha
        self.expected = authorize(cfg, receipt, auth, config_sha, receipt_sha)
        self.core = load_core()
        self.root = Path(cfg['root'])
        self.old = self.relative_run(self.expected['run'])
        self.launched = False
        self.bound_launch = None
        self.stack = contextlib.ExitStack()
        self.lockfd = None
        require(cfg['source_pin'] == SOURCE_PIN and cfg['schema'] == 1, 'source/config pin mismatch')
        require(cfg['operator_lock'] != cfg['server_lock'], 'operator and guard locks must be distinct')
        required_roles = {'guard', 'launcher', 'server', 'source', 'dso', 'interpreter', 'target_config', 'drafter_config'}
        self.identities = {r['role']: r for r in cfg['identities']}
        require(len(self.identities) == len(cfg['identities']) and required_roles <= self.identities.keys(), 'missing/duplicate identity role')
        for key, maximum in [('stop', 120), ('headroom', 120), ('launch', 120), ('ready', 500), ('http', 5)]:
            t = cfg['timeouts'][key]
            require(type(t) in (float, int) and math.isfinite(t) and 0 < t <= maximum, 'invalid bounded timeout: ' + key)

    def __enter__(self):
        try:
            self.fs = self.stack.enter_context(OwnedRoot(str(self.root)))
            self.fs.pin('runs')
            self.fs.pin(self.old)
            self.fs.pin(self.cfg['receipt_dir'])
            self.lockfd = self.stack.enter_context(self.fs.lock(self.cfg['operator_lock']))
            self.check_authority()
            self.validate_old()
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        self.stack.close()
        self.lockfd = None

    def relative_run(self, run):
        p = Path(run)
        require(str(p) == run and p.parent == self.root / 'runs' and p.name not in ('.', '..'), 'run outside owned root')
        return 'runs/' + p.name

    def check_authority(self):
        require(self.lockfd is not None, 'operator lock not held')
        authorize(self.cfg, self.receipt, self.auth, self.config_sha, self.receipt_sha)
        with self.fs.open(self.cfg['operator_lock']) as fd:
            require(fingerprint(os.fstat(fd)) == fingerprint(os.fstat(self.lockfd)), 'operator lock inode changed')

    def active(self):
        raw = self.fs.read(self.cfg['active']).decode()
        run = raw.removesuffix('\n')
        self.relative_run(run)
        return run

    def pids(self, run):
        return strict_json(self.fs.read(self.relative_run(run) + '/pid.json'))

    def process(self, pid):
        expected = next((r for r in self.expected['processes'] if r['pid'] == pid), None)
        require(expected is not None, 'unowned PID lookup')
        sample = proc_snapshot(pid)
        if sample is None:
            return None
        row, scope = sample
        require(row == expected and scope == self.receipt['scopes'][str(pid)], 'process identity/scope/environment changed')
        return row

    def exists(self, relative):
        try:
            self.fs.read(relative)
            return True
        except FileNotFoundError:
            return False

    def verify_identities(self):
        with wall_deadline(60):
            for item in self.cfg['identities']:
                require(file_digest(item['path']) == item['sha256'], 'source/config/DSO hash changed: ' + item['role'])
            if 'source_manifest' in self.identities:
                manifest = strict_json(read_absolute(self.identities['source_manifest']['path']))
                require(manifest['pin'] == SOURCE_PIN and bool(manifest['files']), 'source manifest pin/coverage missing')
                for path, expected in manifest['files'].items():
                    require(file_digest(path) == expected, 'source manifest file changed: ' + path)

    def assert_server_lock(self, pid):
        with self.fs.open(self.cfg['server_lock']) as fd:
            st = os.fstat(fd)
            matches = []
            for line in Path('/proc/locks').read_text().splitlines():
                f = line.split()
                if len(f) < 8 or f[1] == '->':
                    continue
                dev = f[5].split(':')
                if len(dev) == 3 and (int(dev[0], 16), int(dev[1], 16), int(dev[2])) == (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino):
                    matches.append(f)
            require(len(matches) == 1 and matches[0][1] in ('FLOCK', 'POSIX') and
                    matches[0][2:4] == ['ADVISORY', 'WRITE'] and int(matches[0][4]) == pid and matches[0][6:8] == ['0', 'EOF'],
                    'guard server.lock ownership unproved')
        # DrvFS translates flock into a whole-file POSIX lock. Never accept its
        # label alone: prove a separate flock description actually conflicts.
        try:
            with self.fs.lock(self.cfg['server_lock']):
                raise Refusal('guard server.lock is not exclusively held')
        except BlockingIOError:
            pass

    def validate_old(self):
        self.check_authority()
        require(self.active() == self.expected['run'] and self.pids(self.expected['run']) == self.expected['pids'], 'active/PID record changed')
        require(digest(self.fs.read(self.old + '/launch.json')) == self.receipt['launch_sha256'], 'old launch config changed')
        require(digest(self.fs.read(self.cfg['live_config'])) == self.receipt['live_config_sha256'], 'live config changed')
        oldcfg = strict_json(self.fs.read(self.old + '/launch.json'))
        require(oldcfg == strict_json(self.fs.read(self.cfg['live_config'])) and oldcfg['outdir'] == self.expected['run'], 'old config binding mismatch')
        require(oldcfg['command'] == self.expected['pids']['command'] and oldcfg['memory_policy'] == 'uma' and oldcfg['max_seconds'] == 43200,
                'old command/guard policy mismatch')
        for row in self.expected['processes']:
            require(self.process(row['pid']) == row, 'owned process absent before STOP')
        require(self.expected['processes'][0]['args'] == self.cfg['guard_argv'], 'guard argv mismatch')
        serverenv = self.receipt['scopes'][str(self.expected['pids']['server_pid'])]['environment']
        require(all(serverenv.get(k) == v for k, v in oldcfg['env'].items()), 'old configured environment mismatch')
        require(not self.exists(self.old + '/result.json') and not self.exists(self.old + '/STOP'), 'old run already stopping/completed')
        self.assert_server_lock(self.expected['pids']['guard_pid'])
        self.verify_identities()

    def open_health(self, url, timeout):
        require(url == 'http://127.0.0.1:8096/health', 'unexpected core health target')
        row = self.expected['processes'][1]
        require(self.process(row['pid']) == row, 'health process identity changed')
        return fetch_health(row, self.cfg['port'], min(timeout, self.cfg['timeouts']['http']))

    def before_stop(self):
        self.check_authority()

    def write_stop(self, run, text):
        require(run == self.expected['run'], 'STOP outside verified run')
        self.validate_old()
        self.core.recover.health(self)  # final owned idle read immediately before create
        self.check_authority()
        require(self.active() == run and self.pids(run) == self.expected['pids'], 'owner changed during final health')
        for row in self.expected['processes']:
            require(self.process(row['pid']) == row, 'process changed during final health')
        assert_no_unowned_members(self.expected['pids'])
        self.fs.create(self.old + '/STOP', text.encode())

    def read_stop(self, run):
        require(run == self.expected['run'], 'STOP read outside verified run')
        return self.fs.read(self.old + '/STOP').decode()

    def result(self, run):
        require(run == self.expected['run'], 'result outside verified run')
        try:
            result = strict_json(self.fs.read(self.old + '/result.json'))
        except FileNotFoundError:
            return None
        require(result['reason'] == 'requested_stop' and type(result['returncode']) is int and
                result['oom_kill_delta'] == result['host_oom_kill_delta'] == 0, 'guard result/zero OOM deltas unproved')
        return result

    def memory(self):
        raw = read_absolute(self.identities['guard']['path'])
        require(digest(raw) == self.identities['guard']['sha256'], 'guard hash changed before memory read')
        module = {'__name__': 'verified_memory_reader'}
        exec(compile(raw, self.identities['guard']['path'], 'exec'), module)
        sample = module['memory_sample']()
        for key in ('available_gib', 'cgroup_headroom_gib', 'free_gib', 'host_oom_kill', 'cgroup_oom', 'cgroup_oom_kill'):
            require(type(sample[key]) in (int, float) and math.isfinite(sample[key]) and sample[key] >= 0, 'invalid memory sample')
        return sample

    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)

    def no_overlap(self):
        for row in self.expected['processes']:
            require(self.process(row['pid']) is None, 'old process remains; no overlapping load')
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                raw = (entry / 'cmdline').read_bytes()
            except FileNotFoundError:
                continue
            args = raw.decode().rstrip('\0').split('\0')
            require(not any('serve_native.py' in v for v in args) and
                    not ('download' in args and any(Path(v).name == 'hf' for v in args)) and
                    self.identities['guard']['path'] not in args, 'other model/download/guard process present')
        # Acquire and release the SAME guard lock. It cannot be inherited by the
        # unchanged launcher, which opens a separate flock description. The
        # operator lease spans this handoff; the guard acquires server.lock itself.
        with self.fs.lock(self.cfg['server_lock']):
            pass
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', self.cfg['port']))

    def launch_retained(self):
        import subprocess
        require(not self.launched, 'retained launcher may run only once')
        self.launched = True
        self.check_authority()
        self.verify_identities()
        require(self.active() == self.expected['run'], 'active changed before retained launch')
        self.no_overlap()
        sample = self.memory()
        require(sample['available_gib'] >= 102 and sample['cgroup_headroom_gib'] >= 102 and sample['free_gib'] >= 2 and
                all(sample[k] == 0 for k in ('host_oom_kill', 'cgroup_oom', 'cgroup_oom_kill')), 'load headroom changed')
        self.preexisting_runs = {p.name for p in (self.root / 'runs').iterdir()}
        self.launch_tick = int(float(Path('/proc/uptime').read_text().split()[0]) * os.sysconf('SC_CLK_TCK'))
        log = self.cfg['receipt_dir'] + '/launcher-' + self.receipt_sha + '.log'
        # No shell, env inheritance, wrapper session, timeout retry, or candidate target.
        with self.fs.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL) as fd:
            child = subprocess.Popen(self.cfg['launcher_argv'], env=self.cfg['launcher_env'], cwd=str(self.root),
                                     stdin=subprocess.DEVNULL, stdout=fd, stderr=fd, close_fds=True)
            try:
                rc = child.wait(timeout=self.cfg['timeouts']['launch'])
            except subprocess.TimeoutExpired as e:
                child.kill()
                child.wait(timeout=5)
                raise Refusal('launcher_timeout: exact launcher reaped; it may have spawned a guard; retain ownership, no retry') from e
        require(rc == 0, 'retained launcher failed; no retry; inspect exact launcher log')
        lines = self.fs.read(log, 262144).decode().splitlines()
        launched = [strict_json(line[len('LAUNCHED '):]) for line in lines if line.startswith('LAUNCHED ')]
        require(len(launched) == 1, 'launcher new-run receipt missing/ambiguous')
        r = launched[0]
        new = self.relative_run(r['run'])
        require(r['running'] is True and r['run'] != self.expected['run'] and
                Path(r['run']).name not in self.preexisting_runs and Path(r['run']).name.startswith(self.cfg['new_run_prefix']),
                'new run identity unproved')
        self.fs.pin(new)
        require(r['configuration'] == dict(self.cfg['retained'], outdir=r['run']), 'retained launch configuration mismatch')
        require(self.active() == r['run'], 'new active pointer mismatch')
        self.bound_launch = strict_json(json.dumps(r))
        return strict_json(json.dumps(r))

    def guard_parent_pid(self):
        return 1

    def new_snapshot(self, run, pids):
        require(set(pids) == {'guard_pid', 'server_pid', 'command'} and pids['command'] == self.cfg['retained']['command'] and
                pids['guard_pid'] == self.bound_launch['guard_pid'], 'retained PID record mismatch')
        require(pids['guard_pid'] != pids['server_pid'] and
                not ({pids['guard_pid'], pids['server_pid']} & {r['pid'] for r in self.expected['processes']}), 'reused/duplicate new PID')
        rows, scopes = [], {}
        for role, argv, parent in [('guard', self.cfg['guard_argv'], self.guard_parent_pid()),
                                   ('server', pids['command'], pids['guard_pid'])]:
            pid = pids[role + '_pid']
            snapshot = proc_snapshot(pid)
            require(snapshot is not None, 'retained process disappeared')
            row, scope = snapshot
            require(row['args'] == argv and int(row['starttime']) >= self.launch_tick, 'new argv/starttime unproved')
            require(scope['uid'] == os.getuid() and scope['ppid'] == parent and scope['pgid'] == scope['sid'] == pid,
                    'new guard/server process scope mismatch')
            old_scope = self.receipt['scopes'][str(self.expected['pids'][role + '_pid'])]
            for key in ('cgroup', 'cgroupns', 'netns', 'mntns', 'exe'):
                require(scope[key] == old_scope[key], 'new namespace/executable/cgroup mismatch: ' + key)
            expected_env = self.cfg['launcher_env'] if role == 'guard' else {**self.cfg['launcher_env'], **self.cfg['retained']['env']}
            require(scope['environment'] == expected_env, 'new full environment mismatch')
            rows.append(row); scopes[str(pid)] = scope
        require(int(rows[1]['starttime']) >= int(rows[0]['starttime']), 'server predates guard')
        self.assert_server_lock(pids['guard_pid'])
        return {'run': run, 'pids': pids, 'processes': rows}, scopes

    def verify_mapped_dso(self, pid):
        item = self.identities['dso']
        p = Path(item['path'])
        matches = [l.split() for l in Path('/proc', str(pid), 'maps').read_text().splitlines()
                   if p.name in l or 'exllamav3_ext' in l]
        require(matches and {m[-1] for m in matches} == {str(p)}, 'loaded DSO path mismatch/unproved')
        st = os.stat(p, follow_symlinks=False)
        require(all(int(m[4]) == st.st_ino and tuple(int(x, 16) for x in m[3].split(':')) ==
                    (os.major(st.st_dev), os.minor(st.st_dev)) for m in matches), 'mapped DSO inode mismatch')
        require(file_digest(p) == item['sha256'], 'mapped DSO hash mismatch')

    def wait_retained(self, receipt):
        require(self.bound_launch is not None and receipt == self.bound_launch, 'new receipt binding mismatch')
        run = receipt['run']; relative = self.relative_run(run)
        end = time.monotonic() + self.cfg['timeouts']['ready']
        first = None
        while time.monotonic() < end:
            self.check_authority()
            require(self.active() == run, 'retained active pointer changed')
            require(not self.exists(relative + '/result.json') and not self.exists(relative + '/STOP'), 'retained guard stopped')
            raw = self.fs.read(relative + '/launch.json')
            cfg = strict_json(raw)
            require(cfg == receipt['configuration'] == strict_json(self.fs.read(self.cfg['live_config'])), 'retained config readback mismatch')
            try:
                pids = self.pids(run)
            except FileNotFoundError:
                time.sleep(.05)
                continue
            owner, scopes = self.new_snapshot(run, pids)
            if first is None:
                first = (owner, scopes)
            require(first == (owner, scopes), 'new PID/starttime/argv/scope changed while waiting')
            row = owner['processes'][1]
            try:
                response = fetch_health(row, self.cfg['port'], min(self.cfg['timeouts']['http'], max(.001, end-time.monotonic())))
            except urllib.error.HTTPError as e:
                response = e
            except TimeoutError as e:
                code = 'retained_readiness_timeout' if time.monotonic() >= end else 'retained_health_timeout'
                raise Refusal(code + '; no restored claim; keep ownership') from e
            except Refusal as e:
                if str(e) != 'missing or ambiguous listener':
                    raise
                time.sleep(.05)
                continue
            with response:
                status = getattr(response, 'status', response.code if hasattr(response, 'code') else None)
                h = strict_json(response.read())
            require(status in (200, 503) and isinstance(h, dict) and type(h.get('requests')) is int and h['requests'] == 0 and
                    type(h.get('healthy')) is bool, 'retained unknown/busy/malformed health')
            if status == 200 and h['healthy'] is True:
                self.verify_identities()
                self.verify_mapped_dso(row['pid'])
                assert_no_unowned_members(pids)
                require((owner, scopes) == self.new_snapshot(run, self.pids(run)) and self.active() == run, 'retained final identity changed')
                require(time.monotonic() < end, 'retained_readiness_timeout before final health')
                try:
                    final = fetch_health(row, self.cfg['port'], min(self.cfg['timeouts']['http'], end-time.monotonic()))
                except urllib.error.HTTPError as e:
                    e.close()
                    raise Refusal('retained final health not HTTP200') from e
                with final:
                    h = strict_json(final.read())
                    require(final.status == 200 and h.get('healthy') is True and type(h.get('requests')) is int and h['requests'] == 0,
                            'retained final health unknown/busy/unhealthy')
                require((owner, scopes) == self.new_snapshot(run, self.pids(run)) and self.active() == run,
                        'retained final health identity changed')
                bound = {'owner': owner, 'scopes': scopes, 'launch_sha256': digest(raw), 'identities': self.cfg['identities'],
                         'import_readback': getattr(self, 'import_event', None),
                         'config_sha256': self.config_sha, 'receipt_sha256': self.receipt_sha,
                         'authorization_sha256': digest(json.dumps(self.auth, sort_keys=True).encode()),
                         'guard_policy': {'memory_policy': cfg['memory_policy'], 'max_seconds': cfg['max_seconds']},
                         'listener_inode': listener_inode(row, self.cfg['port']), 'health': {'status': status, 'body': h}}
                name = self.cfg['receipt_dir'] + '/restored-' + self.receipt_sha + '.json'
                self.fs.create(name, json.dumps(bound, indent=2).encode())
                require(strict_json(self.fs.read(name)) == bound, 'restored receipt readback failed')
                return {'run': run, 'healthy': True, 'requests': 0, 'receipt': bound, 'readback_path': str(self.root / name)}
            time.sleep(.05)
        raise Refusal('retained_readiness_timeout; no restored claim; keep ownership')


def verify_import_event(raw, identities):
    lines = raw.decode().splitlines()
    events = [strict_json(l[len('ROUND6_IMPORT '):]) for l in lines if l.startswith('ROUND6_IMPORT ')]
    expected = {'python': identities['source']['path'], 'source_sha256': identities['source']['sha256'],
                'extension': identities['dso']['path'], 'elide': True}
    require(events == [expected], 'retained source-import readback missing/changed/ambiguous')
    return events[0]


REQUIRED_SOURCE_PATHS = {
    '/workspace/MiMo-V2.6-Flash-RL-EXL3-recipe/server/serve_native.py',
    '/workspace/mimo-exl3/exllamav3/exllamav3/util/memory.py',
    '/workspace/mimo-exl3/runtime-ready.json',
    '/workspace/mimo-tune/round5-quant-drafter/download-verified.json',
    '/workspace/mimo-tune/sampler-greedy-deployed.json',
    '/workspace/mimo-tune/dflash-pages-deployed.json',
    '/workspace/mimo-tune/dflash-ready.json',
    *('/workspace/mimo-tune/60tps-round3/python-shadow/exllamav3/' + name for name in (
        'generator/job.py', 'generator/pagetable.py', 'generator/generator.py',
        'model/config.py', 'model_init.py', 'modules/arch_specific/dflash.py',
        'modules/block_sparse_mlp.py', 'util/memory.py')),
}


def verify_manifest_shape(manifest, identities):
    require(set(manifest) == {'pin', 'files'} and manifest['pin'] == SOURCE_PIN and isinstance(manifest['files'], dict),
            'source manifest pin/schema mismatch')
    files = manifest['files']
    require(REQUIRED_SOURCE_PATHS <= files.keys(), 'source manifest launch/import closure missing')
    for role in PRODUCTION_PINS:
        item = identities[role]
        require(files.get(item['path']) == item['sha256'], 'source manifest disagrees with pinned identity: ' + role)
    for path, h in files.items():
        require(valid_sha(h) and Path(path).is_absolute() and str(Path(path)) == path and '..' not in Path(path).parts,
                'invalid source manifest entry')


def verify_memory_scope(caller, guard, server):
    for scope in (caller, guard, server):
        require(scope['cgroup'] == '0::/\n' and scope['uid'] == os.getuid(), 'root cgroup-v2 sampler scope unproved')
        for key in ('cgroupns', 'mntns', 'netns'):
            require(scope[key] == caller[key], 'memory/HTTP namespace mismatch: ' + key)


def check_existing_outputs(fs, cfg):
    # Existing cooperative lease required; never silently invent another lock.
    for name in (cfg['operator_lock'], cfg['server_lock'], cfg['active'], cfg['live_config']):
        with fs.open(name):
            pass
    try:
        with fs.open('guard-france-uma.log'):
            pass
    except FileNotFoundError:
        pass


class ProductionIO(LinuxIO):
    """Fixed future-host profile; never used for inert launch/memory substitutions."""
    def __init__(self, cfg, *args):
        validate_production(cfg)
        super().__init__(cfg, *args)

    def __enter__(self):
        with OwnedRoot(str(self.root)) as fs:
            check_existing_outputs(fs, self.cfg)
        return super().__enter__()

    def verify_identities(self):
        manifest = strict_json(read_absolute(self.identities['source_manifest']['path']))
        verify_manifest_shape(manifest, self.identities)
        super().verify_identities()
        interpreter = self.identities['interpreter']['path']
        require(str(Path('/usr/bin/python3').resolve(strict=True)) == interpreter and
                str(Path(self.cfg['retained']['command'][0]).resolve(strict=True)) == interpreter,
                'interpreter symlink target differs from explicit binary identity')

    def validate_old(self):
        super().validate_old()
        check_existing_outputs(self.fs, self.cfg)
        guard, server = (self.receipt['scopes'][str(self.expected['pids'][key])] for key in ('guard_pid', 'server_pid'))
        verify_memory_scope(proc_snapshot(os.getpid())[1], guard, server)
        for scope, row in zip((guard, server), self.expected['processes']):
            require(scope['pgid'] == scope['sid'] == row['pid'] and scope['exe'] == self.identities['interpreter']['path'],
                    'old group/session/executable scope mismatch')
        require(guard['ppid'] == 1 and server['ppid'] == self.expected['pids']['guard_pid'], 'old guard/server lineage mismatch')
        assert_no_unowned_members(self.expected['pids'])
        LinuxIO.verify_mapped_dso(self, self.expected['pids']['server_pid'])

    def memory(self):
        guard, server = (self.receipt['scopes'][str(self.expected['pids'][key])] for key in ('guard_pid', 'server_pid'))
        verify_memory_scope(proc_snapshot(os.getpid())[1], guard, server)
        mounts = [line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup2 ' in line]
        require(len(mounts) == 1 and mounts[0][3:5] == ['/', '/sys/fs/cgroup'], 'sampler cgroup mount root unproved')
        return super().memory()

    def launch_retained(self):
        check_existing_outputs(self.fs, self.cfg)
        return super().launch_retained()

    def verify_mapped_dso(self, pid):
        super().verify_mapped_dso(pid)
        relative = self.relative_run(self.bound_launch['run'])
        self.import_event = verify_import_event(self.fs.read(relative + '/server.log', 8 * 1024 * 1024), self.identities)


def perform(io):
    """Recovery-only operator integration: preserves the reported width outcome.

    The width outcome is data from explicit authorization, not a callable that
    could accidentally launch another native16 candidate.
    """
    width = io.auth['width']
    status = {'width': width['status'], 'width_error': width['error'], 'rollback': 'not_run',
              'rollback_error': None, 'readback_error': None}
    name = io.cfg['receipt_dir'] + '/status-' + io.receipt_sha + '.json'
    status['status_path'] = str(io.root / name)
    try:
        status['recovery'] = io.core.launch(io.expected, io, stop_seconds=io.cfg['timeouts']['stop'],
                                          headroom_seconds=io.cfg['timeouts']['headroom'])
        status['rollback'] = 'restored'
    except BaseException as e:
        status['rollback'] = 'failed'
        status['rollback_error'] = repr(e)
    try:
        io.fs.create(name, json.dumps(status, indent=2).encode())
        require(strict_json(io.fs.read(name)) == status, 'status readback mismatch')
    except BaseException as e:
        status['readback_error'] = repr(e)
    code = 0 if status['width'] == 'passed' and status['rollback'] == 'restored' and status['readback_error'] is None else 1
    return code, status


def cli(argv=None):
    import argparse
    import sys
    ap = argparse.ArgumentParser(description='Owned Linux recovery ONLY. No candidate, inference, retry or receipt capture.')
    for name in ('config', 'receipt', 'authorization'):
        ap.add_argument('--' + name)
        ap.add_argument('--' + name + '-sha256')
    ap.add_argument('--validate-only', action='store_true', help='validate trusted documents/profile only; no process/HTTP/STOP/launch')
    args = ap.parse_args(argv)
    status = {'width': 'not_run', 'width_error': None, 'rollback': 'failed', 'rollback_error': None, 'readback_error': None}
    try:
        require(sys.platform == 'linux' and sys.flags.isolated, 'Linux /usr/bin/python3 -I -B is required')
        documents = {}
        hashes = {}
        for name in ('config', 'receipt', 'authorization'):
            path = getattr(args, name)
            expected = getattr(args, name + '_sha256')
            require(path and valid_sha(expected), 'explicit trusted config, PRE-FAILURE receipt, authorization and all three SHA256 values required')
            raw = read_absolute(path)
            require(digest(raw) == expected, 'explicit trusted document hash mismatch: ' + name)
            documents[name] = strict_json(raw)
            hashes[name] = expected
        cfg, receipt, auth = (documents[name] for name in ('config', 'receipt', 'authorization'))
        validate_production(cfg)
        authorize(cfg, receipt, auth, hashes['config'], hashes['receipt'])
        load_core()
        status['width'] = auth['width']['status']; status['width_error'] = auth['width']['error']
        if args.validate_only:
            print(json.dumps({'validated_documents_only': True, 'restore_attempted': False, 'source_pin': SOURCE_PIN}))
            return 0
        with ProductionIO(cfg, receipt, auth, hashes['config'], hashes['receipt']) as boundary:
            code, status = perform(boundary)
        print(json.dumps(status, indent=2))
        return code
    except BaseException as e:
        status['rollback_error'] = repr(e)
        print(json.dumps(status, indent=2))
        return 1


