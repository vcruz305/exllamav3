"""TEST ONLY inert width-trial stack: guard / server / retained launcher.

Composed from the sealed adapter's `inert_stack.py` shape (same three modes,
same flock guard, same `LAUNCHED` receipt line, same STOP/result.json
contract), extended for the width trial:

* the inert `server` creates the width probe directory and writes
  `events.jsonl` when `ROUND8_WIDTH_OUT` is set, controlled by
  `probe.json` in the fixture root (`{"mode": "absent"}` skips creation);
* the inert `launcher` always creates runs with the production
  `france-round6-quant-readback-` prefix, so `new_run_prefix` keeps its
  production value under test.

The evidence this fixture writes is SYNTHESIZED to the sealed diagnostic's
`emit()` vocabulary. It is a contract fixture, NOT device evidence: see
GAPS.md G-FIXTURE-EVIDENCE. No model, torch, CUDA or network is touched.
"""
import argparse
import fcntl
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['server', 'guard', 'launcher'])
ap.add_argument('path')
ap.add_argument('port', nargs='?', type=int)
a = ap.parse_args()

if a.mode == 'server':
    root = Path(a.path)
    out = os.environ.get('ROUND8_WIDTH_OUT')
    if out:
        probe = Path(out)
        probe.mkdir(parents=True, exist_ok=True)
        plan = json.loads((root / 'probe.json').read_text()) if (root / 'probe.json').exists() else {'mode': 'complete'}
        if plan['mode'] != 'absent':
            with (probe / 'events.jsonl').open('a') as handle:
                for record in plan['records']:
                    handle.write(json.dumps(record) + '\n')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state = json.loads((root / 'health.json').read_text())
            raw = state.get('raw', json.dumps(state['body'])).encode()
            self.send_response(state['status'])
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))

    class Server(HTTPServer):
        allow_reuse_address = True

    with Server(('127.0.0.1', a.port), Handler) as http:
        (root / 'listening').write_text(str(os.getpid()))
        http.serve_forever(poll_interval=.02)
elif a.mode == 'guard':
    cfg = json.loads(Path(a.path).read_text())
    root = Path(a.path).parent
    run = Path(cfg['outdir'])
    with (root / 'server.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (run / 'server.log').open('w') as log:
            child = subprocess.Popen(cfg['command'], env={**os.environ, **cfg['env']}, stdout=log,
                                     stderr=log, start_new_session=True)
        # The sealed adapter's identity check reads /proc/<server_pid>/cmdline as
        # soon as pid.json appears, and a forked-but-not-yet-exec'd child has an
        # EMPTY cmdline. Wait (bounded) for the exec so this fixture is
        # deterministic. The production guard's pid.json/exec ordering is an
        # unverified device-time item: see GAPS.md G-PID-EXEC-RACE.
        deadline = time.monotonic() + 10
        while child.poll() is None and time.monotonic() < deadline:
            try:
                if Path('/proc', str(child.pid), 'cmdline').read_bytes().strip(b'\0'):
                    break
            except OSError:
                break
            time.sleep(.02)
        (run / 'pid.json').write_text(json.dumps({'guard_pid': os.getpid(), 'server_pid': child.pid,
                                                 'command': cfg['command']}))
        try:
            while (not (run / 'STOP').exists() or (root / 'ignore-stop').exists()) and child.poll() is None:
                time.sleep(.02)
        finally:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=3)
            (run / 'result.json').write_text(json.dumps({'reason': 'requested_stop', 'returncode': child.returncode,
                                                        'oom_kill_delta': 0, 'host_oom_kill_delta': 0}))
else:
    root = Path(a.path)
    cfg = json.loads((root / 'retained-template.json').read_text())
    run = root / 'runs' / ('france-round6-quant-readback-' + str(time.time_ns()))
    run.mkdir()
    cfg['outdir'] = str(run)
    (run / 'launch.json').write_text(json.dumps(cfg))
    (root / 'france-uma.json').write_text(json.dumps(cfg))
    (root / 'france-active-run.txt').write_text(str(run))
    (root / 'health.json').write_text(json.dumps({'status': 200, 'body': {'healthy': True, 'requests': 0,
                                                                        'max_active_requests': 1}}))
    (root / 'launch-count').write_text((root / 'launch-count').read_text() + '1\n')
    with (root / 'guard-france-uma.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-B', __file__, 'guard', str(root / 'france-uma.json')],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    time.sleep(.15)
    print('LAUNCHED ' + json.dumps({'guard_pid': child.pid, 'running': child.poll() is None,
                                   'run': str(run), 'configuration': cfg}), flush=True)
    sys.exit(0 if child.poll() is None else 1)
