"""INERT local Linux process fixture, hard expiry, never imports Torch."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    try:
        s = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, starttime=int(s[19]), ppid=int(s[1]), pgid=int(s[2]), state=s[0])
    except FileNotFoundError:
        return None


def main():
    out = Path(os.environ['FIXTURE_OUT'])
    role = sys.argv[1] if len(sys.argv) > 1 else 'leader'
    mode = os.environ.get('FIXTURE_MODE', 'wait')
    signal.signal(signal.SIGALRM, lambda *_: os._exit(99))
    signal.alarm(12)
    if role == 'descendant':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with (out / (role + '.json')).open('x') as f:
        json.dump(identity(os.getpid()), f)
    print('fixture-stdout-' + role, flush=True)
    print('fixture-stderr-' + role, file=sys.stderr, flush=True)
    if role == 'leader' and mode in ('descendant', 'leader_exit', 'escape'):
        subprocess.Popen([sys.executable, '-B', __file__, 'descendant'],
                         start_new_session=mode == 'escape')
        deadline = time.monotonic() + 2
        while not (out / 'descendant.json').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (out / 'descendant.json').exists()
    if role == 'leader':
        (out / 'ready').touch()
        if mode in ('exit0', 'leader_exit'):
            return
        if mode == 'exit7':
            raise SystemExit(7)
    time.sleep(11)


if __name__ == '__main__':
    main()
