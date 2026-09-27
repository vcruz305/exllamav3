"""Bound one future authorized Linux trial; preserves logs and kills only its PGID.

Usage: python bounded_deferred.py --receipt NEW_DIR [--seconds 1800] -- [deferred_gpu args]
No SSH, no server controls, no imports of torch/exllamav3. Does not replace a memory guard.
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=1800)
    p.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if not 1<=a.seconds<=1800:p.error('seconds must be1..1800')
    from build_trial import require_authorization
    require_authorization()
    if os.environ.get('EXL3_THREE_STAGE_GUARD_ACTIVE')!='YES':p.error('GPU owner must launch under the existing memory guard and assert EXL3_THREE_STAGE_GUARD_ACTIVE=YES')
    a.receipt.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B',str(Path(__file__).with_name('deferred_gpu.py'))]+(a.args[1:] if a.args[:1]==['--'] else a.args)
    start=time.monotonic();timed_out=False
    with (a.receipt/'raw.log').open('xb') as log:
        child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=child.wait(timeout=a.seconds)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
    result={'argv':argv,'owned_pgid':child.pid,'returncode':rc,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-start}
    (a.receipt/'process.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    raise SystemExit(124 if timed_out else rc)

if __name__=='__main__':main()
