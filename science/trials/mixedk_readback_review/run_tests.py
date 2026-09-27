"""Run real test subprocesses, keep immutable transcripts and returncodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
p=argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--label', required=True)
p.add_argument('--expect', type=int, default=0)
p.add_argument('tests', nargs='*', default=['test_mixedk'])
a=p.parse_args()
root=Path(__file__).resolve().parent
env=dict(os.environ, REVIEW_SOURCE_ROOT=str(Path(a.source).resolve()), PYTHONDONTWRITEBYTECODE='1')
log=root/'evidence'/f'{a.label}.txt'
with log.open('x',encoding='utf-8') as f:
    cmd=[sys.executable,'-m','unittest','-v',*a.tests]
    r=subprocess.run(cmd,cwd=root/'tests',env=env,text=True,capture_output=True)
    f.write(json.dumps({'command':cmd,'source':a.source,'returncode':r.returncode})+'\n'+r.stdout+r.stderr)
print(r.stdout+r.stderr)
print(json.dumps({'returncode':r.returncode,'expected':a.expect,'log':str(log)}))
sys.exit(0 if r.returncode==a.expect else 1)
