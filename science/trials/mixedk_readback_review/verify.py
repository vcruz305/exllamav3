"""Offline, CPU-only apply/verify. No SSH, GPU imports or production mutations.
Usage: python verify.py --label unique-label [--upstream path-to-read-only-pinned-git]
Requires Python 3.11+, NumPy and git. Every applied tree is NEW and disposable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

R=Path(__file__).resolve().parent
PIN='ca4a880e8918e1985fd25e06c6aff561666d3f14'
p=argparse.ArgumentParser()
p.add_argument('--label',required=True)
p.add_argument('--upstream',type=Path)
a=p.parse_args()
if not a.label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.label):
    p.error('label must be a new alphanumeric/hyphen/underscore token')
work=R/'verified'/a.label
work.mkdir(parents=True,exist_ok=False)
summary=[]

def run(name, cmd, *, cwd=None, env=None, expected=0):
    r=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,text=True,capture_output=True)
    record={'name':name,'command':[str(x) for x in cmd],'returncode':r.returncode}
    with (R/'evidence'/f'{a.label}-{name}.txt').open('x',encoding='utf-8') as f:
        f.write(json.dumps(record)+'\n'+r.stdout+r.stderr)
    print(json.dumps(record))
    summary.append(record)
    if r.returncode!=expected:
        print(r.stdout+r.stderr)
        raise RuntimeError(f'{name}: expected exit {expected}, got {r.returncode}')
    return r

def test(name,root,modules, failures=0):
    env=dict(os.environ,REVIEW_SOURCE_ROOT=str(root),PYTHONDONTWRITEBYTECODE='1')
    r=run(name,[sys.executable,R/'tests/test_driver.py',*modules],cwd=R/'tests',env=env,expected=int(failures!=0))
    line=next(l for l in r.stdout.splitlines() if l.startswith('TEST_RESULT '))
    counts=json.loads(line.removeprefix('TEST_RESULT '))
    assert counts['failures']==failures and counts['errors']==0 and counts['skipped']==0 and counts['unexpected_successes']==0, counts
    assert counts['tests']>0
    summary[-1]['counts']=counts
    print(json.dumps({'test':name,**counts}))

def check_upstream(suffix):
    if a.upstream:
        head=run('upstream-head-'+suffix,['git','-C',a.upstream,'rev-parse','HEAD']).stdout.strip()
        assert head==PIN,(head,PIN)
        status=run('upstream-status-'+suffix,['git','-C',a.upstream,'status','--porcelain']).stdout
        assert status=='',status

manifest=json.loads((R/'MANIFEST.json').read_text(encoding='utf-8'))
assert manifest['pin']==PIN
for rel,digest in manifest['sha256'].items():
    actual=hashlib.sha256((R/rel).read_bytes()).hexdigest()
    assert actual==digest, (rel,actual,digest)
check_upstream('before')
if a.upstream:
    for rel in ('exllamav3/modules/block_sparse_mlp.py','exllamav3/generator/generator.py'):
        r=subprocess.run(['git','-C',str(a.upstream),'show',PIN+':'+rel],capture_output=True,check=True)
        assert r.stdout==(R/'baseline'/rel).read_bytes(),rel

# The baseline's target failures are ASSERTION failures, never harness import/errors.
test('baseline-red',R/'baseline',[
    'test_mixedk.Elision','test_telemetry.RoundDiagnostics.test_final_round_survives_job_removal'],failures=2)
test('baseline-controls',R/'baseline',['test_mixedk.Preservation','test_telemetry.Preservation'])
test('candidate',R/'candidate',['test_mixedk','test_telemetry'])
test('prior-bug',R/'prior-source',['prior_telemetry_repro'],failures=2)

patches={
 'mixedk-dead-readback.patch':'exllamav3/modules/block_sparse_mlp.py',
 'target-round-diagnostics.patch':'exllamav3/generator/generator.py'}
for mode,names,modules in (
    ('mixedk',['mixedk-dead-readback.patch'],['test_mixedk','test_telemetry.Preservation']),
    ('telemetry',['target-round-diagnostics.patch'],['test_mixedk.Preservation','test_telemetry']),
    ('combined',list(patches),['test_mixedk','test_telemetry']),
):
    dest=work/mode
    dest.mkdir()
    with tarfile.open(R/'baseline.tar') as tf: tf.extractall(dest,filter='data')
    run(mode+'-git-init',['git','init','--quiet',dest])
    git=['git','-c','core.autocrlf=false','-c','core.whitespace=cr-at-eol','apply']
    for i,name in enumerate(names):
        run(mode+f'-check-{i}',git+['--check','--whitespace=error-all',R/'patches'/name],cwd=dest)
        run(mode+f'-apply-{i}',git+['--whitespace=error-all',R/'patches'/name],cwd=dest)
    touched={patches[n] for n in names}
    # Verify every archive member; no capacities, kernel, server or unrelated source changes.
    with tarfile.open(R/'baseline.tar') as tf:
        for member in tf.getmembers():
            if member.isfile():
                expected=(R/'candidate'/member.name).read_bytes() if member.name in touched else tf.extractfile(member).read()
                assert (dest/member.name).read_bytes()==expected,(mode,member.name)
    test(mode+'-applied',dest,modules)
check_upstream('after')
with (R/'evidence'/f'{a.label}-summary.json').open('x',encoding='utf-8') as f:
    json.dump({'pin':PIN,'cpu_only':True,'results':summary},f,indent=2)
print('VERIFIED: independent and combined patches; exact files; CPU tests; baseline untouched')
