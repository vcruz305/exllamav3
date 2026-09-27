"""Offline source-only verifier. Python + NumPy + local Git; NEVER GPU/network.
Run: <numpy-python> -B verify_repair.py --label fresh-unique-label
Requires sealed sibling task trees under HERE.parent (no writes there).
"""
from __future__ import annotations
import argparse, ast, difflib, hashlib, io, json, os, re, shutil, subprocess, sys, tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=HERE.parent
ORIGINAL=R/'dflash-cost-aware-candidate'
BASE=R/'dflash-pages-candidate/upstream'
PIN='ca4a880e8918e1985fd25e06c6aff561666d3f14'
FILES=['exllamav3/generator/generator.py','exllamav3/generator/draft_cost.py']
def sha(raw):return hashlib.sha256(raw).hexdigest()
def git(*args,cwd=BASE):
    return subprocess.check_output(['git','-c','core.autocrlf=false','-C',str(cwd),*map(str,args)])
def exclusive(path,raw):
    if path.exists():
        assert path.read_bytes()==raw,'differing existing artifact: '+str(path)
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(raw)
def dump(path,obj):exclusive(path,(json.dumps(obj,indent=2)+'\n').encode())
def entries(raw):
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        return {m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
def diff(old,new,paths):
    lines=[]
    for p in paths:
        a=old.get(p,b'');b=new[p]
        ast.parse(b);assert b'\r\n' not in b, 'changed files must be exact LF'
        lines.extend(difflib.unified_diff(a.decode('latin1').splitlines(keepends=True),b.decode('latin1').splitlines(keepends=True),
                                        fromfile='a/'+p if p in old else '/dev/null',tofile='b/'+p,lineterm='\n'))
    return ''.join(lines).encode('latin1')
def verify_seals():
    original=json.loads((HERE/'sealed-original-manifest.json').read_text())
    now={p.relative_to(R).as_posix():sha(p.read_bytes()) for p in ORIGINAL.rglob('*') if p.is_file()}
    wanted={Path(p).as_posix():v for p,v in original.items()}
    assert now==wanted,'sealed original candidate tree changed'
    dependencies=json.loads((HERE/'sealed-inputs-manifest.json').read_text())
    for p,h in dependencies.items():assert sha((R/p).read_bytes())==h,p
    return dict(original_files=len(wanted),other_input_files=len(dependencies))
def run(label,stem,args,root=None,expected=0):
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['EXL3_DFLASH_COST_AWARE']='0'
    env['DFLASH_SOURCE_ROOT']=str(root or HERE/'candidate')
    proc=subprocess.run([sys.executable,'-B',str(HERE/'safe_run.py'),*map(str,args)],env=env,cwd=HERE,capture_output=True,text=True)
    text=proc.stdout+proc.stderr
    with (HERE/'evidence'/f'{label}-{stem}.log').open('x',encoding='utf8',newline='\n') as f:f.write(text)
    assert proc.returncode==expected,(stem,proc.returncode,text)
    counts=re.findall(r'Ran (\d+) tests?',text)
    assert len(counts)==1,(stem,text)
    result=dict(tests=int(counts[0]),exit_code=proc.returncode)
    if expected:
        assert 'FAILED (failures=1)' in text and 'ERROR:' not in text,text
    else:assert '\nOK\n' in text,text
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+',a.label)
    label=a.label
    with (HERE/'evidence'/f'{label}-started.json').open('x') as f:json.dump({'pin':PIN},f)
    seals=verify_seals()
    assert git('rev-parse','HEAD').decode().strip()==PIN
    assert git('status','--porcelain')==b''
    archived=entries((HERE/'baseline.tar').read_bytes())
    # All baseline archive members, not only changed files, match clean exact-pin LF archive.
    actual=entries(git('archive','--format=tar',PIN))
    assert archived==actual
    for p in FILES[:1]:assert archived[p]==git('show',PIN+':'+p)
    dump(HERE/'baseline-source-manifest.json',{p:sha(raw) for p,raw in archived.items()})
    candidate={p.relative_to(HERE/'candidate').as_posix():p.read_bytes() for p in (HERE/'candidate').rglob('*') if p.is_file()}
    orig={p.relative_to(ORIGINAL/'candidate').as_posix():p.read_bytes() for p in (ORIGINAL/'candidate').rglob('*') if p.is_file()}
    delta=sorted(p for p in set(candidate)|set(orig) if candidate.get(p)!=orig.get(p))
    assert delta==[FILES[1]],delta
    combined=sorted(p for p in set(candidate)|set(archived) if candidate.get(p)!=archived.get(p))
    assert combined==sorted(FILES),combined
    inc=diff(orig,candidate,delta);full=diff(archived,candidate,FILES)
    exclusive(HERE/'incremental.patch',inc);exclusive(HERE/'combined.patch',full)
    for f in FILES:exclusive(HERE/'full-files'/f,candidate[f])
    # Apply both paths in separate disposable LF trees. No Git init/commit necessary.
    applied=HERE/f'applied-{label}';incremental=HERE/f'incremental-applied-{label}'
    applied.mkdir();incremental.mkdir()
    for tree,source,patchfile in ((applied,archived,HERE/'combined.patch'),(incremental,orig,HERE/'incremental.patch')):
        for f,b in source.items():exclusive(tree/f,b)
        git('apply','--check','--whitespace=error-all',patchfile,cwd=tree)
        git('apply','--whitespace=error-all',patchfile,cwd=tree)
        found={p.relative_to(tree).as_posix():p.read_bytes() for p in tree.rglob('*') if p.is_file()}
        assert found==candidate,'full-tree byte mismatch'
    results={}
    results['parent_red']=run(label,'parent-red',[HERE/'test_parent_repro.py'],expected=1)
    results['source_init_red']=run(label,'source-init-red',[HERE/'test_recurrent.py','Initialization.test_retained_config_enabled'],ORIGINAL/'candidate',1)
    results['profile']=run(label,'profile',[HERE/'test_profile.py',label+'-profile'])
    for tag,root in [('baseline',BASE),('original',ORIGINAL/'candidate'),('candidate',HERE/'candidate'),('applied',applied),('incremental',incremental)]:
        results['pages_'+tag]=run(label,'pages-'+tag,[R/'dflash-pages-candidate/tests/run_suite.py','--source',root,'--report',HERE/'evidence'/f'{label}-pages-{tag}.json'],root)
        results['native_'+tag]=run(label,'native-'+tag,[HERE/'run_native_controls.py','--source',root,'--label',label+'-native-'+tag]+(['--baseline'] if tag=='baseline' else []),root)
        results['sampler_'+tag]=run(label,'sampler-'+tag,[root/'tests/test_sampler_batch_verify_cpu.py'],root)
        assert results['pages_'+tag]['tests']==187
        if tag!='baseline':
            results['cost_'+tag]=run(label,'cost-'+tag,[HERE/'test_cost.py',label+'-cost-'+tag],root)
            assert sum(results[k+'_'+tag]['tests'] for k in ('pages','native','sampler','cost'))==240
        if tag in ('candidate','applied','incremental'):
            results['recurrent_'+tag]=run(label,'recurrent-'+tag,[HERE/'test_recurrent.py'],root)
    assert (HERE/'profile.json').read_bytes()==(ORIGINAL/'profile.json').read_bytes()
    assert (HERE/'context.example.json').read_bytes()==(ORIGINAL/'context.example.json').read_bytes()
    assert json.loads((HERE/'context.example.json').read_text())['attest_same_round8_weights_and_runtime'] is False
    assert (HERE/'test_cost.py').read_bytes()==(ORIGINAL/'test_cost.py').read_bytes()
    assert git('status','--porcelain')==b''
    assert verify_seals()==seals
    summary=dict(pin=PIN,results=results,seals=seals,baseline_clean=True,all_baseline_blobs_match=True,
                 both_applied_trees_byte_identical=True,incremental_changed=delta,combined_changed=combined,
                 incremental_sha256=sha(inc),combined_sha256=sha(full),profile_sha256=sha((HERE/'profile.json').read_bytes()),
                 files={f:sha(candidate[f]) for f in FILES},
                 total_per_repaired_tree=sum(results[k+'_candidate']['tests'] for k in ('pages','native','sampler','cost','recurrent')),
                 no_real_torch_or_exllamav3_imports=True,no_gpu_validation=True,no_tps_claim=True,live_attestation=False)
    dump(HERE/'evidence'/f'{label}-summary.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
