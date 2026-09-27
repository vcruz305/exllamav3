"""Reproducible LF patch/apply/byte/CPU verifier. Never imports torch/project."""
import argparse, difflib, hashlib, io, json, os, re, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PIN='ca4a880e8918e1985fd25e06c6aff561666d3f14'

def sha(data):return hashlib.sha256(data).hexdigest()
def files(path):return {p.relative_to(path).as_posix():p.read_bytes() for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--upstream',type=Path,default=ROOT.parent/'dflash-pages-candidate/upstream')
    args=parser.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',args.label):parser.error('simple unique label required')
    out=ROOT/('verify-'+args.label);out.mkdir(exist_ok=False)
    def run(label,command,expected=0,cwd=ROOT,env=None):
        result=subprocess.run(command,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/(label+'.txt')).write_bytes(result.stdout)
        if result.returncode!=expected:raise AssertionError((label,result.returncode,result.stdout[-2000:].decode(errors='replace')))
        return result.stdout
    # Hard assertion against original exact-pin checkout and archive's all-file bytes.
    assert run('head',['git','-C',str(args.upstream),'rev-parse','HEAD']).decode().strip()==PIN
    assert not run('status-before',['git','-C',str(args.upstream),'status','--porcelain'])
    original=files(ROOT/'original');candidate=files(ROOT/'tree')
    prov=json.loads((ROOT/'provenance.json').read_text())
    assert {n:sha(d) for n,d in original.items()}==prov['baseline_hashes']
    request=''.join(PIN+':'+name+'\n' for name in original).encode()
    proc=subprocess.run(['git','-C',str(args.upstream),'cat-file','--batch'],input=request,stdout=subprocess.PIPE,check=True)
    b=io.BytesIO(proc.stdout)
    for name,data in original.items():
        hdr=b.readline().split();assert hdr[1]==b'blob'
        assert b.read(int(hdr[2]))==data,name
        assert b.read(1)==b'\n'
    changed=[n for n in sorted(set(original)|set(candidate)) if original.get(n)!=candidate.get(n)]
    assert len(changed)==8,changed
    assert not any('phased' in n for n in changed)
    chunks=[]
    for name in changed:
        old=original.get(name,b'');new=candidate.get(name,b'')
        assert b'\r' not in new,name
        chunks.extend(difflib.unified_diff(old.decode('latin-1').splitlines(keepends=True),new.decode('latin-1').splitlines(keepends=True),
                    fromfile='a/'+name if name in original else '/dev/null',tofile='b/'+name,lineterm='\n'))
    data=''.join(chunks).encode('latin-1')
    patch=ROOT/'mixedk-three-stage.patch'
    if patch.exists():assert patch.read_bytes()==data,'Source changed: preserve old patch, explicitly reseal before rerun'
    else:patch.write_bytes(data)
    applied=out/'applied';shutil.copytree(ROOT/'original',applied)
    cmd=['git','-c','core.autocrlf=false','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab','apply']
    run('apply-check',cmd+['--check','--whitespace=error-all',str(patch)],cwd=applied)
    run('apply',cmd+['--whitespace=error-all',str(patch)],cwd=applied)
    assert files(applied)==candidate,'Applied tree is not byte-identical'
    totals={}
    for name,tree,suite,expected in (('original-red',ROOT/'original','red',1),('original-preservation',ROOT/'original','preservation',0),
                                    ('candidate',ROOT/'tree','all',0),('applied',applied,'all',0)):
        env=dict(os.environ,CANDIDATE_TREE=str(tree))
        path=out/(name+'.json')
        run(name,[sys.executable,'-B',str(ROOT/'run_cpu_tests.py'),'--suite',suite,'--json',str(path)],expected,env=env)
        totals[name]=json.loads(path.read_text())
        assert not totals[name]['errors'] and not totals[name]['forbidden_imports']
    # Explicit missing-column-ownership mutation must fail (not merely bijective coverage).
    mutation=out/'ownership-mutation';shutil.copytree(applied,mutation)
    k=mutation/'exllamav3/exllamav3_ext/quant/exl3_moe_three_stage.cuh'
    text=k.read_text();assert 'blockIdx.x * 256 + (w % 2) * 128' in text
    k.write_bytes(text.replace('blockIdx.x * 256 + (w % 2) * 128','blockIdx.x * 0 + (w % 2) * 128').encode())
    run('ownership-mutation-red',[sys.executable,'-B','-m','unittest','test_ownership.OwnershipTests.test_all_rows_tiles_partial_tiers_duplicates_exclusions','-v'],1,
        env=dict(os.environ,CANDIDATE_TREE=str(mutation)))
    run('deferred-plan',[sys.executable,'-B',str(ROOT/'deferred_gpu.py'),'--plan'])
    # Compile Python only, in memory (no project import or cache writes).
    for p in [*ROOT.glob('*.py'),ROOT/'tree/exllamav3/modules/mixedk_three_stage.py',ROOT/'tree/exllamav3/modules/block_sparse_mlp.py']:
        compile(p.read_text(),str(p),'exec')
    assert not run('status-after',['git','-C',str(args.upstream),'status','--porcelain'])
    result={'status':'CPU_SOURCE_VALIDATED_CUDA_NOT_COMPILED_NOT_GPU_VALIDATED','pin':PIN,
            'baseline_exact_blobs':len(original),'candidate_files':len(candidate),'changed_files':changed,
            'patch_sha256':sha(data),'patch_bytes':len(data),'applied_all_bytes_equal':True,'tests':totals,
            'changed_hashes':{n:sha(candidate[n]) for n in changed},
            'original_clean':True,'CUDA_compile':False,'GPU_execution':False}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
