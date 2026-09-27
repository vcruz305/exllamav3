"""Actual integrated CLI with inert workload and explicit injected memory samples.
No CUDA/torch/project/compiler use. Real path preflight, lock, guard and processes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from test_preflight import NoGPU,GUARD,SHA
import guarded_deferred as runner
PIN='29a870a4e11c55d185066df788402c283a46409ea08601758d721bb3eeeb1500'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    parser.add_argument('--case',choices=('admitted','receipt_protected','lock_protected','cache_protected','output_protected','alias_receipt','missing_inventory','reused_cache','receipt_parent','old_pin','changed_build','changed_deferred','changed_gpu','changed_scalar','changed_manifest'),default='admitted')
    a=parser.parse_args()
    out=ROOT/'evidence'/a.label;out.mkdir(exist_ok=False)
    package=ROOT/'package'
    protected=out/'other-source';protected.mkdir()
    (protected/'sentinel').write_text('TEST_ONLY_SOURCE')
    receipt=out/'run';cache=out/'cache';output=out/'result';lock=out/'shared-owner.lock'
    lock.write_text('TEST_ONLY_EXISTING_SHARED_LOCK')
    if a.case=='receipt_protected':receipt=protected/'run'
    if a.case=='lock_protected':lock=protected/'new-owner.lock'
    if a.case=='cache_protected':cache=protected/'cache'
    if a.case=='output_protected':output=protected/'result'
    if a.case=='alias_receipt':
        (out/'alias').symlink_to(protected,target_is_directory=True);receipt=out/'alias'/'run'
    if a.case=='reused_cache':cache.mkdir()
    if a.case=='receipt_parent':output=receipt/'result'
    if a.case.startswith('changed_'):
        import shutil
        changed={'changed_build':'build_trial.py','changed_deferred':'deferred_gpu.py','changed_gpu':'gpu_reference.py',
                 'changed_scalar':'scalar_reference.py','changed_manifest':'source-manifest.json'}[a.case]
        mini=out/'tampered-package';mini.mkdir()
        for name in ('build_trial.py','deferred_gpu.py','gpu_reference.py','scalar_reference.py','source-manifest.json'):
            shutil.copyfile(package/name,mini/name)
        with (mini/changed).open('ab') as f:f.write(b'\nTEST_ONLY_TAMPER')
        package=mini
    def lock_snapshot():
        return (lock.stat().st_ino,lock.read_bytes()) if lock.exists() else None
    lock_before=lock_snapshot()
    def file_snapshot():
        return {str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    before=file_snapshot()
    sys.meta_path.insert(0,NoGPU())
    original_spec=importlib.util.spec_from_file_location
    observed={'scope':'LOCAL Linux; injected memory, real processes, INERT command only','invoked':0}

    def spec_for(name,path,*args,**kwargs):
        spec=original_spec(name,path,*args,**kwargs)
        if Path(path).resolve()==GUARD.resolve():
            execute=spec.loader.exec_module
            def inject(module):
                execute(module)
                module.memory_sample=lambda:dict(available_gib=120,free_gib=120,cgroup_headroom_gib=120,
                                                 psi_full10=0,oom_kill=0,host_oom_kill=0)
            spec.loader.exec_module=inject
        return spec

    real_run=runner.run_guarded
    def inert_run(guard,command,folder,**kwargs):
        observed['invoked']+=1
        assert command[:3]==[sys.executable,'-B',str(package/'deferred_gpu.py')]
        observed['production_command_NOT_executed']=command
        observed['replacement_command']=[sys.executable,'-B',str(ROOT/'process_fixture.py')]
        kwargs['env']=dict(kwargs['env'],FIXTURE_OUT=str(out),FIXTURE_MODE='exit7')
        kwargs['sample']=guard.memory_sample
        return real_run(guard,observed['replacement_command'],folder,**kwargs)

    argv=[str(ROOT/'guarded_deferred.py'),'--guard',str(GUARD),'--guard-sha256',SHA,
          '--candidate',str(package),'--deferred-sha256',PIN,'--receipt',str(receipt),
          '--owner-lock',str(lock),'--seconds','2','--sole-gpu-owner','--model-stopped','--headroom-verified',
          '--','--mode','numeric','--cache',str(cache),'--output',str(output),
          '--protected-root',str(protected),'--protected-roots-complete']
    if a.case=='missing_inventory':argv.remove('--protected-roots-complete')
    if a.case=='old_pin':argv[argv.index('--deferred-sha256')+1]='00500630f3f875610d9e9f70d0a80d8f4237f2cf42cb626f6152f08c3d7ad83e'
    with (patch.dict(os.environ,dict(EXL3_TRIAL_MODEL_STOPPED='YES',EXL3_THREE_STAGE_AUTHORIZED='YES',MAX_JOBS='1')),
          patch.object(importlib.util,'spec_from_file_location',spec_for),
          patch.object(runner,'run_guarded',inert_run),patch.object(sys,'argv',argv)):
        try:code=runner.main()
        except SystemExit as exc:
            code=exc.code;observed['system_exit']=str(exc)
    observed['returncode']=code
    observed['lock_identity_unchanged']=lock_snapshot()==lock_before
    observed['files_unchanged_before_receipt']=file_snapshot()==before
    if a.case=='admitted':
        observed['passed']=code==7 and observed['invoked']==1 and observed['lock_identity_unchanged']
    else:
        observed['passed']=code not in (0,7) and observed['invoked']==0 and observed['files_unchanged_before_receipt'] and observed['lock_identity_unchanged']
    (out/'receipt.json').write_text(json.dumps(observed,indent=2))
    print(json.dumps(observed,indent=2))
    assert observed['passed'],'Final hardened package must pass actual CLI binding and propagate inert exit7'


if __name__=='__main__':main()
