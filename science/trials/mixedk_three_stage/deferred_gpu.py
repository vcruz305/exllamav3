"""Deferred bounded GPU harness. --plan/--help are CPU-only; NOT run locally.

Separate numeric, profiler and event-timing modes. An external owner memory guard
and stopped model are mandatory; authorization env vars are assertions, not proof.
"""
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('numeric','profile','timing'),default='numeric')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--poison',action='store_true',help='separate actual-intermediate debug build; numeric only')
    p.add_argument('--cache',type=Path,default=ROOT/'isolated-build-cache')
    p.add_argument('--output',type=Path)
    p.add_argument('--capture',type=Path,action='append',default=[])
    p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--samples',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.iterations<=20 or not 3<=a.samples<=5 or len(a.capture)>6:p.error('bounded iterations 1..20/samples3..5/captures<=6')
    if a.poison and a.mode!='numeric':p.error('poison builds are numeric-only, never timing/profiling evidence')
    return a


def main():
    args=arguments()
    from build_trial import sources,require_authorization,load_trial
    trees={'original':ROOT/'original','five':ROOT/'five-stage-control','three':ROOT/'tree'}
    plan={'status':'DEFERRED_NOT_COMPILED_NOT_GPU_VALIDATED','trees':{k:str(v) for k,v in trees.items()},
          'translation_units':{k:len(sources(v/'exllamav3/exllamav3_ext'))+1 for k,v in trees.items()},
          'tensor_limit_bytes':1536*1024**2,'cuda_fraction_budget_bytes':2*1024**3,
          'separate_modes':['numeric','profile','timing'],'q_shapes':list(range(1,9)),
          'poison':args.poison,'timeout_seconds':1800}
    if args.plan:print(json.dumps(plan,indent=2));return
    require_authorization()
    if not args.output:raise SystemExit('--output NEW_DIRECTORY required')
    args.output.mkdir(parents=True,exist_ok=False)
    # No GPU imports can occur before both authorization and platform checks above.
    import torch
    import gpu_reference as g
    records=[];case=None
    try:
        if torch.cuda.get_device_capability()!=(12,1):raise RuntimeError('sm121 only')
        for key in ('EXL3_MK_BPS','EXL3_MOE_TILE_N','EXL3_MK_SHPIPE','EXL3_MK_SMEM','EXL3_MOE_MIXEDK_CMAX','EXL3_MOE_MIXEDK_GROUPS'):
            if os.environ.get(key) not in (None,'','0'):raise RuntimeError('Remove conflicting knob '+key)
        torch.manual_seed(137)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.mem_get_info()[0]<3*1024**3:raise RuntimeError('Need reported 3GiB for bounded fixture; not model capacity')
        modules={k:load_trial(v,args.cache,poison=args.poison and k=='three') for k,v in trees.items()}
        def run(arm,case):
            module,mode={'original':(modules['original'],'off'),'off':(modules['three'],'off'),
                         'five':(modules['five'],'five'),'three':(modules['three'],'three')}[arm]
            return case.run(module,mode,True).clone()
        def compare(case,label,exact_original):
            outputs={arm:run(arm,case) for arm in ('original','off','five','three')}
            case.check_slots()
            assert torch.equal(outputs['original'],outputs['off']),'default-off not original'
            assert torch.equal(outputs['five'],outputs['three']),'three differs from same-full-K five-stage math'
            if exact_original:assert torch.equal(outputs['original'],outputs['three']),'complete-K original geometry identity failed'
            record=g.metrics(torch,outputs['three'],outputs['original'],label)
            record.update(q=len(case.x),five_three_bitwise=True,off_original_bitwise=True)
            for _ in range(6):
                again=run('three',case);case.check_slots()
                assert torch.equal(again,outputs['three']),'repeated lifetime/determinism failure'
            records.append(record)
            return outputs
        def profile(case,label):
            # Do NOT run with Compute Sanitizer: both subscribe to CUPTI.
            for arm in ('original','off','five','three'):
                run(arm,case)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    run(arm,case);torch.cuda.synchronize()
                names=sorted({e.name for e in prof.events() if 'exl3_moe' in e.name})
                if arm=='three':
                    assert sum('exl3_moe_three_stage_kernel<' in n for n in names)==3,names
                    assert not any('exl3_moe_mixedk_kernel<' in n or 'exl3_moe_phased_kernel<' in n for n in names),names
                elif arm=='five':assert sum('exl3_moe_phased_kernel<' in n for n in names)==5,names
                else:assert any('exl3_moe_mixedk_kernel<' in n for n in names),names
                records.append({'label':label,'arm':arm,'q':len(case.x),'actual_kernel_names':names})
                prof.export_chrome_trace(str(args.output/(label+'-'+arm+'.trace.json')))
        def timing(case,label):
            # Numerical/profiler passes are prerequisites, not performance results.
            for repetition in range(args.samples):
                for arm in ('original','off','five','three','original'):
                    module,mode={'original':(modules['original'],'off'),'off':(modules['three'],'off'),
                                 'five':(modules['five'],'five'),'three':(modules['three'],'three')}[arm]
                    for _ in range(3):case.run(module,mode,False)
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();start.record()
                    for _ in range(args.iterations):case.run(module,mode,False)
                    end.record();torch.cuda.synchronize()
                    records.append({'label':label,'q':len(case.x),'arm':arm,'repetition':repetition,'event_ms':start.elapsed_time(end)/args.iterations})
        pool=g.make_pool(torch);codes,ptrs=g.tables(torch,pool)
        if args.mode=='numeric':records.extend(g.validate_all(torch,modules['original'],modules['three'],pool,codes,ptrs))
        for q in range(1,9):
            for pattern in ('hot','spread'):
                sel,rw=g.routing(torch,q,pattern)
                case=g.Case(torch,torch.randn((q,g.H),device='cuda',dtype=torch.float16)*.1,sel,rw,codes,ptrs)
                label=f'q{q}-{pattern}'
                if args.mode=='numeric':compare(case,label,True)
                elif args.mode=='profile':profile(case,label)
                elif q in (1,8):timing(case,label)
        # Nontrivial live/excluded fixtures (not just num_active=0 early exits).
        for label,pattern,cap,lo,hi,compact in (
                ('overflow-live','mixed_counts',8,1,8,False),('tier-live','mixed_counts',64,9,64,False),
                ('compact-excluded-prefix','mixed_counts',8,1,8,True),('duplicate64','duplicates',64,1,64,False),
                ('all-excluded','duplicates',8,1,8,True)):
            sel,rw=g.routing(torch,8,pattern)
            case=g.Case(torch,torch.randn((8,g.H),device='cuda',dtype=torch.float16)*.1,sel,rw,codes,ptrs,cap,lo,hi,compact)
            if args.mode=='numeric':compare(case,label,case.args[29]>=6 or case.args[29]==0)
            elif args.mode=='profile' and case.args[29]>0:profile(case,label)
        del codes,ptrs,pool,case
        case=None;torch.cuda.empty_cache()
        # Private small selected-expert packs, not full models. No weights downloaded.
        for i,path in enumerate(args.capture):
            meta=json.loads(path.with_suffix(path.suffix+'.json').read_text())
            assert meta['pin']=='ca4a880e8918e1985fd25e06c6aff561666d3f14'
            assert meta['capture_sha256']==sha(path)
            assert meta['sorted_original_expert_ids']==sorted(set(meta['sorted_original_expert_ids']))
            assert meta['layer'] in (27,43,47,46) and 1<=meta['q']<=8
            pool,x,sel,rw=g.load_capture(torch,path);codes,ptrs=g.tables(torch,pool)
            assert len(x)==meta['q'] and len(pool)==len(meta['sorted_original_expert_ids'])
            case=g.Case(torch,x,sel,rw,codes,ptrs);label=f'capture{i}-L{meta["layer"]}-q{len(x)}'
            if args.mode=='numeric':
                outs=compare(case,label,case.args[29]>=6)
                ref=g.dense_reference(torch,modules['original'],pool,case)
                records.append(g.metrics(torch,outs['three'],ref,label+'-independent',.005,.06))
            elif args.mode=='profile':profile(case,label)
            else:timing(case,label)
            del pool,x,sel,rw,codes,ptrs,case
            case=None;torch.cuda.empty_cache()
        torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()
        assert peak<=g.MAX_TENSOR_BYTES,'allocation ceiling'
        result={'status':'executed','mode':args.mode,'poison':args.poison,'records':records,
                'peak_allocated':peak,'peak_reserved':torch.cuda.max_memory_reserved(),
                'note':'allocator caching disabled can report zero; not zero memory. No model throughput conclusion.'}
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
    except BaseException:
        (args.output/'failure.txt').write_text(traceback.format_exc())
        (args.output/'partial-records.json').write_text(json.dumps(records,indent=2))
        if case is None:case=getattr(g,'LAST_CASE',None)
        if case is not None:
            torch.save({k:getattr(case,k).detach().cpu() for k in ('x','sel','rw','counts','starts','base','scratch','out')},args.output/'failure-tensors.pt')
        raise

if __name__=='__main__':main()
