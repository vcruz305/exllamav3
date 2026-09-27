"""Offline round8 whole-round cost pooling. No model/runtime imports or replay.
A provisional instrumented observational profile, NOT a clean live calibration.
"""
import argparse, hashlib, json, math, statistics
from pathlib import Path

PIN='ca4a880e8918e1985fd25e06c6aff561666d3f14'
DRAFT_PIN='d50ead3c6a3dec221e9a595fbdc103ef60db594e'

def build(root):
    root=Path(root);sources={}
    def read(rel,lines=False):
        raw=(root/rel).read_bytes();sources[rel]=hashlib.sha256(raw).hexdigest()
        return [json.loads(s) for s in raw.splitlines()] if lines else json.loads(raw)
    responses=read('mirror/responses.jsonl',True)
    frozen=read('mirror/frozen.json')
    receipt=read('parent01-receipt.json')
    launch=read('mirror/runs/france-round8-cost-1790460677264568082/launch.json')
    proof=read('mirror/width/source-proof.json')
    raw=(root/'cost_diag.py').read_bytes();sources['cost_diag.py']=hashlib.sha256(raw).hexdigest()
    assert len(responses)==len(frozen)==8
    assert receipt['status']=='passed' and receipt['rounds']==797
    all_rows=[];requests=[]
    for ordinal,(response,request) in enumerate(zip(responses,frozen)):
        assert response['ordinal']==ordinal
        assert all(response[k]==request[k] for k in ('case','phase','rep','prompt','cap'))
        assert request['case']==('code' if ordinal%2==0 else 'prose')
        assert request['phase']==('cold' if ordinal<2 else 'warm')
        assert request['rep']==ordinal//2
        rows=read(f'mirror/cost/{ordinal}-rounds.jsonl',True)
        terminal=read(f'mirror/cost/{ordinal}-terminal.json')
        timings=read(f'mirror/cost/{ordinal}-timing.jsonl',True)
        assert terminal['rounds_dropped']==terminal['routes_dropped']==0
        assert len(timings)==2*len(rows)
        for i,row in enumerate(rows):
            assert row['round']==i and row['target_forwards']==row['draft_forwards']==1
            assert type(row['q']) is int and 1<=row['q']<=8 and row['q']==row['proposed']+1
            assert math.isfinite(row['round_host_ms']) and row['round_host_ms']>0
            assert row['unaccounted_proposals']==0
            events=[t for t in timings if t['round']==i]
            assert sorted(t['phase'] for t in events)==['drafter_forward','target_forward']
            all_rows.append(dict(row,phase=request['phase'],case=request['case'],ordinal=ordinal))
        requests.append(dict(ordinal=ordinal,case=request['case'],phase=request['phase'],rep=request['rep'],rounds=len(rows)))
    def aggregate(rows):
        out=[]
        for q in range(1,9):
            group=[r for r in rows if r['q']==q]
            if not group:continue
            values=[r['round_host_ms'] for r in group]
            out.append(dict(q=q,samples=len(group),case_counts={c:sum(r['case']==c for r in group) for c in ('code','prose')},
                            mean_ms=statistics.mean(values),min_ms=min(values),max_ms=max(values)))
        return out
    warm=[r for r in all_rows if r['phase']=='warm'];cold=[r for r in all_rows if r['phase']=='cold']
    assert len(all_rows)==receipt['rounds'] and len(warm)==606
    argv=launch['command'];arg=lambda key:argv[argv.index(key)+1]
    context=dict(source_revision=PIN,model_directory=arg('-m'),model_pack='MiMo-V2.6-Flash-RL-EXL3/2.50bpw',
                 drafter_directory=arg('-dm'),drafter_revision=DRAFT_PIN,drafter_bpw=4.0,native_block=8,
                 cache_bits=int(arg('-cq')),context_tokens=int(arg('-cs')),chunk_size=int(arg('-chunk_size')),
                 batch_size=int(arg('-ambs')),gpu_split=int(arg('-gs')),reserve_mb=int(launch['env']['EXL3_UMA_RESERVE_MB']),
                 dynamic_confidence=float(arg('-dc')),batch_verify=0,handled_elision=1,counts_elision=0,kernel='original-CUDA',
                 hardware='France-GB10',target_weight_identity='owner-attested unchanged round8 2.50bpw pack')
    assert arg('-ndt')=='7' and launch['env']['EXL3_BATCH_VERIFY']=='0'
    assert launch['env']['EXL3_MOE_MIXEDK_ELIDE_HANDLED']=='1'
    return dict(schema=1,units='ms',metric='whole_round_host',population='pooled_warm_actual_q',provisional=True,
                context=context,q_costs=aggregate(warm),cold_q_costs=aggregate(cold),requests=requests,
                warm_rounds=len(warm),cold_rounds=len(cold),total_rounds=len(all_rows),
                zero_proposal_rounds=sum(r['proposed']==0 for r in all_rows),sources=sources,
                measured_source_hashes=proof,
                limitations=['Instrumented observational costs, not clean live cost calibration.',
                  'Policy/state/workload selection bias; q8 warm has code only. No invented prose q8.',
                  'No per-position confidence or proposal IDs: cannot replay candidate windows or acceptance.',
                  'Whole-round span includes native drafter, target, sampling, cache updates. Do not sum overlapping event/readback times.',
                  'Target weight revision/hash absent from round8 evidence; runtime owner must attest identical retained pack. No weight verification performed here.',
                  'Cold code SyntaxError and warm truncation are baseline quality observations, not expected-policy reward.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=build(a.evidence)
    with a.output.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2)
    print(json.dumps(dict(warm=data['warm_rounds'],cold=data['cold_rounds'],costs=data['q_costs']),indent=2))
