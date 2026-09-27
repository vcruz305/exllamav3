import sys,json,os,subprocess,time,traceback,hashlib,urllib.request
from pathlib import Path
B=Path('/workspace/mimo-tune');O=B/'round8-cost-width';sys.path[:0]=[str(B/'round6-quant-readback'),str(B)]
import aba
aba.O=O
from guard_uma import memory_sample
from bench_http import request
ROLL=B/'round6-quant-readback/start_quant_readback.py'
def put(p,s):p.write_text(s);assert p.read_text()==s
put(O/'preflight.json',(B/'round6-quant-readback/preflight.json').read_text())
def identity():
 run=Path((B/'france-active-run.txt').read_text().strip());pids=json.loads((run/'pid.json').read_text());rows=[]
 for name in ('guard_pid','server_pid'):
  p=Path('/proc')/str(pids[name]);rows.append(dict(pid=pids[name],starttime=p.joinpath('stat').read_text().rsplit(') ',1)[1].split()[19],args=p.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')))
 assert rows[1]['args']==pids['command']
 assert aba.health()['requests']==0
 return dict(run=str(run),pids=pids,processes=rows)
W=O/'width'
def launch(tag,launcher):
 before=identity();put(W/(tag+'-before.json'),json.dumps(before,indent=2))
 for r in before['processes']:assert Path('/proc',str(r['pid']),'stat').read_text().rsplit(') ',1)[1].split()[19]==r['starttime']
 p=subprocess.run(['/usr/bin/python3',str(B/'stop_launch.py'),str(launcher)],capture_output=True,text=True,timeout=360)
 put(W/(tag+'-launch.log'),p.stdout+'\n'+p.stderr);assert p.returncode==0 and 'LAUNCH_OK True' in p.stdout,p.stdout+p.stderr
 run=Path((B/'france-active-run.txt').read_text().strip());deadline=time.monotonic()+500
 while True:
  if (run/'result.json').exists():raise RuntimeError((run/'result.json').read_text()+(run/'server.log').read_text()[-4000:])
  try:
   if aba.health()['healthy']:break
  except Exception:pass
  assert time.monotonic()<deadline;time.sleep(2)
 print('WIDTH_READY',tag,run,flush=True);return run
assert (O/'cost-complete.json').exists() and (O/'restored.json').exists()
assert identity()['run']==json.loads((O/'restored.json').read_text())['run']
error=None
try:
 for ndt in (1,7,15):
  assert time.time()<1790460423.9056594+32*60,'reserve recovery time'
  run=launch('ndt'+str(ndt),W/('start_ndt'+str(ndt)+'.py'))
  cases=[dict(case='tiny',prompt='What is 1 plus 1? Reply with only the integer.',cap=1)]+json.loads((W/'prompts.json').read_text())
  cases += [dict(case='arithmetic',prompt='What is 17 multiplied by 19? Reply with only the integer.',cap=16),dict(case='json',prompt='Return only valid JSON with exactly these values: {"name":"oak","count":3}.',cap=32),dict(case='short-code',prompt='Write only a complete Python function add(a, b) that returns their sum. No markdown or explanation.',cap=64)]
  for c in cases:
   m=memory_sample();assert m['available_gib']>16 and m['host_oom_kill']==0
   rr=request(c['prompt'],c['cap']);rr.update(c);rr.update(ndt=ndt,instrumented=True,run=str(run),memory_after=memory_sample())
   with (W/'responses.jsonl').open('a') as f:f.write(json.dumps(rr)+'\n')
   print('WIDTH_RESPONSE',ndt,c['case'],rr['usage'],repr(rr['text']),flush=True)
   if 'prompt_tokens' in c:assert rr['usage']['prompt_tokens']==c['prompt_tokens']
   if c['case']=='arithmetic':assert rr['text'].strip()=='323'
   if c['case']=='json':assert json.loads(rr['text'])=={'name':'oak','count':3}
   if c['case']=='short-code':
    import ast
    tree=ast.parse(rr['text']);assert rr['finish']=='stop' and any(isinstance(x,ast.FunctionDef) and x.name=='add' for x in tree.body)
   ev=[json.loads(l) for l in (W/('ndt'+str(ndt))/'events.jsonl').read_text().splitlines()]
   assert any(r['kind']=='native_samples' and r['proposal_count']==15 for r in ev)
   assert any(r['kind']=='qkv' and r['q_shape']==[1,16,64,128] for r in ev)
  put(W/('ndt'+str(ndt)+'-completed.json'),json.dumps(dict(run=str(run),requests=len(cases))))
except BaseException:
 error=traceback.format_exc();put(W/'failure.txt',error);print(error,flush=True)
finally:
 run=launch('restored',ROLL)
 fixture=json.loads((B/'dynamic-long-fixture.json').read_text());rr=request(fixture['prompt'],24);rr.update(prompt=fixture['prompt'],expected=fixture['expected']);put(W/'restored-prefill.json',json.dumps(rr,indent=2));assert rr['text'].strip()==fixture['expected']
 rr=request('What is 17 multiplied by 19? Reply with only the integer.',16);put(W/'restored-arithmetic.json',json.dumps(rr,indent=2));assert rr['text'].strip()=='323'
 r=identity();r.update(health=aba.health(),memory=memory_sample());put(W/'restored.json',json.dumps(r,indent=2));print('WIDTH_RESTORED',json.dumps(r),flush=True)
if error:raise SystemExit(1)
