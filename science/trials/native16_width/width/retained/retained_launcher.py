import json, os, subprocess, sys, time, socket
from pathlib import Path
base=Path('/workspace/mimo-tune');sys.path.insert(0,str(base))
from guard_uma import memory_sample, require_load_headroom
ready=Path('/workspace/mimo-exl3/runtime-ready.json')
assert ready.is_file(), 'Runtime verification marker missing'
assert json.loads(Path('/workspace/mimo-tune/round5-quant-drafter/download-verified.json').read_text())['revision']=='d50ead3c6a3dec221e9a595fbdc103ef60db594e'
print('RUNTIME_MARKER',str(ready),flush=True)
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        cmd=p.read_bytes()
        if b'serve_native.py' in cmd or b'hf\x00download\x00' in cmd:
            raise SystemExit('Existing model/download process: '+str(p.parent.name))
    except FileNotFoundError: pass
memfile=Path('/workspace/mimo-exl3/exllamav3/exllamav3/util/memory.py')
assert 'EXL3_UMA' in memfile.read_text(), 'UMA runtime patch not deployed'
# Match the native server's bare-bind precheck, including TIME_WAIT.
deadline=time.monotonic()+90
while True:
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',8096))
        break
    except OSError:
        if time.monotonic()>=deadline: raise
        time.sleep(1)
assert (base/'sampler-greedy-deployed.json').exists(), 'Greedy-only batch capability patch required'
assert (base/'dflash-pages-deployed.json').exists(), 'Native block reservation fix required'
draft_ready=json.loads((base/'dflash-ready.json').read_text())
assert draft_ready['config']['dflash_config']['tap_shift']==0
assert draft_ready['config']['block_size']==8
assert Path(draft_ready['fixed'],'mask_embedding.safetensors').is_file()
s=memory_sample();print('LOAD_PREFLIGHT',json.dumps(s),flush=True)
require_load_headroom(s,memory_policy='uma')
run=base/'runs'/('france-round6-quant-readback-'+str(time.time_ns()))
cfg={'outdir':str(run),'memory_policy':'uma','max_seconds':43200,
 'env':{'CUDA_HOME': '/usr/local/cuda', 'PATH': '/workspace/mimo-exl3/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin', 'EXL3_ROOT': '/workspace/mimo-tune/60tps-round3/python-shadow', 'PYTHONPATH': '/workspace/mimo-tune/60tps-round3/python-shadow', 'PYTHONUNBUFFERED': '1', 'EXL3_UMA': '1', 'EXL3_UMA_RESERVE_MB': '8192', 'TORCH_CUDA_ARCH_LIST': '12.1', 'OMP_NUM_THREADS': '8', 'EXL3_BATCH_VERIFY': '0', 'EXL3_MOE_MIXEDK_ELIDE_HANDLED': '1'},
 'command':['/workspace/mimo-exl3/venv/bin/python', '/workspace/mimo-tune/round6-quant-readback/serve_native.py', '-m', '/workspace/MiMo-V2.6-Flash-RL-EXL3/2.50bpw', '-dm', '/workspace/mimo-exl3/models/MiMo-V2.6-Flash-RL-dflash-EXL3-4.0bpw', '-ndt', '7', '-dds', '-dc', '0.6', '-gs', '106', '-cs', '4096', '-cq', '4', '-ambs', '1', '-chunk_size', '4096', '-ccs', '0', '-rcs', '0.25', '--max-active-requests', '1', '--max-pending-requests', '2', '--max-model-len', '4096', '--host', '127.0.0.1', '--port', '8096', '--request-timeout', '600', '-lv']}
run.mkdir(parents=True,exist_ok=True)
(run/'launch.json').write_text(json.dumps(cfg,indent=2))
p=base/'france-uma.json';p.write_text(json.dumps(cfg,indent=2))
(base/'france-active-run.txt').write_text(str(run))
with (base/'guard-france-uma.log').open('w') as log:
    child=subprocess.Popen(['/usr/bin/python3',str(base/'guard_uma.py'),str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(2)
print('LAUNCHED',json.dumps({'guard_pid':child.pid,'running':child.poll() is None,'run':str(run),'configuration':cfg}),flush=True)
if child.poll() is not None:
    print((base/'guard-france-uma.log').read_text(),flush=True)
    raise SystemExit(1)
