"""DEFERRED actual-device micro-preflight. NOT RUN by local verifier.
No model, checkpoint, exllamav3 import or network. CUDA context still uses memory.
Run only under future exclusive ownership and explicit device-probe permission.
"""
import argparse,ast,json
from pathlib import Path
from types import SimpleNamespace as S

def main():
 p=argparse.ArgumentParser();p.add_argument('--authorize-device-probe',action='store_true');p.add_argument('--diag',type=Path,default=Path(__file__).parent/'candidate/width_diag.py');a=p.parse_args()
 if not a.authorize_device_probe:p.error('deferred: requires exclusive-owner permission and --authorize-device-probe')
 import torch  # Deliberately deferred, never imported by local tests/verifier.
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; do not load model')
 tree=ast.parse(a.diag.read_text());selected=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('finite','inp')]
 assert len(selected)==2
 for outdev,maskdev in [('cpu','cuda:0'),('cpu','cpu'),('cuda:0','cuda:0'),('cuda:0','cpu')]:
  for outtype,masktype in [(torch.float16,torch.bfloat16),(torch.float32,torch.float16)]:
   mask=torch.arange(4096,device=maskdev,dtype=torch.float32).remainder(7).to(masktype)
   y=torch.zeros((1,16,4096),device=outdev,dtype=outtype)
   # The production slice-assignment contract is tested, NOT changed to suit capture.
   y[:,1:,:]=mask.to(y.dtype)
   saved=y.clone();ptr=y.data_ptr();rows=[]
   env={'torch':torch,'state':{'active':True},'dm':S(input_layer=S(mask_embedding=mask)),'original_input':lambda *args,**kw:y,'emit':lambda kind,**kw:rows.append({'kind':kind,**kw})}
   exec(compile(ast.Module(body=selected,type_ignores=[]),str(a.diag),'exec'),env)
   result=env['inp'](torch.zeros((1,1),dtype=torch.long),{})
   assert result is y and y.data_ptr()==ptr and torch.equal(y,saved)
   assert rows[-1]['mask_equal'] and rows[-1]['finite']
   # Exact mismatch and nonfinite failures must remain failures.
   for bad in (3.14159,float('nan')):
    y[0,1,0]=bad
    try:env['inp'](torch.zeros((1,1),dtype=torch.long),{})
    except AssertionError:pass
    else:raise AssertionError('invalid output accepted')
    y.copy_(saved)
   print(json.dumps({'status':'device_contract_pass','output_device':outdev,'mask_device':maskdev,'output_dtype':str(outtype),'mask_dtype':str(masktype)}))
 torch.cuda.synchronize()
 print('MICRO_PREFLIGHT_ONLY: not neural/page/stream/kernel correctness or performance proof')
if __name__=='__main__':main()
