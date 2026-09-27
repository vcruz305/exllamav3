"""CPU-only checks for the deferred harness ABI and safety boundaries."""
import ast, importlib.util, json, os, subprocess, sys, unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
from test_candidate import ROOT

class HarnessTests(unittest.TestCase):
    def test_case_modes_and_compact_gather_use_actual_method(self):
        source=ast.parse((ROOT/'gpu_reference.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Case')
        node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        ns={'os':os};exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual Case.run>','exec'),ns)
        calls=[]
        class Tensor:
            def __init__(self,name):self.name=name
            def zero_(self):calls.append(('zero',self.name))
            def fill_(self,v):calls.append(('fill',self.name))
            def __getitem__(self,i):return self
        args=[object() for _ in range(35)]
        case=N(args=args,out=Tensor('out'),scratch=Tensor('scratch'),temps=[Tensor(str(i)) for i in range(4)],
               flat=object(),inv=object(),starts=Tensor('expert_start'),base=Tensor('slot_base'),kind=object(),ws=object())
        module=N(run=lambda *a:calls.append(('run',a)),run_three=lambda *a:calls.append(('run_three',a)),
                 gather=lambda *a:calls.append(('gather',a)))
        ns['E']=256
        for mode in ('off','five','three'):
            calls.clear()
            with patch.dict(os.environ,{},clear=True):
                self.assertIs(ns['run'](case,module,mode,True),case.out)
                self.assertEqual(os.environ['EXL3_MK_THREE_STAGE'],'1' if mode=='three' else '0')
                self.assertEqual(os.environ['EXL3_MK_PHASED'],'1' if mode=='five' else '0')
            run=next(c for c in calls if c[0] in ('run','run_three'))
            self.assertEqual(run[0],'run_three' if mode=='three' else 'run')
            self.assertTrue(all(x is y for x,y in zip(run[1],args)))
            gather=next(c[1] for c in calls if c[0]=='gather')
            self.assertIs(gather[4],case.starts);self.assertIs(gather[5],case.base)
            self.assertIsNot(gather[4],gather[5])
    def test_thresholds_dtype_capture_and_debug_controls(self):
        source=(ROOT/'gpu_reference.py').read_text()
        self.assertIn('rms_limit=0.002, peak_limit=0.025',source)
        self.assertIn('0.005, 0.06',source)
        self.assertIn("weights_only=True",source)
        self.assertIn('dtype=torch.int32',source);self.assertIn('dtype=torch.int64',source)
        build=(ROOT/'build_trial.py').read_text()
        for token in ('-DEXL3_THREE_STAGE_POISON','-fvisibility=hidden','RTLD_GLOBAL','EXL3_THREE_STAGE_AUTHORIZED'):
            self.assertIn(token,build)
        gpu=(ROOT/'deferred_gpu.py').read_text()
        self.assertIn("a.poison and a.mode!='numeric'",gpu)
        self.assertIn("outputs['five'],outputs['three']",gpu)
        self.assertIn("outputs['original'],outputs['off']",gpu)
    def test_plan_is_cpu_only_and_sources_complete(self):
        run=subprocess.run([sys.executable,'-B',str(ROOT/'deferred_gpu.py'),'--plan'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        plan=json.loads(run.stdout)
        self.assertEqual(plan['translation_units'],{'original':46,'five':46,'three':46})
        self.assertEqual(plan['status'],'DEFERRED_NOT_COMPILED_NOT_GPU_VALIDATED')
        self.assertNotIn('torch',sys.modules);self.assertNotIn('exllamav3',sys.modules)
    def test_authorization_refuses_before_gpu_import(self):
        spec=importlib.util.spec_from_file_location('isolated_builder',ROOT/'build_trial.py')
        b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(SystemExit):b.require_authorization()
        with patch.dict(os.environ,{'EXL3_TRIAL_MODEL_STOPPED':'YES','EXL3_THREE_STAGE_AUTHORIZED':'YES'},clear=True),patch.object(sys,'platform','win32'):
            with self.assertRaises(SystemExit):b.require_authorization()
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
