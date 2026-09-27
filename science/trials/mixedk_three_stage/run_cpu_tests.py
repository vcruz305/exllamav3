"""Stdlib-only test runner with explicit no-GPU import enforcement and JSON totals."""
import argparse, importlib.abc, json, sys, unittest
from pathlib import Path

class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','exllamav3'):
            raise RuntimeError('CPU verification forbids importing '+fullname)
        return None
sys.meta_path.insert(0,BlockGPU())

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=('all','preservation','red'),default='all');p.add_argument('--json',type=Path,required=True)
    a=p.parse_args()
    names={'all':['test_candidate','test_guard','test_layout','test_native_guard','test_ownership','test_preservation','test_harness'],
           'preservation':['test_layout','test_preservation'],'red':['test_candidate.FusionContracts']}[a.suite]
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    totals={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
            'success':result.wasSuccessful(),'forbidden_imports':[n for n in sys.modules if n.split('.')[0] in ('torch','exllamav3')]}
    with a.json.open('x') as f:json.dump(totals,f,indent=2)
    print(json.dumps(totals));raise SystemExit(not result.wasSuccessful())

if __name__=='__main__':main()
