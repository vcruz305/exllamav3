"""Run original native-width source controls without writing their study folder.
On a candidate, exclude ONLY the original-study byte/clean-Git assertion; the
artifact verifier checks allowed changes and pristine baseline separately.
"""
import argparse,sys,os,unittest,io,json,importlib.abc
from pathlib import Path
class BlockGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','exllamav3'):
            raise RuntimeError('Forbidden actual runtime import: '+fullname)
sys.meta_path.insert(0,BlockGPUImports());sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--label',required=True);p.add_argument('--baseline',action='store_true');a=p.parse_args()
os.environ['DFLASH_SOURCE_ROOT']=a.source
sys.path.insert(0,str(HERE.parent/'dflash-native-width-study'))
import regressions
names=unittest.defaultTestLoader.getTestCaseNames(regressions.Regressions)
if not a.baseline:names.remove('test_originals_unchanged')
suite=unittest.TestSuite(regressions.Regressions(name) for name in names)
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
text=stream.getvalue();print(text)
with (HERE/'evidence'/f'{a.label}.txt').open('x') as f:f.write(text)
with (HERE/'evidence'/f'{a.label}.json').open('x') as f:json.dump(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful(),source=a.source,excluded=[] if a.baseline else ['test_originals_unchanged']),f,indent=2)
assert 'torch' not in sys.modules and 'exllamav3' not in sys.modules
sys.exit(not result.wasSuccessful())
