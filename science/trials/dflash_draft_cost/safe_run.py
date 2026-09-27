"""Execute a CPU source test with actual torch/exllamav3 imports forbidden."""
import importlib.abc,runpy,sys
class DenyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','exllamav3'):
            raise RuntimeError('Forbidden actual runtime import: '+fullname)
sys.meta_path.insert(0,DenyRuntime());sys.dont_write_bytecode=True
script=sys.argv[1];sys.argv=sys.argv[1:]
from pathlib import Path
sys.path.insert(0,str(Path(script).resolve().parent))
runpy.run_path(script,run_name='__main__')
