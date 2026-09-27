import importlib.util,importlib,sys,runpy,os,json,hashlib
from pathlib import Path
root=os.environ['EXL3_ROOT'];sys.path.insert(0,root)
spec=importlib.util.find_spec('exllamav3_ext')
assert spec and spec.origin and spec.origin.endswith('.so')
assert hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()=='02b0ae5bc8414d335facca41083f561cf24f41d56ef6e1e73a8b41fb16f80207'
sparse=importlib.import_module('exllamav3.modules.block_sparse_mlp')
assert Path(sparse.__file__).resolve()==Path(root+'/exllamav3/modules/block_sparse_mlp.py').resolve()
print('ROUND6_IMPORT',json.dumps(dict(python=sparse.__file__,source_sha256=hashlib.sha256(Path(sparse.__file__).read_bytes()).hexdigest(),extension=spec.origin,elide=getattr(sparse,'MIXEDK_ELIDE_HANDLED',False))),flush=True)
server='/workspace/MiMo-V2.6-Flash-RL-EXL3-recipe/server';sys.path.insert(0,server)
runpy.run_path(server+'/serve_native.py',run_name='__main__')
