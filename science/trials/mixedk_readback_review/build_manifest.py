"""Generate relative-path SHA256 inventory for the local review package."""
from pathlib import Path
import hashlib
import json
R=Path(__file__).resolve().parent
files=[R/'baseline.tar']
files+=list(R.glob('*.py'))+list(R.glob('*.md'))+list((R/'tests').glob('*.py'))+list((R/'patches').glob('*.patch'))
for tree in ('baseline','candidate'):
    files += [R/tree/'exllamav3/modules/block_sparse_mlp.py',R/tree/'exllamav3/generator/generator.py']
files+=list((R/'prior-source/exllamav3/generator').glob('*.py'))
files=sorted(set(files))
manifest={'pin':'ca4a880e8918e1985fd25e06c6aff561666d3f14','sha256':{
    p.relative_to(R).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(R/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':len(files),'manifest_sha256':hashlib.sha256((R/'MANIFEST.json').read_bytes()).hexdigest()}))
