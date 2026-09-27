"""Emit byte-preserving source-only patches; baseline/candidate stay isolated."""
from pathlib import Path
import difflib
import hashlib
import json
R=Path(__file__).resolve().parent
pairs={'mixedk-dead-readback.patch':['exllamav3/modules/block_sparse_mlp.py'],
       'target-round-diagnostics.patch':['exllamav3/generator/generator.py']}
for name, rels in pairs.items():
    parts=[]
    for rel in rels:
        a=(R/'baseline'/rel).read_bytes().decode('latin-1')
        b=(R/'candidate'/rel).read_bytes().decode('latin-1')
        parts.extend(difflib.unified_diff(a.splitlines(keepends=True),b.splitlines(keepends=True),
                                        fromfile='a/'+rel,tofile='b/'+rel,lineterm='\n'))
    if parts:
        p=R/'patches'/name
        p.write_bytes(''.join(parts).encode('latin-1'))
        print(name, hashlib.sha256(p.read_bytes()).hexdigest())
