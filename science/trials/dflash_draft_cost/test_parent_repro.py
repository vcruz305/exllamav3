"""Retained-MiMo configuration eligibility regression; CPU/source only.

Execute actual parser declarations and MiMo recurrent-capability branch (disclosed
source slices), then the complete source-extracted Generator constructor from the
existing harness. RecurrentCache is an opaque allocation boundary, not CUDA state.
No model/network/GPU imports; temporary attestation is TEST-ONLY and deleted.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
R=HERE.parent
P=R/'dflash-cost-aware-candidate'
U=R/'dflash-pages-candidate/upstream'
os.environ['DFLASH_SOURCE_ROOT']=str(P/'candidate')
sys.path.insert(0,str(P))
import test_cost as t


def parser_and_caps():
    model_init=ast.parse((U/'exllamav3/model_init.py').read_text())
    add=next(x for x in model_init.body if isinstance(x,ast.FunctionDef) and x.name=='add_args')
    calls=[]
    for node in ast.walk(add):
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
            c=node.value
            if isinstance(c.func,ast.Attribute) and c.func.attr=='add_argument' and c.args:
                if isinstance(c.args[0],ast.Constant) and c.args[0].value in ('-swa_full','-rcs'):
                    calls.append(node)
    assert len(calls)==2
    parser=argparse.ArgumentParser(allow_abbrev=False)
    exec(compile(ast.Module(body=calls,type_ignores=[]),'<actual two parser declarations>','exec'),
         {'parser':parser,'default_recurrent_cache_size':4.0})
    launch=json.loads((R/'round8-width-harness-repair/source/retained_launch.json').read_text())
    args,_=parser.parse_known_args(launch['command'][2:])
    source=ast.parse((U/'exllamav3/architecture/mimo_v2.py').read_text())
    ctor=next(m for c in source.body if isinstance(c,ast.ClassDef)
              for m in c.body if isinstance(m,ast.FunctionDef) and m.name=='__init__'
              and any(a.arg=='swa_full' for a in m.args.args))
    branch=next(n for n in ctor.body if isinstance(n,ast.If)
                and ast.unparse(n.test)=='not self.swa_full' and 'recurrent_states' in ast.unparse(n))
    target=N(swa_full=args.swa_full,caps={},recurrent_state_cls=None)
    marker=object()
    exec(compile(ast.Module(body=[branch],type_ignores=[]),'<actual MiMo capability branch>','exec'),
         {'self':target,'SWAState':marker})
    assert not args.swa_full and target.caps['recurrent_states'] is True
    assert target.caps['default_recurrent_checkpoint_interval']==2048
    assert target.recurrent_state_cls is marker
    return args,dict(target.caps)


class RetainedConfiguration(unittest.TestCase):
    def construct(self,enabled,recurrent=True):
        args,caps=parser_and_caps()
        path=P/'candidate/exllamav3/generator/draft_cost.py'
        spec=importlib.util.spec_from_file_location('actual_cost_policy',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        C=module.DFlashCostPolicy
        raw=(P/'profile.json').read_bytes();profile=json.loads(raw);ctx=profile['context']
        d=t.n.DraftAdapter(t.BASE);d.config.directory=ctx['drafter_directory']
        cfg=copy.copy(d.config);cfg.directory=ctx['model_directory']
        cache=N(max_num_tokens=4096,layers={0:N(k_bits=4,v_bits=4)},num_slots=1,reset_states=lambda:None)
        dc=t.n.Cache();dc.max_num_tokens=4096;dc.layers={0:N(k_bits=4,v_bits=4)}
        class OpaqueRecurrentAllocation:
            def __init__(self,model,size):self.model=model;self.size=size
        kw=dict(model=N(config=cfg,caps=caps if recurrent else {}),cache=cache,
                tokenizer=N(get_id_to_piece_list=lambda *a:['x']*64),max_batch_size=1,max_chunk_size=4096,
                draft_model=d,draft_cache=dc,num_draft_tokens=7,dynamic_draft_tokens=True,draft_confidence=.6,
                recurrent_cache_size=int(args.recurrent_cache_size*1024**3))
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY-',dir=HERE) as tmp:
            att=Path(tmp)/'test-only-not-runtime-attestation.json'
            att.write_text(json.dumps({'context':ctx,'attest_same_round8_weights_and_runtime':True}))
            env=dict(EXL3_DFLASH_COST_AWARE='1' if enabled else '0',EXL3_DFLASH_COST_PROFILE=str(P/'profile.json'),
                     EXL3_DFLASH_COST_PROFILE_SHA256=hashlib.sha256(raw).hexdigest(),EXL3_DFLASH_COST_CONTEXT=str(att),
                     EXL3_MOE_MIXEDK_ELIDE_HANDLED='1',EXL3_UMA_RESERVE_MB='8192')
            with patch.dict(os.environ,env,clear=True),patch.dict(t.n.h.NS_GLOBALS,
                    {'DFlashCostPolicy':C,'_BATCH_VERIFY':False,'RecurrentCache':OpaqueRecurrentAllocation}):
                return t.n.h.Generator(**kw)

    def test_disabled_retained_profile_constructs_recurrent_cache(self):
        g=self.construct(False)
        self.assertIsNotNone(g.recurrent_cache)
        self.assertEqual(g.recurrent_cache.size,g.recurrent_cache_size)
        self.assertEqual(g.recurrent_checkpoint_interval,2048)
        self.assertIsNone(g.dflash_cost_policy)

    def test_enabled_measured_retained_profile_must_be_eligible(self):
        try:g=self.construct(True)
        except ValueError as exc:
            self.fail('Measured MiMo profile is rejected by its cost-policy guard: '+str(exc))
        self.assertIsNotNone(g.recurrent_cache)
        self.assertIsNotNone(g.dflash_cost_policy)

    def test_nonrecurrent_fixture_is_not_the_retained_profile(self):
        g=self.construct(True,recurrent=False)
        self.assertIsNone(g.recurrent_cache)
        self.assertIsNotNone(g.dflash_cost_policy)


if __name__=='__main__':
    unittest.main(verbosity=2)
