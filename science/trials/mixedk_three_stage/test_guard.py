"""Execute pure source guards + actual nested caller, no project/GPU imports."""
import ast, importlib.util, os, unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
from test_candidate import ROOT, TREE
P = TREE/'exllamav3/modules/mixedk_three_stage.py'

def load_guard():
    if not P.exists(): return None
    spec=importlib.util.spec_from_file_location('candidate_guard',P)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def layer():
    def linear(k,n):
        return N(quant_type='exl3',pre_scale=1.0,post_scale=1.0,weight_scale=1.0,softcap=0.0,
                 in_features=k,in_features_unpadded=k,is_sliced=False,out_features=n,out_features_unpadded=n,trim_padded_out=False,
                 inner=N(bias=None,K=2.0,mcg=False,mul1=True))
    return N(gated=True,is_quantized=True,activation_fn='silu',expert_size=4096,
             intermediate_size=2048,intermediate_size_padded=2048,num_experts=256,
             num_local_experts=256,num_experts_per_tok=8,latent_in=None,latent_out=None,
             gates=[linear(4096,2048) for _ in range(256)],
             ups=[linear(4096,2048) for _ in range(256)],downs=[linear(2048,4096) for _ in range(256)])

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.m=load_guard()
        self.assertIsNotNone(self.m,'Missing production three-stage Python guard')
    def test_metadata_all_formats_and_scalar_rejections(self):
        l=layer(); self.assertTrue(self.m.format_supported(l))
        for rate in (1,1.5,2,2.5,3,3.5,4,5,6,7,8):
            l.downs[148].inner.K=rate
            self.assertTrue(self.m.format_supported(l))
        l.gates[0].inner.K=2;l.ups[0].inner.K=4
        self.assertTrue(self.m.format_supported(l),'Gate/up K must be independent')
        for obj,attr,bad in ((l.downs[148].inner,'K',4.5),(l.gates[0].inner,'bias',object()),
                             (l.ups[1].inner,'mcg',True),(l.downs[1],'out_features_unpadded',4095),
                             (l.gates[1],'pre_scale',0.5),(l.ups[1],'post_scale',2),
                             (l.downs[1],'weight_scale',2),(l.ups[1],'softcap',1),
                             (l,'gated',False),(l,'activation_fn','gelu'),(l,'num_local_experts',128),
                             (l,'expert_size',2048),(l,'intermediate_size_padded',2176)):
            old=getattr(obj,attr);setattr(obj,attr,bad)
            self.assertFalse(self.m.format_supported(l),(attr,bad));setattr(obj,attr,old)
    def test_derived_pack_includes_rare_five_bit_down(self):
        import json, hashlib
        from collections import Counter
        raw=(ROOT/'fixtures/expert-byte-table.json').read_bytes()
        census=json.loads((ROOT/'metadata-census.json').read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),census['sha256'])
        table=json.loads(raw);counts=Counter();outliers=[]
        for layer,experts in table.items():
            self.assertEqual(len(experts),256)
            for expert,entry in experts.items():
                for projection in ('gate_proj','up_proj','down_proj'):
                    k2=entry['tensors'][projection+'.trellis']['shape'][-1]//8
                    counts[str(k2)]+=1
                    if k2 not in (4,6,8):outliers.append(dict(layer=layer,expert=expert,projection=projection,K2=k2))
        self.assertEqual(dict(counts),census['K2_counts'])
        self.assertEqual(outliers,[{'layer':'model.layers.46.mlp','expert':'148','projection':'down_proj','K2':10}])
        self.assertEqual((len(table),sum(counts.values())),(47,36096))
    def test_input_padding_and_sliced_weights_fall_back(self):
        l=layer();l.gates[0].in_features_unpadded=4095
        self.assertFalse(self.m.format_supported(l),'Padded input was incorrectly accepted')
        l.gates[0].in_features_unpadded=4096;l.ups[0].is_sliced=True
        self.assertFalse(self.m.format_supported(l),'Sliced matrix was incorrectly accepted')
    def test_decode_only_default_off_and_original_identity(self):
        l=layer(); l._mk_three_stage_ok=True; routing=object(); l.routing_fn=routing
        orig=object();cand=object();ext=N(exl3_moe_mixedk=orig,exl3_moe_mixedk_three_stage=cand)
        with patch.dict(os.environ,{},clear=True):
            self.assertIs(self.m.select_entry(l,{},1,ext,routing),orig)
        for flag in ('','0','2','true'):
            with patch.dict(os.environ,{'EXL3_MK_THREE_STAGE':flag},clear=True):
                self.assertIs(self.m.select_entry(l,{},1,ext,routing),orig)
        with patch.dict(os.environ,{'EXL3_MK_THREE_STAGE':'1'},clear=True):
            for q in range(1,9):self.assertIs(self.m.select_entry(l,{},q,ext,routing),cand)
            for q in (0,9,16,4096):self.assertIs(self.m.select_entry(l,{},q,ext,routing),orig)
            for params in ({'prefill':False},{'prefill':True},{'autosplit_measure':True},{'tp_warmup':True}):
                self.assertIs(self.m.select_entry(l,params,1,ext,routing),orig)
            self.assertIs(self.m.select_entry(l,{},1,ext,object()),orig)
            self.assertIs(self.m.select_entry(l,{},1,N(exl3_moe_mixedk=orig),routing),orig)
            l._mk_three_stage_ok=False
            self.assertIs(self.m.select_entry(l,{},1,ext,routing),orig)
    def test_actual_caller_arguments_unchanged(self):
        py=TREE/'exllamav3/modules/block_sparse_mlp.py'
        tree=ast.parse(py.read_text()); original=ast.parse((ROOT/'original/exllamav3/modules/block_sparse_mlp.py').read_text())
        def nested(t):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='run_mixedk_fused')
        old=nested(original);new=nested(tree)
        callold=next(n for n in ast.walk(old) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='exl3_moe_mixedk')
        calls=[n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='mixedk_entry']
        self.assertEqual(len(calls),1,'Production caller does not use guarded entry')
        self.assertEqual([ast.dump(a) for a in callold.args],[ast.dump(a) for a in calls[0].args])
        self.assertEqual(len(calls[0].args),35)
        self.assertEqual(ast.unparse(calls[0].args[9]),'self.activation_fn_idx')
        self.assertIn('self._mk_three_stage_ok = format_supported(self)',py.read_text())
        # Execute both complete nested functions against the SAME opaque arguments.
        attrs={n.attr:object() for n in ast.walk(old) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self'}
        attrs['_mkd_bufs']=N(**{name:object() for name in ('temp_state_g','temp_state_u','temp_intermediate_g','temp_intermediate_u')})
        shared=dict(self=N(**attrs),y=object(),final_hidden_states=object(),expert_count=object(),
                    token_sorted=object(),weight_sorted=object(),scratch=object(),tables=[object()],_mkd_rows=64)
        calls=[]
        for node in (old,new):
            ns=dict(shared,ext=N(exl3_moe_mixedk=lambda *a:calls.append(a)),mixedk_entry=lambda *a:calls.append(a))
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual nested dispatch>','exec'),ns)
            ns['run_mixedk_fused'](3,1,64,16)
        self.assertEqual(len(calls),2)
        self.assertEqual(len(calls[0]),35)
        self.assertTrue(all(a is b for a,b in zip(*calls)), 'Dispatch changed argument objects')

if __name__=='__main__':unittest.main(verbosity=2)
