"""CPU source execution only; NumPy fake tensors, no torch/exllamav3 import.
Full DFlash producer + actual calibrator + labeled native_harness verification slice.
"""
import os, sys, json, io, unittest, importlib.util, copy, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('DFLASH_SOURCE_ROOT', HERE/'candidate'))
os.environ['DFLASH_SOURCE_ROOT'] = str(ROOT)
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent/'dflash-native-width-study'))
import native_harness as n
from types import SimpleNamespace as NS
BASE = n.META/'hub-config.json'

def ready():
    g=n.generator(BASE,7,dynamic=True)
    j=n.queue(g,240,100)
    for _ in range(80): g.draft_calibrator.add_label(10,True)
    j.sampler=NS(supports_batch_verify=True,reqs_past_ids=False)
    return g,j

class Integration(unittest.TestCase):
    def test_opt_in_selector_reaches_complete_producer(self):
        g,j=ready()
        calls=[]
        g.dflash_cost_policy=NS(select=lambda *a: calls.append(a) or 0)
        self.assertIsNone(n.h.draft(g), 'old producer ignores cost-aware opt-in')
        self.assertEqual(len(calls),1)
        self.assertEqual(g._draft_conf_round['window'],0)
        self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
        self.assertEqual(g._draft_conf_round['ids'].tolist(),[[2,3,4,5,6,7,8]])
    def test_default_off_dds_preserved(self):
        g,j=ready()
        g.draft_model.head.cut=2
        p=n.h.draft(g)
        self.assertEqual(p.tolist(),[[2,3]])
        self.assertEqual(g._draft_conf_round['window'],2)
        self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)

def policy_class(test):
    path=ROOT/'exllamav3/generator/draft_cost.py'
    test.assertTrue(path.exists(), 'cost-aware policy source missing')
    spec=importlib.util.spec_from_file_location('draft_cost_cpu',path)
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.DFlashCostPolicy

def learned(ps):
    cal=n.Calibrator(.6)
    for score,p in enumerate(ps):
        for i in range(100): cal.add_label(score, i < round(100*p))
    return cal

class Objective(unittest.TestCase):
    def choose(self,ps,costs,cap=None):
        C=policy_class(self); obj=C.__new__(C); obj.costs=tuple(costs)
        return obj.choose(learned(ps),list(range(len(ps))),len(ps) if cap is None else cap)
    def test_modeled_argmax_known_probabilities(self):
        self.assertEqual(self.choose([0]*7,[1,2,3,4,5,6,7,8]),0)
        self.assertEqual(self.choose([1]*7,[1]*8),7)
        self.assertEqual(self.choose([.5]*7,[1,1.1,1.2,100,100,100,100,100]),2)
        self.assertEqual(self.choose([1]*7,[1,2,3,4,5,6,7,8]),0)
        self.assertEqual(self.choose([1]*7,[1]*8,2),2)
    def test_nonmonotone_probability_exhaustive_oracle(self):
        import itertools
        ps=[.8,.2,.9,.4,.7,.1,.9]; costs=[10,11,12,12.1,20,21,22,23]
        yields=[]
        for k in range(8):
            expected=0.
            for bits in itertools.product((False,True),repeat=k):
                probability=1.; tokens=1
                for p,b in zip(ps,bits): probability*= p if b else 1-p
                for b in bits:
                    if not b: break
                    tokens+=1
                expected+=probability*tokens
            yields.append(expected/costs[k])
        self.assertEqual(self.choose(ps,costs), max(range(8),key=lambda k:yields[k]))
    def test_trust_requires_burnin_and_same_populated_bins(self):
        C=policy_class(self); obj=C.__new__(C); obj.costs=(1,)*8
        cal=n.Calibrator(.6)
        self.assertIsNone(obj.choose(cal,[1.]*7,7))
        for _ in range(100):cal.add_label(0,True)
        self.assertEqual(cal.estimate(1),1.) # actual nearest-bin optimism is NOT trusted
        self.assertIsNone(obj.choose(cal,[1.]*7,7))
        for _ in range(8):cal.add_label(1,True)
        self.assertEqual(obj.choose(cal,[1.]*7,7),7)
        cal.decay_step()
        self.assertIsNone(obj.choose(cal,[1.]*7,7))
        for bad in (math.nan,math.inf,-math.inf):
            self.assertIsNone(obj.choose(cal,[bad]*7,7))
        self.assertIsNone(obj.choose(None,[1.]*7,7))
        self.assertIsNone(obj.choose(cal,[1.]*7,8))
        self.assertIsNone(obj.choose(cal,[],1))

class Guards(unittest.TestCase):
    def test_policy_guard_and_feedback(self):
        C=policy_class(self)
        self.assertTrue(hasattr(C,'select'),'missing guarded live selector')
        g,j=ready(); n.equip_full_receive(j)
        g.dflash_cost_policy=C.__new__(C); g.dflash_cost_policy.costs=(1,100,100,100,100,100,100,100)
        p=n.h.draft(g)
        self.assertIsNone(p)
        self.assertEqual(g._draft_conf_round['window'],0)
        before=g.draft_calibrator.total
        r=n.verify(g,j,p)
        self.assertEqual(r.lengths,[1])
        self.assertAlmostEqual(g.draft_calibrator.total,(before+1)*.995)
        self.assertIsNone(g._draft_conf_round)
        g.dflash_cost_policy.costs=(1,)*8
        p=n.h.draft(g)
        self.assertEqual(p.shape,(1,7))
        self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
        before=g.draft_calibrator.total
        n.verify(g,j,p)
        self.assertAlmostEqual(g.draft_calibrator.total,(before+7)*.995)
    def test_unsupported_and_boundary_fallback_matrix(self):
        C=policy_class(self); self.assertTrue(hasattr(C,'select'),'missing guarded live selector')
        changes=[('filters',[object()]),('forced_ids',object()),('return_probs',True),('return_top_tokens',1),
                 ('return_logits',True),('banned_strings',['x']),('checkpoint',{}),('new_tokens',-1),
                 ('max_new_tokens',7),('max_rq_tokens',13)]
        for key,value in changes:
            with self.subTest(key=key):
                g,j=ready(); n.equip_full_receive(j)
                obj=C.__new__(C); obj.costs=(1,100,100,100,100,100,100,100)
                setattr(j,key,value)
                self.assertIsNone(obj.select(g,n.Tensor.wrap([[10.]*7]),7))
        g,j=ready(); n.equip_full_receive(j); obj=C.__new__(C);obj.costs=(1,)*8
        for change in ('multi','multiseq','mtp','native16','request8','recurrent','sampler','shape'):
            g,j=ready(); n.equip_full_receive(j)
            if change=='multi':g.active_jobs.append(j)
            if change=='multiseq':j.sequences.append(j.sequences[0])
            if change=='mtp':g.mtp_draft=True
            if change=='native16':g.draft_model.config.block_size=16
            if change=='request8':g.num_draft_tokens=8
            if change=='recurrent':g.recurrent_cache=object()
            if change=='sampler':j.sampler.supports_batch_verify=False
            conf=n.Tensor.wrap([[10.]* (6 if change=='shape' else 7)])
            self.assertIsNone(obj.select(g,conf,7),change)
    def test_confidence_hard_cap_and_truncated_bonus_label(self):
        C=policy_class(self); self.assertTrue(hasattr(C,'select'),'missing guarded live selector')
        for cap in (0,1,3,7):
            g,j=ready(); n.equip_full_receive(j)
            obj=C.__new__(C); obj.costs=(1,)*8;g.dflash_cost_policy=obj
            original=g.draft_model.sample_from_state
            def sample(state,params):
                out=original(state,params);params['draft_confidence_len']=cap;return out
            g.draft_model.sample_from_state=sample
            before=g.draft_calibrator.total
            p=n.h.draft(g)
            self.assertEqual(None if p is None else p.shape[-1],None if cap==0 else cap)
            self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
            n.verify(g,j,p)
            self.assertAlmostEqual(g.draft_calibrator.total,before if cap==0 else (before+min(cap+1,7))*.995)

class Configuration(unittest.TestCase):
    def test_default_off_constructor_does_not_read_profile(self):
        from unittest.mock import patch
        with patch.dict(os.environ,{'EXL3_DFLASH_COST_AWARE':'0','EXL3_DFLASH_COST_PROFILE':'missing'}):
            g,j=ready()
            self.assertTrue(hasattr(g,'dflash_cost_policy'),'constructor opt-in contract missing')
            self.assertIsNone(g.dflash_cost_policy)
    def test_profile_validation_and_context_binding(self):
        C=policy_class(self)
        self.assertTrue(hasattr(C,'from_profile'),'validated profile loader missing')
        p=json.loads((HERE/'profile.json').read_text());ctx=p['context']
        obj=C.from_profile(p,dict(ctx))
        self.assertEqual(obj.costs,tuple(r['mean_ms'] for r in p['q_costs']))
        bads=[]
        for key,value in [('units','seconds'),('metric','target_event'),('schema',2),('provisional',False),('population','prose')]:
            b=copy.deepcopy(p);b[key]=value;bads.append(b)
        for value in (math.nan,math.inf,0,-1,True,'61'):
            b=copy.deepcopy(p);b['q_costs'][0]['mean_ms']=value;bads.append(b)
        for key,value in [('samples',0),('q',2),('case_counts',{'code':99,'prose':99})]:
            b=copy.deepcopy(p);b['q_costs'][0][key]=value;bads.append(b)
        b=copy.deepcopy(p);b['q_costs'].pop();bads.append(b)
        b=copy.deepcopy(p);b['sources']={};bads.append(b)
        b=copy.deepcopy(p);del b['context']['drafter_revision'];bads.append(b)
        for b in bads:
            with self.subTest(b=str(b)[:80]),self.assertRaises(ValueError):C.from_profile(b,ctx)
        for key,value in [('native_block',16),('model_pack','other'),('source_revision','0'*40),('drafter_revision','0'*40),('cache_bits',8)]:
            context=dict(ctx);context[key]=value
            with self.assertRaises(ValueError):C.from_profile(p,context)
            b=copy.deepcopy(p);b['context']=context
            with self.assertRaises(ValueError):C.from_profile(b,context)
    def test_opt_in_init_requires_profile_and_attestation(self):
        from unittest.mock import patch
        C=policy_class(self);n.h.NS_GLOBALS['DFlashCostPolicy']=C;n.h.NS_GLOBALS['_BATCH_VERIFY']=False
        with patch.dict(os.environ,{'EXL3_DFLASH_COST_AWARE':'1'},clear=True):
            with self.assertRaisesRegex(ValueError,'profile|PROFILE'):ready()
        with patch.dict(os.environ,{'EXL3_DFLASH_COST_AWARE':'yes'},clear=True):
            with self.assertRaises(ValueError):ready()

    def test_full_constructor_opt_in_and_runtime_matrix(self):
        from unittest.mock import patch
        import tempfile,hashlib
        C=policy_class(self);n.h.NS_GLOBALS.update(DFlashCostPolicy=C,_BATCH_VERIFY=False)
        p=json.loads((HERE/'profile.json').read_text());ctx=p['context']
        def construct(change=None):
            d=n.DraftAdapter(BASE);d.config.directory=ctx['drafter_directory']
            target_cfg=copy.copy(d.config);target_cfg.directory=ctx['model_directory']
            cache=NS(max_num_tokens=4096,layers={0:NS(k_bits=4,v_bits=4)})
            dc=n.Cache();dc.max_num_tokens=4096;dc.layers={0:NS(k_bits=4,v_bits=4)}
            kw=dict(model=NS(config=target_cfg,caps={}),cache=cache,tokenizer=NS(get_id_to_piece_list=lambda *a:['x']*64),
                    max_batch_size=1,max_chunk_size=4096,draft_model=d,draft_cache=dc,num_draft_tokens=7,
                    dynamic_draft_tokens=True,draft_confidence=.6)
            if change:change(kw)
            return n.h.Generator(**kw)
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            att=Path(tmp)/'context.json';att.write_text(json.dumps({'context':ctx,'attest_same_round8_weights_and_runtime':True}))
            env=dict(EXL3_DFLASH_COST_AWARE='1',EXL3_DFLASH_COST_PROFILE=str(HERE/'profile.json'),
                     EXL3_DFLASH_COST_PROFILE_SHA256=hashlib.sha256((HERE/'profile.json').read_bytes()).hexdigest(),
                     EXL3_DFLASH_COST_CONTEXT=str(att),EXL3_MOE_MIXEDK_ELIDE_HANDLED='1',EXL3_UMA_RESERVE_MB='8192')
            with patch.dict(os.environ,env,clear=True):
                g=construct();self.assertIsInstance(g.dflash_cost_policy,C)
                j=n.queue(g,240,100);n.equip_full_receive(j);j.sampler=NS(supports_batch_verify=True,reqs_past_ids=False)
                for _ in range(80):g.draft_calibrator.add_label(10,True)
                self.assertEqual(n.h.draft(g).shape,(1,7))
                self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
                mutations=[lambda k:k.update(num_draft_tokens=8),lambda k:k.update(max_batch_size=2),
                           lambda k:k.update(max_chunk_size=2048),lambda k:k.update(dynamic_draft_tokens=False),
                           lambda k:setattr(k['draft_model'].config,'block_size',16),
                           lambda k:setattr(k['model'].config,'directory','different-pack'),
                           lambda k:setattr(k['cache'].layers[0],'k_bits',8)]
                for change in mutations:
                    with self.assertRaises(ValueError):construct(change)
                with patch.dict(os.environ,{'EXL3_DFLASH_COST_PROFILE_SHA256':'0'*64}):
                    with self.assertRaisesRegex(ValueError,'hash'):construct()
                for value in ([],{}, {'context':ctx,'attest_same_round8_weights_and_runtime':False}):
                    att.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):construct()
    def test_additional_malformed_profile_contract(self):
        C=policy_class(self);p=json.loads((HERE/'profile.json').read_text())
        for change in (lambda b:b.update(schema=True),lambda b:b['q_costs'][0].update(min_ms=math.nan),
                       lambda b:b['q_costs'][0].update(max_ms=0)):
            b=copy.deepcopy(p);change(b)
            with self.assertRaises(ValueError):C.from_profile(b,p['context'])

class Preservation(unittest.TestCase):
    def test_target_verifier_job_calibrator_cache_bytes_unchanged(self):
        import ast,hashlib
        base=HERE.parent/'dflash-pages-candidate/upstream'
        rel='exllamav3/generator/generator.py'
        def method(root,name):
            source=(root/rel).read_text();tree=ast.parse(source)
            c=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Generator')
            return ast.get_source_segment(source,next(x for x in c.body if isinstance(x,ast.FunctionDef) and x.name==name))
        for name in ('iterate_gen','iterate_draftmodel_gen','iterate_draftmodel_mtp_gen','iterate_ngram_gen'):
            self.assertEqual(method(ROOT,name),method(base,name))
        for rel in ('exllamav3/generator/job.py','exllamav3/generator/draft_confidence.py','exllamav3/generator/pagetable.py',
                    'exllamav3/architecture/dflash.py','exllamav3/modules/arch_specific/dflash.py','exllamav3/cache/quant.py'):
            self.assertEqual((ROOT/rel).read_bytes(),(base/rel).read_bytes())
    def test_source_calibration_slice_exact_labels_not_unverified_negatives(self):
        C=policy_class(self)
        for k in (0,1,3,7):
            g,j=ready();n.equip_full_receive(j)
            obj=C.__new__(C);obj.costs=tuple([1.]*(k+1)+[100.]*(7-k));g.dflash_cost_policy=obj
            labels=[];old=g.draft_calibrator.add_label
            def label(score,accepted):labels.append((score,accepted));old(score,accepted)
            g.draft_calibrator.add_label=label
            p=n.h.draft(g)
            self.assertEqual(g._draft_conf_round['window'],k)
            self.assertEqual(g._draft_conf_round['conf'].tolist(),[[10.]*7])
            n.verify(g,j,p)
            self.assertEqual(labels,[(10.,True)]*min(k+1,7))
        for k in (0,1,3):
            g,j=ready();n.equip_full_receive(j)
            obj=C.__new__(C);obj.costs=tuple([1.]*(k+1)+[100.]*(7-k));g.dflash_cost_policy=obj
            labels=[];old=g.draft_calibrator.add_label
            def label(score,accepted):labels.append((score,accepted));old(score,accepted)
            g.draft_calibrator.add_label=label
            n.verify(g,j,n.h.draft(g),'mismatch',k)
            self.assertEqual(labels,[(10.,True)]*k+[(10.,False)])
        g,j=ready();n.equip_full_receive(j);obj=C.__new__(C);obj.costs=(1,)*8;g.dflash_cost_policy=obj
        labels=[];old=g.draft_calibrator.add_label
        def label(score,accepted):labels.append((score,accepted));old(score,accepted)
        g.draft_calibrator.add_label=label
        n.verify(g,j,n.h.draft(g),'mismatch',2)
        self.assertEqual(labels,[(10.,True),(10.,True),(10.,False)])
    def test_disabled_untrusted_and_no_confidence_paths(self):
        C=policy_class(self)
        for mode in ('no_data','sparse','no_cal','no_conf','disabled'):
            g,j=ready();n.equip_full_receive(j);obj=C.__new__(C);obj.costs=(1,100,100,100,100,100,100,100)
            g.dflash_cost_policy=None if mode=='disabled' else obj
            if mode=='no_data':g.draft_calibrator=n.Calibrator(.6)
            if mode=='sparse':g.draft_calibrator=learned([1.])
            if mode=='no_cal':g.draft_calibrator=None
            if mode=='no_conf':
                original=g.draft_model.sample_from_state
                def sample(state,params):
                    result=original(state,params);params.pop('draft_conf',None);return result
                g.draft_model.sample_from_state=sample
            self.assertEqual(n.h.draft(g).shape,(1,7),mode)
            self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
    def test_invalid_probability_fails_back_and_does_not_mutate_calibrator(self):
        C=policy_class(self);obj=C.__new__(C);obj.costs=(1,)*8
        for value in (math.nan,math.inf,-1,101):
            cal=learned([1]);cal.bins[0][1]=value
            before=copy.deepcopy(cal.__dict__)
            self.assertIsNone(obj.choose(cal,[0]*7,7))
            # NaN is retained, not clamped into a fabricated probability.
            self.assertEqual(cal.total,before['total']);self.assertEqual(cal.cached_threshold,before['cached_threshold'])
    def test_reused_calibrator_default_dds_until_refilled(self):
        C=policy_class(self);obj=C.__new__(C);obj.costs=(1,)*8
        cal=n.Calibrator(.6)
        for _ in range(64):cal.add_label(10,True)
        self.assertEqual(obj.choose(cal,[10]*7,7),7)
        cal.decay_step();self.assertIsNone(obj.choose(cal,[10]*7,7))
        cal.add_label(10,True);self.assertEqual(obj.choose(cal,[10]*7,7),7)
    def test_no_extra_tensor_readbacks_for_policy(self):
        from unittest.mock import patch
        C=policy_class(self);observed=[]
        for on in (False,True):
            g,j=ready();n.equip_full_receive(j);obj=C.__new__(C);obj.costs=(1,)*8
            g.dflash_cost_policy=obj if on else None
            calls=[];original=n.Tensor.cpu
            def cpu(t):calls.append(t.shape);return original(t)
            with patch.object(n.Tensor,'cpu',cpu):n.h.draft(g)
            observed.append(calls)
        self.assertEqual(observed[0],observed[1])

    def test_rewound_job_skips_all_labels_in_real_update_slice(self):
        import ast
        g,j=ready();p=n.h.draft(g);before=g.draft_calibrator.total
        path=ROOT/'exllamav3/generator/generator.py';tree=ast.parse(path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Generator')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='iterate_gen')
        statement=next(x for x in fn.body if isinstance(x,ast.If) and 'self._draft_conf_round is not None' in ast.unparse(x.test))
        labels=[];g.draft_calibrator.add_label=lambda *args:labels.append(args)
        namespace=dict(self=g,logit_mapping=[0,1],accepted_lengths=[8],rewound_jobs={id(j)})
        exec(compile(ast.fix_missing_locations(ast.Module(body=[statement],type_ignores=[])),str(path),'exec'),namespace)
        self.assertEqual(labels,[]);self.assertAlmostEqual(g.draft_calibrator.total,before*.995)
        self.assertIsNone(g._draft_conf_round)
    def test_policy_rounds_requeue_refill_and_keep_native_pages(self):
        C=policy_class(self);g=n.generator(BASE,7,dynamic=True);j=n.queue(g,240,1000,16)
        n.equip_full_receive(j);j.sampler=NS(supports_batch_verify=True,reqs_past_ids=False)
        for _ in range(80):g.draft_calibrator.add_label(10,True)
        obj=C.__new__(C);obj.costs=(1,100,100,100,100,100,100,100);g.dflash_cost_policy=obj
        for i in range(16):
            p=n.h.draft(g)
            self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
            r=n.verify(g,j,p)
            if r.requeued:break
        else:self.fail('requeue did not trigger')
        self.assertEqual(sum(page.kv_position for page in j.sequences[0].allocated_pages),j.sequences[0].kv_position)
        j=j.prepare_for_requeue();n.h.allocate(g,j);g.draft_cache.sequences=j.sequences
        n.equip_full_receive(j);j.sampler=NS(supports_batch_verify=True,reqs_past_ids=False)
        n.h.draft(g);self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
        self.assertEqual(g.draft_reserve_tokens,7)
    def test_actual_combo_sampler_constructor_capability(self):
        import ast
        saved=sys.argv[:];sys.argv=[sys.argv[0],str(ROOT/'exllamav3/generator/sampler')]
        try:
            spec=importlib.util.spec_from_file_location('sampler_source',ROOT/'tests/cpu_requirements_harness.py')
            m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        finally:sys.argv=saved
        for fused in (False,True):
            ns=m.load_classes(fused)
            # Only float32 conversion/finfo/DRY exponent boundaries; no sampling runs.
            ns.update(torch=NS(tensor=lambda v,**kw:n.np.float32(v),float=float,float32=float,
                               finfo=lambda dtype:NS(max=n.np.finfo(n.np.float32).max)),_dry_max_exponent=lambda base:0)
            source=(ROOT/'exllamav3/generator/sampler/presets.py').read_text()
            combo=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='ComboSampler')
            bias=next(x for x in ast.parse((ROOT/'exllamav3/generator/sampler/custom.py').read_text()).body if isinstance(x,ast.ClassDef) and x.name=='LogitBias')
            mod=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),bias,combo],type_ignores=[])
            exec(compile(ast.fix_missing_locations(mod),'ComboSampler-source','exec'),ns)
            for kwargs,eligible in (({'temperature':0},True),({'top_k':1},True),({'temperature':0,'rep_p':1.1},False),({'temperature':.8},False)):
                sampler=ns['ComboSampler'](**kwargs)
                self.assertEqual(sampler.supports_batch_verify,eligible)
                C=policy_class(self);obj=C.__new__(C);obj.costs=(1,)*8
                g,j=ready();n.equip_full_receive(j);j.sampler=sampler
                self.assertEqual(obj.select(g,n.Tensor.wrap([[10.]*7]),7),7 if eligible else None)

if __name__=='__main__':
    label=sys.argv[1]
    suite=(unittest.defaultTestLoader.loadTestsFromTestCase(Integration) if '--contract-only' in sys.argv
           else unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    text=stream.getvalue(); print(text)
    with (HERE/'evidence'/f'{label}.txt').open('x',encoding='utf-8') as f:f.write(text)
    summary=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),successful=result.wasSuccessful(),root=str(ROOT))
    with (HERE/'evidence'/f'{label}.json').open('x') as f:json.dump(summary,f,indent=2)
    assert 'torch' not in sys.modules and 'exllamav3' not in sys.modules
    sys.exit(not result.wasSuccessful())
