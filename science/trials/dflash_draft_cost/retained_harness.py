"""LOCAL CPU source execution. Tensor allocation, loader/weights, prefill are opaque.
Complete MiMo config/model, SlidingAttention ctor, Cache ctor, SWAState, recurrent
cache ctor and loader cache-configuration slice; original native harness supplies
complete Generator/producer/queue methods and disclosed verification slice.
Explicit import substitution returns source-extracted classes, NEVER real runtime.
"""
from __future__ import annotations
import ast, argparse, builtins, copy, hashlib, importlib.util, json, os, tempfile
from collections import deque, OrderedDict
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
import test_cost as t
n=t.n
HERE=Path(__file__).resolve().parent
ROOT=t.ROOT
R=HERE.parent
TARGET=R/'source/workspace/MiMo-V2.6-Flash-RL-EXL3/2.50bpw/config.json'
LAUNCH=R/'round8-width-harness-repair/source/retained_launch.json'
EXTRACTED=[]

def extract(path,name,methods=None,namespace=None):
    ns=S if namespace is None else namespace
    node=copy.deepcopy(next(x for x in ast.parse((ROOT/path).read_text()).body if getattr(x,'name',None)==name))
    if methods is not None:
        node.body=[x for x in node.body if isinstance(x,(ast.Assign,ast.AnnAssign)) or getattr(x,'name',None) in methods]
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(ROOT/path),'exec'),ns)
    EXTRACTED.append((path,name,methods))
    return ns[name]

class Module:
    def __init__(self,config=None,*a,**kw):
        self.config=config;self.caps={};self.__dict__.update(kw)
        self.cache_layers=[];self.submodules=[]
    def register_submodule(self,m):self.submodules.append(m)
class Model(Module):
    def __init__(self,config,**kw):
        super().__init__(config,**kw);self.modules=[];self.loaded_tp=False
    def get_cache_layers(self):return [b.attn for b in self.modules if hasattr(b,'attn') and b.attn.caps.get('kv_cache')]
    def get_recurrent_layers(self):return [b.attn for b in self.modules if hasattr(b,'attn') and b.attn.caps.get('recurrent_cache')]
    def get_layer_instances(self,idx):return [(idx,0)]
class Attention(Module):
    def __init__(self,**kw):
        super().__init__(**kw);self.caps={'kv_cache':True}
        self.q_proj=Module();self.k_proj=Module();self.v_proj=Module();self.o_proj=Module()
class ShapeTensor:
    def __init__(self,shape,**kw):self.shape=tuple(shape);self.dtype=kw.get('dtype');self.device=kw.get('device')
class QuantLayer:
    def __init__(self,config,module,cache_id,max_num_tokens,**kw):
        self.module=module;self.__dict__.update(kw)
class ConfigBase(n.ConfigBase):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.layer_map=None
S=dict(Config=ConfigBase,Model=Model,Module=Module,Attention=Attention,Linear=Module,
       RMSNorm=Module,Embedding=Module,TransformerBlock=Module,BlockSparseMLP=Module,GatedMLP=Module,
       no_default=n.NSG['no_default'],RopeStyle=n.NSG['RopeStyle'],override=lambda f:f,PAGE_SIZE=256,
       torch=N(empty=lambda shape,**kw:ShapeTensor(shape,**kw),half='half',float='float'),
       deque=deque,OrderedDict=OrderedDict)
extract('exllamav3/architecture/mimo_v2.py','_mimo_v2_qkv_dequant')
S['FP8_BLOCK']=128
SWAState=extract('exllamav3/modules/sliding_attn.py','SWAState')
SWALayerState=extract('exllamav3/modules/sliding_attn.py','SWALayerState',['__init__','get_checkpoint_size'])
SlidingAttention=extract('exllamav3/modules/sliding_attn.py','SlidingAttention',['__init__','_decode_state_prep'])
MiMoV2Config=extract('exllamav3/architecture/mimo_v2.py','MiMoV2Config',['__init__','qkv_dequant'])
MiMoV2Model=extract('exllamav3/architecture/mimo_v2.py','MiMoV2Model',['__init__'])
Cache=extract('exllamav3/cache/cache.py','Cache',['__init__','reset_states','get_new_state','get_all_recurrent_layers'])
Cache.attach_to_model=lambda self:None # loader attachment boundary, not cache routing
RecurrentCache=extract('exllamav3/cache/recurrent.py','RecurrentCache',['__init__'])
Job=extract('exllamav3/generator/job.py','Job',['is_checkpoint_boundary','maybe_stash_recurrent'])
advance=extract('exllamav3/cache/recurrent_util.py','advance_recurrent_states')

REAL_IMPORT=builtins.__import__
def source_import(name,globals=None,locals=None,fromlist=(),level=0):
    if name in ('exllamav3.architecture.mimo_v2','exllamav3.modules.sliding_attn','exllamav3.cache.recurrent'):
        return N(MiMoV2Model=MiMoV2Model,SWAState=SWAState,SWALayerState=SWALayerState,
                 SlidingAttention=SlidingAttention,RecurrentCache=RecurrentCache)
    if name=='fp16' and level==1:return N(CacheLayer_fp16=QuantLayer)
    return REAL_IMPORT(name,globals,locals,fromlist,level)

def args():
    tree=ast.parse((ROOT/'exllamav3/model_init.py').read_text())
    add=next(x for x in tree.body if getattr(x,'name',None)=='add_args')
    calls=[]
    wanted=('-swa_full','-rcs','-cs','-cq','-ambs','-chunk_size','-cca')
    for x in ast.walk(add):
        if isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Attribute):
            c=x.value
            if c.func.attr=='add_argument' and c.args and isinstance(c.args[0],ast.Constant) and c.args[0].value in wanted:calls.append(x)
    assert len(calls)==len(wanted)
    parser=argparse.ArgumentParser(allow_abbrev=False)
    exec(compile(ast.Module(body=calls,type_ignores=[]),'<model_init parser declarations>','exec'),
         dict(parser=parser,default_recurrent_cache_size=4.0,default_cache_size=8192,default_autosplit_max_batch_size=4,default_chunk_size=4096))
    a,_=parser.parse_known_args(json.loads(LAUNCH.read_text())['command'][2:])
    return a

def construct(enabled=True,change=None):
    a=args();ctx=json.loads((HERE/'profile.json').read_text())['context']
    with patch('builtins.__import__',source_import):
        cfg=MiMoV2Config(str(TARGET));cfg.directory=ctx['model_directory']
        model=MiMoV2Model(cfg,swa_full=a.swa_full)
        draft=n.DraftAdapter(t.BASE);draft.config.directory=ctx['drafter_directory']
        draft.get_cache_layers=lambda:draft.model.attn_modules
        draft.get_recurrent_layers=lambda:[]
        draft.get_layer_instances=lambda idx:[(idx,0)]
        draft.recurrent_state_cls=None
        # Source slice: loader's max_history and cache construction branch, not model load.
        tree=ast.parse((ROOT/'exllamav3/model_init.py').read_text())
        loader=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='init')
        begin=next(i for i,x in enumerate(loader.body) if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='max_history' for y in x.targets))
        ns=dict(args=a,min_draft_len=7,draft_model=draft,model=model,draft_model_dir='present',Cache=Cache,CacheLayer_quant=QuantLayer,cache=True)
        exec(compile(ast.Module(body=loader.body[begin:begin+2],type_ignores=[]),'<actual loader cache slice>','exec'),ns)
        cache=ns['cache']
        dc=n.Cache();dc.max_num_tokens=4096;dc.layers=ns['draft_cache'].layers
        kw=dict(model=model,cache=cache,tokenizer=N(get_id_to_piece_list=lambda *a:['x']*64),max_batch_size=1,
                max_chunk_size=a.chunk_size,draft_model=draft,draft_cache=dc,num_draft_tokens=7,dynamic_draft_tokens=True,
                draft_confidence=.6,recurrent_cache_size=int(a.recurrent_cache_size*1024**3))
        if change:change(kw)
        spec=importlib.util.spec_from_file_location('cpu_policy',ROOT/'exllamav3/generator/draft_cost.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory(prefix='TEST_ONLY-',dir=HERE) as tmp:
            att=Path(tmp)/'not-live-attestation.json'
            att.write_text(json.dumps(dict(context=ctx,attest_same_round8_weights_and_runtime=True)))
            env=dict(EXL3_DFLASH_COST_AWARE='1' if enabled else '0',EXL3_DFLASH_COST_PROFILE=str(HERE/'profile.json'),
                     EXL3_DFLASH_COST_PROFILE_SHA256=hashlib.sha256((HERE/'profile.json').read_bytes()).hexdigest(),
                     EXL3_DFLASH_COST_CONTEXT=str(att),EXL3_MOE_MIXEDK_ELIDE_HANDLED='1',EXL3_UMA_RESERVE_MB='8192')
            with patch.dict(os.environ,env,clear=True),patch.dict(n.h.NS_GLOBALS,{'DFlashCostPolicy':m.DFlashCostPolicy,'_BATCH_VERIFY':False,'RecurrentCache':RecurrentCache}):
                return n.h.Generator(**kw)

def queue(g,prompt=23,max_new=3000,max_rq=1):
    j=n.queue(g,prompt,max_new,max_rq);n.equip_full_receive(j)
    # Exact constructor declarations for host lifecycle fields, not whole Job init.
    ctor=next(x for x in ast.parse((ROOT/'exllamav3/generator/job.py').read_text()).body if getattr(x,'name',None)=='Job')
    ctor=next(x for x in ctor.body if getattr(x,'name',None)=='__init__')
    statements=[x for x in ctor.body if (isinstance(x,ast.Assign) and any(isinstance(a,ast.Attribute) and a.attr in ('checkpoint','checkpoint_rewound','is_finished','recurrent_state','last_recurrent_checkpoint_pos') for a in x.targets)) or (isinstance(x,ast.If) and ast.unparse(x.test)=='rq_state is None')]
    exec(compile(ast.Module(body=statements,type_ignores=[]),'<actual job lifecycle declaration slice>','exec'),dict(self=j,rq_state=None))
    j.sampler=N(supports_batch_verify=True,reqs_past_ids=False)
    j.is_checkpoint_boundary=Job.is_checkpoint_boundary.__get__(j)
    j.maybe_stash_recurrent=Job.maybe_stash_recurrent.__get__(j)
    j.recurrent_state=g.cache.get_new_state()
    # Opaque prefill boundary: host positions after prompt ingestion, no tensor claim.
    set_position(j,j.sequences[0].kv_position)
    j.last_recurrent_checkpoint_pos=None
    for _ in range(80):g.draft_calibrator.add_label(10,True)
    return j

def scores():return n.Tensor.wrap([[10.]*7])

def set_position(j,pos,new_tokens=None):
    """Opaque prefill/history fixture; never pretends to create real KV contents."""
    seq=j.sequences[0]
    seq.kv_position=pos;seq.sequence_ids=n.h.SeqTensor.ids(pos+1)
    if new_tokens is not None:j.new_tokens=new_tokens
    j.recurrent_state.position=pos
    j.recurrent_state.window_beg=max(0,(pos//256-2)*256)
    for i,page in enumerate(seq.allocated_pages):page.kv_position=min(256,max(0,pos-i*256))

def verify(g,j,proposals,kind='accept',stop_index=0):
    # Actual host advance + ring page-shift + SWA rewind, not real target tensor math.
    seq=j.sequences[0]
    ids=j.get_input_ids_list(proposals,0,add_to_cache=True)[0];q=ids.shape[1]
    module=next(iter(g.cache.recurrent_layers.values())).module;module.device='cpu'
    # Ring axes collapsed to one lane: shape/control oracle, no attention correctness claim.
    with patch.dict(S,{'torch':n.Torch}):
        module._decode_state_prep([j.recurrent_state],n.Tensor((1,768,1,1)),n.Tensor((1,768,1,1)),q)
        advance(ids,dict(recurrent_states=[j.recurrent_state],recurrent_history=proposals is not None),g.model)
    samples=list(range(2,q+2))
    if kind=='mismatch':samples[stop_index]=31
    if kind=='eos':j.stop_tokens={samples[stop_index]}
    logits=n.Tensor.wrap(n.np.asarray(samples).reshape(1,q,1))
    g.record_draft_stats=False;results=[]
    completed,requeued,lengths=n.verify_slice(g,results,proposals,logits,[0,1],[j.recurrent_state],1)
    return N(completed=completed,requeued=requeued,lengths=lengths,results=results,q=q)

