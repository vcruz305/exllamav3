"""Full Generator ctor/iterate_gen and Job ctor/receive_sample/requeue execution.
CPU stubs: token sampler, target forward, SeqTensor, Sequence/page allocation and masks.
Real get_input_ids_list writes CPU page records; real rejection rewinds those records.
No extracted method bodies are sliced. No torch or exllamav3 imports.
"""
from source_harness import *
import random
import time
PAGE_SIZE=256

class SeqTensor:
    def __init__(self, shape=(1,0), dtype=np.int64, seq_dim=-1):
        self.data=Torch.zeros(shape,dtype=dtype)
        self.dim=seq_dim
    @classmethod
    def from_tensor(cls, t):
        obj=cls(t.shape,dtype=t.dtype)
        obj.data=t.clone()
        return obj
    def __len__(self): return self.data.shape[self.dim]
    def torch(self): return self.data
    def torch_slice(self, start, stop): return self.data[:,start:stop]
    def append(self,t): self.data=Torch.cat([self.data,t],dim=self.dim)
    def clear(self):
        shape=list(self.data.shape); shape[self.dim]=0
        self.data=Torch.zeros(tuple(shape),dtype=self.data.dtype)
    def clone(self, drop=None):
        result=self.from_tensor(self.data)
        result.dim=self.dim
        if drop and drop <= len(self):
            key=[slice(None)]*len(self.data.shape)
            key[self.dim]=slice(None,len(self)-drop)
            result.data=self.data[tuple(key)].clone()
        return result
    def slice(self,start,stop): return self.from_tensor(self.torch_slice(start,stop))
    def truncate(self,n): self.data=self.data[:,:n]

class Sequence:
    def __init__(self, ids, seq_ids):
        self.input_ids=SeqTensor.from_tensor(ids)
        self.sequence_ids=SeqTensor.from_tensor(seq_ids)
        self.kv_position=len(self.sequence_ids)-1
        self.max_cached_pages=None
        self.page_hashes=[]
        self.prefill_complete=True
        self.block_index_tensor=Torch.zeros((1,16),dtype=np.int64)
        self.allocate()
    def allocate(self):
        self.allocated_pages=[NS(kv_position=0, ref_count=1, sequence=Torch.zeros((1,PAGE_SIZE),dtype=np.int64),
                                 can_revert=False,phash=None,access_serial=0,update_hash=lambda h:None) for _ in range(32)]
        left=self.kv_position
        for p in self.allocated_pages:
            p.kv_position=min(left,PAGE_SIZE); left-=p.kv_position
    def prepare(self,*a): return set(), len(self.allocated_pages)

class CPUModel:
    caps={}
    config=NS(vocab_size=128)
    def __init__(self,tokens): self.tokens=tokens; self.calls=0; self.fail=False
    def forward(self,input_ids,params):
        self.calls+=1
        if self.fail: raise RuntimeError('target failed')
        n,q=input_ids.shape
        assert len(self.tokens)>=q
        return Tensor(np.broadcast_to(np.array(self.tokens[:q])[None,:,None],(n,q,1)).copy())


def fixture(tokens=(10,11,12,13), *, record=None, max_new=100, stop=(), draft=True, batch_verify=False):
    ns=dict(torch=Torch, _os=os, time=time, random=random, np=np, PAGE_SIZE=PAGE_SIZE,
            PageTable=lambda gen,cache:NS(max_pages=4096,referenced_pages={},unreferenced_pages={}),
            ThreadPoolExecutor=lambda **kw:None, SeqTensor=SeqTensor, Sequence=Sequence,
            cuda_sync_active=lambda:None, _BATCH_VERIFY=batch_verify,
            tensor_hash_checksum=lambda *args:b'cpu-page',
            _strings_to_utf32=lambda x:(None,None), DefaultSampler=lambda:NS(),
            ext=NS(BC_SAM=lambda:NS()))
    # Load an optional helper only for reproducing the earlier faulty patch.
    import ast
    for rel in ('exllamav3/generator/job.py','exllamav3/generator/generator.py'):
        tree=ast.parse((ROOT/rel).read_text(encoding='utf-8'))
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name in ('draft_round_stats','dflash_draft_geometry'):
                exec(compile(ast.Module(body=[n],type_ignores=[]),'<prior helper>','exec'),ns)
    Gen=extract_class('exllamav3/generator/generator.py','Generator',
        ['__init__','_staging','iterate_gen','num_remaining_jobs'],ns)
    Job=extract_class('exllamav3/generator/job.py','Job',
        ['__init__','prepare_for_queue','receive_sample','prepare_for_requeue',
         'is_prefill_done','get_max_seq_len','get_input_ids_list'],ns)
    # Model/sampler/allocator boundaries only, not telemetry or stream construction.
    Job.prepare_logit_mask=lambda self:None
    Job.prepare_sampling_past_ids=lambda self:None
    Job.receive_logits=lambda self,logits:(Torch.tensor([[int(logits.a[0,0,0])]]),None,None,None)
    Job.deallocate_pages=lambda self:None
    model=CPUModel(list(tokens))
    kwargs={} if record is None else {'record_draft_stats':record}
    gen=Gen(model=model,cache=NS(max_num_tokens=1048576),tokenizer=NS(
        get_id_to_piece_list=lambda *a:[str(i) for i in range(128)]),
        max_batch_size=4,ngram_match_min=2 if draft else 0,num_draft_tokens=3 if draft else 0,**kwargs)
    gen.on_queue_drained=lambda:None
    sampler=NS(reqs_past_ids=False,supports_batch_verify=batch_verify,
               forward=lambda logits,*a,**kw:Tensor(logits.a[:,0,0].astype(np.int64)))
    job=Job(input_ids=Torch.tensor([[1,2,3]]),max_new_tokens=max_new,sampler=sampler,
            stop_conditions=list(stop), identifier='cpu-test')
    job.prepare_for_queue(gen,17)
    job.time_first_prefill=job.time_first_token=job.time_enqueue
    gen.active_jobs=[job]
    return gen,job


def step(gen,draft=(10,11,12)):
    results=[]
    gen.iterate_gen(results,None if draft is None else Torch.tensor([list(draft)],dtype=np.int64))
    return results
