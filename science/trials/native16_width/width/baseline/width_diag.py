"""Bounded native-width safety capture, never a benchmark or runtime edit."""
import os
from width_contract import assigned_positions
def install(g):
 if os.environ.get('ROUND8_WIDTH')!='1':return False
 import json,torch,threading,time
 from pathlib import Path
 from exllamav3.generator.job import Job
 from exllamav3.modules.attention_fn.bc_attn import BCAttn
 from exllamav3.util.tensor import g_tensor_cache
 out=Path(os.environ['ROUND8_WIDTH_OUT']);out.mkdir(parents=True,exist_ok=True)
 state=dict(active=False,sampling=False,round=0,record=None,jobs=0,records=0)
 def emit(kind,**kw):
  assert state['records']<10000
  row=dict(kind=kind,round=state['round'],**kw)
  with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  state['records']+=1
 def finite(t):return bool(torch.isfinite(t).all().item())
 dm=g.draft_model;assert dm.config.block_size==16 and g.draft_reserve_tokens==15
 assert dm.input_layer.native_draft_len==16
 assert dm.input_layer.mask_embedding is not None and list(dm.input_layer.mask_embedding.shape)==[4096]
 emit('installed',model_type=type(dm).__name__,block=dm.config.block_size,requested=g.num_draft_tokens,reserve=g.draft_reserve_tokens,mask_shape=list(dm.input_layer.mask_embedding.shape),mask_finite=finite(dm.input_layer.mask_embedding),tap_shift=dm.config.tap_shift,taps=dm.config.target_layer_ids,model_vocab=g.model.config.vocab_size,actual_vocab=g.tokenizer.actual_vocab_size,window=[a.window_arg() for a in dm.attn_modules])
 original_init=Job.__init__
 def init(job,*a,**kw):
  if kw.get('rq_state') is None:kw['max_rq_tokens']=1;state['jobs']+=1
  original_init(job,*a,**kw)
  emit('job_init',identifier=str(job.identifier),input_tokens=len(job.sequences[0].input_ids),cap=job.max_new_tokens,orig_max_rq=job.orig_max_rq_tokens,requeued=job.is_requeued)
 Job.__init__=init
 original_input=dm.input_layer.forward
 def inp(x,params,*a,**kw):
  y=original_input(x,params,*a,**kw)
  if state['active']:
   row=dict(anchor_shape=list(x.shape),input_shape=list(y.shape),finite=finite(y),mask_equal=bool(torch.equal(y[:,1:,:],dm.input_layer.mask_embedding.to(y.dtype).reshape(1,1,-1).expand(1,15,-1))))
   emit('input',**row);assert row['anchor_shape']==[1,1] and row['input_shape']==[1,16,4096] and row['finite'] and row['mask_equal']
  return y
 dm.input_layer.forward=inp
 attn_ids={id(a) for a in dm.attn_modules};old_bc=BCAttn.step
 def bc(self,x,*a,**kw):
  y=old_bc(self,x,*a,**kw)
  if state['active'] and id(self.module) in attn_ids and y is not None:
   qs=(1,16,64,128);kvs=(2,16,8*128)
   q=g_tensor_cache.cache[g_tensor_cache.make_key(self.device,qs,torch.half,'bca_q')][1]
   kv=g_tensor_cache.cache[g_tensor_cache.make_key(self.device,kvs,torch.half,'bca_kv')][1]
   k,v=kv.reshape(2,1,16,8,128).unbind(0)
   row=dict(layer=self.module.key,path='original_BC_graph',q_shape=list(q.shape),k_shape=list(k.shape),v_shape=list(v.shape),finite_q=finite(q),finite_k=finite(k),finite_v=finite(v),rotary_freq_shape=list(self.module.rope.inv_freq.shape))
   emit('qkv',**row);assert row['finite_q'] and row['finite_k'] and row['finite_v']
  return y
 BCAttn.step=bc
 for attn in dm.attn_modules:
  orig=attn.project_qkv
  def qkv(x,params,orig=orig,key=attn.key):
   vals=orig(x,params)
   if state['active']:
    q,k,v,_=vals;row=dict(layer=key,path='original_eager',q_shape=list(q.shape),k_shape=list(k.shape),v_shape=list(v.shape),finite_q=finite(q),finite_k=finite(k),finite_v=finite(v));emit('qkv',**row)
    assert row['q_shape']==[1,16,64,128] and row['k_shape']==row['v_shape']==[1,16,8,128] and row['finite_q'] and row['finite_k'] and row['finite_v']
   return vals
  attn.project_qkv=qkv
 original_forward=dm.forward
 def forward(*a,**kw):
  assert threading.current_thread().name=='native-generator'
  state['active']=True
  params=kw['params'];seq=g.active_jobs[0].sequences[0];start=int(params['cache_seqlens'][0]);table=params['block_table'][0].tolist();assigned=[p.page_index for p in seq.allocated_pages]
  positions=assigned_positions(start,16,assigned,table)
  assert all(p.ref_count>0 for p in seq.allocated_pages)
  pp=torch.tensor([a for a,b in positions],device='cuda');tt=torch.tensor([b for a,b in positions],device='cuda')
  before={str(key):[t[pp,tt].clone() for t in (layer.qk,layer.qv,layer.sk,layer.sv)] for key,layer in g.draft_cache.layers.items()}
  emit('assigned_pages',start=start,end_exclusive=start+16,assigned=assigned,table=table,positions=positions,max_rq=g.active_jobs[0].max_rq_tokens,requeued=g.active_jobs[0].is_requeued)
  try:
   y=original_forward(*a,**kw);emit('state',shape=list(y.shape),finite=finite(y));assert list(y.shape)==[1,16,4096] and finite(y)
   for key,layer in g.draft_cache.layers.items():
    after=[t[pp,tt] for t in (layer.qk,layer.qv,layer.sk,layer.sv)]
    changed=torch.stack([(a!=b).reshape(16,-1).any(dim=1) for a,b in zip(before[str(key)],after)]).any(dim=0).cpu().tolist()
    emit('native_cache_writes',layer=str(key),changed_rows=changed,finite_scales=finite(after[2]) and finite(after[3]))
    assert finite(after[2]) and finite(after[3])
    # Repeated mask rows may write identical bytes. First tiny capture proves changed physical rows.
    if state['round']==1:assert all(changed)
   return y
  finally:state['active']=False
 dm.forward=forward
 lm=g.model.modules[g.model.logit_layer_idx];old_lm=lm.forward
 def head(*a,**kw):
  y=old_lm(*a,**kw)
  if state['sampling']:
   # Before CustomSampler can perform in-place vocabulary masking.
   emit('pristine_draft_head',shape=list(y.shape),finite_model_vocab=finite(y[...,:g.model.config.vocab_size]));assert finite(y[...,:g.model.config.vocab_size])
  return y
 lm.forward=head
 old_sample=dm.sample_from_state
 def ds(s,params):
  state['sampling']=True
  try:y=old_sample(s,params)
  finally:state['sampling']=False
  ids=y.cpu().tolist();flat=[x for row in ids for x in row]
  emit('native_samples',shape=list(y.shape),ids=ids,proposal_count=len(flat)-1,padded_vocab_proposals=sum(x>=g.tokenizer.actual_vocab_size for x in flat[1:]))
  assert list(y.shape)==[1,16] and all(0<=x<g.model.config.vocab_size for x in flat)
  return y
 dm.sample_from_state=ds
 old_draft=g.iterate_draftmodel_dflash_gen
 def draft(results):
  state['round']+=1;assert state['round']<=512
  y=old_draft(results);emit('proposal_truncation',proposed=0 if y is None else y.shape[-1]);return y
 g.iterate_draftmodel_dflash_gen=draft
 old_target=g.model.forward
 def target(*a,**kw):
  emit('target_forward',q=kw['input_ids'].shape[-1],prefill='prefill' in kw['params'],counts_enabled=False,batch_verify=False)
  return old_target(*a,**kw)
 g.model.forward=target
 old_receive=Job.receive_sample
 def receive(job,*a,**kw):
  token=a[1] if len(a)>1 else kw['next_token'];token=int(token.item());assert 0<=token<g.tokenizer.actual_vocab_size
  before=(job.new_tokens,job.accepted_draft_tokens,job.rejected_draft_tokens)
  result=old_receive(job,*a,**kw)
  emit('received',token=token,before=before,after=(job.new_tokens,job.accepted_draft_tokens,job.rejected_draft_tokens),result=str(result),kv=job.sequences[0].kv_position)
  return result
 Job.receive_sample=receive
 old_iter=g.iterate
 def iterate():
  results=old_iter()
  for r in results:
   if r.get('eos'):emit('terminal',values={k:r[k] for k in ('identifier','new_tokens','accepted_draft_tokens','rejected_draft_tokens','eos_reason','time_generate') if k in r})
  return results
 g.iterate=iterate
 return True
