"""Execute the native C++ eligibility expression with metadata-only CPU objects."""
import re, unittest
from test_candidate import Q

class Tensor:
    def __init__(self,dtype='long',numel=256,contiguous=True):
        self.dtype=dtype;self.n=numel;self.contiguous=contiguous
    def scalar_type(self):return self.dtype
    def is_contiguous(self):return self.contiguous
    def numel(self):return self.n
    def value(self):return self


def environment():
    env=dict(requested=True,major=12,minor=1,bsz=1,hidden_dim=4096,intermediate_dim=2048,
             num_experts=256,num_experts_per_tok=8,gate_mul1=True,act_function=0,MOE_ACT_SILU=0,
             m_tile=16,N_off=1,pipe_sel=0,smem_override=0,blocks_per_sm=1,
             _output_scratch=object(),_fused_base=object(),
             hidden_state=Tensor('half',4096),weight_sorted=Tensor('half',8),token_sorted=Tensor('long',8),
             expert_count=Tensor('long',257),fused_base=Tensor('long',257))
    for projection in ('gate','up','down'):
        env['K_'+projection+'_arr']=Tensor('int',256)
        for kind in ('trellis','suh','svh'):env[projection+'_ptrs_'+kind]=Tensor()
    return env


def guard(env):
    source=(Q/'exl3_moe.cu').read_text()
    expr=re.search(r'const bool three_stage = ([\s\S]*?);',source)[1]
    expr=expr.replace('&&',' and ').replace('||',' or ').replace('nullptr','None').replace('(int64_t)','')
    expr=expr.replace('at::kHalf',repr('half')).replace('at::kLong',repr('long'))
    return eval(' '.join(expr.split()),{'__builtins__':{}},env)

class NativeGuardTests(unittest.TestCase):
    def test_query_architecture_geometry_default_off_fallbacks(self):
        env=environment();self.assertTrue(guard(env))
        for q in range(1,9):
            env['bsz']=q;env['token_sorted'].n=q*8;self.assertTrue(guard(env))
        for key,bad in (('requested',False),('major',8),('minor',0),('bsz',9),('hidden_dim',2048),
                        ('intermediate_dim',4096),('num_experts',128),('num_experts_per_tok',4),
                        ('gate_mul1',False),('act_function',1),('m_tile',32),('N_off',0),('pipe_sel',43),
                        ('smem_override',90000),('blocks_per_sm',2),('_output_scratch',None),('_fused_base',None)):
            e=environment();e[key]=bad;self.assertFalse(guard(e),key)
        for key in ('hidden_state','weight_sorted','token_sorted','K_gate_arr','up_ptrs_svh','expert_count','fused_base'):
            e=environment();e[key].contiguous=False;self.assertFalse(guard(e),key)
    def test_raw_pointer_tables_are_int64(self):
        for projection in ('gate','up','down'):
            for kind in ('trellis','suh','svh'):
                e=environment();key=projection+'_ptrs_'+kind;e[key].dtype='int'
                self.assertFalse(guard(e),'non-int64 pointer table accepted: '+key)

if __name__=='__main__':unittest.main(verbosity=2)
