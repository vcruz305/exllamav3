"""Source-driven ownership enumeration, NOT a CUDA execution/synchronization model.

Expressions are read from the actual candidate loops and the full-K partition.
An independent expected domain checks every eligible assignment and Had128 group.
"""
from collections import Counter
import re, unittest
from test_candidate import Q, ROOT


def expression(text, declaration):
    pattern = re.escape(declaration) + r'\s*=\s*([^;]+);'
    match = re.search(pattern, text)
    assert match, declaration
    return match[1].strip()


def evaluate(expr, **env):
    expr = expr.replace('(size_t)', '').replace('(int)', '')
    expr = expr.replace('blockIdx.x','bx').replace('blockIdx.z','bz').replace('threadIdx.x','tx').replace('blockDim.x','threads')
    expr = expr.replace('&&',' and ').replace('||',' or ').replace('/', '//')
    return eval(' '.join(expr.split()), {'__builtins__':{},'MIN':min,'MAX':max},env)


def source_spans(counts, cap, lo, hi, kernel):
    pred = re.search(r'if \((count > 0[^\n]+)\)',kernel)[1]
    # Execute the scanner's predicate and prefix update, including skipped experts.
    assert 'start += count;' in kernel and 'active++ == (int) blockIdx.z' in kernel
    result=[];start=0
    for expert,count in enumerate(counts[:-1]):
        if evaluate(pred,count=count,max_tokens_per_expert=cap,count_lo=lo,count_hi=hi):
            result.append((expert,start,count))
        start += count
    return result


def source_domain(counts,cap,lo,hi):
    kernel=(Q/'exl3_moe_three_stage.cuh').read_text()
    gemm=(Q/'exl3_gemm_inner.cuh').read_text()
    starts=source_spans(counts,cap,lo,hi,kernel)
    # Extract the full-K arm of the real ternary expressions, not a copied formula.
    beg=expression(gemm,'int slice_beg').split('?')[1].split(':')[0]
    end=expression(gemm,'int slice_end').split('?')[1].split(':')[0]
    col=re.search(r'const int col = (blockIdx.x[^;]+);',kernel)[1]
    row_expr=expression(kernel,'const int r')
    upper=re.search(r'for \(int w = warp; w < (MIN[^;]+); w \+= warps\)',kernel)[1]
    assert 'row += 16' in kernel
    produced=Counter();consumed=Counter();domain=Counter()
    for e,start,rows in starts:
        for stage,k,n in ((1,4096,2048),(2,2048,4096)):
            for bx in range(n//256):
                b=evaluate(beg,tiles_n=n//256,tiles_k=k//32,bx=bx,num_slices=n//256)
                en=evaluate(end,tiles_n=n//256,tiles_k=k//32,bx=bx,num_slices=n//256)
                assert b % (k//32) == en % (k//32) == 0
                # Actual helper write_sum_gl writes both halves of each warp's output.
                # Its row guards exclude r>=MIN(rows-row,16), all 256 columns present.
                assert 'MIN(size_m, MT)' in (Q/'exl3_moe_kernel.cuh').read_text()
                for row in range(0,rows,16):
                    for tile in range(b//(k//32),en//(k//32)):
                        for rr in range(min(16,rows-row)):
                            for group in range(tile*2,tile*2+2):
                                for projection in (('gate','up') if stage==1 else ('down',)):
                                    produced[(e,row+rr,projection,group,bx)] += 1
                    for warp in range(16):
                        for w in range(warp,evaluate(upper,rows=rows,row=row),16):
                            r=evaluate(row_expr,row=row,w=w)
                            c=evaluate(col,bx=bx,w=w)
                            assert 0<=r<rows and c%128==0 and bx*256<=c<c+128<=bx*256+256
                            for projection in (('gate','up') if stage==1 else ('down',)):
                                consumed[(e,r,projection,c//128,bx)] += 1
                            domain[(e,start+r,stage,c//128)] += 1
    return produced,consumed,domain


class OwnershipTests(unittest.TestCase):
    def test_all_rows_tiles_partial_tiers_duplicates_exclusions(self):
        cases=[]
        for q in range(1,9):
            # q rows, top8: hot, spread, all duplicates; counts plus sentinel.
            cases += [([q]*8+[0]*249,64,1,64),([1]*(q*8)+[0]*(257-q*8),64,1,64),([q*8]+[0]*256,64,1,64)]
        # Every remainder across a 16-row boundary, and excluded prefix/middle.
        for count in range(1,65):
            cases.append(([count]+[0]*256,64,1,64))
        cases += [([16,0,8,8,8,8,8,8]+[0]*249,8,1,8),
                  ([16,0,8,8,8,8,8,8]+[0]*249,64,9,64),
                  ([8,17,3,20,16]+[0]*252,64,4,16),
                  ([64]+[0]*256,8,1,8),([8]*8+[0]*249,64,9,64),
                  ([0]*256+[64],64,1,64)]
        checked=0
        for counts,cap,lo,hi in cases:
            p,c,d=source_domain(counts,cap,lo,hi)
            self.assertEqual(p,c)
            self.assertTrue(all(v==1 for v in p.values()))
            self.assertTrue(all(v==1 for v in d.values()))
            expected=Counter();start=0
            for e,count in enumerate(counts[:-1]):
                if 0<count<=cap and lo<=count<=hi:
                    for row in range(count):
                        for stage,groups in ((1,16),(2,32)):
                            for group in range(groups):expected[(e,start+row,stage,group)]+=1
                start+=count
            self.assertEqual(d,expected)
            checked+=sum(p.values())
        print('ownership_cases=',len(cases),'same_CTA_projection_groups=',checked)
    def test_incumbent_complete_K_topology_and_counterexample(self):
        original=(ROOT/'original/exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh').read_text()
        b=expression(original,'int slice_beg');en=expression(original,'int slice_end')
        for k,n in ((4096,2048),(2048,4096)):
            for bx in range(8):
                env=dict(tiles_k=k//32,tiles_n=n//256,bx=bx,num_slices=8)
                self.assertEqual(evaluate(b,**env)%(k//32),0)
                self.assertEqual(evaluate(en,**env)%(k//32),0)
        # group32 can split K: unchanged inner arithmetic alone isn't identity then.
        self.assertNotEqual(evaluate(en,tiles_k=128,tiles_n=8,bx=0,num_slices=32)%128,0)
    def test_stage_boundaries_and_shared_reuse_are_explicit(self):
        k=(Q/'exl3_moe_three_stage.cuh').read_text();g=(Q/'exl3_gemm_inner.cuh').read_text();h=(Q/'exl3_moe.cu').read_text()
        self.assertIn('stage == 0 ? 1 : (stage == 1 ? 8 : 16), 1, experts',h)
        self.assertIn('cp_async_wait<0>();\n        __syncthreads();',g)
        self.assertIn('if (!sub_k) write_sum_gl();',g)
        self.assertIn('__syncwarp();',k)
        self.assertNotIn('group_barrier(',k);self.assertNotIn('atomicAdd(',k)
        self.assertNotIn('gemv',k)
        # Derive worst-case exact shared layout from source constants and formulas.
        tile_m,tile_k,tile_n,stages,base,bits=16,32,256,3,256,8
        frags=2*(tile_n//16)//(base//32)
        a=tile_m*tile_k*2;b=(tile_k//16)*(tile_n//16)*(16*bits)*2
        c=4*base*frags*(tile_m//16)*4
        exact=stages*(a+b)+c
        reserved=stages*(a+b)+2*max(c,tile_m*tile_n*4)
        had=(512//32)*128*4
        self.assertEqual((exact,reserved,had),(44032,60416,8192))
        self.assertGreaterEqual(reserved,max(exact,had))
        self.assertIn('MAX(gemm_smem, output_had_smem)',h)
        self.assertEqual(64*(4096*2+2048*2)*2,1572864)
    def test_compact_gather_source_and_tier_offsets(self):
        host=(Q/'exl3_moe.cu').read_text()
        sig=host.split('void exl3_moe_gather\n')[1].split(')')[0]
        self.assertLess(sig.index('expert_start'),sig.index('slot_base'))
        expr=re.search(r'slot = (slot_base\[e\][^;]+);',host)[1]
        starts=[0,16,16,24];base=[-1,-1,0,8]
        for e,count in ((2,8),(3,8)):
            for r in range(count):
                slot=evaluate(expr,slot_base=base,expert_start=starts,e=e,pos=starts[e]+r)
                self.assertEqual(slot,base[e]+r)
                self.assertNotEqual(evaluate(expr,slot_base=starts,expert_start=base,e=e,pos=starts[e]+r),slot)
        k=(Q/'exl3_moe_three_stage.cuh').read_text()
        self.assertIn('(fused_base[expert] + r) * hidden_dim + col',k)
        self.assertIn('weight_sorted[start + r]',k)

if __name__=='__main__':unittest.main(verbosity=2)
