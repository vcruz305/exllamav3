"""Preservation controls executed on both original and candidate snapshots."""
from pathlib import Path
import ast, re, unittest
from test_candidate import ROOT,Q,TREE

O=ROOT/'original/exllamav3/exllamav3_ext/quant'

def function(text, name):
    start=text.index(name);left=text.index('{',start);depth=1;pos=left+1
    while depth:
        depth+=(text[pos]=='{')-(text[pos]=='}');pos+=1
    return text[left:pos]

class PreservationTests(unittest.TestCase):
    def test_all_had_codebooks_decoder_and_persistent_body_unchanged(self):
        for name in ('hadamard_inner.cuh','codebook.cuh','exl3_dq.cuh','reconstruct.cu','exl3_moe_common.cuh'):
            self.assertEqual((Q/name).read_bytes(),(O/name).read_bytes(),name)
        self.assertEqual(function((Q/'exl3_moe_kernel.cuh').read_text(),'void exl3_moe_mixedk_kernel'),
                         function((O/'exl3_moe_kernel.cuh').read_text(),'void exl3_moe_mixedk_kernel'))
    def test_gemm_pipeline_stores_and_accumulators_unchanged(self):
        old=(O/'exl3_gemm_inner.cuh').read_text();new=(Q/'exl3_gemm_inner.cuh').read_text()
        # Complete pipeline loads/dequant/MMA fragments, CTA sums and FP16 stores.
        for begin,end in (('// Pipe 0','// Output reduction'),('// Wait until','    if constexpr (FRAG_STAGES == 5)')):
            self.assertEqual(old[old.index(begin):old.index(end)],new[new.index(begin):new.index(end)])
        self.assertIn('__CUDA_ARCH__ == 860',new)
        self.assertIn('#else\n    #define EXL3_GEMM_H_ACC 0',new)
        self.assertIn('__floats2half2_rn(frag_c[0][n][0], frag_c[0][n][1])',new)
        self.assertIn('pred_a_gl[i] = m < size_m',new)
        self.assertIn('if (r1 < size_m)',new)
    def test_rounding_activation_clamp_route_association(self):
        text=(Q/'hadamard_inner.cuh').read_text()
        inp=function(text,'void had_hf_r_128_inner')
        self.assertLess(inp.index('__hmul2'),inp.index('__half2float'))
        guad=function(text,'void had_hf_r_128_guad_inner')
        positions=[guad.index(s) for s in ('// Hadamard','// Post scale','// Activation','// Optional activation limits','// Gate','// Pre scale (d)','// Store')]
        self.assertEqual(positions,sorted(positions))
        for token in ('h2exp(neg_x)','__hadd2(one, e)','h2rcp(sum)','__hmul2(vg.x, vu.x)','__hmul2(vg.x, scales_d.x)'):
            self.assertIn(token,guad)
        out=function(text,'void had_hf_r_128_d_inner')
        self.assertLess(out.index('h0 *= r_scale'),out.index('h0 *= __low2float(scales.x)'))
        self.assertIn('warp_id * 128',out);self.assertIn('blockIdx.y * 32 + t',out)
    def test_existing_35arg_abi_gather_and_pointer_types(self):
        old=(O/'exl3_moe.cuh').read_text();new=(Q/'exl3_moe.cuh').read_text()
        sig=lambda s,name: re.search(r'void '+name+r'\s*\((.*?)\);',s,re.S)[1]
        self.assertEqual(sig(old,'exl3_moe_mixedk'),sig(new,'exl3_moe_mixedk'))
        if 'void exl3_moe_mixedk_three_stage' in new:
            self.assertEqual(sig(old,'exl3_moe_mixedk'),sig(new,'exl3_moe_mixedk_three_stage'))
        self.assertEqual(function((Q/'exl3_moe.cu').read_text(),'void exl3_moe_gather\n'),
                         function((O/'exl3_moe.cu').read_text(),'void exl3_moe_gather\n'))
        host=(Q/'exl3_moe.cu').read_text();common=(Q/'exl3_moe_common.cuh').read_text()
        for p in ('gate','up','down'):
            self.assertIn('TORCH_CHECK_DTYPE(K_'+p+'_arr, kInt)',host)
            self.assertIn('const int* __restrict__ K_'+p+'_arr',common)
        args=[s.strip() for s in sig(old,'exl3_moe_mixedk').split(',')]
        self.assertEqual(len(args),35);self.assertEqual(args[9],'const int act_function')
        self.assertEqual(sig(old,'exl3_moe_gather'),sig(new,'exl3_moe_gather'))
    def test_noop_and_existing_host_checks_preserved(self):
        old=(O/'exl3_moe.cu').read_text();new=(Q/'exl3_moe.cu').read_text()
        old=old[old.index('void exl3_moe_mixedk\n'):]
        marker='static void exl3_moe_mixedk_impl\n' if 'static void exl3_moe_mixedk_impl\n' in new else 'void exl3_moe_mixedk\n'
        new=new[new.index(marker):]
        begin='    if (num_active == 0) return;';end='    void* _K_down_arr = K_down_arr.data_ptr();'
        self.assertEqual(old[old.index(begin):old.index(end)+len(end)],new[new.index(begin):new.index(end)+len(end)])
        self.assertIn('count_lo',new);self.assertIn('count_hi',new)

if __name__=='__main__':unittest.main(verbosity=2)
