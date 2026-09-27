"""CPU source/ownership contracts, explicitly not CUDA execution."""
import os
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parent
TREE = Path(os.environ.get('CANDIDATE_TREE', str(ROOT/'tree')))
Q = TREE/'exllamav3/exllamav3_ext/quant'

class FusionContracts(unittest.TestCase):
    def test_distinct_three_stage_dispatch_and_same_cta_epilogues(self):
        host = (Q/'exl3_moe.cu').read_text()
        self.assertIn('EXL3_MK_THREE_STAGE', host, 'No default-off three-stage dispatch')
        p = Q/'exl3_moe_three_stage.cuh'
        self.assertTrue(p.exists(), 'No tile-local fused ownership implementation')
        kernel = p.read_text()
        self.assertIn('exl3_moe_three_stage_kernel', kernel)
        self.assertNotIn('exl3_moe_phased_kernel', host)
        self.assertIn('stage < 3', host)
        self.assertIn('const int col = blockIdx.x * 256 + (w % 2) * 128;', kernel)
        self.assertIn('had_hf_r_128_guad_inner', kernel)
        self.assertIn('had_hf_r_128_d_inner<false>', kernel)
        self.assertEqual(kernel.count('moe_gemm_tile<0, 2, 16, 256, 3, 3, true>'), 3)

class PoisonContracts(unittest.TestCase):
    def test_debug_build_poisons_actual_owned_intermediates(self):
        host=(Q/'exl3_moe.cu').read_text()
        self.assertIn('#ifdef EXL3_THREE_STAGE_POISON',host,'No actual-intermediate poisoning build control')
        poison=host.split('#ifdef EXL3_THREE_STAGE_POISON')[1].split('#endif')[0]
        for buffer in ('phase_g','phase_u','phase_ig','phase_iu'):
            self.assertIn(buffer+'.fill_(poison)',poison)
        self.assertLess(host.index('#ifdef EXL3_THREE_STAGE_POISON'),host.index('_temp_state_g = phase_g.data_ptr()'))

if __name__ == '__main__': unittest.main(verbosity=2)
