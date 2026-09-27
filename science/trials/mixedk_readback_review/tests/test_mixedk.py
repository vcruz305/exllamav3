import unittest
from collections import Counter
import numpy as np
from source_harness import moe_fixture, READBACKS

class Elision(unittest.TestCase):
    def test_opt_in_removes_only_dead_readback(self):
        m,x,ext = moe_fixture([[0,1],[0,2]], enabled=True)
        out = m.forward(x,{})
        self.assertEqual(Counter(ext.rows), Counter([(0,0),(1,0),(0,1),(2,1)]))
        self.assertTrue(np.isfinite(out.a).all())
        self.assertEqual(READBACKS, ['counts'])

class Preservation(unittest.TestCase):
    def check(self, selected, **kwargs):
        outputs = []
        traces = []
        for enabled in (False,True):
            m,x,ext = moe_fixture(selected, enabled=enabled, **kwargs)
            out = m.forward(x,{})
            expected = Counter((e-m.routing_first,t) for t,row in enumerate(selected) for e in row
                               if m.routing_first <= e < m.routing_first+m.num_local_experts)
            self.assertEqual(Counter(ext.rows), expected)
            reference = np.zeros_like(out.a)
            for (e,t), multiplicity in expected.items(): reference[t] += x.a[t]*(e+1)*multiplicity
            np.testing.assert_array_equal(out.a, reference)
            outputs.append(out.a.copy()); traces.append(ext.launches)
        np.testing.assert_array_equal(*outputs)
        self.assertEqual(*traces)
        return traces[0]
    def test_default_retains_both_readbacks(self):
        m,x,_ = moe_fixture([[0,1]])
        m.forward(x,{})
        self.assertEqual(READBACKS, ['counts','nonzero'])
    def test_active_below_concurrency_unchanged(self):
        trace = self.check([[0,0],[0,0]], concurrency=12)
        self.assertEqual(trace[0][1],1)
    def test_empty_local_tp_sentinel(self): self.assertEqual(self.check([[0,7],[7,0]], local=2,total=8,first=3), [])
    def test_partial_local_tp_sentinel(self): self.check([[0,3],[4,7]], local=2,total=8,first=3)
    def test_empty_expert_slice(self): self.check([[0],[1]],local=0,total=4)
    def test_row_cap_and_fallback_exact_once(self):
        self.check([[0],[0],[0],[1],[2]], rowcap=2)
    def test_real_row_cap_boundary_exact_once(self):
        self.check([[0]]*128 + [[1]]*129,rowcap=128)
    def test_mtile_host_counts(self):
        trace = self.check([[0]]*80 + [[1]]*20 + [[2]]*5, mtile=True, rowcap=96)
        self.assertEqual(sorted(t[5] for t in trace), [16,32,64])
    def test_legacy_grouped_unchanged(self): self.check([[0,1],[2,3]], unified=False,legacy=True,det=False,initialized=False)
    def test_first_unified_init(self): self.check([[0,1],[2,3]],initialized=False)
    def test_atomic_mode(self): self.check([[0,1],[2,3]],det=False)
    def test_grouped_then_unified_preexisting_hazard(self):
        # Baseline reuses _mkd_bufs but has no _mkd_fused_rows after legacy init.
        # This is NOT fixed by this narrow patch; no runtime eligibility switching is approved.
        for enabled in (False,True):
            m,x,_ = moe_fixture([[0,1],[2,3]],unified=False,legacy=True,det=False,initialized=False,enabled=enabled)
            m.forward(x,{})
            m.mixedk_unified = True
            with self.assertRaisesRegex(AttributeError, '_mkd_fused_rows'):
                m.forward(x,{})

if __name__ == '__main__': unittest.main(verbosity=2)
