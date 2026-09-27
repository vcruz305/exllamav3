import unittest, importlib.util, json, math, hashlib, sys, io
from pathlib import Path
HERE=Path(__file__).resolve().parent
EVIDENCE=HERE.parent/'round8-cost-width'

class ProfileEvidence(unittest.TestCase):
    def test_builder_pools_real_rounds_and_matches_parent_independently(self):
        path=HERE/'build_profile.py'
        self.assertTrue(path.exists(),'offline profile builder missing')
        spec=importlib.util.spec_from_file_location('builder',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        p=m.build(EVIDENCE)
        parent=json.loads((EVIDENCE/'parent-window-costs.json').read_text())
        self.assertEqual(p['warm_rounds'],606)
        self.assertEqual(p['cold_rounds'],191)
        self.assertEqual(p['total_rounds'],797)
        self.assertEqual(p['zero_proposal_rounds'],132)
        for got,want in zip(p['q_costs'],parent['pooled_whole_round_cost_by_native_q']):
            self.assertEqual(got['q'],want['q'])
            self.assertEqual(got['samples'],want['samples'])
            self.assertEqual(got['case_counts'],{c:want['case_counts'].get(c,0) for c in ('code','prose')})
            self.assertEqual(got['mean_ms'],want['mean_round_wall_ms'])
            self.assertEqual(got['min_ms'],want['min_round_wall_ms'])
            self.assertEqual(got['max_ms'],want['max_round_wall_ms'])
        self.assertEqual(p['q_costs'][-1]['case_counts']['prose'],0)
        for path,sha in p['sources'].items():
            self.assertEqual(hashlib.sha256((EVIDENCE/path).read_bytes()).hexdigest(),sha)
        if (HERE/'profile.json').exists():
            self.assertEqual(json.loads((HERE/'profile.json').read_text()),p)
        else:
            with (HERE/'profile.json').open('x') as f:json.dump(p,f,indent=2)

if __name__=='__main__':
    label=sys.argv[1];stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProfileEvidence))
    text=stream.getvalue();print(text)
    with (HERE/'evidence'/f'{label}.txt').open('x') as f:f.write(text)
    sys.exit(not result.wasSuccessful())
