"""Expected RED against the earlier patched pristine3, not candidate acceptance tests."""
import unittest
from generator_harness import fixture,step

class EarlierFinalRoundBug(unittest.TestCase):
    def test_one_round_final_result_includes_that_round(self):
        gen,job=fixture(record=True,max_new=1)
        results=step(gen)
        self.assertEqual(len(job.draft_stats),1)
        self.assertEqual(gen.model.calls,1)
        self.assertEqual(results[-1].get('draft_rounds'),1)
    def test_two_round_final_result_includes_both_rounds(self):
        gen,job=fixture(record=True,max_new=5)
        step(gen)
        results=step(gen)
        self.assertEqual(len(job.draft_stats),2)
        self.assertEqual(gen.model.calls,2)
        self.assertEqual(results[-1].get('draft_rounds'),2)

if __name__=='__main__': unittest.main(verbosity=2)
