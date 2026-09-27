import unittest
from generator_harness import fixture, step

class RoundDiagnostics(unittest.TestCase):
    def test_final_round_survives_job_removal(self):
        gen,job=fixture(record=True,max_new=1)
        results=step(gen)
        self.assertEqual(gen.model.calls,1)
        self.assertEqual(gen.active_jobs,[])
        self.assertEqual(results[-1]['new_tokens'],1)
        self.assertEqual(len(job.draft_stats),1)
        diag=getattr(gen,'draft_diagnostics',None)
        self.assertIsNotNone(diag, 'completed target pass must be exported outside removed Job')
        self.assertEqual(diag['target_forward_invocations'],1)
        p=diag['passes'][-1]
        self.assertTrue(p['completed'])
        self.assertEqual(p['role'],'target_decode')
        self.assertEqual(p['proposed_window'],3)
        self.assertEqual(p['jobs'],[dict(serial=17, sampled_tokens=1, accepted_prefix=0,
             emitted_token_ids=1,new_tokens_delta=1,eos=True,requeued=False,rewound=False)])

    def test_two_target_passes_include_final_partial_pass(self):
        gen,_=fixture(record=True,max_new=5)
        step(gen); step(gen)
        d=gen.draft_diagnostics
        self.assertEqual(d['target_forward_invocations'],2)
        self.assertEqual([p['jobs'][0]['sampled_tokens'] for p in d['passes']],[4,1])
        self.assertEqual([p['jobs'][0]['emitted_token_ids'] for p in d['passes']],[4,1])
        self.assertTrue(d['passes'][-1]['jobs'][0]['eos'])

    def test_batched_jobs_share_one_actual_target_forward(self):
        gen,job=fixture(record=True,max_new=1)
        _,other=fixture(record=True,max_new=1)
        other.prepare_for_queue(gen,18)
        other.time_first_prefill=other.time_first_token=other.time_enqueue
        gen.active_jobs.append(other)
        step(gen,draft=None)
        self.assertEqual(gen.model.calls,1)
        d=gen.draft_diagnostics
        self.assertEqual(d['target_forward_invocations'],1)
        self.assertEqual([j['serial'] for j in d['passes'][0]['jobs']],[17,18])
        self.assertEqual([j['emitted_token_ids'] for j in d['passes'][0]['jobs']],[1,1])

    def test_records_bounded_including_legacy_history(self):
        gen,job=fixture(record=True,max_new=2000)
        for _ in range(260): step(gen)
        d=gen.draft_diagnostics
        self.assertEqual(d['target_forward_invocations'],gen.model.calls)
        self.assertEqual(d['target_forward_invocations'],260)
        self.assertEqual(len(d['passes']),256)
        self.assertEqual(d['dropped_passes'],4)
        self.assertEqual(d['passes'][0]['pass_id'],5)
        self.assertEqual(len(job.draft_stats),256)

    def test_eos_stop_token_not_emitted(self):
        gen,job=fixture(record=True,stop=(10,))
        result=step(gen)
        p=gen.draft_diagnostics['passes'][-1]['jobs'][0]
        self.assertEqual((p['sampled_tokens'],p['accepted_prefix'],p['emitted_token_ids']),(1,0,0))
        self.assertTrue(p['eos'])
        self.assertNotIn('token_ids',result[-1])

    def test_first_last_mismatch_full_accept_serial_and_batch(self):
        for bv in (False,True):
            for draft,sampled,accepted in (((99,11,12),1,0),((10,11,99),3,2),((10,11,12),4,3)):
                with self.subTest(bv=bv,draft=draft):
                    gen,job=fixture(record=True,batch_verify=bv)
                    step(gen,draft)
                    p=gen.draft_diagnostics['passes'][-1]['jobs'][0]
                    self.assertEqual((p['sampled_tokens'],p['accepted_prefix'],p['emitted_token_ids']),
                                     (sampled,accepted,sampled))
                    self.assertEqual(job.sequences[0].allocated_pages[0].kv_position,2+sampled)
                    self.assertEqual(gen.model.calls,1)

    def test_requeue_finalized_before_job_reinit(self):
        gen,job=fixture(record=True,max_new=7)
        job.max_rq_tokens=gen.draft_reserve_tokens+1
        step(gen)
        p=gen.draft_diagnostics['passes'][-1]['jobs'][0]
        self.assertEqual((p['sampled_tokens'],p['emitted_token_ids'],p['accepted_prefix']),(2,2,1))
        self.assertTrue(p['requeued'])
        self.assertEqual(gen.active_jobs,[])
        self.assertIs(gen.pending_jobs.pop(),job)
        self.assertEqual(job.new_tokens,0)  # actual prepare_for_requeue called actual __init__
        job.time_first_prefill=job.time_first_token=job.time_enqueue
        gen.active_jobs=[job]
        step(gen)
        self.assertEqual(gen.draft_diagnostics['target_forward_invocations'],2)
        self.assertEqual(gen.draft_diagnostics['passes'][0]['jobs'][0],p)
        self.assertEqual([x['jobs'][0]['serial'] for x in gen.draft_diagnostics['passes']],[17,17])

    def test_no_draft_instrumented(self):
        gen,_=fixture(record=True,draft=False,max_new=1)
        step(gen,None)
        p=gen.draft_diagnostics['passes'][-1]
        self.assertEqual(p['proposed_window'],0)
        self.assertEqual(p['jobs'][0]['sampled_tokens'],1)
        self.assertEqual(p['jobs'][0]['accepted_prefix'],0)

    def test_forward_error_counted_without_completed_claim(self):
        gen,_=fixture(record=True)
        gen.model.fail=True
        with self.assertRaisesRegex(RuntimeError,'target failed'): step(gen)
        d=gen.draft_diagnostics
        self.assertEqual(d['target_forward_invocations'],gen.model.calls)
        self.assertFalse(d['passes'][-1]['completed'])

    def test_no_ready_job_no_forward(self):
        gen,job=fixture(record=True)
        job.sequences[0].kv_position=0
        self.assertEqual(step(gen),[])
        self.assertEqual(gen.model.calls,0)
        self.assertEqual(gen.draft_diagnostics['target_forward_invocations'],0)

    def test_held_tokens_emitted_on_later_pass_not_reconstructed(self):
        gen,_=fixture(record=True,draft=False,max_new=10)
        pieces=[str(i) for i in range(128)]; pieces[10]='�'
        gen.tokenizer.get_id_to_piece_list=lambda *a:pieces
        gen.tokenizer.decode=lambda *a,**kw:['�']
        step(gen,None)
        gen.model.tokens=[11]
        step(gen,None)
        jobs=[p['jobs'][0] for p in gen.draft_diagnostics['passes']]
        self.assertEqual([p['sampled_tokens'] for p in jobs],[1,1])
        self.assertEqual([p['emitted_token_ids'] for p in jobs],[0,2])

    def test_abandoned_window_kept_and_labeled(self):
        from types import SimpleNamespace
        gen,job=fixture(record=True)
        job.banned_strings_utf32_offsets=object()
        job.banned_strings_utf32_buffer=object()
        matches=iter([-1,0])
        job.receive_sample.__func__.__globals__['ext']=SimpleNamespace(partial_strings_match=lambda *a:next(matches))
        step(gen)
        p=gen.draft_diagnostics['passes'][-1]['jobs'][0]
        self.assertTrue(p['rewound'])
        self.assertEqual((p['sampled_tokens'],p['accepted_prefix'],p['new_tokens_delta'],p['emitted_token_ids']),
                         (2,1,1,1))
        self.assertEqual(job.draft_stats,[])  # old list skips abandoned window; diagnostics does not

    def test_dynamic_window_is_actual_proposal_not_requested_max(self):
        gen,_=fixture(record=True)
        step(gen,(10,))
        p=gen.draft_diagnostics['passes'][-1]
        self.assertEqual(p['proposed_window'],1)
        self.assertEqual(p['input_tokens_per_row'],2)
        self.assertEqual(p['jobs'][0]['sampled_tokens'],2)

    def test_diagnostic_json_contains_only_metadata(self):
        import json
        gen,_=fixture(record=True,max_new=1)
        step(gen)
        encoded=json.dumps(gen.draft_diagnostics)
        self.assertEqual(json.loads(encoded),gen.draft_diagnostics)
        self.assertNotIn('cpu-test',encoded)

class Preservation(unittest.TestCase):
    def test_instrumentation_off_avoids_output_tensor_scan(self):
        from source_harness import Tensor
        from unittest.mock import patch
        gen,_=fixture(record=False,draft=False,max_new=1)
        with patch.object(Tensor,'numel',side_effect=AssertionError('diagnostic output scan')):
            step(gen,None)
        self.assertIsNone(getattr(gen,'draft_diagnostics',None))

    def test_on_off_generation_identical(self):
        for draft in ((10,11,12),(99,11,12),(10,11,99),None):
            values=[]
            for on in (False,True):
                gen,job=fixture(record=on,max_new=3,draft=draft is not None)
                results=step(gen,draft)
                values.append(([(r.get('text'),r.get('eos'),r.get('new_tokens')) for r in results],
                    job.accepted_draft_tokens,job.rejected_draft_tokens,
                    job.sequences[0].sequence_ids.torch().a.tolist()))
            self.assertEqual(*values)

    def test_default_disabled_no_draft(self):
        gen,job=fixture(draft=False,max_new=1)
        results=step(gen,None)
        self.assertEqual(gen.model.calls,1)
        self.assertEqual(results[-1]['new_tokens'],1)
        self.assertFalse(gen.record_draft_stats)
        self.assertIsNone(getattr(gen,'draft_diagnostics',None))
        self.assertEqual(job.draft_stats,[])

if __name__=='__main__': unittest.main(verbosity=2)
