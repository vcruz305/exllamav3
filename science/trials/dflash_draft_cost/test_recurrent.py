"""Recurrent retained-profile vertical regressions. Run via safe_run.py."""
import unittest
import retained_harness as h

class Initialization(unittest.TestCase):
    def test_retained_config_enabled(self):
        try:g=h.construct()
        except ValueError as e:self.fail('retained recurrent-SWA must initialize: '+str(e))
        self.assertIsNotNone(g.dflash_cost_policy)
        self.assertIs(type(g.model),h.MiMoV2Model)
        self.assertIs(g.model.recurrent_state_cls,h.SWAState)
        self.assertEqual(len(g.cache.layers),9)
        self.assertEqual(len(g.cache.recurrent_layers),39)
        self.assertEqual(g.cache.max_history,7)
        self.assertEqual(g.recurrent_cache.max_size,268435456)
        self.assertEqual(g.recurrent_checkpoint_interval,2048)
    def test_other_recurrence_or_unproved_configuration_rejected(self):
        changes=[lambda k:setattr(k['model'],'swa_full',True),
                 lambda k:setattr(k['model'],'recurrent_state_cls',object),
                 lambda k:setattr(k['cache'],'recurrent_state_cls',object),
                 lambda k:setattr(k['model'],'loaded_tp',True),
                 lambda k:setattr(k['model'].config,'sliding_window',256),
                 lambda k:setattr(k['model'].config,'layer_map',[0]),
                 lambda k:setattr(k['cache'],'max_history',0),
                 lambda k:k.update(recurrent_cache_size=0),
                 lambda k:k.update(recurrent_checkpoint_interval=256),
                 lambda k:k['cache'].recurrent_layers.pop(next(iter(k['cache'].recurrent_layers))),
                 lambda k:setattr(next(iter(k['cache'].recurrent_layers.values())).module,'kv_state_size',256),
                 lambda k:k['cache'].layers.update({(1,0):h.N(k_bits=4,v_bits=4)})]
        for change in changes:
            with self.subTest(change=changes.index(change)),self.assertRaises(ValueError):h.construct(change=change)

    def test_wrong_architecture_and_state_types_are_not_name_allowlisted(self):
        class OtherSWA(h.MiMoV2Model):pass
        def other(k):k['model'].__class__=OtherSWA
        with self.assertRaises(ValueError):h.construct(change=other)
        g=h.construct();j=h.queue(g)
        j.recurrent_state=h.N(**j.recurrent_state.__dict__)
        self.assertIsNone(g.dflash_cost_policy.select(g,h.scores(),7))
        g=h.construct();h.queue(g);g.recurrent_cache=object()
        self.assertIsNone(g.dflash_cost_policy.select(g,h.scores(),7))

    def test_retained_default_off(self):
        g=h.construct(False)
        self.assertIsNone(g.dflash_cost_policy)
        self.assertIsNotNone(g.recurrent_cache)
        self.assertFalse(g.model.swa_full)

class Eligibility(unittest.TestCase):
    def test_unproved_live_recurrent_state_and_lifecycle_fall_back(self):
        changes=[lambda g,j:setattr(j,'recurrent_state',None),
                 lambda g,j:setattr(j.recurrent_state,'position',0),
                 lambda g,j:setattr(j.recurrent_state,'cache',object()),
                 lambda g,j:setattr(j,'is_requeued',True),
                 lambda g,j:setattr(j,'is_finished',True),
                 lambda g,j:setattr(j,'checkpoint_rewound',True),
                 lambda g,j:setattr(j,'stop_strings',{'end'}),
                 lambda g,j:setattr(j,'loop_detector',object()),
                 lambda g,j:setattr(j,'cached_pages',-1),
                 lambda g,j:setattr(g,'recurrent_checkpoint_interval',256)]
        for i,change in enumerate(changes):
            with self.subTest(i=i):
                g=h.construct();j=h.queue(g);change(g,j)
                self.assertIsNone(g.dflash_cost_policy.select(g,h.scores(),7))

    def test_checkpoint_boundary_envelope_uses_cached_prefix_offset(self):
        for cached_pages in (0,1):
            boundary=2048+cached_pages*256
            for delta in (-9,-8,-7,-1,0,1,9):
                pos=boundary+delta
                with self.subTest(cached_pages=cached_pages,delta=delta):
                    g=h.construct();j=h.queue(g,prompt=pos+1,max_new=256,max_rq=None)
                    j.cached_pages=cached_pages
                    expected_fallback=False
                    # Execute actual checkpoint predicate across current + full q envelope.
                    for step in range(9):
                        j.sequences[0].kv_position=pos+step
                        expected_fallback |= j.is_checkpoint_boundary()
                    j.sequences[0].kv_position=pos
                    choice=g.dflash_cost_policy.select(g,h.scores(),7)
                    self.assertEqual(choice is None,expected_fallback)
                    if expected_fallback:
                        g.dflash_cost_policy.costs=(1,100,100,100,100,100,100,100)
                        self.assertEqual(h.n.h.draft(g).shape,(1,7),'boundary must retain DDS')

    def test_ordinary_retained_round_reaches_policy_and_native_producer(self):
        g=h.construct();j=h.queue(g)
        self.assertEqual(j.max_rq_tokens,2025)
        self.assertIsNone(j.checkpoint) # banned-string hold, NOT recurrent state
        self.assertIsNotNone(j.recurrent_state)
        expected=g.dflash_cost_policy.choose(g.draft_calibrator,[10.]*7,7)
        self.assertIsInstance(expected,int)
        self.assertEqual(g.dflash_cost_policy.select(g,h.scores(),7),expected,
                         'ordinary retained-profile round must select, not always fall back')
        # Force a distinct k=0 ranking only after proving observed-cost eligibility.
        g.dflash_cost_policy.costs=(1,100,100,100,100,100,100,100)
        self.assertIsNone(h.n.h.draft(g))
        self.assertEqual(g._draft_conf_round['window'],0)
        self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
        self.assertEqual(g.draft_reserve_tokens,7)

class HostPreservation(unittest.TestCase):
    def test_interior_progresses_to_effective_requeue_not_requested_one(self):
        g=h.construct();j=h.queue(g)
        eligible=fallback=rounds=0
        while rounds<4096:
            choice=g.dflash_cost_policy.select(g,h.scores(),7)
            if choice is None:fallback+=1
            else:eligible+=1
            p=h.n.h.draft(g);r=h.verify(g,j,p);rounds+=1
            self.assertEqual(j.recurrent_state.position,j.sequences[0].kv_position)
            self.assertEqual(sum(x.kv_position for x in j.sequences[0].allocated_pages),j.sequences[0].kv_position)
            if r.requeued:break
            self.assertEqual(r.completed,[])
        else:self.fail('effective requeue was never reached')
        self.assertGreater(eligible,1);self.assertGreater(fallback,0)
        self.assertEqual(j.new_tokens,2019)
        self.assertEqual(j.max_rq_tokens,2025)
        print('ACTUAL_HOST_PROGRESS',dict(rounds=rounds,eligible=eligible,fallback=fallback,
              new_tokens=j.new_tokens,position=j.recurrent_state.position,effective_max_rq=j.max_rq_tokens))

    def test_checkpoint_truncation_and_stash_unchanged(self):
        outcomes=[]
        for on in (False,True):
            g=h.construct(on);j=h.queue(g,prompt=2047,max_new=256,max_rq=None)
            p=h.n.h.draft(g);r=h.verify(g,j,p)
            self.assertEqual(j.sequences[0].kv_position,2048)
            self.assertEqual(j.recurrent_state.position,2048)
            self.assertEqual(sum(x.kv_position for x in j.sequences[0].allocated_pages),2048)
            calls=[];stash=h.N(put=lambda key,state:calls.append((key,state.position)))
            j.maybe_stash_recurrent(stash);j.maybe_stash_recurrent(stash)
            self.assertEqual(len(calls),1)
            outcomes.append((j.accepted_draft_tokens,j.rejected_draft_tokens,j.new_tokens,r.lengths))
        self.assertEqual(outcomes[0],outcomes[1])

    def test_actual_requeue_budget_boundary_and_continuation(self):
        outcomes=[]
        for on in (False,True):
            g=h.construct(on);j=h.queue(g)
            self.assertEqual(j.orig_max_rq_tokens,1);self.assertEqual(j.max_rq_tokens,2025)
            # Recorded 23-token prompt: actual requeue is new_tokens > 2025-7.
            h.set_position(j,22+2018,2018)
            if on:self.assertIsNone(g.dflash_cost_policy.select(g,h.scores(),7))
            p=h.n.h.draft(g);r=h.verify(g,j,p)
            self.assertEqual(r.requeued,[j]);self.assertEqual(j.new_tokens,2019)
            self.assertEqual(j.recurrent_state.position,j.sequences[0].kv_position)
            self.assertEqual(sum(x.kv_position for x in j.sequences[0].allocated_pages),j.sequences[0].kv_position)
            before=(j.new_tokens,j.sequences[0].kv_position)
            continuation=j.prepare_for_requeue() # full source; reinitializes same CPUJob
            self.assertEqual(continuation.orig_max_rq_tokens,1)
            self.assertTrue(continuation.last_init_kwargs['rq_state']['rq_new_tokens']>0)
            h.n.h.allocate(g,continuation)
            # Allocation/prefill remains opaque; populate CPU page records for next forward.
            h.n.equip_full_receive(continuation)
            continuation.is_requeued=True;continuation.is_finished=False
            continuation.sampler=h.N(supports_batch_verify=True,reqs_past_ids=False)
            g.cache.reset_states();continuation.recurrent_state=g.cache.get_new_state()
            h.set_position(continuation,continuation.sequences[0].kv_position)
            g.draft_cache.sequences=continuation.sequences
            for page in continuation.sequences[0].allocated_pages:
                page.ref_count=1;page.sequence=h.n.Tensor((1,256));page.phash=None;page.update_hash=lambda value:None
            self.assertLessEqual(len(continuation.sequences[0].allocated_pages)*256,4096)
            if on:self.assertIsNone(g.dflash_cost_policy.select(g,h.scores(),7))
            p2=h.n.h.draft(g);h.verify(g,continuation,p2)
            self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
            outcomes.append((before,continuation.max_rq_tokens,continuation.sequences[0].kv_position))
        self.assertEqual(outcomes[0],outcomes[1])

    def test_max_new_and_unexpected_eos_preserve_terminal_controls(self):
        for max_new,kind in ((1,'accept'),(8,'accept'),(100,'eos')):
            outcomes=[]
            for on in (False,True):
                g=h.construct(on);j=h.queue(g,max_new=max_new,max_rq=None)
                if on and max_new<=8:self.assertIsNone(g.dflash_cost_policy.select(g,h.scores(),7))
                p=h.n.h.draft(g);r=h.verify(g,j,p,kind)
                self.assertEqual(r.completed,[j]);self.assertEqual(r.requeued,[])
                self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
                outcomes.append((j.new_tokens,j.accepted_draft_tokens,j.rejected_draft_tokens,[x.get('eos_reason') for x in r.results]))
            self.assertEqual(outcomes[0],outcomes[1])

    def test_selected_short_windows_stop_tokens_and_bonus(self):
        for k in (0,1,3,7):
            for stop_index in (0,k):
                g=h.construct();j=h.queue(g,max_new=256,max_rq=None)
                g.dflash_cost_policy.costs=tuple([1.]*(k+1)+[100.]*(7-k))
                p=h.n.h.draft(g)
                self.assertEqual(g._draft_conf_round['window'],k)
                r=h.verify(g,j,p,'eos',stop_index)
                self.assertEqual(r.completed,[j]);self.assertEqual(r.requeued,[])
                self.assertEqual(j.new_tokens,stop_index+1)
                self.assertEqual(j.accepted_draft_tokens+j.rejected_draft_tokens,k)
                self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)

    def test_end_over_requeue_precedence_unchanged_for_recurrence(self):
        for kind in ('eos','max_new'):
            for on in (False,True):
                g=h.construct(on);j=h.queue(g,max_new=1 if kind=='max_new' else 100,max_rq=None)
                # Actual receive_sample controls simultaneous terminal + budget crossing.
                j.max_rq_tokens=7
                if on:self.assertIsNone(g.dflash_cost_policy.select(g,h.scores(),7))
                r=h.verify(g,j,h.n.h.draft(g),'eos' if kind=='eos' else 'accept')
                self.assertEqual(r.completed,[j]);self.assertEqual(r.requeued,[])
                self.assertEqual(j.new_tokens,1)

    def test_zero_and_short_policy_rounds_keep_native_page_coverage(self):
        for k in (0,1,3,7):
            for pos in (22,254,510,766):
                g=h.construct();j=h.queue(g,prompt=pos+1,max_new=256,max_rq=None)
                g.dflash_cost_policy.costs=tuple([1.]*(k+1)+[100.]*(7-k))
                p=h.n.h.draft(g);self.assertEqual(g._draft_conf_round['window'],k)
                r=h.verify(g,j,p,'mismatch',0) if k else h.verify(g,j,p)
                self.assertEqual(j.recurrent_state.position,j.sequences[0].kv_position)
                self.assertEqual(sum(x.kv_position for x in j.sequences[0].allocated_pages),j.sequences[0].kv_position)
                self.assertEqual(g.draft_model.calls[-1]['native_rows'],8)
                self.assertEqual(j.recurrent_state.last_history,0)

if __name__=='__main__':unittest.main(verbosity=2)
