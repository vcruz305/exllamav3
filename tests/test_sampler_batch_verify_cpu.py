"""CPU/source-AST tests of the merged generator's batched spec-window readback gate.

Upstream v1.6.5 (288d0434) replaced the sampler-level supports_batch_verify gate with a
batched speculative-window readback: when no per-position constraint advances between the
window's positions, every sampler still launches in serial-loop order but the W sampled
tokens read back through one pinned buffer and one synchronize. The gate lives in
Generator.iterate_gen as the `batchable` expression; the fork adds `job.token_budget is None`
(phase-budget enforcement runs at each accepted producer position, so a budgeted window can
never be pre-sampled). 85cae3e7 adds the draft_max_rows row ceiling, applied by
Generator.row_limited before the drafter runs.

Everything here is extracted from the sources with ast and run on stub objects, so nothing
is built, no extension is loaded and no CUDA is touched. The sampler classes are loaded
through cpu_requirements_harness, which reads the real sampler sources the same way.
"""
import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
SOURCE = _arg.resolve() if _arg is not None and (_arg / 'exllamav3/generator/sampler').is_dir() \
    else Path(__file__).resolve().parents[1]
GENERATOR = SOURCE / 'exllamav3/generator/generator.py'


def _load_harness():
    saved = sys.argv[:]
    sys.argv = [__file__, str(SOURCE / 'exllamav3/generator/sampler')]
    spec = importlib.util.spec_from_file_location(
        'requirements_harness', Path(__file__).parent / 'cpu_requirements_harness.py')
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    sys.argv = saved
    return harness


def _gate_expression():
    """The one `batchable = (...)` assignment of the merged readback, as an eval-able node."""
    tree = ast.parse(GENERATOR.read_text(encoding='utf-8'))
    tests = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "batchable" for t in node.targets)
    ]
    assert len(tests) == 1, "expected exactly one batchable assignment in generator.py"
    return tests[0]


_GATE = compile(
    ast.Expression(_gate_expression()), str(GENERATOR), 'eval')


def _clean_job(sampler):
    return SimpleNamespace(
        checkpoint=None, forced_ids=None, new_tokens=16, min_new_tokens=0,
        filters=[], sampler=sampler, token_budget=None,
    )


def batch_gate(sampler, **changes):
    """The merged gate's verdict for a job with this sampler and job-level overrides."""
    job = _clean_job(sampler)
    for key, value in changes.items():
        setattr(job, key, value)
    # a/b are the job's rows in the flattened batch; the gate only takes a single row
    namespace = {'_NO_BATCH_READBACK': False, 'a': 0, 'b': 1, 'job': job}
    return eval(_GATE, namespace)


def _row_limited():
    """Generator.row_limited exec-able on a stub generator (logger neutered)."""
    tree = ast.parse(GENERATOR.read_text(encoding='utf-8'))
    original = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "row_limited"
    )
    probe = copy.deepcopy(original)
    probe.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    namespace = {'logger': SimpleNamespace(debug=lambda *args, **kwargs: None)}
    exec(compile(module, str(GENERATOR), 'exec'), namespace)
    return namespace['row_limited']


def _generator_with_rows(draft_max_rows, rows, num_draft_tokens):
    """A stub generator whose active jobs expose `rows` prefill-done sequences."""
    jobs = [] if rows == 0 else \
        [SimpleNamespace(is_prefill_done=lambda: True, sequences=[None] * rows)]
    return SimpleNamespace(
        draft_max_rows=draft_max_rows,
        active_jobs=jobs,
        num_draft_tokens=num_draft_tokens,
        _draft_window_last=None,
    )


class BatchGateTests(unittest.TestCase):
    def test_neutral_greedy_window_is_batch_eligible(self):
        """The readback (not the sampling) is what the gate batches: a clean greedy window
        pays one launch-to-readback round trip."""
        for fused in (False, True):
            with self.subTest(fused=fused):
                harness = _load_harness()
                n = harness.load_classes(fused)
                s = n['CustomSampler']([n['SS_RepP'](), n['SS_PresFreqP'](), n['SS_Argmax']()])
                self.assertFalse(s.reqs_past_ids)
                self.assertTrue(batch_gate(s))

    def test_sampler_past_id_requirements_close_the_gate(self):
        """Repetition penalties and DRY need the window's own sampled tokens, so the
        positions are not independent and the readback stays per-position serial."""
        for fused in (False, True):
            harness = _load_harness()
            n = harness.load_classes(fused)
            for steps in (
                [n['SS_RepP'](1.1), n['SS_Argmax']()],
                [n['SS_PresFreqP'](pres_p=.3), n['SS_Argmax']()],
                [n['SS_PresFreqP'](freq_p=.3), n['SS_Argmax']()],
            ):
                with self.subTest(fused=fused, steps=[type(x).__name__ for x in steps]):
                    s = n['CustomSampler'](steps)
                    self.assertTrue(s.reqs_past_ids)
                    self.assertFalse(batch_gate(s))

    def test_active_dry_closes_the_gate(self):
        for fused in (False, True):
            harness = _load_harness()
            n = harness.load_classes(fused)
            dry = object.__new__(n['SS_DRY'])
            dry.dry_multiplier = .5
            dry.dry_base = 1.75
            s = n['CustomSampler']([dry, n['SS_Argmax']()])
            self.assertTrue(s.reqs_past_ids)
            self.assertFalse(batch_gate(s))

    def test_token_budget_closes_the_gate(self):
        """Fork gate: phase budgets enforce at each accepted producer position, so a
        budgeted window can never be pre-sampled."""
        harness = _load_harness()
        n = harness.load_classes(False)
        s = n['CustomSampler']([n['SS_RepP'](), n['SS_PresFreqP'](), n['SS_Argmax']()])
        self.assertTrue(batch_gate(s))
        self.assertFalse(batch_gate(s, token_budget=object()))

    def test_job_constraints_close_the_gate(self):
        harness = _load_harness()
        n = harness.load_classes(False)
        s = n['CustomSampler']([n['SS_RepP'](), n['SS_PresFreqP'](), n['SS_Argmax']()])
        constraints = [
            ('checkpoint', object()),
            ('forced_ids', object()),
            ('new_tokens', -1),                 # token healing mask window
            ('filters', [SimpleNamespace(is_active=True, trigger_token=None)]),
            ('filters', [SimpleNamespace(is_active=False, trigger_token=object())]),
        ]
        for key, value in constraints:
            with self.subTest(constraint=key):
                self.assertFalse(batch_gate(s, **{key: value}))

    def test_multi_row_jobs_stay_serial(self):
        """A job spanning more than one batch row (b != a + 1) never batches."""
        harness = _load_harness()
        n = harness.load_classes(False)
        s = n['CustomSampler']([n['SS_RepP'](), n['SS_PresFreqP'](), n['SS_Argmax']()])
        job = _clean_job(s)
        namespace = {'_NO_BATCH_READBACK': False, 'a': 0, 'b': 2, 'job': job}
        self.assertFalse(eval(_GATE, namespace))

    def test_the_serial_override_hook_closes_the_gate(self):
        harness = _load_harness()
        n = harness.load_classes(False)
        s = n['CustomSampler']([n['SS_RepP'](), n['SS_PresFreqP'](), n['SS_Argmax']()])
        job = _clean_job(s)
        namespace = {'_NO_BATCH_READBACK': True, 'a': 0, 'b': 1, 'job': job}
        self.assertFalse(eval(_GATE, namespace))

    def test_base_sampler_still_defaults_to_serial_batching(self):
        """Speculative batching stays opt-in at the sampler level; the base sampler never
        claims it."""
        harness = _load_harness()
        n = harness.load_classes(False)
        self.assertFalse(n['Sampler']().supports_batch_verify)


class RowBudgetTests(unittest.TestCase):
    def test_ceiling_is_neutral_when_unset(self):
        row_limited = _row_limited()
        for window in (0, 5):
            with self.subTest(window=window):
                g = _generator_with_rows(0, 2, 5)
                self.assertEqual(row_limited(g, window), window)

    def test_ceiling_applies_per_total_row_count(self):
        row_limited = _row_limited()
        # bsz * (1 + window) rows must stay under draft_max_rows
        for draft_max_rows, rows, window, expected in (
            (8, 2, 5, 3),      # 2 * (1 + 5) = 12 > 8 -> 8 // 2 - 1
            (9, 4, 5, 1),      # 9 // 4 - 1
            (2, 4, 5, 0),      # no room for any draft row
            (64, 1, 5, 5),     # single row already under the ceiling
        ):
            with self.subTest(draft_max_rows=draft_max_rows, rows=rows, window=window):
                g = _generator_with_rows(draft_max_rows, rows, window)
                self.assertEqual(row_limited(g, window), expected)

    def test_ceiling_ignores_jobs_that_are_not_prefill_done(self):
        row_limited = _row_limited()
        g = _generator_with_rows(8, 0, 5)
        g.active_jobs = [SimpleNamespace(is_prefill_done=lambda: False, sequences=[None] * 2)]
        self.assertEqual(row_limited(g, 5), 5)

    def test_trimmed_window_is_recorded_for_the_debug_log(self):
        row_limited = _row_limited()
        g = _generator_with_rows(8, 2, 5)
        self.assertEqual(row_limited(g, 5), 3)
        self.assertEqual(g._draft_window_last, 3)
        # A window below the ceiling leaves the recorded value untouched
        self.assertEqual(row_limited(g, 2), 2)
        self.assertEqual(g._draft_window_last, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2, argv=[sys.argv[0]])
