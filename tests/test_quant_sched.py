"""CPU checks for the quantization work queue. No GPU, no torch, no extension import."""

import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "exllamav3" / "conversion" / "quant_sched.py"
_spec = importlib.util.spec_from_file_location("quant_sched", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pull_order = _mod.pull_order
simulate_queue = _mod.simulate_queue
static_makespan = _mod.static_makespan
tail_is_one_group = _mod.tail_is_one_group


def test_pull_order_keeps_existing_order_without_measured_costs():
    assert pull_order(5, None) == [0, 1, 2, 3, 4]


def test_pull_order_schedules_measured_cost_first_and_does_not_use_size_times_k():
    # Caller passes measured costs. A small high-cost group is pulled before a large cheap one.
    assert pull_order(3, costs=[1.0, 50.0, 3.0]) == [1, 2, 0]


def test_every_group_pulled_once():
    costs = [4.0, 1.0, 9.0, 2.0, 2.0, 7.0]
    assigned, _ = simulate_queue(costs, 4, costs=costs)
    flat = [i for groups in assigned for i in groups]
    assert sorted(flat) == list(range(len(costs)))


def test_worker_error_drops_no_group_identity():
    # The queue hands out each id once. A failed worker must not be required to
    # re-queue a group it already started; the job aborts instead. Unpulled ids remain.
    pulled = []
    pending = pull_order(4, [3, 1, 8, 2])
    def worker(fail_on):
        while pending:
            gid = pending.pop(0)
            pulled.append(gid)
            if gid == fail_on:
                raise RuntimeError("worker failed")
    with pytest.raises(RuntimeError):
        worker(2)
    assert pulled.count(2) == 1
    assert set(pulled).isdisjoint(set(pending))


def test_queue_makespan_beats_static_numel_fit_when_cost_is_not_numel():
    # Small groups are the expensive ones. Static best-fit never sees that, so it
    # piles both onto the device that still has numel budget. The queue pulls
    # measured-expensive groups first. This is not a size*K model.
    numel = [80, 80, 80, 10, 10]
    measured = [5.0, 5.0, 5.0, 40.0, 40.0]
    static_assigned, _ = static_makespan(numel, 2)
    static_clock = max(sum(measured[i] for i in groups) for groups in static_assigned)
    _, queued = simulate_queue(measured, 2, costs=measured)
    assert static_clock == 85
    assert queued == 50
    assert queued < static_clock


def test_giant_group_is_not_divisible_by_the_queue():
    # One group costs 60, seven cost 4. Eight workers. The tail is that one group.
    costs = [60.0] + [4.0] * 7
    assert tail_is_one_group(costs, 8)
    _, makespan = simulate_queue(costs, 8, costs=costs)
    assert makespan == 60.0
