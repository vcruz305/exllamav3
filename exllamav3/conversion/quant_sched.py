"""Scheduling helpers for multi-GPU EXL3 quantization.

Pure Python: no torch, no CUDA. The quantizer call, group membership and group
order of tensors inside a group stay in convert_model.group_quant_linears.
This module only decides which worker pulls which already-built group.
"""

from __future__ import annotations


def static_best_fit(group_numels, n_devices, device_ratios=None):
    """The pre-queue assignment: one static best-fit on numel, no stealing.

    Kept so tests can show the old makespan. Do not use it for new runs.
    """
    if n_devices < 1:
        raise ValueError("n_devices")
    tot = sum(group_numels)
    if device_ratios is None:
        budget = [tot // n_devices for _ in range(n_devices)]
    else:
        if len(device_ratios) != n_devices:
            raise ValueError("device_ratios")
        split = sum(device_ratios)
        budget = [tot * r // split for r in device_ratios]
    assigned = [[] for _ in range(n_devices)]
    for i, numel in enumerate(group_numels):
        fit = [b - numel for b in budget]
        best = max(range(n_devices), key=lambda d: fit[d])
        budget[best] -= numel
        assigned[best].append(i)
    return assigned


def pull_order(n_groups, costs=None):
    """Order in which idle workers pull groups.

    ``costs`` is an optional list of measured costs, one per group, keyed by
    the caller on shape, K, group mode and Hessian preparation. Expensive
    groups are pulled first. Missing costs keep the existing group order.
    Size times K is not a cost: that ratio was not measured.
    """
    if n_groups < 0:
        raise ValueError("n_groups")
    order = list(range(n_groups))
    if costs is None:
        return order
    if len(costs) != n_groups:
        raise ValueError("costs")
    return sorted(order, key=lambda i: (-costs[i], i))


def simulate_queue(group_costs, n_devices, costs=None):
    """Idle workers pull the next group. Returns (per_device_group_ids, makespan).

    A worker never splits a group. The makespan is the max device finish time.
    """
    if n_devices < 1:
        raise ValueError("n_devices")
    # costs is the measured table. group_costs is the duration the simulator adds.
    # Without measured costs the pull order is the existing group order.
    order = list(range(len(group_costs))) if costs is None else pull_order(len(group_costs), costs)
    assigned = [[] for _ in range(n_devices)]
    finish = [0.0] * n_devices
    for i in order:
        worker = min(range(n_devices), key=lambda d: (finish[d], d))
        assigned[worker].append(i)
        finish[worker] += group_costs[i]
    return assigned, (max(finish) if finish else 0.0)


def static_makespan(group_costs, n_devices, device_ratios=None):
    assigned = static_best_fit(group_costs, n_devices, device_ratios)
    finish = [sum(group_costs[i] for i in groups) for groups in assigned]
    return assigned, (max(finish) if finish else 0.0)


def tail_is_one_group(group_costs, n_devices):
    """True when the slowest group is at least the rest of the makespan.

    A queue cannot divide a group that is already one task. Measured on the
    512-expert layers: gate/up share one Hessian and concatenate into one
    group (about 490 tensors), and same-K down projections stack into one
    group (308 at K=6 on layer 24). Those are single tasks.
    """
    if not group_costs:
        return False
    _, makespan = simulate_queue(group_costs, n_devices, costs=group_costs)
    return max(group_costs) >= makespan - 1e-9 and max(group_costs) > 0
