"""Bounded standalone GB10 numerical + A/B/A trial; never loads a model.

Requires explicit operator authorization, Linux, sm_121. CPU --help is safe.
A is compiled from the unmodified PIN; B from the candidate. Independent reference
uses the original reconstruct + explicit half-rounded Hadamard/MLP operations.
Synthetic performance is NOT production throughput. A supplied small layer capture
can replace synthetic inputs, but full unmodified module-input replay is still a
promotion requirement. Do not publish private model captures.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

from build_trial import load_trial, require_authorization
from scalar_reference import scalar_windows, fragment_rc, decode_mul1

def unpack_tile(words, bits):
    tile = [[0.0] * 16 for _ in range(16)]
    codes = scalar_windows(words, bits * 2)
    for lane in range(32):
        for j in range(8):
            r, c = fragment_rc(lane, j)
            tile[r][c] = decode_mul1(codes[lane * 8 + j])
    return tile


H, I, E, TOPK = 4096, 2048, 256, 8
CODES = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16)  # half-bit API units
MAX_TENSOR_BYTES = 1536 * 1024**2


def metrics(torch, got, ref, label, rms_limit=0.002, peak_limit=0.025):
    if not torch.isfinite(got).all() or not torch.isfinite(ref).all():
        raise AssertionError(label + ': non-finite')
    diff = (got.float() - ref.float()).abs()
    rms = ref.float().square().mean().sqrt().item()
    floor = max(rms, 1e-6)
    result = dict(label=label, max_abs=diff.max().item(),
                  nrmse=diff.square().mean().sqrt().item() / floor,
                  peak_over_rms=diff.max().item() / floor,
                  bitwise=torch.equal(got, ref))
    print(json.dumps(result), flush=True)
    if result['nrmse'] > rms_limit or result['peak_over_rms'] > peak_limit:
        raise AssertionError('Numerical gate failed: ' + label)
    return result


def had(torch, x):
    y = x.float().reshape(-1, 128)
    for width in (1, 2, 4, 8, 16, 32, 64):
        z = y.reshape(-1, 128 // (2 * width), 2, width)
        a, b = z[:, :, 0, :], z[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
    return (y * 0.088388347648).reshape(x.shape)


def make_pool(torch):
    # 64 unique expert payloads, not a 256-expert layer/model. Unselected table
    # entries alias these valid payloads. Default spread routes hit all 64 uniquely.
    torch.manual_seed(137)
    projected_bytes = sum((H * I * CODES[(e + p * 3) % len(CODES)] // 16 +
                           (H + I) * 2) for e in range(64) for p in range(3))
    if projected_bytes + 256 * 1024**2 > MAX_TENSOR_BYTES:
        raise AssertionError('synthetic memory budget')
    pool = []
    for e in range(64):
        matrices = []
        for p, (k, n) in enumerate(((H, I), (H, I), (I, H))):
            code = CODES[(e + p * 3) % len(CODES)]
            b = torch.randint(-32768, 32768, (k // 16, n // 16, code * 8),
                              dtype=torch.int16, device='cuda')
            su = (torch.randint(0, 2, (k,), device='cuda') * 2 - 1).half()
            sv = ((torch.rand(n, device='cuda') * 0.5 + 0.75) *
                  (torch.randint(0, 2, (n,), device='cuda') * 2 - 1) * 0.015).half()
            matrices.append((code, b, su, sv))
        pool.append(matrices)
    return pool


def tables(torch, pool):
    codes, ptrs = [], []
    for p in range(3):
        codes.append(torch.tensor([pool[e % len(pool)][p][0] for e in range(E)],
                                  device='cuda', dtype=torch.int32))
        for item in (1, 2, 3):
            ptrs.append(torch.tensor([pool[e % len(pool)][p][item].data_ptr() for e in range(E)],
                                     device='cuda', dtype=torch.int64))
    return codes, ptrs


def routing(torch, rows, pattern):
    if pattern == 'hot':
        sel = torch.arange(8).repeat(rows, 1)
    elif pattern == 'spread':
        sel = (torch.arange(rows * 8).reshape(rows, 8) * 17) % E
    elif pattern == 'sentinel':
        sel = (torch.arange(rows * 8).reshape(rows, 8) * 17) % E
        sel[:, -1] = E
        sel[0, 0] = 255
    elif pattern == 'mixed_counts':
        sel = torch.arange(8).repeat(rows, 1)
        sel[:, 1] = 0
    elif pattern == 'duplicates':
        # Adversarial API test, NOT valid unique router top-k. 64 rows for one
        # expert exercises >8/>16 loops and tier/overflow exclusion.
        sel = torch.zeros((rows, 8), dtype=torch.int64)
    else:
        raise ValueError(pattern)
    rw = torch.arange(1, 9).float().repeat(rows, 1) / 36
    rw[0, 1] = 0  # zero routing weight must not corrupt slots or gather
    return sel.cuda(), rw.half().cuda()


class Case:
    def __init__(self, torch, x, sel, rw, codes, ptrs, cap=64, lo=1, hi=64, compact=False):
        global LAST_CASE
        LAST_CASE = self
        self.torch = torch
        self.x, self.sel, self.rw = x, sel, rw
        self.cap, self.lo, self.hi = cap, lo, hi
        assert x.shape[1] == H and sel.shape == rw.shape == (len(x), 8)
        assert x.dtype == rw.dtype == torch.float16 and sel.dtype == torch.int64
        assert bool(((sel >= 0) & (sel <= E)).all())
        assert torch.isfinite(x).all() and torch.isfinite(rw).all()
        self.flat = sel.flatten()
        order = self.flat.argsort(stable=True)
        self.inv = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device='cuda'))
        self.ts = torch.arange(len(x), device='cuda').repeat_interleave(8)[order]
        self.ws = rw.flatten()[order]
        self.counts = torch.bincount(self.flat, minlength=E + 1)
        self.starts = self.counts.cumsum(0) - self.counts
        self.kind = ((self.counts > 0) & (self.counts <= cap) &
                     (self.counts >= lo) & (self.counts <= hi)).long()[:E].contiguous()
        self.base = self.starts.clone()
        if compact:
            live_counts = self.counts[:E] * self.kind
            self.base[:E] = live_counts.cumsum(0) - live_counts
        self.scratch = torch.empty((sel.numel(), H), dtype=torch.float32, device='cuda')
        self.out = torch.empty_like(x, dtype=torch.float32)
        self.temps = [torch.empty((6, cap, n), dtype=torch.float16, device='cuda') for n in (H, H, I, I)]
        self.args = [x, self.out, self.counts, self.ts, self.ws, *self.temps, 0,
                     *codes, *ptrs, False, True, False, True, False, True,
                     0.0, int(self.kind.sum().item()), self.scratch, self.base, lo, hi, 16]
        # Capture values needed for reference before timed work; no timed readbacks.
        self.counts_cpu = self.counts.cpu().tolist()
        self.ts_cpu = self.ts.cpu().tolist()
        self.order_cpu = order.cpu().tolist()
        self.slot_mask = torch.repeat_interleave(self.kind.bool(), self.counts[:E])
        self.active_slots = int(self.counts[:E].sum().item())
        assert sum(self.counts_cpu) == sel.numel()
        assert sorted(self.order_cpu) == list(range(sel.numel()))

    def run(self, module, phased, poison=False):
        mode = ('three' if phased else 'off') if isinstance(phased, bool) else phased
        assert mode in ('off', 'five', 'three')
        os.environ['EXL3_MK_PHASED'] = '1' if mode == 'five' else '0'
        os.environ['EXL3_MK_THREE_STAGE'] = '1' if mode == 'three' else '0' 
        if poison:
            self.scratch.fill_(float('nan'))
            for buf in self.temps:
                buf.fill_(float('nan'))
        self.out.zero_()
        (module.run_three if mode == 'three' else module.run)(*self.args)
        module.gather(self.out, self.scratch, self.flat, self.inv, self.starts[:E],
                      self.base[:E], self.kind, self.ws)
        return self.out

    def check_slots(self):
        torch = self.torch
        used = torch.zeros(len(self.scratch), device='cuda', dtype=torch.bool)
        for e, count in enumerate(self.counts_cpu[:E]):
            if self.kind[e]:
                start = int(self.base[e]); used[start:start+count] = True
        assert torch.isfinite(self.scratch[used]).all(), 'eligible slot unwritten'
        assert torch.isnan(self.scratch[~used]).all(), 'excluded/unowned slot overwritten'
        # Independent fixed top-k CPU-selected ordering, same fp32 adds as kernel.
        expected = torch.zeros_like(self.out)
        inv = self.inv.cpu().tolist()
        flat = self.flat.cpu().tolist()
        kind = self.kind.cpu().tolist()
        for tok in range(len(self.x)):
            for k in range(8):
                i = tok * 8 + k
                if flat[i] < E and kind[flat[i]]:
                    expected[tok] += self.scratch[int(self.base[flat[i]]) + inv[i] - int(self.starts[flat[i]])]
        assert torch.equal(expected, self.out), 'gather top-k order/routing mismatch'


def dense_reference(torch, original, pool, case):
    scratch = torch.zeros_like(case.scratch)
    start = 0
    for e, rows in enumerate(case.counts_cpu[:E]):
        if rows and rows <= case.cap and case.lo <= rows <= case.hi:
            xx = case.x[case.ts[start:start + rows]]
            proj = []
            for p in range(3):
                code, packed, su, sv = pool[e % len(pool)][p]
                if p == 2:
                    u, g = proj[1], proj[0]
                    # Explicit half intermediates match the production GUAD helper.
                    act = (g * ((1 + (-g).exp()).half().reciprocal()).half()).half()
                    xx = (act * u).half()
                xh = had(torch, (xx * su).half()).half()
                n = I if p < 2 else H
                w = torch.empty((xx.shape[1], n), device='cuda', dtype=torch.float16)
                original.reconstruct(w, packed, code / 2, False, True)
                z = (xh.float() @ w.float()).half()
                del w
                if p < 2:
                    proj.append((had(torch, z).half() * sv).half())
                else:
                    # Production applies routing scale before svh in float.
                    weighted = had(torch, z) * case.ws[start:start + rows, None].float() * sv.float()
                    scratch[start:start + rows] = weighted
        start += rows
    output = torch.zeros_like(case.out)
    inv, flat = case.inv.cpu().tolist(), case.flat.cpu().tolist()
    kind = case.kind.cpu().tolist()
    for tok in range(len(case.x)):
        for k in range(8):
            s = tok * 8 + k
            if flat[s] < E and kind[flat[s]]:
                output[tok] += scratch[inv[s]]
    return output


def validate_format(torch, original):
    # Real reconstruct output versus scalar CPU format/layout reference (integer K).
    for bits in range(1, 9):
        packed = torch.randint(-32768, 32768, (1, 8, bits * 16), dtype=torch.int16)
        expected = torch.empty((16, 128), dtype=torch.float16)
        words = packed.view(torch.int32).to(torch.int64).bitwise_and(0xffffffff)
        for n in range(8):
            expected[:, n * 16:(n + 1) * 16] = torch.tensor(unpack_tile(words[0, n].tolist(), bits))
        got = torch.empty((16, 128), dtype=torch.float16, device='cuda')
        original.reconstruct(got, packed.cuda(), float(bits), False, True)
        assert torch.equal(got.cpu(), expected), f'CPU format/layout mismatch K={bits}'


def validate_all(torch, original, candidate, pool, codes, ptrs):
    validate_format(torch, original)
    results = []
    for rows in range(1, 9):
        for pattern in ('hot', 'spread'):
            sel, rw = routing(torch, rows, pattern)
            x = torch.randn((rows, H), device='cuda', dtype=torch.float16) * 0.1
            case = Case(torch, x, sel, rw, codes, ptrs)
            a = case.run(original, False, True).clone()
            case.check_slots()
            off = case.run(candidate, False, True).clone()
            if not torch.equal(a, off):
                raise AssertionError('candidate default-off changed original result')
            b = case.run(candidate, True, True).clone()
            case.check_slots()
            b2 = case.run(candidate, True, True).clone()
            assert torch.equal(b, b2), 'candidate nondeterministic'
            assert torch.equal(b, a), 'complete-K normal geometry must be bit-identical'
            results.append(metrics(torch, b, a, f'{rows}/{pattern} B vs A'))
            # Dense reference on both normal overlap extremes, no full-layer matrix load.
            if rows in (1, 8):
                ref = dense_reference(torch, original, pool, case)
                results.append(metrics(torch, b, ref, f'{rows}/{pattern} vs reconstruct', 0.005, 0.06))
    # Boundary / sentinel / overflow exclusion: poison verifies skip contract; no silent
    # truncation of eligible >8/>16 counts. The real caller still computes excluded experts.
    for pattern, cap, lo, hi in (('sentinel', 64, 1, 64), ('duplicates', 64, 1, 64),
                                 ('duplicates', 8, 1, 8), ('hot', 64, 9, 64),
                                 ('mixed_counts', 8, 1, 8), ('mixed_counts', 64, 9, 64)):
        sel, rw = routing(torch, 8, pattern)
        x = torch.randn((8, H), device='cuda', dtype=torch.float16) * 0.1
        case = Case(torch, x, sel, rw, codes, ptrs, cap, lo, hi)
        a = case.run(original, False, True).clone()
        b = case.run(candidate, True, True).clone()
        case.check_slots()
        results.append(metrics(torch, b, a, f'{pattern}/cap{cap}/lo{lo}'))
    # Preserve unsupported activation/codebook/row shapes through the original path.
    int_pool = [[next(m[p] for m in pool if m[p][0] == 4) for p in range(3)]]
    ic, ip = tables(torch, int_pool)
    for rows, act, mcg, limit in ((8, 1, False, 0.0), (8, 2, False, 0.0),
                                 (8, 0, True, 0.0), (9, 0, False, 0.0),
                                 (8, 0, False, 0.02)):
        sel, rw = routing(torch, rows, 'hot')
        case = Case(torch, torch.randn((rows, H), device='cuda', dtype=torch.float16) * 0.1,
                    sel, rw, ic, ip)
        case.args[9] = act
        case.args[22:28] = [mcg, not mcg] * 3
        case.args[28] = limit
        a = case.run(original, False, True).clone()
        b = case.run(candidate, True, True).clone()
        case.check_slots()
        if act != 0 or mcg or rows > 8:
            assert torch.equal(a, b), 'unsupported-case fallback changed numerics'
        else:
            results.append(metrics(torch, b, a, 'activation-limit'))
    torch.cuda.synchronize()
    return results


def load_capture(torch, path):
    # Small tensor-only capture: x, selected_experts, routing_weights, pool. Pool is
    # a list of per-expert [(Kcode, trellis, suh, svh)*3]; IDs index that list directly.
    # Capture schema must be validated before any GPU allocation. No raw pointer tables.
    if path.stat().st_size > MAX_TENSOR_BYTES:
        raise ValueError('capture exceeds bounded trial budget')
    d = torch.load(path, map_location='cpu', weights_only=True)
    pool = d['pool']
    if not 1 <= len(pool) <= 256:
        raise ValueError('capture expert count')
    nbytes = 0
    for matrices in pool:
        assert len(matrices) == 3
        for p, (code, b, su, sv) in enumerate(matrices):
            k, n = (I, H) if p == 2 else (H, I)
            assert code in CODES and b.shape == (k // 16, n // 16, code * 8)
            assert b.dtype == torch.int16 and su.dtype == sv.dtype == torch.float16
            assert su.shape == (k,) and sv.shape == (n,)
            nbytes += sum(t.numel() * t.element_size() for t in (b, su, sv))
    assert nbytes + 256 * 1024**2 <= MAX_TENSOR_BYTES
    x, sel, rw = d['x'], d['selected_experts'], d['routing_weights']
    assert 1 <= len(x) <= 8 and x.shape == (len(x), H)
    assert sel.shape == rw.shape == (len(x), 8)
    assert x.dtype == rw.dtype == torch.float16 and sel.dtype == torch.int64
    assert bool(((sel >= 0) & (sel < len(pool))).all())
    pool = [[(code, b.contiguous().cuda(), su.contiguous().cuda(), sv.contiguous().cuda())
             for code, b, su, sv in matrices] for matrices in pool]
    return pool, x.contiguous().cuda(), sel.contiguous().cuda(), rw.contiguous().cuda()

