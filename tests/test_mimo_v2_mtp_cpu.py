"""CPU tests for the MiMo-V2 multi-layer MTP bookkeeping (no GPU, no weights).

A fake forward_layer records, per MTP layer k and position p, the (token, paired-hidden) pair the
layer's draft-cache entry was written from. The simulation then drives prefill chunks and draft /
verify rounds with random acceptance, and checks the DeepSeek-V3 MTP invariant the real kernels
rely on: whenever layer k runs at the newest position, every cache entry it can attend to holds
(t[p + k], h_trunk[p - 1]) for the CURRENT token sequence (committed tokens plus this round's
drafts), and the draft it emits is conditioned on (h_trunk[n - 2], t[n - 1 + k]).
"""
import random
import unittest
import torch

from exllamav3.architecture.mimo_v2_mtp import MiMoV2MTPModel


class FakeIds:
    def __init__(self, ids):
        self.ids = ids

    def torch_slice(self, a, b):
        return torch.tensor([self.ids[a:b]], dtype = torch.long)

    def __len__(self):
        return len(self.ids)


class FakeSeq:
    def __init__(self, ids):
        self.sequence_ids = FakeIds(ids)
        self.block_index_tensor = None


class FakeJob:
    pass


def hid(p):
    # paired hidden for position p is the trunk state at p - 1; encode it as a scalar
    return float(p - 1)


def make_head(num_layers):
    m = MiMoV2MTPModel.__new__(MiMoV2MTPModel)
    m.mtp_layers = list(range(num_layers))
    m.caches = [dict() for _ in range(num_layers)]
    m.log = []

    def forward_layer(k, ids, params):
        p0 = int(params["cache_seqlens"][0])
        th = params["target_hidden"]
        T = ids.shape[-1]
        assert th.shape[1] == T
        for i in range(T):
            m.caches[k][p0 + i] = (int(ids[0, i]), float(th[0, i, 0]))
        m.log.append((k, p0, T))
        # state carries (k, last position) so the fake sampler can emit a checkable id
        return torch.tensor([[[float(k), float(p0 + T - 1)]]]).expand(1, T, 2).clone()

    def sample_from_state(state, params):
        return torch.tensor([[1000 + int(state[0, -1, 1]) * 10 + int(state[0, -1, 0])]])

    m.forward_layer = forward_layer
    m.sample_from_state = sample_from_state
    return m


def paired(p0, p1):
    return torch.tensor([[[hid(p)] for p in range(p0, p1 + 1)]], dtype = torch.half)


class MultiPositionsTests(unittest.TestCase):
    def test_draft_range_ends_at_newest(self):
        self.assertEqual(MiMoV2MTPModel.multi_positions(5, 10, 0), (5, 9))
        self.assertEqual(MiMoV2MTPModel.multi_positions(7, 10, 2), (7, 9))

    def test_prefill_range_only_final_entries(self):
        # layer 0 leaves n-1 for the first draft round; layer k stops at n-1-k
        self.assertEqual(MiMoV2MTPModel.multi_positions(0, 10, 0, upto = 9), (0, 8))
        self.assertEqual(MiMoV2MTPModel.multi_positions(0, 10, 1, upto = 9), (0, 8))
        self.assertEqual(MiMoV2MTPModel.multi_positions(0, 10, 2, upto = 9), (0, 7))
        # a chunk that ends early is bounded by its last paired position
        self.assertEqual(MiMoV2MTPModel.multi_positions(0, 10, 2, upto = 4), (0, 4))

    def test_token_ids_use_drafts_past_committed(self):
        committed = torch.tensor([[10, 11, 12, 13]])
        ids = MiMoV2MTPModel.multi_token_ids(committed, [20, 21], 2, 3, 2)
        self.assertEqual(ids.tolist(), [[20, 21]])
        ids = MiMoV2MTPModel.multi_token_ids(committed, [20], 1, 3, 1)
        self.assertEqual(ids.tolist(), [[12, 13, 20]])


class SimulationTests(unittest.TestCase):

    def check_entries(self, m, k, upto, tokens):
        # every entry at positions <= upto of layer k matches the current sequence
        for p in range(0, upto + 1):
            self.assertIn(p, m.caches[k], f"layer {k} missing position {p}")
            tok, h = m.caches[k][p]
            self.assertEqual(tok, tokens[p + k], f"layer {k} pos {p}: stale token")
            self.assertEqual(h, hid(p), f"layer {k} pos {p}: wrong hidden")

    def run_sim(self, seed, num_layers = 3, window = 3, prompt_len = 37, chunk = 11, rounds = 40):
        rng = random.Random(seed)
        m = make_head(num_layers)
        job = FakeJob()
        tokens = [rng.randrange(100) for _ in range(prompt_len)]
        seq = FakeSeq(tokens)

        # chunked prefill of positions 0 .. prompt_len - 2 (the last token stays unprocessed)
        start = 0
        end_all = prompt_len - 1
        while start < end_all:
            end = min(start + chunk, end_all)
            m.multi_prefill(job, seq, start, paired(start, end), None, window)
            start = end
        n = len(tokens)
        st = job.mtp_multi
        self.assertEqual(st["h0"] + st["h"].shape[1], n)

        for _ in range(rounds):
            n = len(tokens)
            m.log.clear()
            drafts, _ = m.multi_draft(job, seq, None, window)
            self.assertEqual(len(drafts), min(window, num_layers))
            full = tokens + drafts
            for k in range(len(drafts)):
                # the layer attended positions 0 .. n-1 with the current tokens incl. drafts
                self.check_entries(m, k, n - 1, full)
                self.assertEqual(drafts[k], 1000 + (n - 1) * 10 + k)
            # verify: accept a random prefix of the drafts, then a bonus token
            acc = rng.randint(0, len(drafts))
            new = drafts[:acc] + [rng.randrange(100)]
            kv_before = n - 1
            tokens.extend(new)
            m.multi_push_hidden(job, kv_before + 1, paired(kv_before + 1, kv_before + len(new))[:, :, :])
            st = job.mtp_multi
            self.assertEqual(st["h0"] + st["h"].shape[1], len(tokens))
            # history is trimmed to the oldest non-final entry
            self.assertEqual(st["h0"], min(st["front"][:window]) if st["h0"] > 0 else st["h0"])
        return m

    def test_random_rounds(self):
        for seed in range(20):
            self.run_sim(seed)

    def test_short_window_does_not_pin_history(self):
        m = self.run_sim(3, window = 1, rounds = 60)
        self.assertEqual(len(m.caches[1]) + len(m.caches[2]), 0)

    def test_single_chunk_prompt(self):
        self.run_sim(7, prompt_len = 2, chunk = 4096)

    def test_recompute_is_bounded(self):
        # per round, layer k recomputes at most (accepted + 1 + k) positions
        rng = random.Random(1)
        m = make_head(3)
        job = FakeJob()
        tokens = [rng.randrange(100) for _ in range(300)]
        seq = FakeSeq(tokens)
        m.multi_prefill(job, seq, 0, paired(0, 299), None, 3)
        for _ in range(30):
            n = len(tokens)
            m.log.clear()
            drafts, _ = m.multi_draft(job, seq, None, 3)
            for k, p0, T in m.log:
                self.assertLessEqual(T, 4 + k)
            new = drafts[:rng.randint(0, 3)] + [7]
            tokens.extend(new)
            m.multi_push_hidden(job, n, paired(n, n + len(new) - 1))


if __name__ == "__main__":
    unittest.main()
