"""Stdlib-only scalar/layout audit; no project imports, torch, CUDA or weights."""
from pathlib import Path
import importlib.util
import random
import re
import unittest

ROOT = Path(__file__).resolve().parent
from test_candidate import Q

def load_reference():
    path = ROOT / 'scalar_reference.py'
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location('scalar_reference', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def source_store_layout():
    """Execute only labeled reconstruct coordinate/store statements, not a CUDA emulator."""
    text = (Q / 'reconstruct.cu').read_text()
    body = text[text.index('if (!(lane_id & 4))'):text.index('// Store unpacked tile')]
    labels = {}
    for name, half, a, b in re.findall(r'half2 (m\d) = __halves2half2\(__(low|high)2half\(frag\[(\d)\]\[(\d)\]\)', body):
        labels[name] = 4 * int(a) + 2 * int(b) + (half == 'high')
    stores = re.findall(r'tile\[(r\d)\]\[warp_id\]\[(c\d)\] = (m\d);', body)
    assert len(labels) == len(stores) == 8
    result = {}
    for lane in range(32):
        if lane & 4:
            continue
        env = {'lane_id': lane}
        for name, expr in re.findall(r'int ([rc]\d) = ([^;]+);', body):
            env[name] = eval(expr.replace('/', '//'), {'__builtins__': {}}, env)
        for row, col, label in stores:
            j = labels[label]
            result[lane, j] = (env[row], 2 * env[col])
            result[lane + 4, j] = (env[row], 2 * env[col] + 1)
    return result


class LayoutTests(unittest.TestCase):
    def test_scalar_layout_matches_actual_labeled_stores(self):
        m = load_reference()
        self.assertIsNotNone(m, 'Independent scalar_reference.py is not implemented')
        mapping = source_store_layout()
        self.assertEqual(len(mapping), 256)
        for key, value in mapping.items():
            self.assertEqual(m.fragment_rc(*key), value, key)
        self.assertEqual(m.fragment_rc(0, 2), (8, 0))
        self.assertEqual(m.fragment_rc(0, 4), (0, 8))
        self.assertEqual(len(set(mapping.values())), 256)

    def test_bijective_quadrant_swap_is_detected(self):
        correct = source_store_layout()
        broken = {(l, j): ((l % 4) * 2 + (j & 1) + (j >> 2) * 8,
                           (l // 8) * 2 + ((l >> 2) & 1) + ((j >> 1) & 1) * 8)
                  for l in range(32) for j in range(8)}
        self.assertEqual(len(set(broken.values())), 256)
        self.assertEqual(sum(broken[key] != value for key, value in correct.items()), 128)


def source_windows(words, k2):
    """Execute scalar declarations from dq/dq8_half up to codebook calls.

    This slice excludes CUDA fragment intrinsics and the codebook itself.
    It is a second algorithm to the independent MSB-first bitstream reference.
    """
    text = (Q / 'exl3_dq.cuh').read_text()
    half = k2 % 2
    marker = 'void dq8_half(' if half else 'half dq('
    text = text[text.index(marker):]
    text = text[text.index('{') + 1:]
    text = text[:text.index('half2 d0d1' if half else 'return decode_3inst')]
    text = re.sub(r'//[^\n]*', '', text).replace('{', '').replace('}', '')
    text = re.sub(r'\b(?:constexpr|const|uint32_t|int)\s+', '', text)
    def fshift(b, a, shift):
        return (((a << 32) | b) >> shift) & 0xffffffff
    result = []
    for off in range(0, 256, 8 if half else 1):
        env = {'KA': k2 // 2, 'bits': k2 // 2, 'ptr': words, 't_offset': off,
               'fshift': fshift, '__funnelshift_r': fshift}
        for stmt in text.split(';'):
            # Split comma-separated declarations, never commas inside calls.
            parts, start, depth = [], 0, 0
            for i, ch in enumerate(stmt):
                depth += (ch in '([') - (ch in ')]')
                if ch == ',' and depth == 0:
                    parts.append(stmt[start:i]); start = i + 1
            parts.append(stmt[start:])
            for part in parts:
                if '=' not in part:
                    continue
                name, expr = part.split('=', 1)
                env[name.strip()] = eval(expr.strip().replace('/', '//'), {'__builtins__': {}}, env)
        if half:
            result.extend(env['w' + str(j)] & 0xffff for j in range(8))
        else:
            result.append(env['w0'])
    return result


class CodebookTests(unittest.TestCase):
    def test_mul1_scalar_decode_exact_all_indices(self):
        import struct
        from fractions import Fraction
        m = load_reference()
        self.assertTrue(callable(getattr(m, 'decode_mul1', None)), 'Mul1 scalar decode missing')
        cb = (Q / 'codebook.cuh').read_text()
        scalar = cb.split('half decode_3inst(uint32_t x)')[1].split('half2 decode_3inst_2')[0]
        self.assertIn('x *= 0x83DCD12Du;', scalar)
        self.assertIn('__dp4a(x, 0x01010101u, acc)', scalar)
        self.assertIn('__hfma(h.as_half, k_inv_h, k_bias_h)', scalar)
        inv = Fraction(struct.unpack('<e', struct.pack('<H', 0x1eee))[0])
        bias = Fraction(struct.unpack('<e', struct.pack('<H', 0xc931))[0])
        for code in range(65536):
            # Source's uint32 multiply + unsigned dp4a. No separate half multiply:
            # hfma has one rounding, and 0x6400 + byte sum represents 1024 + sum.
            byte_sum = sum(((code * 0x83DCD12D) & 0xffffffff).to_bytes(4, 'little'))
            exact = (1024 + byte_sum) * inv + bias
            expected = struct.unpack('<e', struct.pack('<e', float(exact)))[0]
            self.assertEqual(m.decode_mul1(code), expected, code)


class FormatTests(unittest.TestCase):
    def test_integer_and_half_bit_payload_windows_and_stores(self):
        m = load_reference()
        self.assertTrue(callable(getattr(m, 'scalar_windows', None)), 'Independent bitstream oracle missing')
        rng = random.Random(90371)
        mapping = source_store_layout()
        for k2 in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16):
            for fixture in range(4):
                words = [rng.getrandbits(32) for _ in range(4 * k2)]
                want = source_windows(words, k2)
                got = m.scalar_windows(words, k2)
                self.assertEqual(got, want, (k2, fixture))
                # Actual labeled stores, not merely a coordinate bijection.
                expected_tile = {mapping[l, j]: want[l * 8 + j] for l in range(32) for j in range(8)}
                actual_tile = {m.fragment_rc(l, j): got[l * 8 + j] for l in range(32) for j in range(8)}
                self.assertEqual(actual_tile, expected_tile)
                if callable(getattr(m, 'decode_mul1', None)):
                    self.assertEqual({rc: m.decode_mul1(v) for rc, v in actual_tile.items()},
                                     {rc: m.decode_mul1(v) for rc, v in expected_tile.items()})
        with self.assertRaises(ValueError):
            m.scalar_windows([0], 4)
        with self.assertRaises(ValueError):
            m.scalar_windows([0] * 36, 9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
