"""Independent format reference. Pure Python; does not simulate MMA or CUDA math."""

def fragment_rc(lane, j):
    if not 0 <= lane < 32 or not 0 <= j < 8:
        raise ValueError('fragment coordinate out of range')
    return ((lane % 4) * 2 + (j & 1) + ((j >> 1) & 1) * 8,
            (lane // 8) * 2 + ((lane >> 2) & 1) + (j >> 2) * 8)


def scalar_windows(words, k2):
    """Read the circular MSB-first stream, advancing by each symbol's width.

    K2 is the ABI half-bit code, not an integer bitrate. Odd positions carry
    the extra bit for 1.5/2.5/3.5. Return 16-bit codebook indices before decode.
    """
    if k2 not in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16):
        raise ValueError('unsupported half-bit code')
    if len(words) != 4 * k2 or any(not 0 <= w < 2**32 for w in words):
        raise ValueError('wrong packed tile')
    stream = ''.join(format(w, '032b') for w in words)
    cursor = 0
    result = []
    for position in range(256):
        cursor += k2 // 2 + (position % 2 if k2 % 2 else 0)
        window = ''.join(stream[i % len(stream)] for i in range(cursor - 16, cursor))
        result.append(int(window, 2))
    assert cursor == len(stream)
    return result


def decode_mul1(code):
    """Scalar mul1 decode: unsigned byte sum, half bitcast, fused half RNE.

    Half operands' product plus bias is exactly representable in Python's
    binary64 in this bounded range, so the final pack supplies the single RNE.
    """
    import struct
    if not 0 <= code <= 65535:
        raise ValueError('codebook index out of range')
    product = code * 0x83DCD12D % (1 << 32)
    total = 0x6400 + sum((product >> shift) & 255 for shift in (0, 8, 16, 24))
    def from_bits(bits):
        return struct.unpack('<e', struct.pack('<H', bits))[0]
    value = from_bits(total) * from_bits(0x1eee) + from_bits(0xc931)
    return struct.unpack('<e', struct.pack('<e', value))[0]
