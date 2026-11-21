"""Reference 32-bit arithmetic coder with explicit transmitted CDFs.

    Static per-channel tables, 16-bit total, nonnegative symbols 0..63.
    No entropy-library dependency. CPU symbolic coding is allowed.
"""
import bisect
import json
import math
import struct
import numpy as np

TOTAL = 65536
HALF, QUARTER, THREE_QUARTERS = 1 << 31, 1 << 30, 3 << 30
MASK = (1 << 32) - 1


def make_cdf(probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / p.sum(1, keepdims=True)
    # One count reserved for every symbol: all streams remain encodable.
    scaled = p * (TOTAL - p.shape[1])
    freq = np.floor(scaled).astype(np.int64) + 1
    for c in range(len(freq)):
        missing = TOTAL - int(freq[c].sum())
        indices = np.argsort(-(scaled[c] - np.floor(scaled[c])), kind='stable')[:missing]
        freq[c, indices] += 1
    return np.concatenate([np.zeros((len(p), 1), dtype=np.int64), freq.cumsum(1)], 1)


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.byte = self.used = self.bits = 0

    def put(self, bit):
        self.byte = (self.byte << 1) | bit
        self.used += 1
        self.bits += 1
        if self.used == 8:
            self.data.append(self.byte)
            self.byte = self.used = 0

    def finish(self):
        if self.used:
            self.data.append(self.byte << (8 - self.used))
        return bytes(self.data)


class BitReader:
    def __init__(self, data):
        self.data, self.index = data, 0

    def get(self):
        i = self.index
        self.index += 1
        if i >= len(self.data) * 8:
            return 0
        return (self.data[i // 8] >> (7 - i % 8)) & 1


def arithmetic_encode(symbols, cdfs):
    if symbols.ndim != 2 or cdfs.shape != (len(symbols), 65):
        raise ValueError('Expected [channels,positions] symbols and [channels,65] CDFs')
    low, high, pending = 0, MASK, 0
    writer = BitWriter()
    def emit(bit):
        nonlocal pending
        writer.put(bit)
        for _ in range(pending):
            writer.put(1 - bit)
        pending = 0
    for row, cdf in zip(symbols, cdfs):
        table = [int(v) for v in cdf]
        for symbol in row:
            symbol = int(symbol)
            if not 0 <= symbol < 64:
                raise ValueError('Symbol outside explicit alphabet')
            width = high - low + 1
            high = low + width * table[symbol + 1] // TOTAL - 1
            low = low + width * table[symbol] // TOTAL
            while True:
                if high < HALF:
                    emit(0)
                elif low >= HALF:
                    emit(1)
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < THREE_QUARTERS:
                    pending += 1
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low *= 2
                high = high * 2 + 1
    pending += 1
    emit(0 if low < QUARTER else 1)
    payload = writer.finish()
    return payload, writer.bits


def arithmetic_decode(payload, cdfs, positions):
    reader = BitReader(payload)
    value = 0
    for _ in range(32):
        value = value * 2 + reader.get()
    low, high = 0, MASK
    result = np.empty((len(cdfs), positions), dtype=np.uint8)
    for c, cdf in enumerate(cdfs):
        table = [int(v) for v in cdf]
        for j in range(positions):
            width = high - low + 1
            target = ((value - low + 1) * TOTAL - 1) // width
            symbol = bisect.bisect_right(table, target) - 1
            result[c, j] = symbol
            high = low + width * table[symbol + 1] // TOTAL - 1
            low = low + width * table[symbol] // TOTAL
            while True:
                if high < HALF:
                    pass
                elif low >= HALF:
                    value -= HALF
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high < THREE_QUARTERS:
                    value -= QUARTER
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low *= 2
                high = high * 2 + 1
                value = value * 2 + reader.get()
    return result


def estimated_integer_bits(symbols, cdf):
    freq = np.diff(cdf, axis=1)
    return float(sum((-np.log2(freq[c, row] / TOTAL)).sum() for c, row in enumerate(symbols)))


def save_tables(path, tables):
    # CDF endpoint is implicit TOTAL; interior cumulative values require uint32.
    with open(path, 'wb') as stream:
        np.savez(stream, **{s: c.astype(np.uint32) for s, c in tables.items()})


def write_image(path, codes, cdfs):
    streams, descriptions, metrics = [], [], {}
    for s, code in codes.items():
        code = np.asarray(code, dtype=np.uint8)
        flat = code.reshape(len(code), -1)
        payload, bits = arithmetic_encode(flat, cdfs[s])
        reconstructed = arithmetic_decode(payload, cdfs[s], flat.shape[1])
        if not np.array_equal(flat, reconstructed):
            raise AssertionError('Arithmetic round-trip mismatch')
        integer_estimate = estimated_integer_bits(flat, cdfs[s])
        if abs(len(payload) * 8 - integer_estimate) > 64:
            raise AssertionError('Unexpected arithmetic coding overhead')
        descriptions.append(dict(scale=s, shape=list(code.shape), bytes=len(payload), valid_bits=bits))
        streams.append(payload)
        metrics[s] = dict(integer_estimated_bits=integer_estimate, payload_bits=len(payload) * 8,
                          finalization_bits=bits, quantized_roundtrip_exact=True)
    header = json.dumps(dict(version=1, format='UVG-RD-AC32', scales=descriptions),
                        separators=(',', ':')).encode()
    data = b'UVGRD1' + struct.pack('<I', len(header)) + header + b''.join(streams)
    with open(path, 'wb') as stream:
        stream.write(data)
    return dict(file_bits=len(data) * 8, payload_bits=sum(len(p) * 8 for p in streams),
                header_bits=(10 + len(header)) * 8, scales=metrics)


def read_image(path, cdfs):
    data = open(path, 'rb').read()
    if data[:6] != b'UVGRD1':
        raise ValueError('Invalid stream magic')
    size = struct.unpack('<I', data[6:10])[0]
    header = json.loads(data[10:10 + size])
    offset = 10 + size
    result = {}
    for entry in header['scales']:
        shape = entry['shape']
        payload = data[offset:offset + entry['bytes']]
        offset += entry['bytes']
        result[entry['scale']] = arithmetic_decode(payload, cdfs[entry['scale']], math.prod(shape[1:])).reshape(shape)
    if offset != len(data):
        raise ValueError('Trailing bytes')
    return result
