"""A real fixed-width wire format: disabled slots still cost all their fields."""
import math

import torch


def width(n):
    return math.ceil(math.log2(n)) if n > 1 else 0


def fields(cfg, method):
    m = cfg["model"]
    if method == "b0":
        return [("gene_ids", width(m["gene_count"]), m["gene_count"])]
    result = [("gene_ids", width(m["gene_count"]), m["gene_count"]), ("enabled", 1, 2)]
    if method != "b1":
        result += [("position_ids", width(m["position_grid"]**2), m["position_grid"]**2),
                   ("scale_ids", width(len(m["scale_levels"])), len(m["scale_levels"]))]
    return result


def layout(cfg, method):
    m = cfg["model"]
    if method == "b0":
        max_tokens = m["budget_bits"] // width(m["gene_count"])
        assert max_tokens >= 1, "Budget cannot encode even one B0 token"
        rows = math.isqrt(max_tokens)
        cols = max_tokens // rows
        count = rows*cols
    else:
        count = m["max_instances"]
        rows = cols = None
    bits = count*sum(w for _, w, _ in fields(cfg, method))
    assert bits <= m["budget_bits"], f"{bits} bits exceed hard budget {m['budget_bits']}"
    assert method != "b2" or m["max_instances"] <= m["gene_count"]
    return dict(tokens=count, bits=bits, storage_bytes=(bits+7)//8,
                padding_bits=((bits+7)//8)*8-bits, rows=rows, cols=cols)


def pack_program(program, cfg, method, index=0):
    info = layout(cfg, method)
    bits = []
    for slot in range(info["tokens"]):
        for name, nbits, cardinality in fields(cfg, method):
            tensor = program[name]
            assert not tensor.is_floating_point(), "Only integer scene parameters may be serialized"
            val = int(tensor[index, slot])
            assert 0 <= val < cardinality
            bits.extend((val >> shift) & 1 for shift in reversed(range(nbits)))
    bits.extend([0]*info["padding_bits"])
    return bytes(sum(bits[i+j] << (7-j) for j in range(8)) for i in range(0, len(bits), 8))


def unpack_program(payload, cfg, method):
    info = layout(cfg, method)
    assert len(payload) == info["storage_bytes"]
    bits = [(byte >> shift) & 1 for byte in payload for shift in reversed(range(8))]
    out = {k: [] for k, _, _ in fields(cfg, method)}
    cursor = 0
    for _ in range(info["tokens"]):
        for name, nbits, cardinality in fields(cfg, method):
            val = 0
            for bit in bits[cursor:cursor+nbits]:
                val = (val << 1) | bit
            cursor += nbits
            assert val < cardinality, "Invalid unused categorical code"
            out[name].append(val)
    assert not any(bits[cursor:]), "Nonzero padding could carry unaccounted information"
    return {k: torch.tensor([v], dtype=torch.long) for k, v in out.items()}
