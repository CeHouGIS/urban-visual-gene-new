"""Aligned, localized evaluation triples; target access is scoring-only.

Foreground vegetation is restricted to y>=176 to isolate it from buildings.
This is a declared paired-evaluation variant of the V0 generator, not an
assertion that arbitrary overlapping objects are segmentable by programs.
"""
from dataclasses import dataclass

import numpy as np
import torch
from PIL import ImageDraw

from urbangene.data.synthetic import SyntheticSceneGenerator, FACTORS, SEEN, HELDOUT

REGION = (0, 176, 256, 256)


def render_aligned(spec):
    image, _ = SyntheticSceneGenerator().sample(dict(spec, tree_count=0))
    rng = np.random.default_rng(spec["seed"]+12345)
    d = ImageDraw.Draw(image); n = spec["tree_count"]; layout = spec["vegetation_layout"]
    for j in range(n):
        if layout == 0:
            x = int(256*(j+1)/(n+1))
        elif layout == 1:
            x = 18+25*j
        else:
            x = 22+22*(j//2) if j%2 == 0 else 234-22*(j//2)
        y = 205+int(rng.integers(-3, 4)); r = int(rng.integers(9, 13))
        d.rectangle((x-2, y, x+2, y+24), fill=(90, 60, 35))
        color = tuple(int(v) for v in rng.integers([25, 90, 25], [80, 160, 80]))
        d.ellipse((x-r, y-r*2, x+r, y+3), fill=color)
    return torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float()/255


@dataclass
class AlignedTriplet:
    source_a_spec: dict
    source_b_spec: dict
    split: str
    region: tuple = REGION

    def sources(self):
        return render_aligned(self.source_a_spec), render_aligned(self.source_b_spec)

    def target_for_scoring(self):
        # This method must be invoked only after target-free prediction is complete.
        target_spec = dict(self.source_a_spec,
                           vegetation_layout=self.source_b_spec["vegetation_layout"])
        return render_aligned(target_spec)


def make_triplets(count, split="heldout", seed=42):
    assert split in ("seen", "heldout")
    targets = HELDOUT if split == "heldout" else SEEN
    triples = []
    namespace = 6 if split == "seen" else 7
    for i in range(count):
        a, b, v = targets[i % len(targets)]
        va = next(x[2] for x in SEEN if x[0] == a and x[1] == b and x[2] != v)
        bb = next(x[1] for x in SEEN if x[0] == a and x[2] == v and x[1] != b)
        common = dict(seed=seed*10**9+namespace*10**7+i, split=f"paired_{split}",
                      road_width=i%3, building_relation=(i//3)%3, tree_count=1+i%3)
        sa = dict(common, **dict(zip(FACTORS, (a, b, va))))
        sb = dict(common, **dict(zip(FACTORS, (a, bb, v))))
        triples.append(AlignedTriplet(sa, sb, split))
    return triples
