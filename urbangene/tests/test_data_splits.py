import itertools

import numpy as np

from urbangene.data.synthetic import FACTORS, HELDOUT, SEEN, SyntheticSceneGenerator, generate_manifests


def test_holdouts_and_seed_isolation(cfg):
    splits = generate_manifests(cfg, cfg["data"]["root"])
    seeds = [s["seed"] for specs in splits.values() for s in specs]
    assert len(seeds) == len(set(seeds))
    assert not set(HELDOUT) & set(SEEN)
    # Isolate third-order combinations, not unseen values or unseen factor pairs.
    for i, j in itertools.combinations(range(3), 2):
        assert {(x[i], x[j]) for x in SEEN} == set(itertools.product(range(3), repeat=2))
    for spec in splits["test_unseen_combo"]:
        assert tuple(spec[k] for k in FACTORS) in HELDOUT
    for spec in splits["test_higher_complexity"]:
        assert 7 <= spec["tree_count"]+3 <= cfg["model"]["max_instances"]
    gen = SyntheticSceneGenerator()
    a, _ = gen.sample(splits["train"][0]); b, _ = gen.sample(splits["train"][0])
    assert np.array_equal(a, b)
    # Independently regenerated specifications and renderings are identical.
    assert generate_manifests(cfg, cfg["data"]["root"]) == splits
