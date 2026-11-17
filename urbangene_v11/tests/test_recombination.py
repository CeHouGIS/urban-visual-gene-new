import inspect

import torch

from urbangene.data.synthetic import FACTORS, SEEN, HELDOUT
from urbangene.models.image_autoencoder import ImageAutoencoder
from urbangene_v11.model import UrbanGeneV11
from urbangene_v11.paired_composition import make_triplets
from urbangene_v11.recombination import merge_programs, predict_recombination, wire_roundtrip


def test_aligned_triplets_and_heldout_rules():
    for split in ["seen", "heldout"]:
        triples = make_triplets(27, split)
        for t in triples:
            a, b = t.sources(); c = t.target_for_scoring()
            assert tuple(t.source_a_spec[k] for k in FACTORS) in SEEN
            assert tuple(t.source_b_spec[k] for k in FACTORS) in SEEN
            target_combo = (t.source_a_spec["building_shape"], t.source_a_spec["facade"],
                            t.source_b_spec["vegetation_layout"])
            assert target_combo in (HELDOUT if split == "heldout" else SEEN)
            x0, y0, x1, y1 = t.region
            expected = a.clone(); expected[:, y0:y1, x0:x1] = b[:, y0:y1, x0:x1]
            assert torch.equal(c, expected), "The locked region must have an exactly aligned target"


def test_oracle_geometry_merge_not_slot_numbers(cfg, program):
    pa = {k: v.clone() for k, v in program.items()}; pb = {k: v.clone() for k, v in program.items()}
    for p in [pa, pb]:
        p["enabled"].zero_(); p["enabled"][0, :2] = 1
        p["position_ids"][0, :2] = torch.tensor([52, 212])
        p["scale_ids"].fill_(0)
    pa["gene_ids"][0, :2] = torch.tensor([1, 2]); pb["gene_ids"][0, :2] = torch.tensor([3, 4])
    merged = merge_programs(pa, pb, (0, 176, 256, 256), cfg)
    assert not merged.overflow and merged.requested_instances == 2
    assert set(merged.program["gene_ids"][0, :2].tolist()) == {1, 4}
    permutation = torch.randperm(16)
    other = merge_programs({k: v[:, permutation] for k, v in pa.items()}, pb, (0, 176, 256, 256), cfg)
    assert set(other.program["gene_ids"][0, :2].tolist()) == {1, 4}
    assert all(torch.equal(v, wire_roundtrip(merged.program, cfg)[k]) for k, v in merged.program.items())


def test_overflow_is_retained_and_not_truncated(cfg, program):
    a = {k: v.clone() for k, v in program.items()}; b = {k: v.clone() for k, v in program.items()}
    a["enabled"].fill_(1); b["enabled"].fill_(1)
    a["position_ids"].fill_(52); b["position_ids"].fill_(212)
    a["scale_ids"].zero_(); b["scale_ids"].zero_()
    result = merge_programs(a, b, (0, 176, 256, 256), cfg)
    assert result.overflow and result.requested_instances == 32 and result.program is None


def test_no_target_access_in_prediction(cfg):
    assert list(inspect.signature(predict_recombination).parameters) == ["model", "source_a", "source_b", "region", "threshold"]
    model = UrbanGeneV11(cfg, ImageAutoencoder(8, 8)).eval()
    triple = make_triplets(1, "heldout")[0]
    a, b = triple.sources()
    def forbidden():
        raise AssertionError("C* was accessed during inference")
    triple.target_for_scoring = forbidden
    calls = []
    handle = model.ae.encoder.register_forward_pre_hook(lambda mod, inputs: calls.append(inputs[0].shape[0]))
    result = predict_recombination(model, a[None], b[None], triple.region)
    handle.remove()
    assert calls == [2], "Only A and B may be encoded"
    assert "merge" in result
