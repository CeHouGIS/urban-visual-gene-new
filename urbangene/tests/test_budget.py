import copy

import pytest
import torch

from urbangene.budget import layout, pack_program, unpack_program
from urbangene.models.image_autoencoder import ImageAutoencoder
from urbangene.models.urbangene import UrbanGene


@pytest.mark.parametrize("method", ["urbangene", "b0", "b1", "b2"])
def test_wire_roundtrip_and_hard_cap(cfg, method):
    model = UrbanGene(cfg, ImageAutoencoder(4, 8), method).eval()
    program, _ = model.parser(torch.randn(2, 4, 32, 32))
    info = layout(cfg, method)
    assert info["bits"] <= cfg["model"]["budget_bits"]
    blob = pack_program(program, cfg, method)
    assert len(blob) == info["storage_bytes"]
    restored = unpack_program(blob, cfg, method)
    for name, val in restored.items():
        assert torch.equal(val, program[name][:1])
    bad = copy.deepcopy(cfg); bad["model"]["budget_bits"] = 1
    with pytest.raises((AssertionError, ZeroDivisionError)):
        layout(bad, method)


def test_noninteger_scene_parameters_rejected(cfg):
    model = UrbanGene(cfg, ImageAutoencoder(4, 8)).eval()
    program, _ = model.parser(torch.randn(1, 4, 32, 32))
    program["enabled"] = program["enabled"].float()
    with pytest.raises(AssertionError):
        pack_program(program, cfg, "urbangene")
