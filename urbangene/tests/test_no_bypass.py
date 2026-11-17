import torch

from urbangene.budget import pack_program, unpack_program
from urbangene.models.image_autoencoder import ImageAutoencoder
from urbangene.models.urbangene import UrbanGene


def test_fixed_program_ignores_changed_input_and_ae_stays_frozen(cfg):
    model = UrbanGene(cfg, ImageAutoencoder(4, 8)).eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        first = model(x)
        p = first["program"]
        wire = pack_program(p, cfg, "urbangene")
        from_wire = model.reconstruct_program(unpack_program(wire, cfg, "urbangene"))
        # Processing a different image cannot modify output for a saved program.
        model(1-x)
        same = model.reconstruct_program(p)
    assert torch.equal(first["image_hat"], same["image_hat"])
    assert torch.equal(from_wire["image_hat"], same["image_hat"])
    assert set(p) == {"gene_ids", "enabled", "position_ids", "scale_ids"}
    assert all(not value.is_floating_point() for value in p.values())
    assert all(not param.requires_grad for param in model.image_ae.parameters())
