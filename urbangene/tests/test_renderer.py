import pytest
import torch

from urbangene.models.image_autoencoder import ImageAutoencoder
from urbangene.models.urbangene import UrbanGene


def program(cfg):
    n = cfg["model"]["max_instances"]
    return dict(gene_ids=torch.zeros(1, n, dtype=torch.long),
                enabled=torch.zeros(1, n, dtype=torch.long),
                position_ids=torch.full((1, n), 136, dtype=torch.long),
                scale_ids=torch.ones(1, n, dtype=torch.long))


def test_empty_repeat_overlay_and_order(cfg):
    m = UrbanGene(cfg, ImageAutoencoder(4, 8)).eval()
    p = program(cfg)
    with torch.no_grad():
        m.library.templates.fill_(1.)
    assert not m.renderer(p, m.library).any()
    p["enabled"][0, 0] = 1
    one = m.renderer(p, m.library)
    p["enabled"][0, 1] = 1
    twice = m.renderer(p, m.library)
    # Exact duplicate footprint remains bounded under weighted overlay.
    assert twice.max() <= 1.00001
    p["position_ids"][0, 1] = 68
    repeated = m.renderer(p, m.library)
    assert (repeated > 0).sum() > (one > 0).sum()
    reversed_slots = {k: v.flip(1) for k, v in p.items()}
    assert torch.allclose(m.renderer(reversed_slots, m.library), repeated)


@pytest.mark.parametrize("method", ["urbangene", "b0", "b1", "b2"])
def test_all_methods_have_parser_and_library_gradients(cfg, method):
    torch.manual_seed(21)
    m = UrbanGene(cfg, ImageAutoencoder(4, 8), method).train()
    z = torch.randn(3, 4, 32, 32)
    out = m.forward_latent(z)
    (out["latent_hat"]-z).abs().mean().backward()
    assert m.library.templates.grad.norm() > 0
    assert sum(p.grad.norm() for p in m.parser.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in m.image_ae.parameters())
    if method == "b2":
        for row in out["program"]["gene_ids"]:
            assert len(set(row.tolist())) == len(row)
    if method == "urbangene":
        h = m.parser.head.weight.grad.reshape(cfg["model"]["max_instances"], m.parser.size, -1)
        k = cfg["model"]["gene_count"]
        assert h[:, k+2:k+2+256].abs().sum() > 0, "Position choices must receive a learning signal"
        assert h[:, -3:].abs().sum() > 0, "Scale choices must receive a learning signal"
