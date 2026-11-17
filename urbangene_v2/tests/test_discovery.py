import pytest
import torch

from urbangene_v2.backbone import coordinates
from urbangene_v2.discovery import DiscoveryModel, FeatureReconstructionDecoder


@pytest.mark.parametrize("slots,background", [(1, False), (8, True), (32, True)])
def test_shapes_competition_and_gradients(slots, background):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    features = torch.randn(2, 16, 4, 5)
    model = DiscoveryModel(16, 24, slots, 3, 32, background_slot=background)
    output = model(features)
    assert output["slot_features"].shape == (2, slots, 24)
    assert output["soft_masks"].shape == (2, slots, 4, 5)
    assert output["exist_logits"].shape == (2, slots)
    assert output["estimated_geometry"].shape == (2, slots, 4)
    assert output["reconstructed_features"].shape == features.shape
    assert torch.allclose(output["soft_masks"].sum(1), torch.ones(2, 4, 5), atol=1e-6)
    assert torch.allclose(output["attention_masks"].sum(1), torch.ones(2, 4, 5), atol=1e-6)
    torch.nn.functional.mse_loss(output["reconstructed_features"], features).backward()
    modules = [model.encoder.input_net, model.encoder.gru, model.decoder]
    if slots > 1:
        modules.append(model.encoder.query)
    for module in modules:
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    if slots > 1 and background:
        assert model.encoder.presence.weight.grad.abs().sum() > 0


def test_empty_gates_are_finite():
    model = DiscoveryModel(16, 16, 4, 2, 16)
    with torch.no_grad():
        model.encoder.presence.bias.fill_(-1000)
    output = model(torch.randn(1, 16, 4, 4))
    assert torch.isfinite(output["reconstructed_features"]).all()
    assert torch.isfinite(output["soft_masks"]).all()
    assert output["soft_masks"][:, 0].min() > .99


def test_slot_permutation_equivariance():
    torch.manual_seed(7)
    model = DiscoveryModel(16, 16, 4, 3, 16, background_slot=False).eval()
    features = torch.randn(2, 16, 4, 5)
    initial = torch.randn(2, 4, 16)
    permutation = torch.tensor([3, 1, 0, 2])
    a = model(features, initial)
    b = model(features, initial[:, permutation])
    assert torch.allclose(a["soft_masks"][:, permutation], b["soft_masks"], atol=1e-6)
    assert torch.allclose(a["reconstructed_features"], b["reconstructed_features"], atol=1e-6)


def test_invalid_features():
    model = DiscoveryModel(16, 16, 4, 2, 16)
    with pytest.raises(ValueError):
        model(torch.full((1, 16, 4, 4), float("nan")))


@pytest.mark.parametrize("background", [False, True])
def test_shared_projection_optimization_preserves_outputs_and_gradients(background):
    """Check equivalence to the original materialized per-slot feature decoder."""
    torch.manual_seed(91)
    decoder = FeatureReconstructionDecoder(17, 8, 12, background).double()
    slots = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    geometry = torch.cat([torch.randn(2, 5, 2), torch.rand(2, 5, 2) + .1], -1).double()
    exist = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    height, width = 3, 4
    actual, actual_masks = decoder(slots, geometry, exist, height, width)
    xy = coordinates(height, width, dtype=torch.float64).reshape(1, 1, -1, 2).expand(2, 5, -1, -1)
    relative = ((xy - geometry[:, :, None, :2]) / geometry[:, :, None, 2:]).clamp(-10, 10)
    embedded = torch.cat([slots[:, :, None].expand(-1, -1, height * width, -1), xy, relative], -1)
    raw = decoder.network(embedded)
    logits = raw[..., -1] + torch.nn.functional.logsigmoid(exist)[..., None]
    if background:
        logits = torch.cat([raw[:, :1, :, -1], logits[:, 1:]], 1)
    masks = logits.softmax(1)
    expected = (raw[..., :-1] * masks[..., None]).sum(1).transpose(1, 2).reshape(2, 17, height, width)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.allclose(actual_masks, masks.reshape(2, 5, height, width), atol=1e-12, rtol=1e-12)
    parameters = [slots, exist, *decoder.parameters()]
    left = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
    right = torch.autograd.grad(expected.square().mean(), parameters)
    assert all(torch.allclose(a, b, atol=1e-12, rtol=1e-10) for a, b in zip(left, right))


def test_spatial_seeds_are_distinct_and_masks_remain_adaptive():
    torch.manual_seed(17)
    model = DiscoveryModel(16, 16, 8, 3, 16, spatial_initial_scale=.18)
    grid = coordinates(8, 8)
    initial = model.encoder.initial_geometry(grid, 2)
    assert initial.shape == (2, 8, 4)
    for row in initial:
        assert len(torch.unique(row[1:, :2], dim=0)) == 7
        assert row[1:, :2].std(0).min() > .3
    features = torch.randn(2, 16, 8, 8)
    output = model(features)
    assert torch.allclose(output['attention_masks'].sum(1), torch.ones(2, 8, 8), atol=1e-6)
    assert output['estimated_geometry'].requires_grad
    output['reconstructed_features'].square().mean().backward()
    assert model.encoder.query.weight.grad.abs().sum() > 0


def test_spatial_seed_configuration_rejects_invalid_scale_or_capacity():
    with pytest.raises(ValueError, match="must be positive"):
        DiscoveryModel(16, spatial_initial_scale=0)
    model = DiscoveryModel(16, 16, 8, 3, 16, spatial_initial_scale=.2)
    with pytest.raises(ValueError, match="exceed patch count"):
        model(torch.rand(1, 16, 2, 2))
