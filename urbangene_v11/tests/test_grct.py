import copy
import inspect

import pytest
import torch

from urbangene.budget import layout, pack_program, unpack_program
from urbangene.models.gene_library import GeneLibrary
from urbangene.models.image_autoencoder import ImageAutoencoder
from urbangene.models.renderer import CompositionalRenderer
from urbangene_v11.config import PROJECT, load_config
from urbangene_v11.model import UrbanGeneV11
from urbangene_v11.models.composition import GRCTComposer, FixedAverage, FixedAlpha, ResidualCNN
from urbangene_v11.models.composition.geometry import ProgramGeometry, safe_masked_softmax
from urbangene_v11.models.composition.spatial_cross_attention import SpatialCrossLayer


def library():
    return GeneLibrary(64, 8, 8)


def activate_learned_branch(composer):
    with torch.no_grad():
        composer.spatial.output.weight.normal_(std=0.02)
        composer.spatial.output.bias.normal_(std=0.02)


@pytest.mark.parametrize("batch", [1, 2])
def test_shape_full_default_architecture(cfg, program, batch):
    full = load_config(PROJECT / "configs/v1_r3_grct.yaml")
    composer = GRCTComposer(full).eval()
    assert len(composer.relation.layers) == 6 and len(composer.spatial.layers) == 4
    assert composer.spatial.output.in_features == 256
    p = {k: v.expand(batch, -1) for k, v in program.items()}
    with torch.no_grad():
        z = composer(library(), p)
    assert z.shape == (batch, 8, 32, 32) and torch.isfinite(z).all()


def test_v0_initialization_and_wire(cfg, program):
    g = library(); composer = GRCTComposer(cfg).eval()
    v0 = CompositionalRenderer(cfg).eval()
    with torch.no_grad():
        reference = v0(program, g); actual = composer(g, program)
        assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
        blob = pack_program(program, cfg, "urbangene")
        assert len(blob) == 34 and layout(cfg, "urbangene")["bits"] == 272
        restored = unpack_program(blob, cfg, "urbangene")
        assert torch.equal(actual, composer(g, restored))


def test_permutation_invariance_and_relation_equivariance(cfg, program):
    g = library(); composer = GRCTComposer(cfg).eval(); activate_learned_branch(composer)
    permutation = torch.randperm(16)
    pp = {k: v[:, permutation] for k, v in program.items()}
    with torch.no_grad():
        a = composer(g, program); b = composer(g, pp)
        assert (a-b).abs().max() < 1e-5
        def tokens(p):
            ins = composer.geometry(g, p)
            return composer.relation(composer.embedding(ins), composer.geometry.relations(ins), ins.enabled_hard)
        assert torch.allclose(tokens(program)[:, permutation], tokens(pp), atol=2e-6, rtol=1e-5)


def test_geometry_bias_and_quantized_positions(cfg, program):
    g = library(); composer = GRCTComposer(cfg).eval(); activate_learned_branch(composer)
    changed = {k: v.clone() for k, v in program.items()}
    changed["position_ids"][0, 0] = 220; changed["scale_ids"][0, 0] = 2
    i = composer.geometry(g, program); j = composer.geometry(g, changed)
    bias = composer.relation.layers[0].relation_bias
    assert not torch.equal(bias(composer.geometry.relations(i)), bias(composer.geometry.relations(j)))
    assert not torch.equal(composer(g, program), composer(g, changed))
    floats = dict(program, position_ids=program["position_ids"].float())
    with pytest.raises(ValueError):
        composer(g, floats)


def test_all_empty_and_inactive_isolation(cfg, program):
    composer = GRCTComposer(cfg).eval(); activate_learned_branch(composer); g = library()
    blank = {k: v.clone() for k, v in program.items()}; blank["enabled"].zero_()
    mixed = {k: torch.cat([program[k], blank[k]]) for k in program}
    with torch.no_grad():
        output = composer(g, mixed)
        assert torch.isfinite(output).all() and not output[1].any()
        assert torch.allclose(output[:1], composer(g, program), atol=2e-6)
        changed = {k: v.clone() for k, v in program.items()}
        changed["gene_ids"][:, 8:] = 63; changed["position_ids"][:, 8:] = 0; changed["scale_ids"][:, 8:] = 2
        assert torch.allclose(output[:1], composer(g, changed), atol=2e-6)
        sample = composer.geometry.sample(composer.geometry(g, program))
        coverage = sample["coverage"].transpose(1, 2).reshape(1, 1, 32, 32)
        assert not output[:1].masked_select(~coverage).any()
    program["gene_ids"][:, 8:] = 63
    composer(g, program).square().mean().backward()
    assert not g.templates.grad[63].any(), "Inactive gene content cannot influence output"


def test_no_bypass_and_repeated_gene(cfg, program):
    model = UrbanGeneV11(cfg, ImageAutoencoder(8, 8)).eval()
    assert list(inspect.signature(model.composer.forward).parameters) == ["library", "program", "st"]
    with torch.no_grad():
        first = model.compose_program(program)["latent_hat"]
        model(torch.rand(1, 3, 256, 256))
        assert torch.equal(first, model.compose_program(program)["latent_hat"])
        altered = {k: v.clone() for k, v in program.items()}
        altered["gene_ids"][:, 1] = altered["gene_ids"][:, 0]
        altered["position_ids"][:, 1] = 220
        assert not torch.equal(first, model.compose_program(altered)["latent_hat"])
    assert all(not p.requires_grad for p in model.ae.parameters())


def test_parser_heads_and_interior_grads_after_zero_init(cfg):
    model = UrbanGeneV11(cfg, ImageAutoencoder(8, 8)).train()
    before = {k: v.clone() for k, v in model.ae.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    latent = torch.randn(2, 8, 32, 32)
    # First update opens the zero-initialized residual head. Check interiors next.
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_latent(latent)
        (output["latent_hat"]-latent).abs().mean().backward()
        grad = model.parser.head.weight.grad.reshape(16, model.parser.size, -1)
        for field, sl in {"gene": slice(0, 64), "active": slice(64, 66),
                          "position": slice(66, 322), "scale": slice(322, 325)}.items():
            assert torch.isfinite(grad[:, sl]).all() and grad[:, sl].abs().sum() > 0, field
        assert model.library.templates.grad.abs().sum() > 0
        if step == 1:
            for part in [model.composer.embedding, model.composer.relation, model.composer.spatial.layers]:
                assert sum(p.grad.abs().sum() for p in part.parameters() if p.grad is not None) > 0
        optimizer.step()
    assert all(p.grad is None for p in model.ae.parameters())
    assert all(torch.equal(before[k], v) for k, v in model.ae.state_dict().items())


def test_st_is_forward_identical_and_soft_sidechannel_rejected(cfg):
    model = UrbanGeneV11(cfg, ImageAutoencoder(8, 8)).train()
    p, st = model.parse_latent(torch.randn(2, 8, 32, 32))
    assert torch.allclose(model.composer(model.library, p, st), model.composer(model.library, p), atol=1e-6)
    bad = dict(st, position_weights=st["position_weights"]*0.8+0.2/256)
    with pytest.raises(ValueError, match="ST forward"):
        model.composer(model.library, p, bad)


def test_all_inactive_program_keeps_active_st_gradient(cfg, program):
    composer = GRCTComposer(cfg).train(); g = library()
    p = {k: v.clone() for k, v in program.items()}; p["enabled"].zero_()
    enabled_logit = torch.randn(p["enabled"].shape, requires_grad=True)
    enabled_st = enabled_logit-enabled_logit.detach()
    st = dict(gene_weights=torch.nn.functional.one_hot(p["gene_ids"], 64).float(),
              position_weights=torch.nn.functional.one_hot(p["position_ids"], 256).float(),
              scale_weights=torch.nn.functional.one_hot(p["scale_ids"], 3).float(),
              enabled=enabled_st)
    target = torch.ones(1, 8, 32, 32)
    output = composer(g, p, st)
    assert not output.any(), "Hard all-inactive forward remains exactly zero"
    (output-target).abs().mean().backward()
    assert enabled_logit.grad.abs().sum() > 0, "The parser must be able to recover from an empty program"


def test_parser_never_emits_empty_scene_and_fallback_is_slot_equivariant(cfg):
    model = UrbanGeneV11(cfg, ImageAutoencoder(8, 8)).eval()
    latent = torch.randn(2, 8, 32, 32)
    with torch.no_grad():
        program, st = model.parse_latent(latent)
    assert program["enabled"].bool().any(-1).all()
    assert torch.equal(st["enabled"], program["enabled"].float())
    empty = {"gene_ids": torch.arange(16).view(1, 16).expand(2, -1).clone(),
             "position_ids": torch.arange(16).view(1, 16).expand(2, -1).clone(),
             "scale_ids": torch.zeros(2, 16, dtype=torch.long),
             "enabled": torch.zeros(2, 16, dtype=torch.long)}
    surrogate = {"enabled": torch.zeros(2, 16, requires_grad=True)}
    expected, expected_st = model._ensure_nonempty(empty, surrogate)
    permutation = torch.randperm(16)
    permuted = {k: v[:, permutation] for k, v in empty.items()}
    moved_st = {"enabled": surrogate["enabled"][:, permutation].clone().detach().requires_grad_()}
    actual, _ = model._ensure_nonempty(permuted, moved_st)
    inverse = torch.argsort(permutation)
    assert torch.equal(actual["enabled"][:, inverse], expected["enabled"])
    assert torch.equal(expected_st["enabled"], expected["enabled"].float())


def test_pixel_instance_values_are_not_broadcast():
    torch.manual_seed(4)
    layer = SpatialCrossLayer(8, 2, 2, ratio=2).eval()
    q = torch.zeros(1, 2, 8); context = torch.zeros(1, 1, 8)
    s = dict(relative=torch.zeros(1, 2, 1, 4), valid=torch.ones(1, 2, 1, dtype=torch.bool),
             coverage=torch.ones(1, 2, 1, dtype=torch.bool), local_features=torch.zeros(1, 2, 1, 2))
    first = layer(q, context, s)
    s["local_features"][0, 1, 0, 0] = 1
    changed = layer(q, context, s)
    assert torch.equal(first[:, 0], changed[:, 0])
    assert not torch.equal(first[:, 1], changed[:, 1])


@pytest.mark.parametrize("composer_class", [FixedAverage, FixedAlpha, ResidualCNN])
def test_controls_shape_empty_and_gradients(cfg, program, composer_class):
    g = library(); composer = composer_class(cfg)
    out = composer(g, program)
    assert out.shape == (1, 8, 32, 32) and torch.isfinite(out).all()
    out.square().mean().backward(); assert g.templates.grad.abs().sum() > 0
    program["enabled"].zero_()
    assert not composer(g, program).any()


def test_local_arrangement_retained_and_hand_sample(cfg, program):
    geo = ProgramGeometry(cfg); g = library()
    with torch.no_grad():
        g.templates.zero_(); g.templates[0, :, :, 4:] = 1
    program["enabled"].zero_(); program["enabled"][0, 0] = 1
    program["gene_ids"].zero_(); program["scale_ids"].fill_(1)
    sample = geo.sample(geo(g, program))
    assert sample["local_features"].max() == 1 and sample["local_features"].min() == 0
    left = sample["relative"][0, :, 0, 0] < -0.5
    right = (sample["relative"][0, :, 0, 0] > 0.5) & sample["valid"][0, :, 0]
    assert sample["local_features"][0, left, 0].max() == 0
    assert sample["local_features"][0, right, 0].max() > 0


def test_safe_empty_attention():
    logits = torch.randn(2, 4, requires_grad=True)
    a = safe_masked_softmax(logits, torch.zeros_like(logits, dtype=torch.bool))
    assert not a.any() and torch.isfinite(a).all()
    a.sum().backward(); assert not logits.grad.any()


def test_parameter_matched_controls():
    from urbangene_v11.config import load_config

    root = PROJECT / "configs"
    models = [UrbanGeneV11(load_config(root/name), ImageAutoencoder(8, 24))
              for name in ["v1_r3_grct.yaml", "v1_r0_parameter_matched.yaml",
                           "v1_r2_parameter_matched.yaml"]]
    totals = [sum(model.parameter_counts()[key] for key in ("parser", "library", "composer"))
              for model in models]
    for value in totals[1:]:
        assert abs(value-totals[0])/totals[0] <= 0.05, totals
