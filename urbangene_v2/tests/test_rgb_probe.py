from types import SimpleNamespace

import pytest
import torch
from torch import nn

from urbangene_v2.composition import SlotProgram
from urbangene_v2.discovery import DiscoveryModel
from urbangene_v2.rgb_probe import SlotRGBBridge, freeze_module, render_slot_program


def test_shared_projection_preserves_discovery_output_subspace():
    torch.manual_seed(5)
    torch.set_num_threads(2)
    weight, bias = torch.randn(20, 12), torch.randn(20)
    bridge = SlotRGBBridge(weight, bias, width=16)
    hidden = torch.randn(2, 12, 4, 6)
    feature = torch.einsum('cd,bdhw->bchw', weight, hidden) + bias[None, :, None, None]
    recovered = torch.einsum('cd,bdhw->bchw', bridge.basis, bridge.project(feature))
    torch.testing.assert_close(recovered + bias[None, :, None, None], feature)
    assert bridge(feature).shape == (2, 4, 8, 12)
    with pytest.raises(ValueError):
        bridge(torch.randn(2, 4, 4, 6))


def test_program_only_replay_and_frozen_decoder_gradient_flow(tmp_path):
    torch.manual_seed(12)
    torch.set_num_threads(2)
    model = freeze_module(DiscoveryModel(20, 8, 4, 2, 12))
    with torch.no_grad():
        encoding = model(torch.randn(1, 20, 4, 6))
    program = SlotProgram.from_encoding(encoding, 4, 6, 'test')
    projection = model.decoder.network[-1]
    bridge = SlotRGBBridge(projection.weight[:-1], projection.bias[:-1], width=16)

    class DummyVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Conv2d(4, 3, 1)

        def decode(self, latent):
            return SimpleNamespace(sample=self.decoder(latent))

    vae = freeze_module(DummyVAE())
    output = render_slot_program(program, model.decoder, bridge, vae)
    path = tmp_path / 'program.pt'
    program.save(path)
    replay = render_slot_program(SlotProgram.load(path), model.decoder, bridge, vae)
    torch.testing.assert_close(output, replay, rtol=0, atol=0)
    output.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bridge.parameters())
    assert all(p.grad is None for p in vae.parameters())
    assert all(p.grad is None for p in model.parameters())
