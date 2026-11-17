"""Run only after the supervisor owns an idle GPU pair."""
import os
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('RUN_GPU_TESTS') != '1', reason='Requires leased CUDA GPUs')


@pytest.mark.parametrize('variant', ['single_vector', 'multiscale_vector', 'multiscale_conv', 'dense_ae'])
def test_real_architecture_gradient_paths_and_budget(variant):
    import torch
    from model import MultiscaleGeneAE
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = MultiscaleGeneAE(variant).cuda()
    image = torch.rand(2, 3, 32, 32, device='cuda')
    result = model(image)
    assert result['reconstruction'].shape == image.shape
    (result['reconstruction'] - image).square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    if model.sparse:
        for scale, act in result['activations'].items():
            assert act.min() >= 0
            assert int((act > 0).flatten(1).sum(1).max()) <= max(1, round(model.budgets[scale] / 64))
        assert sum(model.gene_counts.values()) == 256


@pytest.mark.parametrize('kernel', [1, 3])
def test_dictionary_addition_gene_deletion_and_unit_norm(kernel):
    import torch
    from model import GeneDictionary
    dictionary = GeneDictionary(7, 8, kernel).cuda()
    activation = torch.rand(2, 7, 9, 9, device='cuda')
    complete = dictionary(activation)
    explicit = sum(dictionary.contribution(activation, k) for k in range(7))
    torch.testing.assert_close(complete, explicit, atol=3e-6, rtol=1e-5)
    deleted = activation.clone()
    deleted[:, 3] = 0
    torch.testing.assert_close(dictionary(deleted), complete - dictionary.contribution(activation, 3), atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(dictionary.atoms.flatten(1).norm(dim=1), torch.ones(7, device='cuda'))
    torch.testing.assert_close(dictionary(activation + deleted), dictionary(activation) + dictionary(deleted), atol=3e-6, rtol=1e-5)


def test_spatial_atom_has_nonzero_neighbors():
    import torch
    from model import GeneDictionary
    dictionary = GeneDictionary(1, 2, 3).cuda()
    impulse = torch.zeros(1, 1, 9, 9, device='cuda')
    impulse[:, :, 4, 4] = 1
    value = dictionary(impulse)
    assert (value != 0).sum().item() == 18
    assert value[:, :, :3].count_nonzero() == 0


def test_common_trunks_and_no_encoder_bypass():
    import torch
    from model import MultiscaleGeneAE
    torch.manual_seed(42)
    single = MultiscaleGeneAE('single_vector').cuda()
    torch.manual_seed(42)
    multi = MultiscaleGeneAE('multiscale_vector').cuda()
    for a, b in zip(single.encoder.parameters(), multi.encoder.parameters()):
        torch.testing.assert_close(a, b)
    for a, b in zip(single.decoder.net.parameters(), multi.decoder.net.parameters()):
        torch.testing.assert_close(a, b)
    with torch.no_grad():
        for head in multi.heads.values():
            head.weight.zero_()
            head.bias.zero_()
        output = multi(torch.rand(2, 3, 32, 32, device='cuda'))
        for feature in output['features'].values():
            assert feature.count_nonzero() == 0
        torch.testing.assert_close(output['reconstruction'][0], output['reconstruction'][1])


def test_topk_is_per_image_and_clips_negative_coefficients():
    import torch
    from model import positive_topk_per_image
    scores = torch.tensor([[-1., 4., 2., 3.], [9., -2., 1., -3.]], device='cuda').view(2, 2, 1, 2)
    expected = torch.tensor([[0., 4., 0., 3.], [9., 0., 1., 0.]], device='cuda').view_as(scores)
    torch.testing.assert_close(positive_topk_per_image(scores, 2), expected)
