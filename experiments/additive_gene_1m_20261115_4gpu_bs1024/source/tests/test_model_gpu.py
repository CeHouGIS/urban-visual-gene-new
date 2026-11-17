"""Run only after the queue has leased an idle GPU. No CPU model fallback."""
import os
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('RUN_GPU_TESTS')!='1',reason='Requires explicitly leased GPU')


def test_sparse_additive_bottleneck_and_gradients():
    import torch
    from model import GeneAE, sum_contributions
    torch.set_num_threads(4)
    torch.manual_seed(7)
    model=GeneAE('sparse_n256_s4',dictionary_size=16,latent_channels=16,active_per_location=3).cuda()
    inputs=torch.rand(2,3,32,32,device='cuda')
    result=model(inputs)
    assert result['reconstruction'].shape==inputs.shape
    assert torch.all((result['activations']>0).sum(1)<=3)
    assert torch.all(result['activations']>=0)
    explicit=sum(model.contribution(result['activations'],k) for k in range(16))
    torch.testing.assert_close(explicit,result['latent'])
    torch.testing.assert_close(model.dictionary.norm(dim=1),torch.ones(16,device='cuda'))
    loss=(result['reconstruction']-inputs).square().mean()
    loss.backward()
    for parameter in [model.dictionary_raw,model.encoder.net[0].weight,model.head.weight,model.decoder.net[0].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum()>0
    activations=result['activations'].detach().clone()
    gene=int(activations.sum((0,2,3)).argmax())
    activations[:,gene]=0
    remaining=sum_contributions(activations,model.dictionary)
    torch.testing.assert_close(remaining,result['latent'].detach()-model.contribution(result['activations'].detach(),gene))


def test_fixed_topk_at_each_location():
    import torch
    from model import sparse_nonnegative
    scores=torch.tensor([1.,-2.,3.,4.,-1.],device='cuda')[None,:,None,None].expand(2,5,3,3)
    result=sparse_nonnegative(scores,2)
    torch.testing.assert_close(result[:,2],torch.full((2,3,3),3.,device='cuda'))
    torch.testing.assert_close(result[:,3],torch.full((2,3,3),4.,device='cuda'))
    assert (result[:,[0,1,4]]==0).all()


def test_variants_share_trunk_initialization_and_dense_path():
    import torch
    from model import GeneAE
    torch.manual_seed(42)
    dense=GeneAE('dense_ae').cuda()
    torch.manual_seed(42)
    sparse=GeneAE('sparse_n256_s4').cuda()
    for a,b in zip(dense.encoder.parameters(),sparse.encoder.parameters()):
        torch.testing.assert_close(a,b)
    for a,b in zip(dense.decoder.parameters(),sparse.decoder.parameters()):
        torch.testing.assert_close(a,b)
    image=torch.rand(2,3,32,32,device='cuda')
    assert dense(image)['reconstruction'].shape==image.shape


def test_ssim_identical_and_perturbed_images():
    import torch
    from metrics import ssim_per_image
    image=torch.rand(2,3,32,32,device='cuda')
    torch.testing.assert_close(ssim_per_image(image,image),torch.ones(2,device='cuda'))
    assert (ssim_per_image(image,1-image)<0.9).all()
