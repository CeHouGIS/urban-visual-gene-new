import json
import os
from pathlib import Path
import pytest
pytestmark=pytest.mark.skipif(os.environ.get('RUN_GPU_TESTS')!='1',reason='Only leased CUDA GPUs')


def settings():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/experiment.json').read_text())


@pytest.mark.parametrize('kernel',[1,3])
def test_synthesis_additivity_adjoint_and_unit_atoms(kernel):
    import torch
    from model import GeneDictionary
    torch.manual_seed(41)
    dictionary=GeneDictionary(7,8,kernel).cuda()
    a=torch.rand(2,7,9,9,device='cuda');b=torch.randn(2,8,9,9,device='cuda')
    torch.testing.assert_close(dictionary(a),sum(dictionary.contribution(a,k) for k in range(7)),rtol=2e-5,atol=3e-6)
    torch.testing.assert_close((dictionary(a)*b).sum(),(a*dictionary.adjoint(b)).sum(),rtol=3e-5,atol=3e-5)
    torch.testing.assert_close(dictionary.atoms.flatten(1).norm(dim=1),torch.ones(7,device='cuda'))
    deleted=a.clone();deleted[:,3]=0
    torch.testing.assert_close(dictionary(deleted),dictionary(a)-dictionary.contribution(a,3),rtol=2e-5,atol=3e-6)


@pytest.mark.parametrize('kernel',[1,3])
def test_proximal_iterations_reduce_fixed_sparse_coding_objective(kernel):
    import torch
    from model import GeneDictionary,ISTACode
    torch.manual_seed(73)
    d=GeneDictionary(12,8,kernel).cuda()
    f=torch.randn(2,8,8,8,device='cuda')
    with torch.no_grad():
        energies=[]
        for steps in [1,2,3,6]:
            z,a=ISTACode(steps,.9).cuda()(f,d,.3)
            assert a.min()>=0
            energy=.5*(z-f).square().sum()+.3*a.abs().sum()
            energies.append(float(energy))
        assert energies[0]<float(.5*f.square().sum())
        assert all(y<=x+1e-5 for x,y in zip(energies,energies[1:]))


@pytest.mark.parametrize('variant',['single_scale_ista','multiscale_residual_ista'])
def test_all_gradient_paths_and_finite_normalized_features(variant):
    import torch
    from model import create_model
    from losses import objective
    c=settings();m=create_model(c,variant);m.set_step(1200)
    image=torch.rand(2,3,32,32,device='cuda')
    result=m(image)
    assert result['reconstruction'].shape==image.shape
    loss,_=objective(result,image,m,c)
    loss.backward()
    for name,p in m.named_parameters():
        assert p.grad is not None,name
        assert torch.isfinite(p.grad).all(),name
        assert p.grad.abs().sum()>0,name
    for target in result['feature_targets'].values():
        torch.testing.assert_close(target.square().mean(1),torch.ones_like(target[:,0]),rtol=2e-5,atol=2e-5)
    assert sum(m.gene_counts.values())==512


def test_fixed_pyramid_reconstructs_original_by_telescoping():
    import torch
    from torch.nn import functional as F
    from model import pyramid_targets
    image=torch.rand(2,3,64,64,device='cuda')
    r=pyramid_targets(image)
    medium=F.interpolate(r['coarse'],size=r['medium'].shape[-2:],mode='bilinear',align_corners=False)+r['medium']
    recovered=F.interpolate(medium,size=image.shape[-2:],mode='bilinear',align_corners=False)+r['fine']
    torch.testing.assert_close(recovered,image)
    assert r['fine'].min()<0 and r['fine'].max()>0


def test_no_encoder_bypass_and_matched_random_trunks():
    import torch
    from model import create_model
    c=settings();a=create_model(c,'single_scale_ista');b=create_model(c,'multiscale_residual_ista')
    for x,y in zip(a.encoder.parameters(),b.encoder.parameters()):
        torch.testing.assert_close(x,y)
    for x,y in zip(a.decoder.net.parameters(),b.decoder.net.parameters()):
        torch.testing.assert_close(x,y)
    b.set_step(1200)
    with torch.no_grad():
        for head in b.heads.values():
            head.weight.zero_()
        result=b(torch.rand(2,3,32,32,device='cuda'))
        for feature in result['features'].values():
            assert feature.count_nonzero()==0
        torch.testing.assert_close(result['reconstruction'][0],result['reconstruction'][1])


def test_curriculum_and_sparse_penalty_are_deterministic():
    from model import create_model
    m=create_model(settings(),'multiscale_residual_ista')
    assert m.gates=={'coarse':1.,'medium':0.,'fine':0.}
    assert m.penalty==pytest.approx(.15)
    m.set_step(400);assert m.gates['medium']==.5 and m.gates['fine']==0
    m.set_step(1200);assert all(x==1 for x in m.gates.values())
    assert m.penalty==pytest.approx(.5)
