"""GPU checks for the increased input dimension and unchanged additive/rate contract."""
import os
import pytest
import torch
from model import Codec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU required')
M = int(os.environ.get('SAE_TEST_ATOMS','2048'))


def test_full_dimension_and_hard_codes():
    model = Codec(atoms=M,channels=1024).cuda().eval()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        f = torch.randn(1,1024,8,8,device='cuda')
        restored,q,_ = model(f)
    assert restored.shape == f.shape and q.shape == (1,M,8,8)
    assert model.dictionary.raw.shape == (M,1024,3,3)
    assert torch.equal(q,q.round()) and q.min() >= 0 and q.max() <= 63


def test_gradients_target_frozen_and_rate():
    model = Codec(atoms=M,channels=1024).cuda().train()
    f = torch.randn(1,1024,8,8,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out = model(f)
        loss,fit,r = model.objective(f.detach(),out,.015,image_hw=512)
    (loss+fit).backward()
    assert f.grad is None
    for p in [model.analysis.weight,model.dictionary.raw,model.entropy.location]:
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
    assert r['bpp'] == pytest.approx(r['bits_per_coefficient']*M*8*8/512**2,rel=2e-6)
    assert r['active_atoms_per_position'] == pytest.approx(r['nonzero_pct']*M/100)
    assert float(loss.detach()) == pytest.approx(r['nmse']+.015*r['bpp'],rel=2e-6)


def test_synthesis_additive_and_unit_norm():
    model = Codec(atoms=M,channels=1024).cuda().eval()
    a,b = [torch.rand(1,M,4,4,device='cuda')*.001 for _ in range(2)]
    with torch.no_grad(),torch.backends.cudnn.flags(allow_tf32=False):
        assert torch.allclose(model.dictionary(a+b),model.dictionary(a)+model.dictionary(b),atol=3e-6,rtol=5e-4)
        assert torch.allclose(model.dictionary.atoms().flatten(1).norm(dim=1),torch.ones(M,device='cuda'),atol=1e-6)
