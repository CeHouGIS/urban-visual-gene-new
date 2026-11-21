import pytest
import torch
from downstream_model import RGBDecoder,source_features,embedding
from model import Codec,tensor_hash

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU required; no CPU neural fallback')


def test_decoder_updates_without_upstream_gradients():
    torch.manual_seed(17)
    codec=Codec(atoms=64,channels=1024).cuda().eval().requires_grad_(False)
    decoder=RGBDecoder().cuda().train()
    upstream=tensor_hash(codec)
    original=tensor_hash(decoder)
    optimizer=torch.optim.Adam(decoder.parameters(),lr=.001)
    f=torch.randn(2,1024,8,8,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        restored=source_features(f,'compressed',{'compressed':codec})
        rgb=decoder(restored)
        assert rgb.shape==(2,3,64,64) and rgb.min()>=0 and rgb.max()<=1
        target=torch.rand_like(rgb)
        loss=(rgb.float()-target.float()).abs().mean()
    loss.backward();optimizer.step()
    assert f.grad is None and all(p.grad is None for p in codec.parameters())
    assert tensor_hash(codec)==upstream and tensor_hash(decoder)!=original


def test_raw_reference_is_detached_and_retrieval_keeps_spatial_information():
    x=torch.randn(2,1024,8,8,device='cuda',requires_grad=True)
    assert not source_features(x,'dino_reference',{}).requires_grad
    with torch.no_grad():
        vectors=embedding(x)
        assert vectors.shape==(2,1024*21)
        assert torch.allclose(vectors.norm(dim=1),torch.ones(2,device='cuda'),atol=1e-6)
        changed=embedding(x.flip(-1))
        assert not torch.allclose(vectors,changed)
