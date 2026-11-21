"""Fresh RGB probe with full 1024-channel input; frozen upstream synthesis."""
from pathlib import Path
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from blocks import Residual
from model import Codec,tensor_hash


class RGBDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Conv2d(1024,128,3,padding=1),Residual(128),Residual(128)]
        for source,target in [(128,96),(96,64),(64,32)]:
            layers.extend([nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                           nn.Conv2d(source,target,3,padding=1),Residual(target)])
        layers.extend([nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,3,3,padding=1),nn.Sigmoid()])
        self.net=nn.Sequential(*layers)

    def forward(self,features):
        return self.net(features)


def load_codec(c,atoms,final=True):
    base=c['base']
    name='checkpoint_final.pt' if final else 'checkpoint_latest.pt'
    path=Path(base['root'])/'runs'/f'atoms_{atoms:04d}'/name
    saved=torch.load(path,map_location='cpu',weights_only=False)
    expected=hashlib.sha256(json.dumps(dict(config=base,atoms=atoms),sort_keys=True).encode()).hexdigest()
    if saved['fingerprint']!=expected or saved['atoms']!=atoms:
        raise ValueError('Incompatible SAE checkpoint')
    if final and saved['step']!=base['compression']['steps']:
        raise ValueError('Require final SAE, never silently use the step-50 preview')
    ready=json.loads((Path(base['root'])/'features/READY.json').read_text())
    if saved['encoder_hash']!=ready['signature']:
        raise ValueError('SAE and DINO features have different provenance')
    model=Codec(step=base['compression']['quantization_step'],atoms=atoms,channels=1024).cuda()
    model.load_state_dict(saved['model'],strict=True)
    model.eval().requires_grad_(False)
    return model,dict(path=str(path),step=saved['step'],model_hash=tensor_hash(model),
                     encoder_signature=saved['encoder_hash'],checkpoint_bytes=path.stat().st_size)


@torch.no_grad()
def source_features(features,source,codecs):
    if source=='dino_reference':return features.detach()
    return codecs[source](features)[0].detach()


def embedding(x):
    # A readout from ONE learned feature grid, not additional learned scales.
    return F.normalize(torch.cat([F.adaptive_avg_pool2d(x.float(),n).flatten(1) for n in (1,2,4)],1),dim=1)
