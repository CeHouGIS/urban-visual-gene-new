"""Spatial image attention and sparse shared dictionary for SDXL.

This is a custom adapter, not an official IP-Adapter/SlotAdapt reproduction.
Only the added branches are trainable; no original-image bypass is provided.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

def positions(side, dim, device, dtype):
    assert dim % 4 == 0
    p = torch.linspace(-1, 1, side, device=device, dtype=torch.float32)
    y,x = torch.meshgrid(p,p,indexing='ij')
    freq = torch.exp(torch.linspace(0, math.log(100), dim//4,device=device))
    return torch.cat([(x.flatten()[:,None]*freq).sin(),(x.flatten()[:,None]*freq).cos(),
                      (y.flatten()[:,None]*freq).sin(),(y.flatten()[:,None]*freq).cos()],dim=-1).to(dtype)

class DenseConditioner(nn.Module):
    def __init__(self, mode='DV', dim=256, resolution=1024):
        super().__init__(); self.mode=mode;self.patch=resolution//8//16
        self.dino = nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,dim),nn.GELU(),nn.Linear(dim,dim))
        # Each spatial token retains all local latent cells (8x8 at1024).
        input_dim=4*self.patch*self.patch
        self.appearance = nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.fuse = nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Linear(dim,dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self,dino,latent):
        if self.mode in ('D','DV'): d=self.dino(dino)
        if self.mode in ('V','DV'):
            v=F.pixel_unshuffle(latent,self.patch).flatten(2).transpose(1,2)
            v=self.appearance(v)
        out=d if self.mode=='D' else v if self.mode=='V' else self.fuse(torch.cat([d,v],-1))
        return self.norm(out)

class GeneConditioner(nn.Module):
    def __init__(self, topk=8, learned=True, dim=256, size=512, seed=17):
        super().__init__();self.topk=topk;self.learned=learned
        generator=torch.Generator().manual_seed(seed)
        self.bank=nn.Parameter(F.normalize(torch.randn(size,dim,generator=generator),dim=1),requires_grad=learned)
        self.coefficients=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,512),nn.GELU(),nn.Linear(512,size))
    def forward(self, dense, return_code=False):
        scores=self.coefficients(dense)
        ids=scores.abs().topk(self.topk,dim=-1).indices
        values=scores.gather(-1,ids)
        vectors=F.embedding(ids,F.normalize(self.bank,dim=1))
        out=(vectors*values[...,None]).sum(-2)
        return (out,ids,values) if return_code else out
    @torch.no_grad()
    def normalize_bank(self):self.bank.copy_(F.normalize(self.bank,dim=1))

class SpatialAttention(nn.Module):
    def __init__(self, hidden_size=None, dim=256, cross=True, initial_gain=.01, locality_bias=0.):
        super().__init__();self.cross=cross;self.base=AttnProcessor2_0();self.locality_bias=locality_bias
        if cross:
            self.q=nn.Linear(hidden_size,dim,bias=False)
            self.k=nn.Linear(dim,dim,bias=False);self.v=nn.Linear(dim,dim,bias=False)
            self.out=nn.Linear(dim,hidden_size,bias=False)
            self.gain=nn.Parameter(torch.tensor(float(initial_gain)))
            self.norm=nn.LayerNorm(hidden_size)
    def __call__(self,attn,hidden_states,encoder_hidden_states=None,attention_mask=None,temb=None,image_tokens=None,*args,**kwargs):
        base=self.base(attn,hidden_states,encoder_hidden_states,attention_mask,temb)
        if not self.cross or image_tokens is None:return base
        if hidden_states.ndim != 3:raise ValueError('Expected transformer spatial tokens')
        b,n,_=hidden_states.shape;side=math.isqrt(n)
        if side*side!=n:raise ValueError('Current experiment requires square latent query grids')
        dtype=hidden_states.dtype
        # Image parameters stay float32 for optimization; matmuls use autocast.
        q=self.q(self.norm(hidden_states.float()))
        pos=positions(side,256,q.device,q.dtype)
        q=q+pos[None]
        tokens=image_tokens+positions(16,256,image_tokens.device,image_tokens.dtype)[None]
        k,v=self.k(tokens),self.v(tokens)
        q=q.view(b,n,4,64).transpose(1,2)
        k=k.view(b,256,4,64).transpose(1,2);v=v.view(b,256,4,64).transpose(1,2)
        bias=None
        if self.locality_bias:
            # Soft coordinate prior, not a category or segmentation label.
            axis=torch.linspace(-1,1,side,device=q.device);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            query_xy=torch.stack([xx.flatten(),yy.flatten()],-1)
            axis=torch.linspace(-1,1,16,device=q.device);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
            key_xy=torch.stack([xx.flatten(),yy.flatten()],-1)
            bias=(-self.locality_bias*torch.cdist(query_xy,key_xy).square()).to(q.dtype)
        branch=F.scaled_dot_product_attention(q,k,v,attn_mask=bias).transpose(1,2).reshape(b,n,256)
        return base+self.gain*self.out(branch).to(dtype)

def install_adapters(unet,initial_gain=.01,locality_bias=0.):
    processors={}
    for name in unet.attn_processors:
        if name.endswith('attn1.processor'): processors[name]=SpatialAttention(cross=False)
        else:
            if name.startswith('mid_block'):hidden=unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):hidden=list(reversed(unet.config.block_out_channels))[int(name.split('.')[1])]
            else:hidden=unet.config.block_out_channels[int(name.split('.')[1])]
            processors[name]=SpatialAttention(hidden_size=hidden,initial_gain=initial_gain,locality_bias=locality_bias)
    unet.set_attn_processor(processors)
    return nn.ModuleList([p for p in unet.attn_processors.values() if isinstance(p,SpatialAttention) and p.cross])
