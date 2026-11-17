"""Three variable-rate controls with identical on-wire code sizes and RGB loss."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from base_model import ResidualISTAGeneAE, pyramid_targets

class BudgetAE(ResidualISTAGeneAE):
 def __init__(self,method,settings):
  super().__init__('multiscale_residual_ista',settings)
  self.method=method;self.sparse=method!='dense';self.rate_index=0
  if method=='random':
   self.dictionaries.requires_grad_(False)
  if method=='dense':
   self.dictionaries=nn.ModuleDict();self.inference=nn.ModuleDict()
   self.heads=nn.ModuleDict({s:nn.Conv2d(c,max(settings['dense_widths']),1,bias=False) for s,c in [('coarse',192),('medium',128),('fine',96)]})
   self.dense_synthesis=nn.ModuleDict({s:nn.Conv2d(max(settings['dense_widths']),settings['latent_channels'],1,bias=False) for s in self.scales})
  self.set_step(0)

 def set_step(self,step):
  super().set_step(step)
  # Every block of three updates visits each rate once; no random sampling drift.
  self.rate_index=max(0,int(step)-1)%3

 def encode(self,image):
  encoded=self.encoder(image);features={};activations={};targets={}
  for s in self.scales:
   projected=self.heads[s](encoded[s])
   if self.sparse:
    target=F.normalize(projected.float(),dim=1,eps=1e-6)*math.sqrt(projected.shape[1])
    _,a=self.inference[s](target,self.dictionaries[s],self.penalty)
    k=min(a[0].numel(),a.shape[-1]*a.shape[-2]*self.settings['sparse_per_spatial'][self.rate_index])
    flat=a.flatten(1);idx=flat.topk(k,dim=1,sorted=False).indices
    sparse=torch.zeros_like(flat).scatter(1,idx,flat.gather(1,idx)).view_as(a)
    # Training and evaluation decode FP16-rounded coefficients. STE only affects gradients.
    quant=sparse.float().to(torch.float16).float()
    a=sparse.float()+(quant-sparse.float()).detach()
    features[s]=self.dictionaries[s](a);activations[s]=a;targets[s]=target
   else:
    width=self.settings['dense_widths'][self.rate_index]
    a=projected[:,:width].float();quant=a.to(torch.float16).float()
    a=a+(quant-a).detach()
    # Zero trailing channels are known from the shared rate header and cost no payload.
    padded=F.pad(a,(0,0,0,0,0,max(self.settings['dense_widths'])-width))
    features[s]=self.dense_synthesis[s](padded);activations[s]=a;targets[s]=features[s]
  return features,activations,targets

 def spec(self,resolution=256):
  return {'method':self.method,'parameters':sum(p.numel() for p in self.parameters()),'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),'frozen_dictionary':self.method=='random','from_scratch':True,'shared_trunk_initialization':True,'rate_widths':self.settings['dense_widths'],'budget_bytes':[32*3+6*k*sum((resolution//d)**2 for d in [16,8,4]) for k in self.settings['sparse_per_spatial']],'coefficient_quantization':'float16 with straight-through training','support':'per-scale global magnitude TopK for dictionaries; dense channel prefix for AE','no_encoder_bypass':True,'dictionary_genes':512 if self.sparse else None,'scale_grids':dict(zip(self.scales,[resolution//16,resolution//8,resolution//4]))}

def create_model(config,variant):
 if not torch.cuda.is_available():raise RuntimeError('GPU required; CPU model execution forbidden')
 method,seed=variant.rsplit('_s',1);torch.manual_seed(int(seed))
 if method not in ['learned','random','dense']:raise ValueError(method)
 return BudgetAE(method,config['model']).cuda()
