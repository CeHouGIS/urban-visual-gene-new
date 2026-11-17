"""Image-conditioned whole-gene hard-concrete gates and bounded uint8 maps."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from reference_model import PyramidEncoder,PyramidDecoder,GeneDictionary,pyramid_targets

SCALES=('coarse','medium','fine')

def ramp(step,bounds):return max(0.,min(1.,(step-bounds[0])/(bounds[1]-bounds[0])))

class GroupSparseGeneAE(nn.Module):
 def __init__(self,settings,method='learned'):
  super().__init__()
  if method not in ['learned','random']:raise ValueError(method)
  self.settings=settings;self.method=method;self.sparse=True;self.scales=SCALES
  self.gene_counts=dict(zip(SCALES,settings['dictionary_sizes']))
  self.encoder=PyramidEncoder();self.decoder=PyramidDecoder(settings['latent_channels'],True)
  self.heads=nn.ModuleDict();self.selectors=nn.ModuleDict();self.dictionaries=nn.ModuleDict();self.residual_heads=nn.ModuleDict()
  beta=settings['hard_concrete_temperature'];gamma=settings['hard_concrete_gamma'];zeta=settings['hard_concrete_zeta'];p=settings['initial_gate_probability']
  initial_log_alpha=math.log(p/(1-p))+beta*math.log(-gamma/zeta)
  for s,cin,g,kernel in zip(SCALES,[192,128,96],settings['dictionary_sizes'],settings['dictionary_kernels']):
   self.heads[s]=nn.Conv2d(cin,g,1)
   nn.init.normal_(self.heads[s].weight,std=.01);nn.init.constant_(self.heads[s].bias,-3.)
   self.selectors[s]=nn.Linear(cin,g)
   nn.init.normal_(self.selectors[s].weight,std=.01);nn.init.constant_(self.selectors[s].bias,initial_log_alpha)
   self.dictionaries[s]=GeneDictionary(g,settings['latent_channels'],kernel)
   self.residual_heads[s]=nn.Conv2d(settings['latent_channels'],3,3,padding=1)
  if method=='random':self.dictionaries.requires_grad_(False)
  self.set_step(0)

 def set_step(self,step):
  self.step=int(step)
  schedule=self.settings.get('rate_schedule',list(range(len(self.settings['budgets']))))
  self.rate_index=schedule[max(0,self.step-1)%len(schedule)]
  self.gates={'coarse':1.,'medium':ramp(step,self.settings['medium_ramp_updates']),'fine':ramp(step,self.settings['fine_ramp_updates'])}
  self.penalty=0. # Compatibility with shared training log; complexity uses explicit group/byte prices.

 def quota(self,scale):
  index=self.scales.index(scale);final=self.settings['budget_allocations'][self.rate_index][index]
  if not self.training:return final
  fraction=ramp(self.step,self.settings['group_cap_ramp_updates'])
  return min(self.gene_counts[scale],math.ceil((1-fraction)*self.gene_counts[scale]+fraction*final))

 def group_gate(self,log_alpha,quota):
  """One sampled gate controls an entire spatial map; analytic L0 precedes cap."""
  beta=self.settings['hard_concrete_temperature'];gamma=self.settings['hard_concrete_gamma'];zeta=self.settings['hard_concrete_zeta']
  prob=torch.sigmoid(log_alpha-beta*math.log(-gamma/zeta))
  if self.training:
   u=torch.rand_like(log_alpha).clamp(1e-6,1-1e-6)
   ranking=(log_alpha+torch.log(u)-torch.log1p(-u))/beta
  else:ranking=log_alpha/beta # Hard-concrete median (u=.5) at inference.
  concrete=(torch.sigmoid(ranking)*(zeta-gamma)+gamma).clamp(0,1)
  fraction=ramp(self.step,self.settings['gate_ramp_updates']) if self.training else 1.
  gate=(1-fraction)+fraction*concrete
  # Rank BEFORE noise so training does not systematically select saturated
  # noise outliers that have almost-zero deterministic inference gates.
  selected=torch.zeros_like(gate).scatter(1,log_alpha.topk(quota,dim=1).indices,1)
  capped=gate*selected
  # Hard support in the forward pass; soft hard-concrete gradients explore
  # unselected groups instead of permanently starving their selector logits.
  return capped.detach()+(gate-gate.detach()),prob

 def synthesize_scale(self,scale,activation):
  up=F.interpolate(activation,scale_factor=self.settings['expression_downsample'],mode='bilinear',align_corners=False)
  return self.dictionaries[scale](up)

 def synthesize_codes(self,activations,apply_scale_gates=True):
  return {s:self.synthesize_scale(s,activations[s])*(self.gates[s] if apply_scale_gates else 1.) for s in self.scales}

 def encode_details(self,image):
  if image.ndim!=4 or image.shape[1]!=3 or any(x%32 for x in image.shape[-2:]):raise ValueError('RGB BCHW sides must be divisible by 32')
  encoded=self.encoder(image);features={};acts={};probabilities={};sampled={}
  maximum=self.settings['expression_max'];quant_step=maximum/255.
  for s in self.scales:
   value=encoded[s];small=F.avg_pool2d(value,self.settings['expression_downsample'])
   # Bounded, positive maps cannot hide an unbounded image-dependent amplitude.
   raw=maximum*torch.sigmoid(self.heads[s](small).float())
   with torch.autocast('cuda',enabled=False):
    log_alpha=self.selectors[s](value.float().mean((2,3)))
   gate,prob=self.group_gate(log_alpha,self.quota(s))
   gated=raw*gate[:,:,None,None]
   quant=torch.round((gated/maximum).clamp(0,1)*255)*quant_step
   act=quant.detach()+(gated-gated.detach()) # Exactly quantized forward; STE backward.
   acts[s]=act;features[s]=self.synthesize_scale(s,act);probabilities[s]=prob;sampled[s]=gate
  return features,acts,probabilities,sampled

 def encode(self,image):
  features,acts,_,_=self.encode_details(image)
  return features,acts,features

 def forward(self,image):
  features,acts,probs,gates=self.encode_details(image)
  decoder_features={s:v*self.gates[s] for s,v in features.items()}
  divisors={'coarse':4,'medium':2,'fine':1}
  residuals={s:F.interpolate(self.residual_heads[s](features[s]),size=tuple(x//divisors[s] for x in image.shape[-2:]),mode='bilinear',align_corners=False) for s in self.scales}
  return {'reconstruction':self.decoder(decoder_features),'features':features,'decoder_features':decoder_features,'activations':acts,'feature_targets':features,'residual_predictions':residuals,'latent':features['coarse'],'gate_probabilities':probs,'sampled_group_gates':gates}

 def expected_complexity(self,output):
  """Training distribution expectation, excluding the cap/rounding: conservative."""
  batch=next(iter(output['activations'].values())).shape[0]
  count=torch.zeros(batch,device=next(self.parameters()).device);payload=torch.full_like(count,32.*len(self.scales))
  f=ramp(self.step,self.settings['gate_ramp_updates']) if self.training else 1.
  for s in self.scales:
   p=output['gate_probabilities'][s]
   # During gate warmup any residual 1-f makes an untruncated gate nonzero.
   if f<1:p=p*0+1
   n=p.sum(1);h,w=output['activations'][s].shape[-2:]
   count=count+n;payload=payload+n*(2+h*w)
  return count,payload

 def actual_complexity(self,output):
  used={s:(a.detach()!=0).any((2,3)) for s,a in output['activations'].items()}
  count=torch.stack([v.sum(1) for v in used.values()]).sum(0)
  payload=torch.full_like(count,32*len(self.scales))
  for s,a in output['activations'].items():payload=payload+used[s].sum(1)*(2+a.shape[-2]*a.shape[-1])
  return count,payload

 def usage(self,activations):return torch.cat([(activations[s]>0).sum((0,2,3)) for s in self.scales])

 def contribution(self,activations,scale,gene):
  act=F.interpolate(activations[scale][:,gene:gene+1],scale_factor=self.settings['expression_downsample'],mode='bilinear',align_corners=False)
  return F.conv2d(act,self.dictionaries[scale].atoms[gene].unsqueeze(1),padding=self.dictionaries[scale].kernel//2)

 def spec(self,resolution=256):
  return {'method':self.method,'parameters':sum(p.numel() for p in self.parameters()),'trainable_parameters':sum(p.numel() for p in self.parameters() if p.requires_grad),'dictionary_genes':sum(self.gene_counts.values()),'scale_gene_counts':self.gene_counts,'dictionary_kernels':dict(zip(self.scales,self.settings['dictionary_kernels'])),'gene_caps':self.settings['budgets'],'per_scale_caps':self.settings['budget_allocations'],'expression_grids':{s:resolution//d//2 for s,d in zip(self.scales,[16,8,4])},'expression_quantization':'uint8; global fixed bound '+str(self.settings['expression_max']),'gate':'image-conditioned stochastic hard-concrete during training, deterministic at evaluation; TopK cap on entire maps','expected_cost':'before cap and uint8 rounding; upper bound on expected post-cap TRAINING cost','encoder_bypass':False,'feature_additivity':True,'rgb_decoder':'nonlinear','from_scratch':True,'dictionary_frozen':self.method=='random'}

def create_model(config,variant):
 if not torch.cuda.is_available():raise RuntimeError('GPU required; no CPU model execution')
 method,seed=variant.rsplit('_s',1);torch.manual_seed(int(seed))
 return GroupSparseGeneAE(config['model'],method).cuda()
