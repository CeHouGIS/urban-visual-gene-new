"""Identical RGB/perceptual/auxiliary objective across all three controls."""
from torch.nn import functional as F
from model import pyramid_targets

def objective(output,target,model,config,perceptual=None):
 pred=output['reconstruction'].float();l1=F.l1_loss(pred,target)
 lp=perceptual(pred*2-1,target*2-1).mean() if perceptual is not None else l1*0
 rgb=l1+config['training']['perceptual_weight']*lp;py=l1*0
 targets=pyramid_targets(target)
 for s in model.scales:
  r=targets[s];norm=r.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(.03).detach()
  py=py+model.gates[s]*((output['residual_predictions'][s].float()-r).abs()/norm).mean()
 py=py/len(model.scales);loss=rgb+config['training']['pyramid_weight']*py
 return loss,{'rgb_loss':rgb,'l1':l1,'lpips':lp,'coding_objective':l1*0,'pyramid_loss':py,'total_loss':loss}
