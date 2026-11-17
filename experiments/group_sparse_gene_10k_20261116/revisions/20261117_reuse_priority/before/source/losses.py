"""Same reconstruction loss for both methods; whole-gene and byte prices."""
from torch.nn import functional as F
from model import pyramid_targets,ramp

def objective(output,target,model,config,perceptual=None):
 pred=output['reconstruction'].float();l1=F.l1_loss(pred,target)
 lp=perceptual(pred*2-1,target*2-1).mean() if perceptual is not None else l1*0
 rgb=l1+config['training']['perceptual_weight']*lp;pyramid=l1*0
 for scale,residual in pyramid_targets(target).items():
  norm=residual.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(.03).detach()
  pyramid=pyramid+model.gates[scale]*((output['residual_predictions'][scale].float()-residual).abs()/norm).mean()
 pyramid=pyramid/len(model.scales)
 count,byte_cost=model.expected_complexity(output)
 complexity=ramp(model.step,config['training']['complexity_ramp_updates'])*(config['training']['gene_price']*count.mean()+config['training']['byte_price']*byte_cost.mean())
 total=rgb+config['training']['pyramid_weight']*pyramid+complexity
 return total,{'rgb_loss':rgb,'l1':l1,'lpips':lp,'coding_objective':complexity,'pyramid_loss':pyramid,'total_loss':total,'expected_pre_cap_genes':count.mean(),'expected_pre_cap_bytes':byte_cost.mean()}
