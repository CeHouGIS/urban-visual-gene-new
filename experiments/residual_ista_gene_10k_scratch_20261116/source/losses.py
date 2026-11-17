"""Reconstruction, normalized sparse coding, fixed RGB residual supervision."""
import torch
from torch.nn import functional as F
from model import pyramid_targets


def objective(output, target, model, config, perceptual=None):
    prediction = output['reconstruction'].float()
    l1 = F.l1_loss(prediction, target)
    lp = perceptual(prediction*2-1,target*2-1).mean() if perceptual is not None else l1*0
    rgb = l1 + config['training']['perceptual_weight'] * lp
    coding = l1 * 0
    pyramid = l1 * 0
    targets = pyramid_targets(target)
    for scale in model.scales:
        feature = output['features'][scale].float()
        ft = output['feature_targets'][scale]
        act = output['activations'][scale].float()
        # Both terms have the same normalization as the per-location ISTA
        # objective: channel SUM, then spatial/batch mean, divided by C.
        fit = 0.5 * (feature-ft).square().mean()
        sparse = model.penalty * act.sum(1).mean() / feature.shape[1]
        coding = coding + model.gates[scale] * (fit+sparse)
        residual = targets[scale]
        # Fixed target energy keeps low-amplitude fine residuals consequential.
        normalization = residual.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(0.03).detach()
        error = ((output['residual_predictions'][scale].float()-residual).abs()/normalization).mean()
        pyramid = pyramid + model.gates[scale] * error
    # Fixed divisor avoids changing existing loss weights as a scale ramps up.
    coding = coding / len(model.scales)
    pyramid = pyramid / len(model.scales)
    total = rgb + config['training']['feature_objective_weight']*coding + config['training']['pyramid_weight']*pyramid
    return total, {'rgb_loss':rgb, 'l1':l1, 'lpips':lp, 'coding_objective':coding,
                   'pyramid_loss':pyramid, 'total_loss':total}
