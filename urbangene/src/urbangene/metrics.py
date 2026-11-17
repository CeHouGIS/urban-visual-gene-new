import torch
import torch.nn.functional as F


def image_metrics(target, prediction):
    mse = (target-prediction).square().mean((1, 2, 3))
    mu_x = F.avg_pool2d(target, 7, 1)
    mu_y = F.avg_pool2d(prediction, 7, 1)
    var_x = F.avg_pool2d(target.square(), 7, 1)-mu_x.square()
    var_y = F.avg_pool2d(prediction.square(), 7, 1)-mu_y.square()
    cov = F.avg_pool2d(target*prediction, 7, 1)-mu_x*mu_y
    ssim = ((2*mu_x*mu_y+0.01**2)*(2*cov+0.03**2)/
            ((mu_x.square()+mu_y.square()+0.01**2)*(var_x+var_y+0.03**2))).mean((1, 2, 3))
    return {"psnr": -10*torch.log10(mse.clamp_min(1e-12)), "ssim_7x7_uniform": ssim,
            "rgb_l1": (target-prediction).abs().mean((1, 2, 3))}
