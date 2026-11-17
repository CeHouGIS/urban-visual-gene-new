import torch
from torch.nn import functional as F


def ssim_per_image(x,y):
    positions=torch.arange(11,device=x.device,dtype=x.dtype)-5
    kernel=torch.exp(-positions.square()/(2*1.5**2))
    kernel=kernel/kernel.sum()
    window=(kernel[:,None]*kernel[None,:])[None,None].expand(x.shape[1],1,11,11)
    def smooth(z):
        return F.conv2d(z,window,groups=x.shape[1])
    ux,uy=smooth(x),smooth(y)
    vx=(smooth(x*x)-ux*ux).clamp_min(0)
    vy=(smooth(y*y)-uy*uy).clamp_min(0)
    cov=smooth(x*y)-ux*uy
    score=((2*ux*uy+0.01**2)*(2*cov+0.03**2))/((ux*ux+uy*uy+0.01**2)*(vx+vy+0.03**2))
    return score.mean((1,2,3))
