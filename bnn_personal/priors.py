import torch
import math

def gaussian_logprior(p, scale):
    var = scale ** 2
    return torch.sum(-0.5 * (p ** 2) / var - 0.5 * math.log(2 * math.pi * var))

def laplace_logprior(p, scale):
    b = scale
    return torch.sum(-torch.abs(p) / b - math.log(2.0 * b))

def horseshoe_logprior(p, scale):
    # simple heavy-tailed proxy for horseshoe (unnormalized)
    x = (p / scale) ** 2
    return torch.sum(-0.5 * torch.log1p(x))

PRIORS = {
    "gaussian": gaussian_logprior,
    "laplace": laplace_logprior,
    "horseshoe": horseshoe_logprior,
}
