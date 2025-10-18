import torch
import math

def gaussian_lp(p, scale):
    var = scale ** 2
    return torch.sum(-0.5 * (p ** 2) / var - 0.5 * math.log(2 * math.pi * var))

def laplace_lp(p, scale):
    b = scale
    return torch.sum(-torch.abs(p) / b - math.log(2.0 * b))

def horseshoe_lp(p, scale, eps=1e-8):
    # Regularized Horseshoe prior (unnormalized, good for SGD)
    # Vehtari 2017 apparently
    return -torch.sum(torch.log1p((p / (scale + eps))**2 / 2))

def corrnormal_lp(p , scale):
    return None

PRIORS = {
    "gaussian": gaussian_lp,
    "laplace": laplace_lp,
    "horseshoe": horseshoe_lp,
    "corr_normal": corrnormal_lp
}


def log_prior_for_model(model, prior_name, global_scale, bias_global_scale):
    # Return 0 if no prior is specified
    if not prior_name or prior_name.upper() == "NA":
        return 0.0
    
    prior_fn = PRIORS.get(prior_name)
    if prior_fn is None:
        raise ValueError(f"Prior '{prior_name}' not found in PRIORS.")

    lp = 0.0
    for name, p in model.named_parameters():
        if "bias" in name:
            lp = lp + PRIORS["gaussian"](p, bias_global_scale)
        else:
            if p.dim() > 1:
                fan_in = p.size(1)
            else:
                fan_in = p.numel()
            layer_scale = global_scale / math.sqrt(fan_in)
            lp = lp + prior_fn(p, layer_scale)
    return lp