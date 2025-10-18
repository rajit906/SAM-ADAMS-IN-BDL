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

def student_t_lp(p, scale, nu=3.0):
    """
    Log-probability of a Student's t-distribution.
    This is a heavy-tailed distribution.
    - scale: The scale parameter (sigma)
    - nu: The degrees of freedom. nu=1 is a Cauchy distribution.
          nu->inf approaches a Gaussian. A common default is nu=3.
    """
    if nu <= 0:
        raise ValueError("Degrees of freedom 'nu' must be positive.")
    
    # Pre-compute scalar constants as TENSORS
    half_nu = torch.tensor(nu / 2.0)
    half_nu_plus_1 = torch.tensor((nu + 1.0) / 2.0)
    
    # Log of the normalization constant
    log_norm_const = (
        torch.lgamma(half_nu_plus_1) - 
        torch.lgamma(half_nu) - 
        0.5 * math.log(nu * math.pi) - 
        torch.log(torch.tensor(scale))
    )
    
    # Log-kernel
    # We must use the tensor half_nu_plus_1 here
    log_kernel = -half_nu_plus_1 * torch.log1p((p / scale)**2 / nu)
    
    # Sum up all log-probabilities
    return torch.sum(log_norm_const + log_kernel)

PRIORS = {
    "gaussian": gaussian_lp,
    "laplace": laplace_lp,
    "horseshoe": horseshoe_lp,
    "student-t": student_t_lp
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
        if p.dim() > 1:
            fan_in = p.size(1)
        else:
            fan_in = p.numel()
        if "bias" in name:
            lp = lp + PRIORS["gaussian"](p, bias_global_scale / math.sqrt(fan_in))
        else:
            layer_scale = global_scale / math.sqrt(fan_in)
            lp = lp + prior_fn(p, layer_scale)
    return lp