import os
import json
import torch
import random
import numpy as np
from datetime import datetime
from scipy.stats import t
import torch.nn.functional as F
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_results_dir(base="results"):
    os.makedirs(base, exist_ok=True)
    return base

def make_run_name(cfg):
    parts = [
        #cfg.get("sampler", "sgld"),
        cfg.get("prior", "gaussian"),
        f"scale{cfg.get('prior_scale', 1.0)}",
        f"lr{cfg.get('lr', 1e-1)}",
        f"H{cfg.get('hidden', 400)}",
        f"bs{cfg.get('batch_size', 100)}",
        #f"runs{cfg.get('run_id','0')}",
        #datetime.now().strftime("%Y%m%d-%H%M%S"),
    ]
    if cfg["sampler"] == "sasgld":
        parts += [f"m{cfg['m']}", f"M{cfg['M']}", f"alpha{cfg['alpha']}", f"r{cfg['r']}", f"s{cfg['s']}"]
    return "_".join(map(str, parts))

def ensure_parent(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_pt(obj, path):
    ensure_parent(path)
    torch.save(obj, path)

def load_pt(path, map_location=None):
    return torch.load(path, map_location=map_location)

def dump_json(d, path):
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def mean_confidence_interval(data, confidence=0.95):
    """
    Compute mean and confidence interval along axis=0.
    """
    data = np.array(data)
    mean = np.mean(data, axis=0)
    sem = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))
    h = sem * t.ppf((1 + confidence) / 2., len(data)-1)
    return mean, h

def get_nll_trace(model_lambda, samples, dataloader, device):
    """
    Calculate the negative log-likelihood for each sample on a given dataset.
    """
    print("Calculating NLL trace for ESS...")
    model = model_lambda().to(device).eval()
    nll_trace = []
    
    all_x = torch.cat([xb for xb, _ in dataloader], dim=0).to(device)
    all_y = torch.cat([yb for _, yb in dataloader], dim=0).to(device)

    with torch.no_grad():
        for s in tqdm(samples, desc="Evaluating samples"):
            model.load_state_dict(s)
            logits = model(all_x)
            nll = F.cross_entropy(logits, all_y, reduction='mean').item()
            nll_trace.append(nll)
    return np.array(nll_trace)
