import os
import json
import torch
import random
import numpy as np
from datetime import datetime

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
        cfg.get("sampler", "sgld"),
        cfg.get("prior", "gaussian"),
        f"scale{cfg.get('prior_scale', 1.0)}",
        f"lr{cfg.get('lr', 1e-2)}",
        f"H{cfg.get('hidden', 128)}",
        f"bs{cfg.get('batch_size', 128)}",
        f"runs{cfg.get('run_id','0')}",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    ]
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
