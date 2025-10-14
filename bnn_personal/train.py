# TODO: Fix horseshoe later. Make sure everything runs on GPU before parallel run.
# Determine whether to fix initialization or not.
# When sampler == sasgld, include the config for m,M,r,s,alpha. Remove double sampler.
import os
import torch.multiprocessing as mp
import math
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models import MLP
from priors import PRIORS
from samplers import SGLD, SASGLD
from utils import set_seed, make_results_dir, make_run_name, save_pt, dump_json
from eval import predictive_metrics_from_weight_dicts
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # Main options
    parser.add_argument("--sampler", type=str, default="sgld")
    parser.add_argument("--prior", type=str, default="laplace")
    parser.add_argument("--prior_scale", type=float, default=1.0)
    parser.add_argument("--bias_prior_scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)

    # SA-SGLD specific
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--M", type=float, default=10.0)
    parser.add_argument("--r", type=float, default=0.25)
    parser.add_argument("--s", type=float, default=2.0)
    parser.add_argument("--init_z", type=float, default=0.0)
    parser.add_argument("--Omega", type=int, default=50000)

    # Model/training
    parser.add_argument("--hidden", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--burnin_batches", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=5)

    # System
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return vars(args)  # return as dictionary like before

cfg = get_config()

def build_dataloaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    trainval = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    n = len(trainval)
    n_val = 10000
    n_train = n - n_val
    train_set, val_set = random_split(trainval, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n

def log_prior_for_model(model, prior_name, global_scale, bias_global_scale):
    lp = 0.0
    prior_fn = PRIORS[prior_name]
    for name, p in model.named_parameters():
        # Bias prior stays simple
        if "bias" in name:
            lp = lp + PRIORS["gaussian"](p, bias_global_scale)
        else:
            # === scale based on layer width (fan-in) ===
            if p.dim() > 1:
                fan_in = p.size(1)  # weight matrix shape [out_dim, in_dim]
            else:
                fan_in = p.numel()
            layer_scale = global_scale / math.sqrt(fan_in)
            # === apply the chosen prior ===
            lp = lp + prior_fn(p, layer_scale)
    return lp


def run_single(cfg, run_id=0):
    set_seed(cfg.get("seed", 0) + run_id)
    device = torch.device(cfg["device"])
    model = MLP(hidden=cfg["hidden"]).to(device)
    train_loader, val_loader, test_loader, num_data = build_dataloaders(cfg["batch_size"])

    # choose optimizer
    if cfg["sampler"].lower() == "sgld":
        opt = SGLD(model.parameters(), lr=cfg["lr"], temperature=cfg["temperature"], num_data=num_data)
    else:
        opt = SASGLD(
            model.parameters(), lr=cfg["lr"], temperature=cfg["temperature"], num_data=num_data,
            alpha=cfg["alpha"], m=cfg["m"], M=cfg["M"], r=cfg["r"], s=cfg["s"], Omega=cfg['Omega'], init_z=cfg["init_z"]
        )

    samples = []
    z_history = []    # list of dicts per saved sample: z->param_name dict
    psi_history = []
    dt_history = []
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "steps": []}

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    total_batches = 0
    model.train()

    for epoch in range(cfg["epochs"]):
        pbar = tqdm(train_loader, desc=f"Run {run_id} Epoch {epoch}", leave=False)
        for xb, yb in pbar:
            total_batches += 1
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            per_example_loss = criterion(logits, yb)
            nll = per_example_loss.mean() * num_data
            lp = log_prior_for_model(model, cfg["prior"], cfg["prior_scale"], cfg["bias_prior_scale"])
            potential = (nll - lp)

            opt.zero_grad()
            potential.backward()
            opt.step()

            # training metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == yb).float().mean().item()
                history["train_loss"].append(per_example_loss.mean().item())
                history["train_acc"].append(acc)
                history["steps"].append(total_batches)

            if hasattr(opt, "get_state_samples_info"):
                    info = opt.get_state_samples_info(model)
                    z_history.append(info["z"])
                    psi_history.append(info["psi"])
                    dt_history.append(info["dt"])
            else:
                    z_history.append({})
                    psi_history.append({})
                    dt_history.append({})

            # sampling logic
            if total_batches > cfg["burnin_batches"] and (total_batches - cfg["burnin_batches"]) % cfg["save_every"] == 0:
                # save weights only (map param name -> tensor.cpu())
                state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                samples.append(state)

            # validation periodic
            if total_batches % len(train_loader) == 0:
                model.eval()
                val_loss = 0.0
                val_acc = 0.0
                n_val = 0
                with torch.no_grad():
                    for xb2, yb2 in val_loader:
                        xb2 = xb2.to(device)
                        yb2 = yb2.to(device)
                        logits2 = model(xb2)
                        l = criterion(logits2, yb2).sum().item()
                        val_loss += l
                        preds2 = logits2.argmax(dim=1)
                        val_acc += (preds2 == yb2).sum().item()
                        n_val += xb2.size(0)
                history["val_loss"].append(val_loss / n_val)
                history["val_acc"].append(val_acc / n_val)
                model.train()

    # After training, run evaluation on test set (using saved samples)
    eval_device = cfg["device"]
    test_metrics = {}
    if len(samples) > 0:
        test_metrics =  predictive_metrics_from_weight_dicts(
                                lambda: MLP(hidden=cfg["hidden"]),
                                samples,
                                test_loader,
                                device=eval_device,
                                psi_history=psi_history)
    
    # Create experiment folder structure: results/sampler/experiment_name/
    results_dir = make_results_dir(cfg["results_dir"])
    sampler_dir = os.path.join(results_dir, cfg["sampler"])
    os.makedirs(sampler_dir, exist_ok=True)
    
    # Create experiment name (without run_id)
    run_cfg_no_id = {k: v for k, v in cfg.items() if k != "num_runs"}
    exp_name = make_run_name(run_cfg_no_id)
    exp_dir = os.path.join(sampler_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save as run_id.pt
    save_path = os.path.join(exp_dir, f"{run_id}.pt")
    run_cfg = {**cfg, "run_id": run_id}
    payload = {
        "config": run_cfg,
        "samples": samples,
        "z_history": z_history,
        "psi_history": psi_history,
        "dt_history": dt_history,
        "train_val_history": history,
        "test_metrics": test_metrics,
    }
    save_pt(payload, save_path)
    print(f"Saved run {run_id} -> {save_path}; n_samples={len(samples)}")
    return save_path, exp_dir

def run_worker(run_id, cfg):
    print(f"Starting run {run_id} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')}")
    run_single(cfg, run_id)

def main():
    set_seed(cfg.get("seed", 0))
    _ = make_results_dir(cfg["results_dir"])
    processes = []

    for run_id in range(cfg["num_runs"]):
        p = mp.Process(target=run_worker, args=(run_id, cfg))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()