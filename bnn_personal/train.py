# TODO: Test with and without reweighting for SA-SGLD.
# Non-global learning rate,lower scale of prior.
# Why is configurational temp wrong? Also reweight for SA-SGLD. Put second Z-step (?)
#train.py
import os
import torch.multiprocessing as mp
import torch
from models import MLP
from priors import log_prior_for_model
from samplers import SGLD, SASGLD
from utils import set_seed, make_results_dir, make_run_name, save_pt
from eval import predictive_metrics_from_weight_dicts
from dataloader import build_dataloaders
from config import get_config
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

cfg = get_config()

def run_single(cfg, run_id=0):
    set_seed(cfg.get("seed", 0) + run_id)
    device = torch.device(cfg["device"])
    model = MLP(hidden=cfg["hidden"]).to(device)

    total_params = sum([p.numel() for p in model.parameters()])

    train_loader, val_loader, test_loader, num_data = build_dataloaders(cfg["batch_size"])

    # Choose optimizer/sampler
    sampler_name = cfg["sampler"].lower()
    if sampler_name == "sgld":
        opt = SGLD(model.parameters(), lr=cfg["lr"], temperature=cfg["temperature"]) 
    elif sampler_name == "sasgld":
        opt = SASGLD(
            model.parameters(), lr=cfg["lr"], temperature=cfg["temperature"],
            alpha=cfg["alpha"], m=cfg["m"], M=cfg["M"], r=cfg["r"], s=cfg["s"], Omega=cfg['Omega'], init_z=cfg["init_z"]
        )
    else:
        raise ValueError(f"Unknown sampler: {cfg['sampler']}")

    samples = []
    z_history = []
    psi_history = []
    dt_history = []
    grad_norm_history = []
    config_temp_history = []
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "steps": []}

    criterion = torch.nn.CrossEntropyLoss()
    total_batches = 0
    model.train()

    for epoch in range(cfg["epochs"]):
        pbar = tqdm(train_loader, desc=f"Run {run_id} Epoch {epoch+1}", leave=False)
        for xb, yb in pbar:
            total_batches += 1
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            
            lp = log_prior_for_model(model, cfg["prior"], cfg["prior_scale"], cfg["bias_prior_scale"])
            potential = loss - (lp / num_data)

            opt.zero_grad()
            potential.backward()

            with torch.no_grad():
                # Track gradient norms
                current_grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
                grad_norm_history.append(current_grad_norms)

                # Calculate configurational temp using <theta, grad U> / d
                if total_params > 0:
                    config_temp = sum(torch.sum(p * p.grad) for p in model.parameters() if p.grad is not None) / total_params
                    config_temp_history.append(config_temp.item())

            opt.step()

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                acc = (preds == yb).float().mean().item()
                history["train_loss"].append(loss.item())
                history["train_acc"].append(acc)
                history["steps"].append(total_batches)

            if hasattr(opt, "get_state_samples_info"):
                info = opt.get_state_samples_info(model)
                z_history.append(info["z"])
                psi_history.append(info["psi"])
                dt_history.append(info["dt"])
            
            if total_batches > cfg["burnin_batches"] and (total_batches - cfg["burnin_batches"]) % cfg["save_every"] == 0:
                state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                samples.append(state)

            if total_batches % len(train_loader) == 0:
                model.eval()
                val_loss, val_acc, n_val = 0.0, 0.0, 0
                with torch.no_grad():
                    for xb2, yb2 in val_loader:
                        xb2, yb2 = xb2.to(device), yb2.to(device)
                        logits2 = model(xb2)
                        val_loss += criterion(logits2, yb2).sum().item()
                        val_acc += (logits2.argmax(dim=1) == yb2).sum().item()
                        n_val += xb2.size(0)
                history["val_loss"].append(val_loss / n_val)
                history["val_acc"].append(val_acc / n_val)
                model.train()

    # --- Post-training evaluation and saving ---
    test_metrics = {}
    if len(samples) > 0:
        test_metrics = predictive_metrics_from_weight_dicts(
            lambda: MLP(hidden=cfg["hidden"]),
            samples,
            test_loader,
            device=cfg["device"])
    
    results_dir = make_results_dir(cfg["results_dir"])
    sampler_dir = os.path.join(results_dir, cfg["sampler"])
    os.makedirs(sampler_dir, exist_ok=True)
    
    run_cfg_no_id = {k: v for k, v in cfg.items() if k not in ["num_runs", "device"]}
    exp_name = make_run_name(run_cfg_no_id)
    exp_dir = os.path.join(sampler_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    save_path = os.path.join(exp_dir, f"{run_id}.pt")
    payload = {
        "config": {**cfg, "run_id": run_id},
        "samples": samples,
        "z_history": z_history,
        "psi_history": psi_history,
        "dt_history": dt_history,
        "grad_norm_history": grad_norm_history,
        "config_temp_history": config_temp_history,
        "train_val_history": history,
        "test_metrics": test_metrics,
    }
    save_pt(payload, save_path)
    print(f"✅ Saved run {run_id} -> {save_path}; n_samples={len(samples)}")

def run_worker(run_id, cfg):
    print(f"Starting run {run_id}...")
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