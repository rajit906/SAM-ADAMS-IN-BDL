import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# --- Assumed to be in your project structure ---
from utils import mean_confidence_interval, set_seed, get_nll_trace
from models import MLP
from eval import predictive_metrics_from_weight_dicts, calculate_ess

def analyze_experiment(
    experiment_folder, 
    confidence=0.95, 
    device="cuda",
    visualize=True
):
    """
    Runs the full analysis pipeline for a given experiment folder.
    - Loads all run data
    - Aggregates and prints final test metrics (NLL, ECE, Brier, Acc)
    - Calculates and prints ESS from the first run
    - Generates a 4x2 summary plot of all traces
    """
    
    # ------------------------
    # Setup and Validation
    # ------------------------
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    if not os.path.exists(experiment_folder):
        raise ValueError(f"Experiment folder not found: {experiment_folder}")
    print(f"--- Analyzing Experiment: {experiment_folder} ---")

    # ------------------------
    # Load runs
    # ------------------------
    run_paths = sorted(glob(os.path.join(experiment_folder, "*.pt")))
    if not run_paths:
        raise ValueError(f"No .pt files found in {experiment_folder}")

    print(f"\nFound {len(run_paths)} run files in {experiment_folder}")
    runs = [torch.load(p, map_location="cpu") for p in run_paths]
    valid_runs = [r for r in runs if "samples" in r and len(r["samples"]) > 0]
    print(f"Valid runs with samples: {len(valid_runs)}/{len(runs)}")
    if not valid_runs:
        raise ValueError("No valid runs with samples to analyze.")

    # ------------------------
    # Prepare for Analysis
    # ------------------------
    main_run = valid_runs[0]
    cfg = main_run['config']
    print(f"\nAnalyzing Run ID: {cfg.get('run_id', 0)} from a total of {len(valid_runs)} runs.")

    set_seed(cfg.get('seed', 42))
    transform = transforms.Compose([transforms.ToTensor()])
    trainval = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    _, val_set = random_split(trainval, [50000, 10000])
    val_loader = DataLoader(val_set, batch_size=500, shuffle=False)
    test_loader = DataLoader(testset, batch_size=500, shuffle=False)
    model_lambda = lambda: MLP(hidden=cfg['hidden'])


    # ------------------------
    # Aggregate Final Test Metrics
    # ------------------------
    print("\nCalculating final test metrics for all valid runs...")
    all_metrics = {'nll': [], 'acc': [], 'brier': [], 'ece': []}
    for run in tqdm(valid_runs, desc="Processing runs"):
        metrics = predictive_metrics_from_weight_dicts(
            model_class=model_lambda,
            weight_dicts=run['samples'],
            dataloader=test_loader,
            device=device
        )
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    print("\n--- Aggregated Test Metrics ---")
    for key, values in all_metrics.items():
        mean = np.mean(values)
        if len(values) > 1:
            _, ci = mean_confidence_interval(values, confidence=confidence)
            print(f"{key.upper():<7}: {mean:.4f} ± {ci:.4f}")
        else:
            print(f"{key.upper():<7}: {mean:.4f} (single run)")
            
    # # ------------------------
    # # Calculate ESS (from first run)
    # # ------------------------
    # print("\n--- MCMC Diagnostics (from first run) ---")
    # nll_trace = get_nll_trace(model_lambda, main_run['samples'], val_loader, device)
    # ess = calculate_ess(nll_trace)
    # print(f"Effective Sample Size (ESS) on validation NLL: {ess:.2f} / {len(main_run['samples'])}")

    if visualize:
        # ------------------------
        # Process data for plots
        # ------------------------
        history_keys = ["train_loss", "train_acc", "val_loss", "val_acc"]
        agg_curves = {}
        for key in history_keys:
            min_len = min(len(r["train_val_history"][key]) for r in valid_runs)
            aligned = np.array([r["train_val_history"][key][:min_len] for r in valid_runs])
            mean, ci = mean_confidence_interval(aligned, confidence)
            agg_curves[key] = {"mean": mean, "ci": ci}

        dt_hist = main_run.get("dt_history", [])
        dt_traces = {name: [s.get(name, 0) for s in dt_hist] for name in dt_hist[0]} if dt_hist else {}

        grad_norm_hist = main_run.get("grad_norm_history", [])
        grad_norm_traces = {name: [s.get(name, 0) for s in grad_norm_hist] for name in grad_norm_hist[0]} if grad_norm_hist else {}

        z_hist = main_run.get("z_history", [])
        z_traces = {name: [s.get(name, 0) for s in z_hist] for name in z_hist[0]} if z_hist else {}

        config_temp_hist = main_run.get("config_temp_history", [])

        # ------------------------
        # Combined figure: 4x2 grid
        # ------------------------
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        fig.suptitle(f"Analysis for {os.path.basename(experiment_folder)}", fontsize=16)

        # === Top 2 rows: Training curves ===
        titles = ["Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"]
        for ax, key, title in zip(axes[:2, :].flatten(), history_keys, titles):
            mean, ci = agg_curves[key]["mean"], agg_curves[key]["ci"]
            steps = np.arange(len(mean))
            ax.plot(steps, mean)
            ax.fill_between(steps, mean - ci, mean + ci, alpha=0.3)
            ax.set_title(title)
            ax.set_xlabel("Steps (x validation frequency)")
            ax.grid(True)

        # === Row 3: Config Temp and Gradient Norm ===
        if config_temp_hist:
            ax = axes[2, 0]
            ax.plot(config_temp_hist, label=f"Mean: {np.mean(config_temp_hist):.3f}")
            if 'temperature' in cfg:
                ax.axhline(cfg['temperature'], color='r', linestyle='--', label=f"Target T={cfg['temperature']}")
            ax.set_title("Configurational Temperature Trace")
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Temperature")
            ax.set_ylim(bottom=min(0, np.min(config_temp_hist)))
            ax.grid(True)
            ax.legend(fontsize=8)
        else:
            axes[2, 0].axis('off')
            axes[2, 0].set_title("No Config Temp Data")

        if grad_norm_traces:
            ax = axes[2, 1]
            for name, trace in grad_norm_traces.items():
                ax.plot(trace, label=name)
            ax.set_title("Gradient Norm Trace (per layer)")
            ax.set_xlabel("Training Step")
            ax.set_yscale("log")
            ax.grid(True)
            ax.legend(fontsize=8)
        else:
            axes[2, 1].axis('off')
            axes[2, 1].set_title("No Grad Norm Data")
        
        # === Row 4: Z Trace and Step-size Trace ===
        if z_traces:
            ax = axes[3, 0]
            for name, trace in z_traces.items():
                ax.plot(trace, label=name)
            ax.set_title("Z Trace (per layer)")
            ax.set_xlabel("Training Step")
            ax.grid(True)
            ax.legend(fontsize=8)
        else:
            axes[3, 0].axis('off')
            axes[3, 0].set_title("No Z Trace Data (not SASGLD)")

        if dt_traces:
            ax = axes[3, 1]
            for name, trace in dt_traces.items():
                ax.plot(trace, label=name)
            ax.set_title("Step-size Trace (per layer)")
            ax.set_xlabel("Training Step")
            ax.set_yscale("log")
            ax.grid(True)
            ax.legend(fontsize=8)
        else:
            axes[3, 1].axis('off')
            axes[3, 1].set_title("No Step-size Data (not SASGLD)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()