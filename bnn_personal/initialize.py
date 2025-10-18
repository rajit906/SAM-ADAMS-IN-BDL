# initialize.py
# TODO: Train NA, Gaussian, Laplace, Horseshoe. Adjust prior scale accordingly.
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import ssl
import matplotlib.pyplot as plt

# Assuming these are in your project structure
from models import MLP
from priors import PRIORS
from utils import set_seed
from eval import predictive_metrics_from_weight_dicts
ssl._create_default_https_context = ssl._create_unverified_context

def get_config():
    """Parses command line arguments for the optimization script."""
    parser = argparse.ArgumentParser(description="SGD Optimizer for finding a good initialization")

    # Configuration based on your request
    parser.add_argument("--prior", type=str, default="NA")
    parser.add_argument("--prior_scale", type=float, default=1.0)
    parser.add_argument("--bias_prior_scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate for SGD.")
    parser.add_argument("--hidden", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--initial_dir", type=str, default="initial", help="Directory to save the checkpoint and plots.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return vars(args)

def build_dataloaders(batch_size):
    """Builds MNIST dataloaders."""
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
    """Calculates the log prior for the entire model."""
    lp = 0.0
    # Return 0 if no prior is specified
    if prior_name.upper() == "NA" or prior_name is None:
        return 0.0
    
    prior_fn = PRIORS.get(prior_name)
    if prior_fn is None:
        raise ValueError(f"Prior '{prior_name}' not found in PRIORS.")

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

def make_checkpoint_name(cfg):
    """Creates a descriptive filename based on the configuration."""
    name = f"prior_{cfg['prior']}"
    if cfg['prior'].upper() != 'NA':
        name += f"_scale_{cfg['prior_scale']}"
    name += f"_lr_{cfg['lr']}_H_{cfg['hidden']}"
    return name

def main():
    cfg = get_config()
    set_seed(cfg['seed'])
    
    # --- Setup ---
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg['initial_dir'], exist_ok=True)
    
    train_loader, val_loader, test_loader, num_data = build_dataloaders(cfg['batch_size'])
    model_lambda = lambda: MLP(hidden=cfg['hidden']) # Needed for eval function
    model = model_lambda().to(device)
    
    # --- Optimizer and Scheduler ---
    optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # --- Training Loop ---
    print("Starting optimization...")
    for epoch in range(cfg['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False)
        epoch_train_loss, epoch_train_acc = 0.0, 0.0
        
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            
            lp = log_prior_for_model(model, cfg['prior'], cfg['prior_scale'], cfg['bias_prior_scale'])
            potential = loss - (lp / num_data)
            
            potential.backward()
            optimizer.step()
            
            preds = torch.argmax(logits, dim=1)
            epoch_train_acc += (preds == yb).float().mean().item()
            epoch_train_loss += loss.item()

        scheduler.step()
        
        history['train_loss'].append(epoch_train_loss / len(train_loader))
        history['train_acc'].append(epoch_train_acc / len(train_loader))

        # --- Simple Validation Loop ---
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).sum().item()
                preds = logits.argmax(dim=1)
                val_acc += (preds == yb).sum().item()
                n_val += xb.size(0)
        
        history['val_loss'].append(val_loss / n_val)
        history['val_acc'].append(val_acc / n_val)
        
        print(f"Epoch {epoch+1}: Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {history['train_acc'][-1]:.4f}, Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {history['val_acc'][-1]:.4f}")

    print("\n--- Final Evaluation on Test Set ---")
    model.eval()
    # Create a list containing just the final model's state dict
    final_model_dict = [{k: v.cpu() for k, v in model.state_dict().items()}]
    
    test_metrics = predictive_metrics_from_weight_dicts(
        model_class=model_lambda,
        weight_dicts=final_model_dict,
        dataloader=test_loader,
        device=device,
        psi_history=None
    )
    print("Final Test Metrics:")
    for key, val in test_metrics.items():
        print(f"  {key.upper():<7}: {val:.4f}")

    # --- Save Checkpoint and Plot ---
    base_name = make_checkpoint_name(cfg)
    
    checkpoint_path = os.path.join(cfg['initial_dir'], f"{base_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Saved final model checkpoint to: {checkpoint_path}")
    
    # Create and save plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['val_acc'], label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    # Create a title string with the final test metrics
    title = (
        f"Optimizer Results | Final Test Metrics:\n"
        f"NLL: {test_metrics['nll']:.4f}, Acc: {test_metrics['acc']:.4f}, "
        f"ECE: {test_metrics['ece']:.4f}, Brier: {test_metrics['brier']:.4f}"
    )
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout to make space for the suptitle
    plot_path = os.path.join(cfg['initial_dir'], f"{base_name}_curves.png")
    plt.savefig(plot_path)
    print(f"📈 Saved training curves plot to: {plot_path}")

if __name__ == "__main__":
    main()