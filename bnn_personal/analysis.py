import os
import torch
from viz import analyze_experiment

# ------------------------
# Config
# ------------------------
results_dir = "./results"
sgld_dir = os.path.join(results_dir, "sgld")
confidence = 0.95
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Loop through all experiment folders in ./results/sgld
# ------------------------
for folder_name in os.listdir(sgld_dir):
    experiment_folder = os.path.join(sgld_dir, folder_name)
    if os.path.isdir(experiment_folder):  # Only process directories
        try:
            analyze_experiment(experiment_folder=experiment_folder, visualize=False)
        except Exception as e:
            print(f"⚠️ Failed to analyze {experiment_folder}: {e}")
