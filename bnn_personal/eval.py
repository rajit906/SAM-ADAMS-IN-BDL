import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import brier_score_loss
import math

def _expected_calibration_error(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    n = len(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        prop_in_bin = in_bin.mean() if n > 0 else 0.0
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence = confidences[in_bin].mean()
            ece += abs(avg_confidence - accuracy_in_bin) * prop_in_bin
    return ece

def predictive_metrics_from_weight_dicts(model_class, weight_dicts, dataloader, device="cpu"):
    """
    weight_dicts: list of dicts compatible with model.load_state_dict (weights only)
    """
    if len(weight_dicts) == 0:
        raise ValueError("No samples provided for evaluation")
    device = torch.device(device)
    model = model_class().to(device).eval()
    probs_accum = None
    ys = None
    with torch.no_grad():
        for w in weight_dicts:
            model.load_state_dict(w)
            all_probs = []
            all_labels = []
            for xb, yb in dataloader:
                xb = xb.to(device)
                logits = model(xb)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(yb.numpy())
            probs_arr = np.concatenate(all_probs, axis=0)
            labels_arr = np.concatenate(all_labels, axis=0)
            if probs_accum is None:
                probs_accum = probs_arr
                ys = labels_arr
            else:
                probs_accum = probs_accum + probs_arr
    probs_mean = probs_accum / len(weight_dicts)
    eps = 1e-12
    nll = -np.mean(np.log(probs_mean[np.arange(len(ys)), ys] + eps)) # NOTE: NLL or Log posterior?
    acc = float((np.argmax(probs_mean, axis=1) == ys).mean())
    n_classes = probs_mean.shape[1]
    onehot = np.eye(n_classes)[ys]
    brier = float(np.mean(np.sum((probs_mean - onehot) ** 2, axis=1)))
    ece = float(_expected_calibration_error(probs_mean, ys, n_bins=15))
    return {"nll": float(nll), "acc": float(acc), "brier": float(brier), "ece": float(ece)}
