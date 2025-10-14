import torch
import torch.nn.functional as F
import numpy as np

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

def predictive_metrics_from_weight_dicts(model_class, weight_dicts, dataloader,
                                         device="cpu", psi_history=None):
    """
    weight_dicts: list of dicts with saved model weights
    psi_history: list of dicts mapping param_name -> psi value (optional, SA-SGLD)
    """
    if len(weight_dicts) == 0:
        raise ValueError("No samples provided for evaluation")

    # ---- helper: detect empty psi_history (SGLD case) ----
    def _is_effectively_empty_psi(ph):
        return ph is None or len(ph) == 0 or all(len(d) == 0 for d in ph)

    # ---- compute sample weights ----
    if _is_effectively_empty_psi(psi_history):
        sample_weights = np.ones(len(weight_dicts), dtype=np.float64)
    else:
        w_list = []
        for psi_dict in psi_history:
            vals = [v for v in psi_dict.values() if v is not None]
            w_list.append(float(np.mean(vals)) if len(vals) > 0 else 1.0)
        sample_weights = np.array(w_list, dtype=np.float64)

    sample_weights = sample_weights / np.sum(sample_weights)  # normalize

    # ---- BMA predictive loop ----
    device = torch.device(device)
    model = model_class().to(device).eval()
    all_probs_weighted = None
    ys = None

    with torch.no_grad():
        for i, w in enumerate(weight_dicts):
            model.load_state_dict(w)

            probs_list = []
            labels_list = []
            for xb, yb in dataloader:
                xb = xb.to(device)
                logits = model(xb)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
                labels_list.append(yb.numpy())

            probs_arr = np.concatenate(probs_list, axis=0)
            labels_arr = np.concatenate(labels_list, axis=0)

            if ys is None:
                ys = labels_arr

            weighted_probs = sample_weights[i] * probs_arr
            if all_probs_weighted is None:
                all_probs_weighted = weighted_probs
            else:
                all_probs_weighted += weighted_probs

    # ---- compute metrics ----
    eps = 1e-12
    nll = -np.mean(np.log(all_probs_weighted[np.arange(len(ys)), ys] + eps))
    acc = float((np.argmax(all_probs_weighted, axis=1) == ys).mean())

    n_classes = all_probs_weighted.shape[1]
    onehot = np.eye(n_classes)[ys]
    brier = float(np.mean(np.sum((all_probs_weighted - onehot) ** 2, axis=1)))

    ece = float(_expected_calibration_error(all_probs_weighted, ys, n_bins=15))

    return {"nll": nll, "acc": acc, "brier": brier, "ece": ece}

