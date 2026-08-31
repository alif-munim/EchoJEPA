"""Regression metrics for EchoBench (MAE, R-squared, Pearson correlation)."""

import numpy as np


def compute_regression_metrics(preds, labels, target_mean=None, target_std=None):
    """
    Compute regression metrics.

    If target_mean and target_std are provided, predictions are un-normalized
    from z-score space back to original units before computing metrics.

    Args:
        preds: np.ndarray of shape [N] or [N, 1]
        labels: np.ndarray of shape [N]
        target_mean: float, optional. Z-score mean used during training.
        target_std: float, optional. Z-score std used during training.

    Returns:
        dict with keys: mae, r2, pearson_r
    """
    preds = np.asarray(preds).squeeze()
    labels = np.asarray(labels).squeeze()

    if len(preds) == 0:
        return {"mae": float("nan"), "r2": float("nan"), "pearson_r": float("nan")}

    # Un-normalize z-scored predictions
    if target_mean is not None and target_std is not None:
        preds = preds * target_std + target_mean

    residuals = labels - preds
    mae = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    if len(labels) > 2:
        from scipy.stats import pearsonr

        pearson_r, _ = pearsonr(labels, preds)
        pearson_r = float(pearson_r)
    else:
        pearson_r = float("nan")

    return {"mae": mae, "r2": r2, "pearson_r": pearson_r}
