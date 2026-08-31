"""Segmentation metrics for EchoBench (Dice score, Hausdorff distance)."""

import torch


def compute_dice(pred, target, num_classes=4):
    """
    Per-class Dice score, excluding background (class 0).

    Args:
        pred: integer tensor [H, W] with class labels
        target: integer tensor [H, W] with class labels
        num_classes: total number of classes including background

    Returns:
        dict mapping class index (1..num_classes-1) to Dice score
    """
    scores = {}
    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            scores[c] = 1.0 if target_c.sum() == 0 else 0.0
        else:
            scores[c] = (2 * intersection / union).item()
    return scores


def compute_hausdorff_95(pred, target, spacing=1.0, max_points=2000):
    """
    95th percentile Hausdorff distance between binary masks.

    Args:
        pred: binary tensor [H, W]
        target: binary tensor [H, W]
        spacing: pixel spacing (default 1.0)
        max_points: subsample point clouds if larger (for speed)

    Returns:
        float: HD95 distance
    """
    pred_pts = torch.nonzero(pred).float()
    target_pts = torch.nonzero(target).float()

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float("inf") if len(pred_pts) != len(target_pts) else 0.0

    if len(pred_pts) > max_points:
        idx = torch.randperm(len(pred_pts))[:max_points]
        pred_pts = pred_pts[idx]
    if len(target_pts) > max_points:
        idx = torch.randperm(len(target_pts))[:max_points]
        target_pts = target_pts[idx]

    d_pred_to_target = torch.cdist(pred_pts, target_pts).min(dim=1).values
    d_target_to_pred = torch.cdist(target_pts, pred_pts).min(dim=1).values

    hd95_fwd = torch.quantile(d_pred_to_target, 0.95).item() * spacing
    hd95_bwd = torch.quantile(d_target_to_pred, 0.95).item() * spacing
    return max(hd95_fwd, hd95_bwd)
