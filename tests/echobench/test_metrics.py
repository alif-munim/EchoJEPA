"""Tests for echobench.metrics module."""

import numpy as np
import torch
import pytest

from echobench.metrics.regression import compute_regression_metrics
from echobench.metrics.segmentation import compute_dice, compute_hausdorff_95


class TestRegressionMetrics:
    def test_perfect_predictions(self):
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_regression_metrics(preds, labels)
        assert m["mae"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)
        assert m["pearson_r"] == pytest.approx(1.0)

    def test_known_mae(self):
        labels = np.array([1.0, 2.0, 3.0])
        preds = np.array([2.0, 3.0, 4.0])
        m = compute_regression_metrics(preds, labels)
        assert m["mae"] == pytest.approx(1.0)

    def test_unnormalize_zscore(self):
        labels = np.array([50.0, 60.0, 70.0])
        # Predictions in z-score space: mean=60, std=10
        preds_z = np.array([-1.0, 0.0, 1.0])
        m = compute_regression_metrics(preds_z, labels, target_mean=60.0, target_std=10.0)
        assert m["mae"] == pytest.approx(0.0)

    def test_empty_input(self):
        m = compute_regression_metrics(np.array([]), np.array([]))
        assert np.isnan(m["mae"])

    def test_2d_preds(self):
        labels = np.array([1.0, 2.0, 3.0])
        preds = np.array([[1.0], [2.0], [3.0]])
        m = compute_regression_metrics(preds, labels)
        assert m["mae"] == pytest.approx(0.0)


class TestDice:
    def test_perfect_segmentation(self):
        pred = torch.tensor([[0, 1, 1], [0, 2, 2], [0, 3, 3]])
        target = torch.tensor([[0, 1, 1], [0, 2, 2], [0, 3, 3]])
        scores = compute_dice(pred, target, num_classes=4)
        assert scores[1] == pytest.approx(1.0)
        assert scores[2] == pytest.approx(1.0)
        assert scores[3] == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = torch.tensor([[1, 1], [1, 1]])
        target = torch.tensor([[2, 2], [2, 2]])
        scores = compute_dice(pred, target, num_classes=3)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(0.0)

    def test_both_empty(self):
        pred = torch.zeros(4, 4, dtype=torch.long)
        target = torch.zeros(4, 4, dtype=torch.long)
        scores = compute_dice(pred, target, num_classes=2)
        # Class 1 is absent in both → Dice = 1.0 by convention
        assert scores[1] == pytest.approx(1.0)


class TestHausdorff:
    def test_identical_masks(self):
        mask = torch.zeros(32, 32, dtype=torch.bool)
        mask[10:20, 10:20] = True
        hd = compute_hausdorff_95(mask, mask)
        assert hd == pytest.approx(0.0)

    def test_separated_masks(self):
        pred = torch.zeros(32, 32, dtype=torch.bool)
        target = torch.zeros(32, 32, dtype=torch.bool)
        pred[0:5, 0:5] = True
        target[25:30, 25:30] = True
        hd = compute_hausdorff_95(pred, target)
        assert hd > 0

    def test_empty_pred(self):
        pred = torch.zeros(16, 16, dtype=torch.bool)
        target = torch.ones(16, 16, dtype=torch.bool)
        hd = compute_hausdorff_95(pred, target)
        assert hd == float("inf")

    def test_both_empty(self):
        pred = torch.zeros(16, 16, dtype=torch.bool)
        target = torch.zeros(16, 16, dtype=torch.bool)
        hd = compute_hausdorff_95(pred, target)
        assert hd == 0.0
