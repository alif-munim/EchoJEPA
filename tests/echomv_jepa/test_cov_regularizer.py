"""Tests for the covariance regularizer (Arm A)."""

from __future__ import annotations

import torch

from src.models.echomv_jepa import covariance_penalty


def test_cov_zero_on_identity_rows():
    """If every row is identical, the centered matrix is zero → cov = 0."""
    h = torch.ones(16, 8)
    l_cov, l_var = covariance_penalty(h, var_floor=0.0)
    assert l_cov.item() == 0.0
    assert l_var.item() == 0.0


def test_cov_nonzero_on_correlated_dims():
    """Two perfectly-correlated dims should show large off-diag covariance."""
    N = 128
    base = torch.randn(N, 1)
    h = torch.cat([base, base, base * 0.5, base * -0.7], dim=-1)  # all 4 corr
    l_cov, _ = covariance_penalty(h, var_floor=0.0)
    assert l_cov.item() > 0.01, f"expected significant cov, got {l_cov.item()}"


def test_cov_small_on_independent_dims():
    """Independent Gaussian dims → off-diag ~ 0."""
    torch.manual_seed(0)
    N, D = 4096, 4
    h = torch.randn(N, D)
    l_cov, _ = covariance_penalty(h, var_floor=0.0)
    # Finite-sample noise; expect well below 0.05.
    assert l_cov.item() < 0.05


def test_var_floor_triggers_penalty_when_std_below():
    # All-constant data has std=0 < 1.0 → penalty is (1 - 0)^2 = 1.0 per dim.
    h = torch.full((32, 4), 0.5)
    _, l_var = covariance_penalty(h, var_floor=1.0)
    # mean over dims → 1.0
    assert abs(l_var.item() - 1.0) < 1e-5


def test_var_floor_no_penalty_when_std_above():
    torch.manual_seed(0)
    h = torch.randn(4096, 4) * 2.0  # std ~ 2.0 per dim
    _, l_var = covariance_penalty(h, var_floor=0.5)
    assert l_var.item() == 0.0  # std > floor → clamp = 0


def test_cov_nan_safety_on_n_lt_2():
    h = torch.randn(1, 8)
    l_cov, l_var = covariance_penalty(h)
    assert l_cov.item() == 0.0
    assert l_var.item() == 0.0


def test_cov_is_differentiable():
    torch.manual_seed(1)
    h = torch.randn(32, 4, requires_grad=True)
    l_cov, l_var = covariance_penalty(h, var_floor=0.2)
    loss = l_cov + l_var
    loss.backward()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
