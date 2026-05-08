"""token_set_infonce_with_hard_neg — correctness + cross-view safety."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.token_relational_loss import (  # noqa: E402
    token_set_infonce_with_hard_neg,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _normed(x):
    return F.normalize(x, dim=-1)


def test_finite_loss_and_shapes():
    B, K, D = 4, 8, 32
    q = _normed(torch.randn(B, K, D))
    p = _normed(torch.randn(B, K, D))
    h = _normed(torch.randn(B, K, D))
    hashes = torch.arange(B, dtype=torch.long)
    out = token_set_infonce_with_hard_neg(q, p, h, hashes, tau=0.1)
    assert torch.isfinite(out["token_rel_loss"])
    assert 0.0 <= out["token_rel_top1_with_hard"].item() <= 1.0
    assert out["token_rel_valid_rows"].item() == B


def test_label_col0_positive():
    """When q == y_pos exactly, column 0 must be argmax for every row."""
    B, K, D = 5, 6, 16
    q = _normed(torch.randn(B, K, D))
    p = q.clone()  # exact match: same token positions, same directions
    h = _normed(torch.randn(B, K, D))
    hashes = torch.arange(B, dtype=torch.long)
    out = token_set_infonce_with_hard_neg(q, p, h, hashes, tau=0.1)
    # With identical q and p, self-row positive is uniquely highest.
    assert out["token_rel_top1_with_hard"].item() == 1.0
    assert out["token_rel_pos_minus_hard_gap"].item() > 0.0


def test_same_study_batch_negatives_are_masked():
    """Same-study off-diagonal batch-negatives shouldn't contribute.
    Flip two rows into the same study — the batch-neg cell must be -inf."""
    B, K, D = 4, 4, 16
    q = _normed(torch.randn(B, K, D))
    p = _normed(torch.randn(B, K, D))
    h = _normed(torch.randn(B, K, D))
    # rows 0 and 1 are same study
    hashes = torch.tensor([7, 7, 1, 2], dtype=torch.long)
    out = token_set_infonce_with_hard_neg(
        q, p, h, hashes, tau=0.1, mask_same_study_batch_negatives=True
    )
    # Off-diagonal same-study pairs: (0,1) and (1,0). Count = 2.
    assert out["token_rel_same_study_masked_count"].item() == 2.0


def test_cross_view_token_set_matching_no_index_assumption():
    """A scrambled token-order positive should still be recoverable via
    token-set matching. We construct: y_pos = p[random permutation along K].
    Set-wise logsumexp should still assign high similarity to self-row."""
    B, K, D = 3, 12, 16
    q = _normed(torch.randn(B, K, D))
    # Permute tokens per row on the positive, same content but different order.
    p_list = []
    for b in range(B):
        perm = torch.randperm(K)
        p_list.append(q[b, perm, :].clone())
    p = torch.stack(p_list, dim=0)
    h = _normed(torch.randn(B, K, D))
    hashes = torch.arange(B, dtype=torch.long)
    out = token_set_infonce_with_hard_neg(q, p, h, hashes, tau=0.1)
    # Set matching: self-row still wins even though token index order differs.
    assert out["token_rel_top1_with_hard"].item() == 1.0


def test_hard_neg_column_not_masked():
    """Even if all batch negatives are masked (single-row batch),
    the hard negative column must remain and the loss stay finite."""
    B, K, D = 2, 4, 8
    q = _normed(torch.randn(B, K, D))
    p = _normed(torch.randn(B, K, D))
    h = _normed(torch.randn(B, K, D))
    # All rows same study → all off-diagonals masked, only self+hard remain.
    hashes = torch.full((B,), 42, dtype=torch.long)
    out = token_set_infonce_with_hard_neg(q, p, h, hashes, tau=0.1)
    assert torch.isfinite(out["token_rel_loss"])
