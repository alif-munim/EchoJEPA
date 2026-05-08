"""Tests for _relational_infonce_with_hard_neg in both negative-set modes.

Covers the two supported modes of rel_negative_mode:

  hard_plus_batch (default, the method):
    - column 0 = positive, column 1 = hard negative, columns 2..B+1 = batch
    - labels = 0 everywhere
    - hard negative contributes to CE loss; gradient flows through y_hard

  no_hardneg (ablation):
    - same candidate layout, but column 1 logit is masked to -inf
    - hard column contributes zero to CE loss; no gradient through y_hard
    - diagnostics (rel_hard_neg_sim_mean, rel_pos_minus_hard_gap) are based
      on the real hard_dot cosine, so they remain finite and meaningful

Edge case: all batch negatives masked out (worst-case same-study batch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.vjepa_multiview.train import _relational_infonce_with_hard_neg  # noqa: E402


def _normed(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _make_batch(B: int = 8, D: int = 16):
    q = _normed(torch.randn(B, D))
    y_pos = _normed(torch.randn(B, D))
    y_hard = _normed(torch.randn(B, D))
    # All distinct study hashes so same-study masking is a no-op in this
    # baseline setup (we flip it on separately in dedicated tests).
    study_hashes = torch.arange(B, dtype=torch.long)
    return q, y_pos, y_hard, study_hashes


# --------------------------------------------------------------------- #
# Mode A: hard_plus_batch (method)
# --------------------------------------------------------------------- #

def test_hard_plus_batch_candidate_shape():
    q, y_pos, y_hard, sh = _make_batch(B=8, D=16)
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard,
        study_hashes=sh, tau=0.1,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=False,
    )
    # Shape + sanity
    assert out["rel_loss"].dim() == 0
    assert torch.isfinite(out["rel_loss"])
    # Real cosine diagnostic is finite
    assert torch.isfinite(out["rel_hard_neg_sim_mean"])
    assert torch.isfinite(out["rel_pos_minus_hard_gap"])


def test_hard_plus_batch_grad_flows_through_y_hard():
    # If the hard column participates in CE, gradient flows through y_hard.
    q, y_pos, y_hard, sh = _make_batch(B=4, D=8)
    q = q.requires_grad_(True)
    y_hard = y_hard.requires_grad_(True)
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard,
        study_hashes=sh, tau=0.1,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=False,
    )
    out["rel_loss"].backward()
    assert y_hard.grad is not None, "expected gradient on y_hard in hard_plus_batch mode"
    assert torch.isfinite(y_hard.grad).all()
    assert (y_hard.grad.abs().sum() > 0), "y_hard gradient is zero; hard column not wired into CE"


def test_hard_plus_batch_labels_are_zero():
    # Candidate column 0 is the positive; labels must always be 0.
    q, y_pos, y_hard, sh = _make_batch(B=4, D=8)
    # Make positives overwhelmingly larger than other candidates so argmax=0.
    y_pos = _normed(q.clone())
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard,
        study_hashes=sh, tau=0.01,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=False,
    )
    # With positives == q and small τ, the positive dominates so top-1 should be
    # very close to 1.0.
    assert out["rel_top1_with_hard"].item() > 0.99


# --------------------------------------------------------------------- #
# Mode B: no_hardneg (ablation)
# --------------------------------------------------------------------- #

def test_no_hardneg_hard_column_masked_to_neg_inf():
    # We can't inspect the internal logits directly; instead verify behavior:
    # argmax can never land on column 1 because it's -inf. Construct a case
    # where y_hard would otherwise dominate (q ≈ y_hard), and confirm the
    # top-1 argmax is not column 1.
    torch.manual_seed(42)
    B, D = 8, 16
    q = _normed(torch.randn(B, D))
    y_hard = _normed(q.clone() + 1e-3 * torch.randn(B, D))  # y_hard ≈ q
    # Random positives and batch so everything else is less correlated.
    y_pos = _normed(torch.randn(B, D))
    sh = torch.arange(B, dtype=torch.long)

    out_hpb = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=sh, tau=0.05,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=False,
    )
    out_nhn = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=sh, tau=0.05,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=True,
    )
    # In hard_plus_batch mode, hard-neg dominates → high loss, top1 is low
    # (argmax picks column 1 often, but label=0).
    # In no_hardneg mode, hard column is -inf so softmax ignores it → loss
    # is strictly lower than hard_plus_batch here (by construction).
    assert out_nhn["rel_loss"].item() < out_hpb["rel_loss"].item()


def test_no_hardneg_diagnostics_are_finite_and_real():
    # The critical bug we're guarding against: hard_neg_sim_mean / gap_hard
    # must be real cosine-scale scalars, NOT -inf or dtype-min.
    q, y_pos, y_hard, sh = _make_batch(B=8, D=16)
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=sh, tau=0.1,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=True,
    )
    hn = out["rel_hard_neg_sim_mean"].item()
    gap = out["rel_pos_minus_hard_gap"].item()
    # Real cosine / τ is bounded roughly within [-1/τ, 1/τ]. Anything past
    # that magnitude (like dtype_min / τ ≈ -3.4e38 for bf16 min at τ=0.1)
    # signals the bug.
    assert abs(hn) < 100.0, f"hard_neg_sim_mean polluted (|{hn}|>100)"
    assert abs(gap) < 100.0, f"rel_pos_minus_hard_gap polluted (|{gap}|>100)"
    # And plain finiteness:
    assert torch.isfinite(torch.tensor(hn))
    assert torch.isfinite(torch.tensor(gap))


def test_no_hardneg_no_gradient_through_y_hard():
    # Verify the core ablation contract: y_hard receives no gradient from
    # the contrastive loss when column 1 is masked to -inf. The current
    # implementation rebinds `hard_logit` to `torch.full_like(...)` (a
    # fresh leaf), so the autograd subgraph through `y_hard` is pruned
    # and `y_hard.grad` stays None after backward(). Accept either
    # outcome: grad is None, or grad exists and is all zeros. Reject
    # nonzero gradients.
    q, y_pos, y_hard, sh = _make_batch(B=4, D=8)
    q = q.requires_grad_(True)
    y_hard = y_hard.requires_grad_(True)
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard,
        study_hashes=sh, tau=0.1,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=True,
    )
    out["rel_loss"].backward()
    if y_hard.grad is None:
        # Stronger outcome: autograd didn't even build the subgraph.
        pass
    else:
        assert y_hard.grad.abs().max().item() < 1e-12, (
            f"expected zero gradient on y_hard in no_hardneg mode; "
            f"got max|grad|={y_hard.grad.abs().max().item()}"
        )


def test_no_hardneg_loss_finite_when_all_batch_negs_masked():
    # Edge case: same-study masking + small batch + study_hashes all equal
    # → the only valid candidate per row is column 0 (positive). In that
    # degenerate case, CE reduces to -log(softmax(pos_logit)) with no
    # negatives, which is mathematically 0 (softmax of a single valid
    # entry is 1.0). The loss must remain finite.
    B, D = 4, 8
    q = _normed(torch.randn(B, D))
    y_pos = _normed(torch.randn(B, D))
    y_hard = _normed(torch.randn(B, D))
    sh = torch.zeros(B, dtype=torch.long)  # all same study

    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=sh, tau=0.1,
        mask_same_study_batch_negatives=True,   # mask off-diagonals
        disable_hard_negative=True,             # AND disable hard neg
    )
    assert torch.isfinite(out["rel_loss"]), (
        "loss must be finite when only the positive column is valid"
    )
    # Loss should be ~0 because softmax of a single valid entry = 1.0.
    assert out["rel_loss"].item() < 1e-3, (
        f"all-negatives-masked edge case: expected ~0 loss, got "
        f"{out['rel_loss'].item()}"
    )
    # same_study_masked_count should record the B*(B-1) off-diagonal mask.
    assert int(out["same_study_masked_count"].item()) == B * (B - 1)


# --------------------------------------------------------------------- #
# Cross-mode sanity: same inputs produce different losses
# --------------------------------------------------------------------- #

def test_mode_toggle_changes_loss():
    q, y_pos, y_hard, sh = _make_batch(B=8, D=16)
    out_hpb = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=sh, tau=0.1,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=False,
    )
    out_nhn = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=sh, tau=0.1,
        mask_same_study_batch_negatives=False,
        disable_hard_negative=True,
    )
    # The two modes should not produce identical losses on random inputs.
    assert not torch.allclose(out_hpb["rel_loss"], out_nhn["rel_loss"])
