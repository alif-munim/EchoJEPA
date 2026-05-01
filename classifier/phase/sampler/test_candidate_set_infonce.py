"""Unit tests for candidate-set InfoNCE + _build_predictor_inputs.

Covers the train.py pieces that don't require GPU / real data:

  1. Candidate ordering: column 0 is positive, column 1 is hard negative.
  2. Labels are all zero.
  3. Same-study batch-negative masking zeros out off-diagonal same-study
     entries.
  4. Loss is finite when every batch negative is masked and only
     [positive, hard] remain in the candidate set.
  5. _build_predictor_inputs returns exactly four tensors (hard-capped
     arity — guards against metadata-shortcut regressions).
  6. Aligned-input sanity: when q == y_pos, argmax(logits) == 0.

No real encoder/video/data — pure torch tensor math.

Run:
    python classifier/phase/sampler/test_candidate_set_infonce.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
VJEPA_ROOT = HERE.parents[2]
if str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))

from app.vjepa_multiview.train import (  # noqa: E402
    _relational_infonce_with_hard_neg,
    _build_predictor_inputs,
)
from app.vjepa_multiview.phase_relational_head import PhaseRelationalHead  # noqa: E402


def _make_normed(B, D, seed):
    torch.manual_seed(seed)
    x = torch.randn(B, D)
    return F.normalize(x, dim=-1)


def test_candidate_ordering_column_0_is_positive():
    """If q[i] is aligned with y_pos[i], argmax(logits[i]) == 0."""
    B, D = 8, 32
    # Make q == y_pos (column 0 should strictly dominate all other columns).
    q = _make_normed(B, D, seed=0)
    y_pos = q.clone()
    # y_hard is unrelated random.
    y_hard = _make_normed(B, D, seed=1)
    # Distinct study ids so no batch masking kicks in.
    study = torch.arange(B, dtype=torch.long)

    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=study, tau=0.10,
        mask_same_study_batch_negatives=False,
    )
    # Reconstruct logits manually to inspect ordering.
    tau = 0.10
    pos = (q * y_pos).sum(-1, keepdim=True) / tau
    hard = (q * y_hard).sum(-1, keepdim=True) / tau
    batch = (q @ y_pos.t()) / tau
    batch.masked_fill_(torch.eye(B, dtype=torch.bool), float("-inf"))
    logits = torch.cat([pos, hard, batch], dim=1)
    assert logits.shape == (B, B + 2), f"shape {tuple(logits.shape)}"
    # Col 0 must be the maximum (q perfectly aligned with y_pos).
    argmax = logits.argmax(dim=1)
    assert (argmax == 0).all().item(), f"argmax {argmax.tolist()}"
    # Loss finite + nonzero (log B+2 when uniform — here it's small).
    assert torch.isfinite(out["rel_loss"]).all().item()
    # rel_top1_with_hard should be 1.0 with perfect alignment.
    assert float(out["rel_top1_with_hard"]) == 1.0
    print(f"[pass] column 0 is positive (argmax=0 for all B={B}); rel_top1=1.0")


def test_labels_all_zero():
    """The contract: every label is 0 (positive is column 0)."""
    # Reproduce logits + labels inside the loss function; verify label vector.
    B = 4
    q = _make_normed(B, 8, seed=2)
    y_pos = _make_normed(B, 8, seed=3)
    y_hard = _make_normed(B, 8, seed=4)
    study = torch.arange(B, dtype=torch.long)
    # Run through the loss and inspect via a small monkey-patch: we trust the
    # assertion inside the function (it asserts (labels == 0).all().item()).
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=study, tau=0.10,
        mask_same_study_batch_negatives=True,
    )
    assert torch.isfinite(out["rel_loss"]).all().item()
    print("[pass] labels=0 assertion inside loss function did not raise")


def test_same_study_batch_negative_masking():
    """Off-diagonal same-study entries in the batch block become -inf;
    hard negative column is never masked."""
    B, D = 4, 8
    q = _make_normed(B, D, seed=5)
    y_pos = _make_normed(B, D, seed=6)
    y_hard = _make_normed(B, D, seed=7)
    # Samples 0 and 1 are same-study; 2 and 3 are same-study.
    study = torch.tensor([100, 100, 200, 200], dtype=torch.long)
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=study, tau=0.10,
        mask_same_study_batch_negatives=True,
    )
    # The same_study_masked_count must equal 4 (two off-diag per study pair).
    masked = int(out["same_study_masked_count"].item())
    assert masked == 4, f"expected 4 masked entries, got {masked}"
    assert torch.isfinite(out["rel_loss"]).all().item()
    print(f"[pass] same-study batch-neg masking: {masked} entries masked to -inf, loss finite")


def test_all_batch_negatives_masked_finite_loss():
    """When every sample is same-study (study vector constant), every
    off-diagonal batch entry is masked. Only [positive, hard] remain —
    loss must still be finite."""
    B, D = 4, 8
    q = _make_normed(B, D, seed=8)
    y_pos = _make_normed(B, D, seed=9)
    y_hard = _make_normed(B, D, seed=10)
    # Everyone is same-study.
    study = torch.full((B,), 777, dtype=torch.long)
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=study, tau=0.10,
        mask_same_study_batch_negatives=True,
    )
    # All B*(B-1) off-diagonal entries masked:
    masked = int(out["same_study_masked_count"].item())
    assert masked == B * (B - 1), f"masked={masked} expected {B*(B-1)}"
    assert torch.isfinite(out["rel_loss"]).all().item(), (
        f"loss not finite: {out['rel_loss']}"
    )
    # Hard negative column must still contribute; otherwise logits for many
    # samples collapse to a single finite entry (positive only) and the
    # softmax is degenerate. Pos + hard = 2 real logits per row.
    print(
        f"[pass] all-batch-masked edge case: {masked} entries masked, "
        f"rel_loss={float(out['rel_loss']):.4f} (finite)"
    )


def test_build_predictor_inputs_arity():
    """Hard invariant: _build_predictor_inputs returns exactly 4 tensors.
    Never 5+. This guards against metadata-shortcut regressions."""
    meta = [
        {
            "clip_a_view": "A4C",
            "clip_b_view": "A4C",
            "target_phi_a": 0.10,
            "target_phi_b": 0.35,
            "study_id": "stud_1",
        },
        {
            "clip_a_view": "PLAX",
            "clip_b_view": "PSAX-AV",
            "target_phi_a": 0.50,
            "target_phi_b": 0.75,
            "study_id": "stud_2",
        },
    ]
    device = torch.device("cpu")
    result = _build_predictor_inputs(meta, device)
    assert isinstance(result, tuple)
    assert len(result) == 4, f"expected 4-tuple, got {len(result)}"
    va, vbp, dphi, studies = result
    assert va.shape == (2,) and va.dtype == torch.long
    assert vbp.shape == (2,) and vbp.dtype == torch.long
    assert dphi.shape == (2,) and dphi.dtype == torch.float32
    assert studies.shape == (2,) and studies.dtype == torch.long
    # Check values.
    # view_to_id: A4C=2, PLAX=4, PSAX-AV=5
    assert va.tolist() == [2, 4]
    assert vbp.tolist() == [2, 5]
    # Δφ = (0.35 - 0.10) % 1 = 0.25; (0.75 - 0.50) % 1 = 0.25
    assert torch.allclose(dphi, torch.tensor([0.25, 0.25]), atol=1e-4)
    print("[pass] _build_predictor_inputs returns exactly 4 tensors with correct shapes")


def test_gradient_flow_through_head():
    """Head must receive gradients; teacher projector input must be detached."""
    B, D = 4, 1024
    head = PhaseRelationalHead(embed_dim=D, rel_dim=128, hidden_dim=256)
    # Synthetic pooled latents.
    c_a = torch.randn(B, D, requires_grad=True)
    h_pos_raw = torch.randn(B, D)           # simulating teacher output
    h_neg_raw = torch.randn(B, D)
    # Emulate the caller: detach before target projector.
    y_pos_raw = h_pos_raw.detach()
    y_neg_raw = h_neg_raw.detach()
    # Forward
    view_a = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    view_bp = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    dphi = torch.tensor([0.10, 0.25, 0.125, 0.5])
    q_pre = head.query(c_a, view_a, view_bp, dphi)
    y_pos_pre = head.target(y_pos_raw)
    y_hard_pre = head.target(y_neg_raw)
    q = F.normalize(q_pre.float(), dim=-1)
    y_pos = F.normalize(y_pos_pre.float(), dim=-1)
    y_hard = F.normalize(y_hard_pre.float(), dim=-1)
    study = torch.arange(B, dtype=torch.long)
    out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard, study_hashes=study, tau=0.10,
    )
    out["rel_loss"].backward()
    # Head params must have grads.
    for n, p in head.named_parameters():
        assert p.grad is not None, f"param {n} has no grad"
        assert torch.isfinite(p.grad).all().item(), f"param {n} has non-finite grads"
    # c_a must have grad (student-side signal).
    assert c_a.grad is not None and torch.isfinite(c_a.grad).all().item()
    # y_pos_raw / y_neg_raw are detached before entering target_proj, so they
    # are leaves with requires_grad=False and never see grads.
    assert not y_pos_raw.requires_grad
    assert not y_neg_raw.requires_grad
    print("[pass] gradients flow to head + c_a; detached teacher inputs never see grads")


def main():
    print("[INFO] Running candidate-set InfoNCE unit tests...")
    test_candidate_ordering_column_0_is_positive()
    test_labels_all_zero()
    test_same_study_batch_negative_masking()
    test_all_batch_negatives_masked_finite_loss()
    test_build_predictor_inputs_arity()
    test_gradient_flow_through_head()
    print("\nALL UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
