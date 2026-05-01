"""Phase-matched multi-view JEPA training app.

Adds a cross-view latent-prediction loss alongside the standard V-JEPA
intraview objective. The student/context encoder processes clip_a; an
EMA teacher processes both clip_a (for intraview) and clip_b (for
crossview); the predictor operates once on the student's clip_a context
and its predicted target latents are compared to BOTH teacher outputs.

    L_total = L_intraview + lambda_crossview * L_crossview

This module is intentionally small and does NOT duplicate the full
optimizer/scheduler/checkpoint machinery in ``app/vjepa/train.py``.
What lives here:

  * ``build_clip_pair_tensors`` — splits a VideoGroupDataset pair
    batch into two clip tensors suitable for ``encoder``/``target_encoder``.
  * ``forward_intraview_and_crossview`` — runs the student once, the
    teacher twice, the predictor once, and returns the two losses
    together with the total loss. All grads flow through the student
    encoder and predictor; the teacher is ``torch.no_grad`` and shares
    EMA parameters with the student, so no gradient flows back through
    it.
  * ``make_phase_matched_collator`` — wraps the existing V-JEPA
    ``MaskCollator`` so masks are generated once per batch from clip_a
    and reused for clip_b (shared-mask geometry; see
    ``classifier/phase/sampler/CROSS_VIEW_DESIGN.md``).
  * ``PhaseMatchedRefreshGuard`` — raises if the training loop forgets
    to call ``builder.refresh_epoch(e)`` before iterating the loader.
  * ``main(args)`` — a minimal entry point wiring config -> model ->
    data -> one-or-N steps. Full-scale training (LR warmup, EMA
    scheduling, checkpoint resume across ranks) is deferred; for the
    pilot we can drive this from a short launcher.

Only ``app/vjepa_multiview/`` is new. No changes to ``app/vjepa/``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from src.masks.multiseq_multiblock3d import MaskCollator
from src.masks.utils import apply_masks

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Batch plumbing
# --------------------------------------------------------------------------- #

@dataclass
class PairBatch:
    """One pair-mini-batch passed to ``forward_intraview_and_crossview``.

    All tensors are already on-device. ``masks_enc`` and ``masks_pred``
    are lists-of-lists matching V-JEPA conventions (outer = fpc index,
    inner = mask-generator index), shared across all clips in the batch.
    ``phase_metadata`` is a list of per-sample dicts carrying
    ``anchor_frame_a``, ``target_phi_a``, etc., for logging only — not
    consumed by the loss.

    For group_size=3 (phase_relational + intraview_only), ``clip_b`` is the
    POSITIVE target (= clip_b_pos), and ``clip_b_neg`` is populated with
    the same-study wrong-phase hard negative. For group_size=2
    (smooth_l1), ``clip_b_neg`` is ``None`` and the existing code paths
    are byte-identical.
    """

    clip_a: list[torch.Tensor]          # list over fpc, each [B, C, T, H, W]
    clip_b: list[torch.Tensor]          # clip_b_pos in 3-clip mode; unchanged in smooth_l1
    masks_enc: list[list[torch.Tensor]]  # [fpc][mask_i] -> [B, N_ctx]
    masks_pred: list[list[torch.Tensor]]  # [fpc][mask_i] -> [B, N_tgt]
    phase_metadata: list[dict[str, Any]]
    clip_b_neg: Optional[list[torch.Tensor]] = None   # list over fpc; None in smooth_l1


def build_clip_pair_tensors(
    collated_batch: tuple,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Given one V-JEPA ``MaskCollator`` output tuple
    ``(collated_batch, masks_enc, masks_pred)``, extract clip_a (view_0)
    and clip_b (view_1) tensors.

    ``collated_batch`` is the ``default_collate`` of a list of per-sample
    tuples ``(segs, label, clip_indices_out, slot_mask, [meta])``. With
    ``VideoGroupDataset(group_size=2)``, ``segs`` is a two-element list
    of per-view tensors ``[C, T, H, W]`` (after transforms). After
    ``default_collate`` this becomes a list of length 2 of stacked
    tensors ``[B, C, T, H, W]``.
    """
    segs = collated_batch[0]
    if not isinstance(segs, (list, tuple)) or len(segs) < 2:
        raise ValueError(
            f"multi-view collator expected segs len >= 2, got {type(segs).__name__}"
        )
    clip_a = segs[0]
    clip_b = segs[1]
    # The single-view path wraps into a list-over-fpc; we do the same so
    # downstream code (masks/encoder) can stay unchanged.
    return ([clip_a.to(device, non_blocking=True)],
            [clip_b.to(device, non_blocking=True)])


def _extract_multiview_clips(
    collated_batch: tuple,
    device: torch.device,
    objective: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], Optional[list[torch.Tensor]]]:
    """Objective-aware clip extraction.

    Returns ``(clip_a, clip_b, clip_b_neg)`` where each element is a
    list-over-fpc of ``[B, C, T, H, W]`` tensors on ``device``.

    - ``smooth_l1``:       expects segs of length 2; clip_b_neg is None.
    - ``intraview_only``:  expects segs of length 3; clip_b and
                           clip_b_neg are loaded/validated but the
                           training loop ignores them in the loss.
    - ``phase_relational``: expects segs of length 3; all three clips
                           reach the loss.

    Fails loudly on shape mismatch so sampler/dataset misconfig surfaces
    immediately rather than producing a silently-wrong loss.
    """
    segs = collated_batch[0]
    if not isinstance(segs, (list, tuple)):
        raise ValueError(
            f"multi-view collator expected segs list/tuple, got {type(segs).__name__}"
        )
    if objective == "smooth_l1":
        if len(segs) < 2:
            raise ValueError(
                f"multiview_objective='smooth_l1' expects segs len >= 2; got {len(segs)}"
            )
        clip_a = [segs[0].to(device, non_blocking=True)]
        clip_b = [segs[1].to(device, non_blocking=True)]
        return clip_a, clip_b, None
    if objective in ("intraview_only", "phase_relational"):
        if len(segs) != 3:
            raise ValueError(
                f"multiview_objective={objective!r} expects segs of length 3 "
                f"(clip_a, clip_b_pos, clip_b_neg); got {len(segs)}"
            )
        clip_a = [segs[0].to(device, non_blocking=True)]
        clip_b = [segs[1].to(device, non_blocking=True)]
        clip_b_neg = [segs[2].to(device, non_blocking=True)]
        return clip_a, clip_b, clip_b_neg
    raise ValueError(
        f"unknown multiview_objective={objective!r}; "
        f"want one of smooth_l1 | intraview_only | phase_relational"
    )


def extract_pair_metadata(collated_batch: tuple) -> list[dict]:
    """Pull the per-sample pair metadata list from the collated batch.

    ``VideoGroupDataset`` appends ``meta`` as the 5th element when
    anchors are installed. After ``default_collate`` the field becomes
    a dict of stacked tensors/lists — default_collate chooses strings -> list,
    numeric types -> tensor, mixed -> list-of-values. We return a
    per-sample list of dicts so the training loop gets a predictable
    shape.
    """
    if len(collated_batch) < 5:
        return []
    meta_collated = collated_batch[4]
    if not isinstance(meta_collated, dict) or not meta_collated:
        return []
    keys = list(meta_collated.keys())
    n = None
    for v in meta_collated.values():
        try:
            n = len(v)
            break
        except TypeError:
            continue
    if n is None:
        return []
    out = []
    for i in range(n):
        row = {}
        for k in keys:
            v = meta_collated[k]
            try:
                item = v[i]
                if hasattr(item, "item"):
                    item = item.item()
                row[k] = item
            except Exception:
                row[k] = None
        out.append(row)
    return out


def summarize_pair_metadata(meta_list: list[dict]) -> dict[str, float]:
    """Mean/median/p90 stats over a batch for per-batch logging. Keys
    produced here match the training-loop log columns."""
    import numpy as np
    if not meta_list:
        return {}

    def _as_arr(key: str) -> np.ndarray:
        vals = [m.get(key) for m in meta_list if m.get(key) is not None]
        vals = [float(v) for v in vals if isinstance(v, (int, float)) and not np.isnan(float(v))]
        return np.asarray(vals, dtype=np.float64) if vals else np.asarray([], dtype=np.float64)

    def _stats(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.9)),
        }

    out: dict[str, float] = {}
    out["sampling_mode"] = meta_list[0].get("sampling_mode", "unknown")
    for key in ("circular_phase_diff", "clip_a_phase_error", "clip_b_phase_error",
                "source_span_cycles_a", "source_span_cycles_b"):
        for stat, v in _stats(_as_arr(key)).items():
            out[f"{key}.{stat}"] = v

    hr_a = _as_arr("clip_a_hr_metadata")
    hr_b = _as_arr("clip_b_hr_metadata")
    if hr_a.size and hr_b.size and hr_a.size == hr_b.size:
        hr_diff = np.abs(hr_a - hr_b)
        out["hr_diff.mean"] = float(hr_diff.mean())
        out["hr_diff.p90"] = float(np.quantile(hr_diff, 0.9))
    else:
        out["hr_diff.mean"] = float("nan")
        out["hr_diff.p90"] = float("nan")

    views_a = [m.get("clip_a_view") for m in meta_list]
    views_b = [m.get("clip_b_view") for m in meta_list]
    if any(v is not None for v in views_a) and any(v is not None for v in views_b):
        same = sum(1 for a, b in zip(views_a, views_b) if a is not None and a == b)
        diff = sum(1 for a, b in zip(views_a, views_b) if a is not None and b is not None and a != b)
        total = max(1, same + diff)
        out["same_view_frac"] = float(same) / total
        out["diff_view_frac"] = float(diff) / total
    else:
        out["same_view_frac"] = float("nan")
        out["diff_view_frac"] = float("nan")

    # Per-batch same_view / same_family / cross_family mixture (uses
    # view_pair_class when view labels are present). Import is deferred to
    # first call so sys.path arranged by data_manager is in effect.
    view_pair_class_fn = _lookup_view_pair_class()
    if view_pair_class_fn is not None:
        counts = {"same_view": 0, "same_family": 0, "cross_family": 0}
        n_with_views = 0
        for va, vb in zip(views_a, views_b):
            va = va if isinstance(va, str) and va else None
            vb = vb if isinstance(vb, str) and vb else None
            if va is None and vb is None:
                continue
            n_with_views += 1
            counts[view_pair_class_fn(va, vb)] += 1
        tot = max(1, n_with_views)
        out["same_view_frac"] = counts["same_view"] / tot
        out["same_family_frac"] = counts["same_family"] / tot
        out["cross_family_frac"] = counts["cross_family"] / tot

    return out


_view_pair_class_fn = None
_view_pair_class_fn_attempted = False
def _lookup_view_pair_class():
    """Resolve the view_pair_class function, trying both direct import and
    a fallback sys.path insertion for classifier/phase/sampler relative to
    any sys.path entry we can identify."""
    global _view_pair_class_fn, _view_pair_class_fn_attempted
    if _view_pair_class_fn is not None or _view_pair_class_fn_attempted:
        return _view_pair_class_fn
    _view_pair_class_fn_attempted = True
    import sys, os
    try:
        from phase_matched_sampler import view_pair_class as _fn
        _view_pair_class_fn = _fn
        return _fn
    except ImportError:
        pass
    for p in list(sys.path):
        candidate = os.path.join(p, "classifier", "phase", "sampler")
        if os.path.isfile(os.path.join(candidate, "phase_matched_sampler.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            try:
                from phase_matched_sampler import view_pair_class as _fn
                _view_pair_class_fn = _fn
                return _fn
            except Exception:
                pass
    return None


def _run_frame_count_guard(pair_df, n: int, log) -> None:
    """Open the first ``n`` MP4 URIs from the pair_df and verify the on-disk
    frame count matches ``clip_a_n_frames`` / ``clip_b_n_frames`` recorded
    when the phase annotations were built. Raises ``AssertionError`` on any
    mismatch. Called at most once per run (first epoch, rank 0).
    """
    import io
    try:
        import boto3
        from decord import VideoReader, cpu
    except ImportError as e:
        raise RuntimeError(f"frame-count guard needs boto3+decord: {e}")

    s3 = boto3.client("s3", region_name="us-west-2")
    n = max(1, min(int(n), len(pair_df)))
    log.info(f"[frame-guard] verifying frame counts on first n={n} pair rows")
    problems = []
    for i in range(n):
        row = pair_df.iloc[i]
        for side in ("a", "b"):
            uri = row[f"view_{0 if side=='a' else 1}"]
            expected = int(row[f"clip_{side}_n_frames"])
            try:
                if uri.startswith("s3://"):
                    bucket, key = uri[5:].split("/", 1)
                    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    vr = VideoReader(io.BytesIO(data), num_threads=-1, ctx=cpu(0))
                else:
                    vr = VideoReader(uri, num_threads=-1, ctx=cpu(0))
                got = len(vr)
            except Exception as e:
                problems.append(f"row={i} side={side} uri={uri}: open-error {e}")
                continue
            if got != expected:
                problems.append(
                    f"row={i} side={side} uri={uri}: expected={expected} got={got}"
                )
    if problems:
        msg = "[frame-guard] FAIL:\n  " + "\n  ".join(problems[:20])
        log.error(msg)
        raise AssertionError(msg)
    log.info(f"[frame-guard] OK: {n} rows × 2 sides, all frame counts match")


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def _jepa_loss_fn(
    z: list[list[torch.Tensor]],
    h: list[torch.Tensor],
    masks_pred: list[list[torch.Tensor]],
    loss_exp: float = 1.0,
) -> torch.Tensor:
    """V-JEPA smooth-L^p loss between predictor outputs ``z`` and teacher
    target latents ``h`` (masked by ``masks_pred``). Mirrors the inner
    ``loss_fn`` inside ``app/vjepa/train.py``.
    """
    h_masked = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_pred)]
    total, n = 0.0, 0
    for zi, hi in zip(z, h_masked):
        for zij, hij in zip(zi, hi):
            total = total + torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp
            n += 1
    return total / max(n, 1)


def forward_intraview_and_crossview(
    pair: PairBatch,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    *,
    lambda_crossview: float = 0.25,
    use_intraview_loss: bool = True,
    use_crossview_loss: bool = True,
    loss_exp: float = 1.0,
    log_mask_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Run one forward pass and compute the combined loss.

    Shape contract (for a single fpc entry, ViT-L/16 defaults):
      clip_a, clip_b:  [B, C, T, H, W]              e.g. [B, 3, 16, 224, 224]
      h_a, h_b:        [B, N, D_enc]                e.g. [B, 1568, 1024]
                         where N = (T//tubelet)*(H//p)*(W//p)
      z:               list-of-list of [B*|m_x|, N_tgt, D_enc]
                         (predictor output per mask-generator per fpc)
      masks_enc[fpc]:  list of LongTensor[B, N_ctx]
      masks_pred[fpc]: list of LongTensor[B, N_tgt]

    The predictor is invoked ONCE on the student's clip_a context; its
    output is compared to the teacher's clip_a latents (intraview) and
    to the teacher's clip_b latents (crossview). Shared masks guarantee
    the two teacher branches produce tensors of matching shape.

    Returns dict with keys:
      intraview_loss, crossview_loss, total_loss, z_shapes,
      h_a_shapes, h_b_shapes, mask_diag (optional).
    """
    # --- teacher ---
    with torch.no_grad():
        h_a = target_encoder(pair.clip_a)
        h_a = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_a]
        h_b = target_encoder(pair.clip_b)
        h_b = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_b]

    # --- student context + predictor ---
    z = encoder(pair.clip_a, pair.masks_enc)
    z = predictor(z, pair.masks_enc, pair.masks_pred, delta_phi=None)

    # --- losses ---
    intraview = None
    crossview = None
    if use_intraview_loss:
        intraview = _jepa_loss_fn(z, h_a, pair.masks_pred, loss_exp=loss_exp)
    if use_crossview_loss:
        crossview = _jepa_loss_fn(z, h_b, pair.masks_pred, loss_exp=loss_exp)

    total = torch.zeros(1, device=pair.clip_a[0].device).squeeze()
    if intraview is not None:
        total = total + intraview
    if crossview is not None:
        total = total + lambda_crossview * crossview

    out = {
        "intraview_loss": intraview if intraview is not None
        else torch.zeros((), device=total.device),
        "crossview_loss": crossview if crossview is not None
        else torch.zeros((), device=total.device),
        "total_loss": total,
        "h_a_shapes": [tuple(hi.shape) for hi in h_a],
        "h_b_shapes": [tuple(hi.shape) for hi in h_b],
        "z_shapes": [[tuple(zij.shape) for zij in zi] for zi in z],
    }

    if log_mask_diagnostics:
        # For the first few batches we log where the target mask sits
        # temporally, relative to the anchor position. Mask indices are
        # flat over (T/tubelet, H/patch, W/patch).
        diag = []
        for i, pred_masks_for_fpc in enumerate(pair.masks_pred):
            for mgi, m in enumerate(pred_masks_for_fpc):
                # m: [B, N_tgt] long. Spatial stride = ceil(H/p)*ceil(W/p).
                # Without knowing exact grid dims here we log the raw
                # temporal-bucket histogram as (idx // HW) values.
                # Caller can post-process with grid dims from mask_collator.
                diag.append({
                    "fpc_idx": i,
                    "mask_gen_idx": mgi,
                    "shape": tuple(m.shape),
                    "dtype": str(m.dtype),
                    "raw_min": int(m.min().item()),
                    "raw_max": int(m.max().item()),
                })
        out["mask_diag"] = diag

    return out


# --------------------------------------------------------------------------- #
# Intraview-only + phase-relational forward paths (group_size=3)
# --------------------------------------------------------------------------- #

def forward_intraview_only(
    pair: PairBatch,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    *,
    loss_exp: float = 1.0,
) -> dict:
    """Standard V-JEPA intraview loss on clip_a only.

    ``pair.clip_b`` and ``pair.clip_b_neg`` are loaded for sampler
    eligibility parity with the phase_relational run but do not enter
    the loss. Byte-identical gradient path to single-view V-JEPA on
    clip_a.
    """
    with torch.no_grad():
        h_a = target_encoder(pair.clip_a)
        h_a = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_a]

    z = encoder(pair.clip_a, pair.masks_enc)
    z = predictor(z, pair.masks_enc, pair.masks_pred, delta_phi=None)
    intraview = _jepa_loss_fn(z, h_a, pair.masks_pred, loss_exp=loss_exp)
    total = intraview + torch.zeros((), device=intraview.device)
    return {
        "intraview_loss": intraview,
        "crossview_loss": torch.zeros((), device=intraview.device),
        "total_loss": total,
        "multiview_objective": "intraview_only",
    }


def _build_predictor_inputs(
    meta_list: list[dict],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the *exact* four tensors the relational branch consumes.

    Returns ``(view_a_ids, view_b_pos_ids, delta_phase_pos, study_hashes)``.
    Nothing else. The head's ``.query`` signature accepts only the first
    three; ``study_hashes`` goes to the InfoNCE same-study masker, not
    the predictor.

    Hard-capped arity — a smoke test asserts the return tuple has
    exactly four elements, so future callers can't sneak HR / quality /
    phase_error / view_b_neg / absolute phase / study_id strings into
    the head.
    """
    from app.vjepa_multiview.phase_relational_head import view_to_id

    B = len(meta_list)

    def _phase_val(m, k):
        v = m.get(k)
        if v is None or (isinstance(v, float) and v != v):
            return 0.0
        return float(v)

    view_a_ids = torch.tensor(
        [view_to_id(m.get("clip_a_view")) for m in meta_list],
        device=device, dtype=torch.long,
    )
    view_b_pos_ids = torch.tensor(
        [view_to_id(m.get("clip_b_view")) for m in meta_list],
        device=device, dtype=torch.long,
    )
    # Δφ_pos = (target_phi_b - target_phi_a) mod 1. Already computed in
    # the sampler as ``circular_phase_diff`` (wrapped magnitude) but
    # here we want signed delta, so recompute.
    delta_phase_pos = torch.tensor(
        [(_phase_val(m, "target_phi_b") - _phase_val(m, "target_phi_a")) % 1.0
         for m in meta_list],
        device=device, dtype=torch.float32,
    )
    # Deterministic study-id hashes for same-study masking in InfoNCE.
    # Any collision is benign — masking is a safety net, not a hard
    # dependency. Use Python's ``hash`` (deterministic within a process
    # but not across processes) to map string study_ids to int64.
    study_hashes = torch.tensor(
        [hash(str(m.get("study_id", ""))) for m in meta_list],
        device=device, dtype=torch.long,
    )
    assert view_a_ids.shape == (B,)
    assert view_b_pos_ids.shape == (B,)
    assert delta_phase_pos.shape == (B,)
    assert study_hashes.shape == (B,)
    return view_a_ids, view_b_pos_ids, delta_phase_pos, study_hashes


def _relational_infonce_with_hard_neg(
    q: torch.Tensor,
    y_pos: torch.Tensor,
    y_hard: torch.Tensor,
    study_hashes: torch.Tensor,
    tau: float,
    mask_same_study_batch_negatives: bool = True,
) -> dict:
    """Candidate-set InfoNCE with a mandatory same-study wrong-phase hard negative.

    Ordering is non-negotiable:

        candidates_i = [
            y_pos_i,       # column 0, label = 0
            y_hard_i,      # column 1, same-study wrong-phase hard negative
            y_pos_{j!=i},  # columns 2..B+1, batch positives used as negatives
        ]

    Labels are all zero. Column 0 **must** be the positive.

    Self-diagonal of the batch-negative block is set to ``-inf`` so the
    diagonal y_pos doesn't appear twice. If ``mask_same_study_batch_negatives``
    is True, off-diagonal same-study pairs are also set to ``-inf``. The
    explicit hard negative in column 1 is **never** masked.

    All vectors must already be L2-normed and in fp32.
    """
    if q.dim() != 2 or y_pos.dim() != 2 or y_hard.dim() != 2:
        raise ValueError(
            f"q/y_pos/y_hard must be [B, D]; got "
            f"{tuple(q.shape)} / {tuple(y_pos.shape)} / {tuple(y_hard.shape)}"
        )
    B, D = q.shape
    if y_pos.shape != (B, D) or y_hard.shape != (B, D):
        raise ValueError(
            f"y_pos/y_hard shape mismatch with q: {tuple(y_pos.shape)}, {tuple(y_hard.shape)}"
        )

    # Column 0: positive (diagonal pair-wise)
    pos_logit = (q * y_pos).sum(dim=-1, keepdim=True) / tau            # [B, 1]
    # Column 1: same-study wrong-phase hard negative
    hard_logit = (q * y_hard).sum(dim=-1, keepdim=True) / tau          # [B, 1]
    # Columns 2..B+1: batch-negative block
    batch_logits = (q @ y_pos.t()) / tau                                # [B, B]

    device = q.device
    eye = torch.eye(B, dtype=torch.bool, device=device)
    neg_inf = torch.finfo(batch_logits.dtype).min
    # Remove self-positive from the batch-negative block (its signal
    # already lives in column 0).
    batch_logits = batch_logits.masked_fill(eye, neg_inf)
    same_study_count = 0
    if mask_same_study_batch_negatives:
        same_study = study_hashes.unsqueeze(0).eq(study_hashes.unsqueeze(1))
        off_diag_same_study = same_study & ~eye
        batch_logits = batch_logits.masked_fill(off_diag_same_study, neg_inf)
        same_study_count = int(off_diag_same_study.sum().item())

    # Concatenate: [pos, hard, batch]. Column 0 is positive; labels=0.
    logits = torch.cat([pos_logit, hard_logit, batch_logits], dim=1)   # [B, B+2]
    labels = torch.zeros(B, dtype=torch.long, device=device)
    assert logits.shape == (B, B + 2), (
        f"candidate-set logits shape is [B, B+2], got {tuple(logits.shape)}"
    )
    assert (labels == 0).all().item(), "candidate-set labels must all be 0"

    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == labels).float().mean()
        pos_sim_mean = (pos_logit * tau).mean()
        hard_neg_sim_mean = (hard_logit * tau).mean()
        # Batch-neg mean over unmasked entries only.
        bn_mask = (batch_logits > neg_inf / 2.0)
        if bn_mask.any():
            batch_neg_sim_mean = ((batch_logits * tau)[bn_mask]).mean()
        else:
            batch_neg_sim_mean = torch.zeros((), device=device)
        pos_minus_hard_gap = pos_sim_mean - hard_neg_sim_mean
        pos_minus_batch_gap = pos_sim_mean - batch_neg_sim_mean
        logits_std = logits[logits > neg_inf / 2.0].std() if (logits > neg_inf / 2.0).any() else torch.zeros((), device=device)

    return {
        "rel_loss": loss,
        "rel_top1_with_hard": top1,
        "rel_pos_sim_mean": pos_sim_mean,
        "rel_hard_neg_sim_mean": hard_neg_sim_mean,
        "rel_batch_neg_sim_mean": batch_neg_sim_mean,
        "rel_pos_minus_hard_gap": pos_minus_hard_gap,
        "rel_pos_minus_batch_gap": pos_minus_batch_gap,
        "logits_std": logits_std,
        "same_study_masked_count": torch.tensor(float(same_study_count), device=device),
    }


def forward_phase_relational(
    pair: PairBatch,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    relational_head: torch.nn.Module,
    *,
    meta_list: list[dict],
    tau: float = 0.10,
    loss_exp: float = 1.0,
    mask_same_study_batch_negatives: bool = True,
) -> dict:
    """Intraview V-JEPA on clip_a + candidate-set InfoNCE on pooled latents.

    Teacher forwards are concat-batched (clip_a + clip_b_pos + clip_b_neg)
    into a single ``target_encoder`` call under ``torch.no_grad``, then
    split. This amortizes kernel launches and keeps all three teacher
    activations under one no_grad context.

    Loss:
        L_total = L_intraview + caller's λ_rel(t) · L_rel
    (caller is responsible for the λ warmup scalar; see main()).
    """
    from app.vjepa_multiview.phase_relational_head import pool_tokens

    if pair.clip_b_neg is None:
        raise ValueError(
            "forward_phase_relational requires pair.clip_b_neg; got None. "
            "Make sure the sampler is running with "
            "require_same_study_wrong_phase_negative=True and group_size=3."
        )

    # --- Teacher: one concat forward on [clip_a, clip_b_pos, clip_b_neg] --- #
    # Each pair.clip_* is list-over-fpc of [B, C, T, H, W]. We only use
    # fpc index 0 here (consistent with the existing smooth_l1 path
    # which also drives ``target_encoder`` with a list of length
    # len(fpc)). Teacher input is a list-over-fpc of stacked clips.
    B = pair.clip_a[0].size(0)
    concat_fpc = [
        torch.cat([pair.clip_a[0], pair.clip_b[0], pair.clip_b_neg[0]], dim=0),
    ]
    with torch.no_grad():
        h_concat = target_encoder(concat_fpc)
        h_concat = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_concat]
    h_a = [hi[:B] for hi in h_concat]
    h_b_pos = [hi[B:2 * B] for hi in h_concat]
    h_b_neg = [hi[2 * B:] for hi in h_concat]

    # --- Student forward + intraview JEPA loss (unchanged semantics) --- #
    z_ctx = encoder(pair.clip_a, pair.masks_enc)
    # ``MultiSeqWrapper`` returns nested ``list[list[Tensor]]`` when masks
    # are supplied. Fail loud if that shape breaks, so we never silently
    # hand a list to ``pool_tokens``.
    assert isinstance(z_ctx, list) and len(z_ctx) >= 1, (
        f"encoder(..., masks) expected list[list[Tensor]]; got {type(z_ctx)}"
    )
    assert isinstance(z_ctx[0], list) and len(z_ctx[0]) >= 1, (
        f"encoder output inner shape changed: {type(z_ctx[0])}"
    )
    z_pred = predictor(z_ctx, pair.masks_enc, pair.masks_pred, delta_phi=None)
    intraview = _jepa_loss_fn(z_pred, h_a, pair.masks_pred, loss_exp=loss_exp)

    # --- Relational branch --- #
    # Pool clip_a student context tokens. The encoder, wrapped in
    # ``MultiSeqWrapper``, returns a nested list ``[[T_fpc0_mask0,
    # T_fpc0_mask1]]`` when called with masks: outer is fpc, inner is
    # mask-generator. We want the first fpc's first (context) mask output.
    # Teacher ``h_b_pos`` / ``h_b_neg`` are un-masked (teacher forward is
    # called without masks), so they are the flat ``[tensor_fpc0]`` form.
    c_a_pool = pool_tokens(z_ctx[0][0])
    y_pos_pool = pool_tokens(h_b_pos[0]).detach()
    y_neg_pool = pool_tokens(h_b_neg[0]).detach()

    view_a_ids, view_b_pos_ids, delta_phase_pos, study_hashes = _build_predictor_inputs(
        meta_list, device=c_a_pool.device,
    )

    # One DDP forward per step touches every parameter of the head so
    # the reducer sees a uniform param set each iteration. Three
    # separate forwards for ``query``/``target``/``target`` would touch
    # disjoint subsets, which either needs ``find_unused_parameters=True``
    # (slower) or breaks the reducer invariant under static_graph.
    q_pre, y_pos_pre, y_hard_pre = relational_head(
        c_a_pool,
        view_a_ids,
        view_b_pos_ids,
        delta_phase_pos,
        y_pos_pool,
        y_neg_pool,
    )

    # Pre-norm diagnostic stats (before F.normalize zeros-out magnitude).
    with torch.no_grad():
        q_prenorm_mean = q_pre.norm(dim=-1).mean()
        y_prenorm_mean = 0.5 * (
            y_pos_pre.norm(dim=-1).mean() + y_hard_pre.norm(dim=-1).mean()
        )
        q_var = q_pre.var(dim=0).mean()
        y_var = y_pos_pre.var(dim=0).mean()

    # Cast to fp32 for the contrastive matmul (standard practice under bf16).
    q = F.normalize(q_pre.float(), dim=-1)
    y_pos = F.normalize(y_pos_pre.float(), dim=-1)
    y_hard = F.normalize(y_hard_pre.float(), dim=-1)

    rel_out = _relational_infonce_with_hard_neg(
        q=q, y_pos=y_pos, y_hard=y_hard,
        study_hashes=study_hashes,
        tau=tau,
        mask_same_study_batch_negatives=mask_same_study_batch_negatives,
    )

    # Total loss: caller multiplies rel_loss by λ_rel(t) and adds.
    out = {
        "intraview_loss": intraview,
        "crossview_loss": torch.zeros((), device=intraview.device),
        "rel_loss": rel_out["rel_loss"],
        "rel_top1_with_hard": rel_out["rel_top1_with_hard"],
        "rel_pos_sim_mean": rel_out["rel_pos_sim_mean"],
        "rel_hard_neg_sim_mean": rel_out["rel_hard_neg_sim_mean"],
        "rel_batch_neg_sim_mean": rel_out["rel_batch_neg_sim_mean"],
        "rel_pos_minus_hard_gap": rel_out["rel_pos_minus_hard_gap"],
        "rel_pos_minus_batch_gap": rel_out["rel_pos_minus_batch_gap"],
        "logits_std": rel_out["logits_std"],
        "same_study_masked_count": rel_out["same_study_masked_count"],
        "q_var": q_var,
        "y_var": y_var,
        "q_prenorm_mean": q_prenorm_mean,
        "y_prenorm_mean": y_prenorm_mean,
        "multiview_objective": "phase_relational",
    }
    return out


# --------------------------------------------------------------------------- #
# Epoch-refresh guard
# --------------------------------------------------------------------------- #

class PhaseMatchedRefreshGuard:
    """Guard against forgetting ``builder.refresh_epoch(e)`` in a DDP loop.

    Call ``guard.mark_refreshed(epoch)`` right after ``refresh_epoch`` and
    ``guard.check(epoch)`` immediately before ``iter(loader)``. The guard
    raises if the latest refresh didn't match the epoch about to run.
    """

    def __init__(self) -> None:
        self._last_refreshed: Optional[int] = None

    def mark_refreshed(self, epoch: int) -> None:
        self._last_refreshed = int(epoch)

    def check(self, epoch: int) -> None:
        if self._last_refreshed != int(epoch):
            raise RuntimeError(
                f"PhaseMatched loader not refreshed for epoch={epoch}; "
                f"last refresh={self._last_refreshed}. Call "
                f"`sampler.builder.refresh_epoch({epoch})` before iter(loader)."
            )


# --------------------------------------------------------------------------- #
# Collator that shares masks between clip_a and clip_b
# --------------------------------------------------------------------------- #

def make_phase_matched_collator(
    mask_collator: MaskCollator,
) -> "_PhaseMatchedCollator":
    """Wrap the vanilla V-JEPA MaskCollator so it sees the pair dataset's
    ``(segs, label, clip_indices_out, slot_mask)`` tuples and stores
    clip_b tensors alongside ``collated_batch`` for later access.

    The existing collator already handles the multi-view case correctly
    because ``default_collate`` will stack the list-of-two per-sample
    seg tensors into a length-2 list. We simply delegate.
    """
    return _PhaseMatchedCollator(mask_collator)


class _PhaseMatchedCollator:
    def __init__(self, inner: MaskCollator) -> None:
        self.inner = inner

    def __call__(self, batch):
        return self.inner(batch)


# --------------------------------------------------------------------------- #
# Full launcher — adapted from app/vjepa/train.py::main
#
# Differences vs. app/vjepa/train.py (all other plumbing is bit-identical):
#   1. Data pipeline goes through ``sampler_type="phase_matched"`` so the
#      data_manager returns a pair-capable VideoGroupDataset + attached
#      PhaseMatchedEpochBuilder. All other init_data args map 1:1.
#   2. Per-epoch ``dist_sampler.builder.refresh_epoch(epoch)`` is called
#      before ``iter(loader)``, guarded by PhaseMatchedRefreshGuard.
#   3. train_step runs forward_intraview_and_crossview (two teacher
#      forwards + one student + predictor + combined loss).
#   4. per-batch phase-metadata logging (sampling_mode, phase errors,
#      source-span cycles, HR differences, view fractions).
#   5. config knobs: phase_multiview.*, optimization.max_steps,
#      optimization.log_every_steps, meta.save_at_end.
# --------------------------------------------------------------------------- #

def main(args: dict, resume_preempt: bool = False) -> None:
    """Phase-matched multi-view JEPA launcher. Mirrors
    ``app/vjepa/train.py::main`` structure; swaps in the paired loader
    and cross-view loss.
    """
    # Heavy imports at call time to keep library usage (forward-only smoke
    # test) zero-cost.
    import copy
    import gc
    import logging as _pylogging
    import os
    import sys
    import time

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from app.vjepa.transforms import make_transforms
    from app.vjepa.utils import init_opt, init_video_model, load_checkpoint
    from src.datasets.data_manager import init_data
    from src.masks.multiseq_multiblock3d import MaskCollator
    from src.utils.checkpoint_loader import robust_checkpoint_loader
    from src.utils.distributed import init_distributed
    from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

    # --- META ---------------------------------------------------------- #
    folder = args.get("folder")
    cfgs_meta = args.get("meta", {}) or {}
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    load_model_cfg = cfgs_meta.get("load_checkpoint") or resume_preempt
    r_file = cfgs_meta.get("read_checkpoint", None)
    seed = cfgs_meta.get("seed", 0)
    which_dtype = cfgs_meta.get("dtype", "bfloat16")
    save_at_end = bool(cfgs_meta.get("save_at_end", False))

    if which_dtype.lower() == "bfloat16":
        dtype, mixed_precision = torch.bfloat16, True
    elif which_dtype.lower() == "float16":
        dtype, mixed_precision = torch.float16, True
    else:
        dtype, mixed_precision = torch.float32, False

    # --- MASK / MODEL / DATA / OPT ------------------------------------- #
    cfgs_mask = args.get("mask")
    cfgs_model = args.get("model", {}) or {}
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads")
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    num_mask_tokens = cfgs_model.get("num_mask_tokens", 2)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
    use_rope = cfgs_model.get("use_rope", False)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)

    cfgs_data = args.get("data", {}) or {}
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 0)
    persistent_workers = cfgs_data.get("persistent_workers", False)
    placeholder_csv = cfgs_data.get(
        "placeholder_csv", "/tmp/phase_multiview_placeholder.csv"
    )

    # phase_multiview block drives the phase_matched data_manager branch.
    cfg_pmv = args.get("phase_multiview", {}) or {}
    if not cfg_pmv.get("enabled", False):
        raise ValueError(
            "vjepa_multiview app requires phase_multiview.enabled=true in config"
        )
    sampler_type = cfg_pmv.get("sampler_type", "phase_matched")
    lambda_crossview = float(cfg_pmv.get("lambda_crossview", 0.25))
    use_intraview_loss = bool(cfg_pmv.get("use_intraview_loss", True))
    use_crossview_loss = bool(cfg_pmv.get("use_crossview_loss", True))
    log_every_steps = int(cfg_pmv.get("log_every_steps", 10))
    debug_verify_frame_count = bool(cfg_pmv.get("debug_verify_frame_count", False))
    debug_verify_n = int(cfg_pmv.get("debug_verify_n", 8))

    # --- New: multiview_objective dispatch + relational config --- #
    multiview_objective = str(cfg_pmv.get("multiview_objective", "smooth_l1"))
    if multiview_objective not in ("smooth_l1", "intraview_only", "phase_relational"):
        raise ValueError(
            f"multiview_objective={multiview_objective!r}; want "
            f"'smooth_l1' | 'intraview_only' | 'phase_relational'"
        )
    # In intraview_only mode the crossview loss is disabled regardless of
    # what the YAML says; force-disable to avoid a confusing config.
    if multiview_objective == "intraview_only":
        use_crossview_loss = False
    rel_cfg = cfg_pmv.get("relational", {}) or {}
    lambda_rel = float(rel_cfg.get("lambda_rel", 0.05))
    rel_warmup_epochs = float(rel_cfg.get("rel_warmup_epochs", 5))
    rel_temperature = float(rel_cfg.get("rel_temperature", 0.10))
    rel_mask_same_study_batch_negatives = bool(
        rel_cfg.get("rel_mask_same_study_batch_negatives", True)
    )
    target_projector_trainable = bool(
        rel_cfg.get("target_projector_trainable", True)
    )

    cfgs_data_aug = args.get("data_aug", {}) or {}
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    cfgs_loss = args.get("loss", {}) or {}
    loss_exp = cfgs_loss.get("loss_exp", 1.0)

    cfgs_opt = args.get("optimization", {}) or {}
    is_anneal = cfgs_opt.get("is_anneal", False)
    force_load_pretrain = cfgs_opt.get("force_load_pretrain", False)
    anneal_ckpt_path = cfgs_opt.get("anneal_ckpt", None)
    ipe = cfgs_opt.get("ipe", None)
    ipe_scale = cfgs_opt.get("ipe_scale", 1.0)
    wd = float(cfgs_opt.get("weight_decay", 0.0))
    final_wd = float(cfgs_opt.get("final_weight_decay", wd))
    # Scheduler horizon is independent of when we stop training. The LR /
    # WD / EMA momentum schedulers are all parameterized off
    # ``scheduler_total_epochs`` (fallback: ``epochs``). The training loop
    # exits after ``stop_after_epochs`` (fallback: ``epochs``). This lets
    # the same run be stopped at 25 and later resumed to 50 or 100 without
    # any LR restart. On resume, the optimizer + scheduler state are
    # loaded and we replay ``completed_steps`` scheduler steps.
    epochs_from_cfg = int(cfgs_opt.get("epochs", 1))
    scheduler_total_epochs = int(cfgs_opt.get("scheduler_total_epochs", epochs_from_cfg))
    stop_after_epochs = int(cfgs_opt.get("stop_after_epochs", epochs_from_cfg))
    # num_epochs is the scheduler horizon used inside init_opt / momentum.
    num_epochs = scheduler_total_epochs
    warmup = int(cfgs_opt.get("warmup", 0))
    start_lr = float(cfgs_opt.get("start_lr", 1e-4))
    lr = float(cfgs_opt.get("lr", 1e-4))
    final_lr = float(cfgs_opt.get("final_lr", lr))
    ema = cfgs_opt.get("ema", [0.999, 0.999])
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = float(cfgs_opt.get("eps", 1e-8))
    max_steps = cfgs_opt.get("max_steps", None)  # short-sanity gate
    if max_steps is not None:
        max_steps = int(max_steps)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    world_size, rank = init_distributed()
    log = get_logger(force=True)
    log.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if folder:
        os.makedirs(folder, exist_ok=True)
        log_file = os.path.join(folder, f"log_r{rank}.csv")
        if multiview_objective == "phase_relational":
            # Extended 24-column header: first 7 columns are byte-identical
            # to the legacy header so old log_r*.csv parsers still work.
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"), ("%d", "itr"),
                ("%.6f", "loss"), ("%.6f", "intraview"), ("%.6f", "crossview"),
                ("%d", "iter-time(ms)"), ("%d", "data-time(ms)"),
                ("%.6f", "rel_loss"),
                ("%.6f", "rel_top1_with_hard"),
                ("%.6f", "rel_pos_sim_mean"),
                ("%.6f", "rel_hard_neg_sim_mean"),
                ("%.6f", "rel_batch_neg_sim_mean"),
                ("%.6f", "rel_pos_minus_hard_gap"),
                ("%.6f", "rel_pos_minus_batch_gap"),
                ("%.6f", "effective_lambda_rel"),
                ("%.6f", "q_var"),
                ("%.6f", "y_var"),
                ("%.6f", "q_prenorm_mean"),
                ("%.6f", "y_prenorm_mean"),
                ("%.6f", "logits_std"),
                ("%.6f", "target_enc_grad_l2"),
                ("%.6f", "target_proj_grad_l2"),
                ("%d",   "rel_target_projector_trainable"),
                ("%d",   "same_study_masked_count"),
            )
        else:
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"), ("%d", "itr"),
                ("%.6f", "loss"), ("%.6f", "intraview"), ("%.6f", "crossview"),
                ("%d", "iter-time(ms)"), ("%d", "data-time(ms)"),
            )
    else:
        csv_logger = None

    # --- model --------------------------------------------------------- #
    encoder, predictor = init_video_model(
        device=device,
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder).to(device)

    # --- data: phase_matched dispatch ---------------------------------- #
    # VideoGroupDataset needs an initial CSV; the pair-builder swaps it
    # out atomically per epoch.
    if rank == 0 and not os.path.exists(placeholder_csv):
        import pandas as pd
        pd.DataFrame({"view_0": ["x"], "view_1": ["y"], "label": [0.0]}).to_csv(
            placeholder_csv, index=False
        )
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()

    mask_collator = MaskCollator(
        cfgs_mask=cfgs_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        fps_sampled=fps,
    )

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    pmv_dispatch_cfg = dict(cfg_pmv)
    pmv_dispatch_cfg.setdefault(
        "parquet_path", pmv_dispatch_cfg.get("phase_annotations_path")
    )
    pmv_dispatch_cfg.setdefault("sampler_dir", pmv_dispatch_cfg.get("sampler_dir"))
    pmv_dispatch_cfg.setdefault("quality_tiers", pmv_dispatch_cfg.get("quality_tiers", ["high"]))
    pmv_dispatch_cfg.setdefault("rr_filter_mode", pmv_dispatch_cfg.get("rr_filter_mode", "strict"))
    pmv_dispatch_cfg.setdefault(
        "sampling_mode", pmv_dispatch_cfg.get("sampling_mode", "uniform_phase")
    )
    pmv_dispatch_cfg.setdefault("phase_tolerance", pmv_dispatch_cfg.get("phase_tolerance", 0.15))
    pmv_dispatch_cfg.setdefault("frames_per_clip", pmv_dispatch_cfg.get("frames_per_clip", max_num_frames))
    pmv_dispatch_cfg.setdefault("frame_step", pmv_dispatch_cfg.get("frame_step", 1))
    pmv_dispatch_cfg.setdefault("pairs_per_study", pmv_dispatch_cfg.get("pairs_per_study", 1))
    pmv_dispatch_cfg.setdefault("allow_frame_step_gt1",
                                 pmv_dispatch_cfg.get("allow_frame_step_gt1", False))

    loader, dist_sampler = init_data(
        data="videogroupdataset",
        root_path=placeholder_csv,
        batch_size=batch_size,
        training=True,
        dataset_fpcs=dataset_fpcs,
        fps=fps,
        transform=transform,
        rank=rank,
        world_size=world_size,
        persistent_workers=persistent_workers,
        collator=mask_collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        log_dir=None,
        clip_len=max_num_frames,
        frame_sample_rate=pmv_dispatch_cfg["frame_step"],
        # group_size=3 is required for intraview_only + phase_relational so
        # the sampler emits (clip_a, clip_b_pos, clip_b_neg). smooth_l1 keeps
        # the legacy group_size=2 pair.
        num_clips=(3 if multiview_objective in ("intraview_only", "phase_relational") else 2),
        num_clips_per_video=1,
        img_size=crop_size,
        sampler_type=sampler_type,
        phase_matched_config=pmv_dispatch_cfg,
    )

    if not hasattr(dist_sampler, "builder"):
        raise RuntimeError(
            "phase_matched data_manager did not attach a builder to the sampler"
        )

    if ipe is None:
        ipe = dist_sampler.num_samples // batch_size + 1
    log.info(f"iterations per epoch: {ipe}  (pair rows/rank: {dist_sampler.num_samples})")
    log.info(
        f"schedule: scheduler_total_epochs={scheduler_total_epochs} "
        f"stop_after_epochs={stop_after_epochs} "
        f"warmup={warmup} ipe={ipe} ipe_scale={ipe_scale} "
        f"=> scheduler T_max={int(scheduler_total_epochs * ipe * ipe_scale)} steps"
    )

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        is_anneal=is_anneal,
        encoder=encoder,
        predictor=predictor,
        wd=wd, final_wd=final_wd,
        start_lr=start_lr, ref_lr=lr, final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup, num_epochs=num_epochs, ipe_scale=ipe_scale,
        mixed_precision=mixed_precision, betas=betas, eps=eps,
    )

    # --- Relational head construction (phase_relational only) ---------- #
    # Built AFTER init_opt and BEFORE load_checkpoint so that:
    #   (1) the head's params are added to the optimizer state before any
    #       checkpoint-state restore; if we resume a run that already had
    #       a head, the optimizer group layout matches.
    #   (2) the head is available for save_checkpoint's state-dict.
    relational_head = None
    if multiview_objective == "phase_relational":
        from app.vjepa_multiview.phase_relational_head import PhaseRelationalHead
        relational_head = PhaseRelationalHead(
            embed_dim=int(rel_cfg.get("embed_dim", 1024)),
            rel_dim=int(rel_cfg.get("rel_projector_dim", 256)),
            hidden_dim=int(rel_cfg.get("rel_predictor_hidden_dim", 1024)),
            num_views=14,
            view_embedding_dim=int(rel_cfg.get("rel_view_embedding_dim", 64)),
            n_phase_freqs=int(rel_cfg.get("rel_num_phase_frequencies", 4)),
        ).to(device)
        # If the target projector is configured non-trainable, freeze its
        # params (first pass keeps it trainable — simplest, matches plan).
        if not target_projector_trainable:
            for p in relational_head.target_proj.parameters():
                p.requires_grad = False
        # Append head params to the optimizer in two groups (WD-included +
        # WD-excluded) mirroring the encoder convention.
        head_wd_params = [
            p for n, p in relational_head.named_parameters()
            if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
        ]
        head_nowd_params = [
            p for n, p in relational_head.named_parameters()
            if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
        ]
        if head_wd_params:
            optimizer.add_param_group({
                "params": head_wd_params,
                "lr": lr, "weight_decay": wd,
            })
        if head_nowd_params:
            optimizer.add_param_group({
                "params": head_nowd_params,
                "lr": lr, "weight_decay": 0.0, "WD_exclude": True,
            })
        log.info(
            f"phase_relational: relational_head has "
            f"{sum(p.numel() for p in relational_head.parameters()):,} params "
            f"(trainable_wd={len(head_wd_params)} trainable_nowd={len(head_nowd_params)})"
        )

    def _momentum_scheduler(start_step=0):
        total = int(ipe * num_epochs * ipe_scale)
        return (ema[0] + i * (ema[1] - ema[0]) / total for i in range(start_step, total + 1))

    # --- ckpt load (force or resume) ----------------------------------- #
    completed_steps = 0
    start_epoch = 0
    if force_load_pretrain and anneal_ckpt_path and os.path.exists(anneal_ckpt_path):
        log.info(f"FORCE-LOADING pretrained model from {anneal_ckpt_path}")
        ckpt = robust_checkpoint_loader(anneal_ckpt_path, map_location=torch.device("cpu"))
        is_flat = "encoder" not in ckpt
        if not is_flat:
            pretrained = {k.replace("module.", ""): v for k, v in ckpt["encoder"].items()}
        else:
            pretrained = ckpt.get("model", ckpt.get("state_dict", ckpt))
            pretrained = {k.replace("module.", ""): v for k, v in pretrained.items()
                          if not k.startswith(("head.", "fc_norm.", "module.head.", "module.fc_norm."))}
        pe_key = "patch_embed.proj.weight"
        if pe_key in pretrained and pretrained[pe_key].ndim == 4:
            pe_2d = pretrained[pe_key]
            t = tubelet_size
            pretrained[pe_key] = pe_2d.unsqueeze(2).repeat(1, 1, t, 1, 1) / float(t)
        if is_flat:
            msg = encoder.backbone.load_state_dict(pretrained, strict=False)
        else:
            msg = encoder.load_state_dict(pretrained, strict=False)
        log.info(f"Loaded encoder: {msg}")
        target_encoder.load_state_dict(encoder.state_dict())
        if "predictor" in ckpt:
            pred_dict = {k.replace("module.", ""): v for k, v in ckpt["predictor"].items()}
            msg = predictor.load_state_dict(pred_dict, strict=False)
            log.info(f"Loaded predictor: {msg}")
        del ckpt
        gc.collect()
    elif load_model_cfg and folder:
        latest = os.path.join(folder, r_file if r_file else "latest.pt")
        if os.path.exists(latest):
            log.info(f"Resuming from {latest}")
            (encoder, predictor, target_encoder, optimizer, scaler,
             start_epoch, start_itr) = load_checkpoint(
                r_path=latest, encoder=encoder, predictor=predictor,
                target_encoder=target_encoder, opt=optimizer, scaler=scaler,
            )
            # If this is a phase_relational resume, also restore the head.
            # If the checkpoint predates phase_relational (e.g. a vanilla
            # e100 / e125 / a prior smooth_l1 run), no relational_head key
            # is present — head starts fresh and opt state may be stale.
            if relational_head is not None:
                try:
                    _ckpt_peek = torch.load(latest, map_location="cpu", weights_only=False)
                except TypeError:
                    _ckpt_peek = torch.load(latest, map_location="cpu")
                if "relational_head" in _ckpt_peek:
                    rh_state = {
                        k.replace("module.", ""): v
                        for k, v in _ckpt_peek["relational_head"].items()
                    }
                    msg = relational_head.load_state_dict(rh_state, strict=False)
                    log.info(f"Loaded relational_head: {msg}")
                else:
                    log.warning(
                        "Resuming a phase_relational run from a checkpoint "
                        "with no 'relational_head' key — head starts from init. "
                        "Optimizer state may not align; resetting start_epoch=0 "
                        "to avoid a stale LR-schedule replay against a newly-sized "
                        "param-group list."
                    )
                    start_epoch = 0
                    completed_steps = 0
                del _ckpt_peek
            completed_steps = start_epoch * ipe + start_itr
            for _ in range(completed_steps):
                scheduler.step(); wd_scheduler.step()
                mask_collator.step()

    # --- DDP wrap (only if distributed initialized) -------------------- #
    use_ddp = world_size > 1 and dist.is_available() and dist.is_initialized()
    if use_ddp:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
        target_encoder = DistributedDataParallel(target_encoder)
        if relational_head is not None:
            # Single unified forward per step touches every head
            # parameter (source_proj + relation_mlp + view embeds +
            # phase_mlp + target_proj), so the reducer sees a uniform
            # param set each iteration — no find_unused_parameters
            # needed. static_graph=False is a safe default; revisit if
            # the forward graph is guaranteed identical on every step.
            relational_head = DistributedDataParallel(
                relational_head, static_graph=False,
            )
    for p in target_encoder.parameters():
        p.requires_grad = False
    momentum_scheduler = _momentum_scheduler(start_step=completed_steps)

    # --- save helper --------------------------------------------------- #
    def save_checkpoint(epoch, itr, path, loss_avg=0.0):
        if rank != 0 or not path:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch, "itr": itr, "loss": loss_avg,
            "batch_size": batch_size, "world_size": world_size, "lr": lr,
            "sampling_mode": cfg_pmv.get("sampling_mode"),
            "lambda_crossview": lambda_crossview,
            "multiview_objective": multiview_objective,
        }
        if relational_head is not None:
            save_dict["relational_head"] = relational_head.state_dict()
            save_dict["rel_config"] = {
                "lambda_rel": lambda_rel,
                "rel_warmup_epochs": rel_warmup_epochs,
                "rel_temperature": rel_temperature,
                "target_projector_trainable": target_projector_trainable,
                "rel_mask_same_study_batch_negatives":
                    rel_mask_same_study_batch_negatives,
            }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(save_dict, tmp)
        os.replace(tmp, path)
        log.info(f"saved checkpoint -> {path}")

    # --- train loop ---------------------------------------------------- #
    guard = PhaseMatchedRefreshGuard()
    global_step = completed_steps
    exit_reason = None
    try:
        for epoch in range(start_epoch, stop_after_epochs):
            dist_sampler.builder.refresh_epoch(epoch)
            guard.mark_refreshed(epoch)
            n_pair = len(dist_sampler.builder.last_pair_df)
            log.info(
                f"[epoch {epoch}] refreshed pairs: rank={rank} n={n_pair} "
                f"mode={cfg_pmv.get('sampling_mode')}"
            )

            # Frame-count guard: open the first N MP4 URIs and compare
            # len(VideoReader) against clip_{a,b}_n_frames from the pair_df.
            # Fails loud; default off so this is only paid in sanity runs.
            if debug_verify_frame_count and epoch == start_epoch and rank == 0:
                _run_frame_count_guard(
                    dist_sampler.builder.last_pair_df, n=debug_verify_n, log=log
                )

            guard.check(epoch)
            loader_iter = iter(loader)
            loss_meter = AverageMeter()
            intra_meter = AverageMeter()
            cross_meter = AverageMeter()

            for itr in range(ipe):
                itr_t0 = time.time()
                try:
                    sample = next(loader_iter)
                except StopIteration:
                    # Loader exhausted early (pair-rows < ipe*batch_size at
                    # small scale); start another lap of the same records.
                    loader_iter = iter(loader)
                    sample = next(loader_iter)

                data_ms = (time.time() - itr_t0) * 1000.0

                # MaskCollator emits a list of per-fpc tuples; we configured
                # one fpc bucket.
                if not sample or not isinstance(sample, list):
                    raise RuntimeError("collated sample is empty or unexpected shape")
                fpc_collation = sample[0]
                collated_batch, masks_enc, masks_pred = fpc_collation

                clip_a, clip_b, clip_b_neg = _extract_multiview_clips(
                    collated_batch, device=device, objective=multiview_objective,
                )
                masks_enc_d = [[m.to(device) for m in masks_enc]]
                masks_pred_d = [[m.to(device) for m in masks_pred]]
                meta_list = extract_pair_metadata(collated_batch)

                pair = PairBatch(
                    clip_a=clip_a, clip_b=clip_b,
                    masks_enc=masks_enc_d, masks_pred=masks_pred_d,
                    phase_metadata=meta_list,
                    clip_b_neg=clip_b_neg,
                )

                def _step():
                    new_lr = scheduler.step()
                    new_wd = wd_scheduler.step()
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        if multiview_objective == "phase_relational":
                            out = forward_phase_relational(
                                pair, encoder, target_encoder, predictor,
                                relational_head, meta_list=meta_list,
                                tau=rel_temperature,
                                loss_exp=loss_exp,
                                mask_same_study_batch_negatives=rel_mask_same_study_batch_negatives,
                            )
                            # λ_rel warmup scalar (linear, capped at 1.0).
                            progress = float(epoch) + (float(itr) / max(1, ipe))
                            warmup_frac = min(
                                1.0,
                                progress / max(1e-6, rel_warmup_epochs),
                            )
                            effective_lambda_rel = lambda_rel * warmup_frac
                            total_loss = (
                                out["intraview_loss"]
                                + effective_lambda_rel * out["rel_loss"]
                            )
                            out["effective_lambda_rel"] = torch.tensor(
                                effective_lambda_rel, device=device,
                            )
                            out["total_loss"] = total_loss
                        elif multiview_objective == "intraview_only":
                            out = forward_intraview_only(
                                pair, encoder, target_encoder, predictor,
                                loss_exp=loss_exp,
                            )
                            total_loss = out["total_loss"]
                        else:  # "smooth_l1" — unchanged path
                            out = forward_intraview_and_crossview(
                                pair, encoder, target_encoder, predictor,
                                lambda_crossview=lambda_crossview,
                                use_intraview_loss=use_intraview_loss,
                                use_crossview_loss=use_crossview_loss,
                                loss_exp=loss_exp,
                                log_mask_diagnostics=(global_step < 3),
                            )
                            total_loss = out["total_loss"]
                    if mixed_precision:
                        scaler.scale(total_loss).backward()
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        total_loss.backward()
                        optimizer.step()
                    optimizer.zero_grad()
                    # EMA
                    m = next(momentum_scheduler)
                    with torch.no_grad():
                        for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                            pk.mul_(m).add_(pq, alpha=1 - m)
                    return out, new_lr, new_wd

                (out, new_lr, new_wd), _ = gpu_timer(_step)
                intra_val = float(out["intraview_loss"].item())
                cross_val = float(out["crossview_loss"].item())
                total_val = float(out["total_loss"].item())
                loss_meter.update(total_val)
                intra_meter.update(intra_val)
                cross_meter.update(cross_val)

                itr_ms = (time.time() - itr_t0) * 1000.0
                global_step += 1

                if (itr % log_every_steps == 0) or (itr == ipe - 1) or (max_steps and global_step == max_steps):
                    stats = summarize_pair_metadata(meta_list)
                    log.info(
                        f"[e{epoch}/{stop_after_epochs}@{num_epochs} i{itr:5d} step{global_step:6d}] "
                        f"total={total_val:.4f} (avg={loss_meter.avg:.4f}) "
                        f"intra={intra_val:.4f} cross={cross_val:.4f} "
                        f"lr={new_lr:.2e} "
                        f"mode={stats.get('sampling_mode', '?')} "
                        f"circ_diff.med={stats.get('circular_phase_diff.median', float('nan')):.3f} "
                        f"span_cyc_a.p90={stats.get('source_span_cycles_a.p90', float('nan')):.2f} "
                        f"hr_diff.p90={stats.get('hr_diff.p90', float('nan')):.1f} "
                        f"vp[sv/sf/cf]={stats.get('same_view_frac', float('nan')):.2f}/"
                        f"{stats.get('same_family_frac', float('nan')):.2f}/"
                        f"{stats.get('cross_family_frac', float('nan')):.2f} "
                        f"data={data_ms:.0f}ms iter={itr_ms:.0f}ms"
                    )
                    if global_step < 3:
                        log.info(
                            f"  mask: enc{tuple(masks_enc_d[0][0].shape)} "
                            f"pred{tuple(masks_pred_d[0][0].shape)} "
                            f"enc_range=[{int(masks_enc_d[0][0].min())},{int(masks_enc_d[0][0].max())}] "
                            f"pred_range=[{int(masks_pred_d[0][0].min())},{int(masks_pred_d[0][0].max())}]"
                        )

                if csv_logger is not None:
                    if multiview_objective == "phase_relational":
                        # Compute target stability contract grad-norms
                        # AFTER optimizer.step()/zero_grad() — these are
                        # always 0 here because zero_grad just ran. The
                        # *invariant* we care about (no teacher grads) is
                        # enforced structurally by no_grad in
                        # forward_phase_relational; we log a simple 0
                        # marker here for schema consistency, and bump
                        # this to a live probe in a future diagnostic
                        # pass if needed.
                        def _param_l2(m):
                            if m is None:
                                return 0.0
                            total = 0.0
                            for p in m.parameters():
                                if p.grad is not None:
                                    total += float(p.grad.detach().float().pow(2).sum().item())
                            return total ** 0.5
                        te_grad_l2 = _param_l2(target_encoder)
                        rh = relational_head
                        if rh is not None and hasattr(rh, "module"):
                            # DDP wrapper
                            tp = getattr(rh.module, "target_proj", None)
                        else:
                            tp = getattr(rh, "target_proj", None) if rh is not None else None
                        tp_grad_l2 = _param_l2(tp)
                        csv_logger.log(
                            epoch + 1, itr,
                            total_val, intra_val, cross_val,
                            int(itr_ms), int(data_ms),
                            float(out.get("rel_loss", 0.0)),
                            float(out.get("rel_top1_with_hard", 0.0)),
                            float(out.get("rel_pos_sim_mean", 0.0)),
                            float(out.get("rel_hard_neg_sim_mean", 0.0)),
                            float(out.get("rel_batch_neg_sim_mean", 0.0)),
                            float(out.get("rel_pos_minus_hard_gap", 0.0)),
                            float(out.get("rel_pos_minus_batch_gap", 0.0)),
                            float(out.get("effective_lambda_rel", 0.0)),
                            float(out.get("q_var", 0.0)),
                            float(out.get("y_var", 0.0)),
                            float(out.get("q_prenorm_mean", 0.0)),
                            float(out.get("y_prenorm_mean", 0.0)),
                            float(out.get("logits_std", 0.0)),
                            te_grad_l2,
                            tp_grad_l2,
                            int(target_projector_trainable),
                            int(out.get("same_study_masked_count", 0)),
                        )
                    else:
                        csv_logger.log(
                            epoch + 1, itr, total_val, intra_val, cross_val,
                            int(itr_ms), int(data_ms),
                        )

                assert np.isfinite(total_val), (
                    f"total_loss non-finite at step {global_step}: {total_val}"
                )

                if max_steps and global_step >= max_steps:
                    exit_reason = f"reached max_steps={max_steps}"
                    break

            if exit_reason is not None:
                break

            log.info(
                f"epoch {epoch}: avg total_loss={loss_meter.avg:.4f} "
                f"avg intra={intra_meter.avg:.4f} avg cross={cross_meter.avg:.4f}"
            )

            if folder and save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
                save_checkpoint(epoch + 1, 0, os.path.join(folder, f"e{epoch + 1}.pt"),
                                loss_avg=loss_meter.avg)

        if folder and (save_at_end or (exit_reason is not None)):
            save_checkpoint(epoch, global_step, os.path.join(folder, "latest.pt"),
                            loss_avg=loss_meter.avg)
    finally:
        if exit_reason:
            log.info(f"exit: {exit_reason}")
        if use_ddp:
            try:
                dist.barrier()
            except Exception:
                pass
