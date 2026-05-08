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

    clip_a: list[torch.Tensor]  # list over fpc, each [B, C, T, H, W]
    clip_b: list[torch.Tensor]  # clip_b_pos in 3-clip mode; unchanged in smooth_l1
    masks_enc: list[list[torch.Tensor]]  # [fpc][mask_i] -> [B, N_ctx]
    masks_pred: list[list[torch.Tensor]]  # [fpc][mask_i] -> [B, N_tgt]
    phase_metadata: list[dict[str, Any]]
    clip_b_neg: Optional[list[torch.Tensor]] = None  # list over fpc; None in smooth_l1
    # --- MV2SV (Fix 1) extensions. Populated only when the sampler is
    # configured with ``mv2sv.enabled=True``. When populated, the MV2SV
    # forward consumes ``target_clip`` as the pair-pred target and uses
    # the fused_clips list as the fused-pool input. When not populated
    # (legacy phase_relational / intraview_only), the forward falls
    # back to clip_b / clip_b_neg. ---
    target_clip: Optional[list[torch.Tensor]] = None  # list-over-fpc of [B, C, T, H, W]
    target_views: Optional[list[str]] = None  # per-sample target view id-str (len B)
    target_delta_phase: Optional[torch.Tensor] = None  # [B] float
    target_clip_present: Optional[torch.Tensor] = None  # [B] bool — 1 if row has real target_clip
    # Fused pool is per-sample variable length in principle; in practice
    # the sampler emits a constant N_fused per batch (pads missing slots
    # and records validity in fused_valid_mask).
    fused_clips: Optional[list[torch.Tensor]] = None  # list-over-fpc of [B, N_fused, C, T, H, W]
    fused_views: Optional[list[list[str]]] = None  # len-B list of len-N_fused view strings
    fused_phases: Optional[torch.Tensor] = None  # [B, N_fused] float
    fused_valid_mask: Optional[torch.Tensor] = None  # [B, N_fused] bool


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
        raise ValueError(f"multi-view collator expected segs len >= 2, got {type(segs).__name__}")
    clip_a = segs[0]
    clip_b = segs[1]
    # The single-view path wraps into a list-over-fpc; we do the same so
    # downstream code (masks/encoder) can stay unchanged.
    return ([clip_a.to(device, non_blocking=True)], [clip_b.to(device, non_blocking=True)])


def _extract_multiview_clips(
    collated_batch: tuple,
    device: torch.device,
    objective: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], Optional[list[torch.Tensor]]]:
    """Objective-aware clip extraction (legacy 3-clip shape).

    Returns ``(clip_a, clip_b, clip_b_neg)`` where each element is a
    list-over-fpc of ``[B, C, T, H, W]`` tensors on ``device``. MV2SV
    callers should use ``_extract_mv2sv_clips`` which also surfaces
    target_clip + fused_clips.

    - ``smooth_l1``:       expects segs of length 2; clip_b_neg is None.
    - ``intraview_only`` / ``phase_relational`` / ``privileged_multiview``:
        expects segs of length >= 3; the first 3 are returned, any
        additional views (MV2SV target/fused) are ignored in this
        entry point.
    """
    segs = collated_batch[0]
    if not isinstance(segs, (list, tuple)):
        raise ValueError(f"multi-view collator expected segs list/tuple, got {type(segs).__name__}")
    if objective in ("smooth_l1", "mcc_jepa"):
        if len(segs) < 2:
            raise ValueError(f"multiview_objective={objective!r} expects segs len >= 2; got {len(segs)}")
        clip_a = [segs[0].to(device, non_blocking=True)]
        clip_b = [segs[1].to(device, non_blocking=True)]
        return clip_a, clip_b, None
    if objective in (
        "intraview_only",
        "phase_relational",
        "privileged_multiview",
        "token_phase_relational",
    ):
        if len(segs) < 3:
            raise ValueError(
                f"multiview_objective={objective!r} expects segs of length >= 3 "
                f"(clip_a, clip_b_pos, clip_b_neg); got {len(segs)}"
            )
        clip_a = [segs[0].to(device, non_blocking=True)]
        clip_b = [segs[1].to(device, non_blocking=True)]
        clip_b_neg = [segs[2].to(device, non_blocking=True)]
        return clip_a, clip_b, clip_b_neg
    raise ValueError(
        f"unknown multiview_objective={objective!r}; "
        f"want one of smooth_l1 | intraview_only | phase_relational | "
        f"privileged_multiview | token_phase_relational | mcc_jepa"
    )


def _extract_mv2sv_clips(
    collated_batch: tuple,
    meta_list: list[dict],
    device: torch.device,
    expected_n_fused: int = 0,
) -> dict:
    """MV2SV-aware clip extraction.

    Returns a dict with:
        clip_a          : list-over-fpc of [B, C, T, H, W]
        clip_b          : list-over-fpc of [B, C, T, H, W] (clip_b_pos)
        clip_b_neg      : list-over-fpc of [B, C, T, H, W] | None
        target_clip     : list-over-fpc of [B, C, T, H, W] | None
        fused_clips     : list-over-fpc of [B, N_fused_extra, C, T, H, W] | None
                          where fused_clips pool includes the target as slot 0
                          logically, and fused[1..] live at view_4..
        target_views    : list[str] (len B) — from meta target_clip_view
        target_delta_phase : FloatTensor [B]
        target_clip_present : BoolTensor [B]  (1 if target loaded, 0 if placeholder)
        fused_valid_mask : BoolTensor [B, N_fused_total] — N_fused_total ==
                          1 (target) + expected_n_fused
        fused_views      : list[B] of list[N_fused_total] of view strings
        fused_phases     : FloatTensor [B, N_fused_total]

    Segs layout in the collated batch (see _records_to_pair_dataframe):
        segs[0] = clip_a
        segs[1] = clip_b (positive)
        segs[2] = clip_b_neg (hard negative; always present when group_size >= 3)
        segs[3] = target_clip   (if any record had one)
        segs[4..] = fused_clips[1..] (if expected_n_fused > 0)
    """
    segs = collated_batch[0]
    if not isinstance(segs, (list, tuple)) or len(segs) < 3:
        raise ValueError(
            f"_extract_mv2sv_clips needs segs of length >= 3 "
            f"(clip_a, clip_b, clip_b_neg); got {len(segs) if hasattr(segs, '__len__') else type(segs)}"
        )
    clip_a = [segs[0].to(device, non_blocking=True)]
    clip_b = [segs[1].to(device, non_blocking=True)]
    clip_b_neg = [segs[2].to(device, non_blocking=True)]
    out: dict = {
        "clip_a": clip_a,
        "clip_b": clip_b,
        "clip_b_neg": clip_b_neg,
        "target_clip": None,
        "fused_clips": None,
        "target_views": None,
        "target_delta_phase": None,
        "target_clip_present": None,
        "fused_valid_mask": None,
        "fused_views": None,
        "fused_phases": None,
    }

    # --- Target clip (view_3) ---
    has_target_slot = len(segs) >= 4
    if has_target_slot:
        out["target_clip"] = [segs[3].to(device, non_blocking=True)]
        B = segs[3].size(0)
        # Per-sample metadata pulled from the meta list.
        tv = [m.get("target_clip_view") or "" for m in meta_list]
        tdp = [
            (
                float(m.get("target_delta_phase", float("nan")))
                if m.get("target_delta_phase") is not None
                else float("nan")
            )
            for m in meta_list
        ]
        tp = [int(m.get("target_clip_present", 0) or 0) for m in meta_list]
        out["target_views"] = tv
        out["target_delta_phase"] = torch.tensor(tdp, dtype=torch.float32, device=device)
        out["target_clip_present"] = torch.tensor(
            [bool(x) for x in tp],
            dtype=torch.bool,
            device=device,
        )

    # --- Fused pool (view_3 = target as slot 0; view_4..view_{3+N} = fused[1..]) ---
    if expected_n_fused > 0 and has_target_slot:
        # Stack target + extra fused into [B, N_fused_total, C, T, H, W].
        n_extra = expected_n_fused
        extra_segs = segs[4 : 4 + n_extra]
        if len(extra_segs) != n_extra:
            # Fewer segs than expected → sampler didn't populate extras;
            # the caller should treat fused as unavailable.
            out["fused_clips"] = None
        else:
            target_seg = segs[3].unsqueeze(1)  # [B, 1, C, T, H, W]
            extra_stack = torch.stack(extra_segs, dim=1)  # [B, n_extra, C, T, H, W]
            fused_stack = torch.cat([target_seg, extra_stack], dim=1)
            out["fused_clips"] = [fused_stack.to(device, non_blocking=True)]
            # Validity mask: target is valid iff target_clip_present; per-
            # extra slot validity read from meta (fused_clip_K_valid).
            B = target_seg.size(0)
            mask = torch.zeros(B, 1 + n_extra, dtype=torch.bool, device=device)
            if out["target_clip_present"] is not None:
                mask[:, 0] = out["target_clip_present"]
            else:
                mask[:, 0] = True
            fv = []
            fp = []
            for b_i, m in enumerate(meta_list):
                row_views: list[str] = [m.get("target_clip_view") or ""]
                row_phases: list[float] = [
                    (
                        float(m.get("target_delta_phase", float("nan")))
                        if m.get("target_delta_phase") is not None
                        else float("nan")
                    )
                ]
                for k in range(1, n_extra + 1):
                    valid = int(m.get(f"fused_clip_{k}_valid", 0) or 0)
                    mask[b_i, k] = bool(valid)
                    row_views.append(m.get(f"fused_clip_{k}_view") or "")
                    row_phases.append(
                        float(m.get(f"fused_clip_{k}_delta_phase", float("nan")))
                        if m.get(f"fused_clip_{k}_delta_phase") is not None
                        else float("nan")
                    )
                fv.append(row_views)
                fp.append(row_phases)
            out["fused_views"] = fv
            out["fused_phases"] = torch.tensor(fp, dtype=torch.float32, device=device)
            out["fused_valid_mask"] = mask

    return out


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
    for key in (
        "circular_phase_diff",
        "clip_a_phase_error",
        "clip_b_phase_error",
        "source_span_cycles_a",
        "source_span_cycles_b",
    ):
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
                problems.append(f"row={i} side={side} uri={uri}: expected={expected} got={got}")
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
        "intraview_loss": intraview if intraview is not None else torch.zeros((), device=total.device),
        "crossview_loss": crossview if crossview is not None else torch.zeros((), device=total.device),
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
                diag.append(
                    {
                        "fpc_idx": i,
                        "mask_gen_idx": mgi,
                        "shape": tuple(m.shape),
                        "dtype": str(m.dtype),
                        "raw_min": int(m.min().item()),
                        "raw_max": int(m.max().item()),
                    }
                )
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
        device=device,
        dtype=torch.long,
    )
    view_b_pos_ids = torch.tensor(
        [view_to_id(m.get("clip_b_view")) for m in meta_list],
        device=device,
        dtype=torch.long,
    )
    # Δφ_pos = (target_phi_b - target_phi_a) mod 1. Already computed in
    # the sampler as ``circular_phase_diff`` (wrapped magnitude) but
    # here we want signed delta, so recompute.
    delta_phase_pos = torch.tensor(
        [(_phase_val(m, "target_phi_b") - _phase_val(m, "target_phi_a")) % 1.0 for m in meta_list],
        device=device,
        dtype=torch.float32,
    )
    # Deterministic study-id hashes for same-study masking in InfoNCE.
    # Any collision is benign — masking is a safety net, not a hard
    # dependency. Use Python's ``hash`` (deterministic within a process
    # but not across processes) to map string study_ids to int64.
    study_hashes = torch.tensor(
        [hash(str(m.get("study_id", ""))) for m in meta_list],
        device=device,
        dtype=torch.long,
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
    disable_hard_negative: bool = False,
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

    ``disable_hard_negative`` (no_hardneg ablation): column 1 is masked to
    ``-inf`` in the logits passed to cross-entropy, so the hard-neg column
    contributes zero to the loss and receives no gradient through ``y_hard``.
    The raw cosine ``q·y_hard/τ`` is still computed as ``hard_dot`` and is
    what the ``rel_hard_neg_sim_mean`` / ``rel_pos_minus_hard_gap``
    diagnostics report — so those remain meaningful scalars in the
    ablation (measuring incidental hard-neg separation, not a trained one).
    In no_hardneg mode, ``rel_top1_with_hard`` cannot be won by column 1
    (it is ``-inf``), so the metric effectively becomes "top-1 over
    positive vs batch negatives" while retaining its schema name.

    All vectors must already be L2-normed and in fp32.
    """
    if q.dim() != 2 or y_pos.dim() != 2 or y_hard.dim() != 2:
        raise ValueError(
            f"q/y_pos/y_hard must be [B, D]; got " f"{tuple(q.shape)} / {tuple(y_pos.shape)} / {tuple(y_hard.shape)}"
        )
    B, D = q.shape
    if y_pos.shape != (B, D) or y_hard.shape != (B, D):
        raise ValueError(f"y_pos/y_hard shape mismatch with q: {tuple(y_pos.shape)}, {tuple(y_hard.shape)}")

    # Column 0: positive (diagonal pair-wise)
    pos_logit = (q * y_pos).sum(dim=-1, keepdim=True) / tau  # [B, 1]
    # Column 1: same-study wrong-phase hard negative. Always compute the
    # real cosine (``hard_dot``) for diagnostics; if disabled, the logit
    # used in CE is -inf so the column contributes zero to the loss and
    # no gradient flows through ``y_hard`` via the softmax.
    hard_dot = (q * y_hard).sum(dim=-1, keepdim=True) / tau  # [B, 1]
    if disable_hard_negative:
        hard_logit = torch.full_like(
            hard_dot,
            torch.finfo(hard_dot.dtype).min,
        )
    else:
        hard_logit = hard_dot
    # Columns 2..B+1: batch-negative block
    batch_logits = (q @ y_pos.t()) / tau  # [B, B]

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
    logits = torch.cat([pos_logit, hard_logit, batch_logits], dim=1)  # [B, B+2]
    labels = torch.zeros(B, dtype=torch.long, device=device)
    assert logits.shape == (B, B + 2), f"candidate-set logits shape is [B, B+2], got {tuple(logits.shape)}"
    assert (labels == 0).all().item(), "candidate-set labels must all be 0"

    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == labels).float().mean()
        pos_sim_mean = (pos_logit * tau).mean()
        # Use hard_dot (real cosine), not hard_logit, so the diagnostic
        # stays finite in the no_hardneg ablation where hard_logit=-inf.
        hard_neg_sim_mean = (hard_dot * tau).mean()
        # Batch-neg mean over unmasked entries only.
        bn_mask = batch_logits > neg_inf / 2.0
        if bn_mask.any():
            batch_neg_sim_mean = ((batch_logits * tau)[bn_mask]).mean()
        else:
            batch_neg_sim_mean = torch.zeros((), device=device)
        pos_minus_hard_gap = pos_sim_mean - hard_neg_sim_mean
        pos_minus_batch_gap = pos_sim_mean - batch_neg_sim_mean
        logits_std = (
            logits[logits > neg_inf / 2.0].std() if (logits > neg_inf / 2.0).any() else torch.zeros((), device=device)
        )

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
    disable_hard_negative: bool = False,
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
    h_b_pos = [hi[B : 2 * B] for hi in h_concat]
    h_b_neg = [hi[2 * B :] for hi in h_concat]

    # --- Student forward + intraview JEPA loss (unchanged semantics) --- #
    z_ctx = encoder(pair.clip_a, pair.masks_enc)
    # ``MultiSeqWrapper`` returns nested ``list[list[Tensor]]`` when masks
    # are supplied. Fail loud if that shape breaks, so we never silently
    # hand a list to ``pool_tokens``.
    assert (
        isinstance(z_ctx, list) and len(z_ctx) >= 1
    ), f"encoder(..., masks) expected list[list[Tensor]]; got {type(z_ctx)}"
    assert isinstance(z_ctx[0], list) and len(z_ctx[0]) >= 1, f"encoder output inner shape changed: {type(z_ctx[0])}"
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
        meta_list,
        device=c_a_pool.device,
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
        y_prenorm_mean = 0.5 * (y_pos_pre.norm(dim=-1).mean() + y_hard_pre.norm(dim=-1).mean())
        q_var = q_pre.var(dim=0).mean()
        y_var = y_pos_pre.var(dim=0).mean()

    # Cast to fp32 for the contrastive matmul (standard practice under bf16).
    q = F.normalize(q_pre.float(), dim=-1)
    y_pos = F.normalize(y_pos_pre.float(), dim=-1)
    y_hard = F.normalize(y_hard_pre.float(), dim=-1)

    rel_out = _relational_infonce_with_hard_neg(
        q=q,
        y_pos=y_pos,
        y_hard=y_hard,
        study_hashes=study_hashes,
        tau=tau,
        mask_same_study_batch_negatives=mask_same_study_batch_negatives,
        disable_hard_negative=disable_hard_negative,
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
# Token phase-relational forward (EchoJEPA-TokenRel / -Motion)
# --------------------------------------------------------------------------- #


def forward_token_phase_relational(
    pair: "PairBatch",  # type: ignore[name-defined]
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    token_rel_head: torch.nn.Module,
    relational_head_safety: torch.nn.Module | None,
    motion_delta_head: torch.nn.Module | None,
    delta_target_projector: torch.nn.Module | None,
    *,
    meta_list: list[dict],
    token_subsample_k: int = 64,
    tau_token: float = 0.10,
    tau_delta: float = 0.10,
    loss_exp: float = 1.0,
    lambda_token_rel: float = 0.02,
    lambda_pool_rel: float = 0.005,
    lambda_delta: float = 0.0,
    lambda_delta_l1: float = 1.0,
    lambda_delta_nce: float = 1.0,
    mask_same_study_batch_negatives: bool = True,
    pool_rel_temperature: float = 0.10,
) -> dict:
    """EchoJEPA-TokenRel forward.

    Student sees clip_a only. Teacher concat-forwards
    [clip_a, clip_b_pos, clip_b_neg] under no_grad. Loss:

        L_total = L_intra
                + lambda_token_rel * L_token_phase_rel
                + lambda_delta     * L_latent_motion_delta        (optional)
                + lambda_pool_rel  * L_pool_rel_safety            (optional)

    ``pool_rel_safety`` is V4's InfoNCE at a tiny weight to preserve the
    LVEF-positive pooled phase signal. Set lambda_pool_rel=0 to disable.

    Caller (main()) applies a linear warmup multiplier to the three
    auxiliary lambdas.
    """
    from app.vjepa_multiview.phase_relational_head import pool_tokens
    from app.vjepa_multiview.token_relational_loss import (
        motion_delta_loss,
        token_set_infonce_with_hard_neg,
    )

    if pair.clip_b_neg is None:
        raise ValueError(
            "forward_token_phase_relational requires pair.clip_b_neg "
            "(3-clip sampler path). Run with group_size>=3."
        )

    B = pair.clip_a[0].size(0)
    device = pair.clip_a[0].device

    # --- Teacher concat forward (one no_grad pass on 3 clips) ---
    concat_fpc = [
        torch.cat([pair.clip_a[0], pair.clip_b[0], pair.clip_b_neg[0]], dim=0),
    ]
    with torch.no_grad():
        h_concat = target_encoder(concat_fpc)
        h_concat = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_concat]
    h_a = [hi[:B] for hi in h_concat]
    h_b_pos = [hi[B : 2 * B] for hi in h_concat]
    h_b_neg = [hi[2 * B :] for hi in h_concat]

    # --- Student forward + L_intra ---
    z_ctx = encoder(pair.clip_a, pair.masks_enc)
    assert isinstance(z_ctx, list) and len(z_ctx) >= 1
    assert isinstance(z_ctx[0], list) and len(z_ctx[0]) >= 1
    z_pred = predictor(z_ctx, pair.masks_enc, pair.masks_pred, delta_phi=None)
    intraview = _jepa_loss_fn(z_pred, h_a, pair.masks_pred, loss_exp=loss_exp)

    # Student context tokens (full, unmasked) for pooled+token projections.
    z_tokens_full = z_ctx[0][0]  # [B, N, D]
    h_a_full = h_a[0]
    h_pos_full = h_b_pos[0]
    h_neg_full = h_b_neg[0]

    # --- Build metadata tensors (reuses V4 helper) ---
    src_view_ids, tgt_view_ids, delta_phase_pos, study_hashes = _build_predictor_inputs(
        meta_list,
        device=device,
    )

    # --- Token subsample (shared indices across batch rows) ---
    from app.vjepa_multiview.token_relational_head import subsample_tokens

    z_tokens_sub, token_idx = subsample_tokens(z_tokens_full, token_subsample_k)
    if token_idx.numel() < z_tokens_full.shape[1]:
        h_a_sub = h_a_full.index_select(dim=1, index=token_idx)
        h_pos_sub = h_pos_full.index_select(dim=1, index=token_idx)
        h_neg_sub = h_neg_full.index_select(dim=1, index=token_idx)
    else:
        h_a_sub = h_a_full
        h_pos_sub = h_pos_full
        h_neg_sub = h_neg_full

    # --- Token-rel InfoNCE ---
    q_tok_pre, y_pos_pre, y_hard_pre = token_rel_head(
        z_tokens_sub,
        src_view_ids,
        tgt_view_ids,
        delta_phase_pos,
        h_pos_sub.detach(),
        h_neg_sub.detach(),
    )
    q_tok = F.normalize(q_tok_pre.float(), dim=-1)
    y_pos_tok = F.normalize(y_pos_pre.float(), dim=-1)
    y_hard_tok = F.normalize(y_hard_pre.float(), dim=-1)
    token_rel_out = token_set_infonce_with_hard_neg(
        q_tok,
        y_pos_tok,
        y_hard_tok,
        study_hashes,
        tau=tau_token,
        mask_same_study_batch_negatives=mask_same_study_batch_negatives,
    )
    L_token_rel = token_rel_out["token_rel_loss"]

    # --- Pool-rel safety (V4-style InfoNCE on pooled features) ---
    if relational_head_safety is not None:
        c_a_pool = pool_tokens(z_tokens_full)
        y_pos_pool = pool_tokens(h_pos_full).detach()
        y_neg_pool = pool_tokens(h_neg_full).detach()
        q_pool_pre, y_pos_pool_pre, y_hard_pool_pre = relational_head_safety(
            c_a_pool,
            src_view_ids,
            tgt_view_ids,
            delta_phase_pos,
            y_pos_pool,
            y_neg_pool,
        )
        q_p = F.normalize(q_pool_pre.float(), dim=-1)
        y_p = F.normalize(y_pos_pool_pre.float(), dim=-1)
        y_h = F.normalize(y_hard_pool_pre.float(), dim=-1)
        pool_rel_out = _relational_infonce_with_hard_neg(
            q_p,
            y_p,
            y_h,
            study_hashes,
            tau=pool_rel_temperature,
            mask_same_study_batch_negatives=mask_same_study_batch_negatives,
        )
        L_pool_rel = pool_rel_out["rel_loss"]
        pool_rel_top1 = pool_rel_out["rel_top1_with_hard"]
        pool_rel_gap = pool_rel_out["rel_pos_minus_hard_gap"]
    else:
        L_pool_rel = torch.zeros((), device=device)
        pool_rel_top1 = torch.zeros((), device=device)
        pool_rel_gap = torch.zeros((), device=device)

    # --- Motion-delta (same-view only) ---
    if motion_delta_head is not None and delta_target_projector is not None and lambda_delta > 0.0:
        delta_out = motion_delta_loss(
            z_tokens_sub=z_tokens_sub,
            h_a_tokens_sub=h_a_sub.detach(),
            h_pos_tokens_sub=h_pos_sub.detach(),
            h_neg_tokens_sub=h_neg_sub.detach(),
            src_view_ids=src_view_ids,
            tgt_view_ids=tgt_view_ids,
            delta_phase=delta_phase_pos,
            motion_delta_head=motion_delta_head,
            delta_target_projector=delta_target_projector,
            tau=tau_delta,
            lambda_l1=lambda_delta_l1,
            lambda_nce=lambda_delta_nce,
        )
        L_delta = delta_out["delta_loss"]
    else:
        # Still touch every param via a zero-loss proxy if modules exist
        # (so DDP reducer is deterministic across steps). Otherwise no-op.
        if motion_delta_head is not None and delta_target_projector is not None:
            dummy_q = motion_delta_head(
                z_tokens_sub[:1],
                src_view_ids[:1],
                delta_phase_pos[:1],
            )
            dummy_t = delta_target_projector(z_tokens_sub[:1].detach())
            L_delta = 0.0 * (dummy_q.sum() + dummy_t.sum())
        else:
            L_delta = torch.zeros((), device=device)
        delta_out = {
            "delta_loss": L_delta,
            "delta_l1": torch.zeros((), device=device),
            "delta_nce": torch.zeros((), device=device),
            "delta_valid_rows": torch.zeros((), device=device),
            "delta_pos_sim_mean": torch.zeros((), device=device),
            "delta_hard_sim_mean": torch.zeros((), device=device),
            "delta_pos_minus_hard_gap": torch.zeros((), device=device),
            "delta_q_var": torch.zeros((), device=device),
            "delta_target_var": torch.zeros((), device=device),
        }

    total = (
        intraview
        + lambda_token_rel * L_token_rel
        + lambda_delta * L_delta
        + lambda_pool_rel * L_pool_rel
    )

    # Sampler bucket diagnostics (same-view / same-family / cross-family).
    with torch.no_grad():
        from app.vjepa_multiview.phase_relational_head import family_of

        same_view = src_view_ids.eq(tgt_view_ids)
        n = same_view.shape[0]
        same_fam_cnt = 0
        cross_fam_cnt = 0
        for i in range(n):
            if same_view[i].item():
                continue
            fa = family_of(int(src_view_ids[i].item()))
            fb = family_of(int(tgt_view_ids[i].item()))
            if fa == fb:
                same_fam_cnt += 1
            else:
                cross_fam_cnt += 1
        same_view_frac = torch.tensor(float(same_view.sum().item()) / max(1, n), device=device)
        same_fam_frac = torch.tensor(float(same_fam_cnt) / max(1, n), device=device)
        cross_fam_frac = torch.tensor(float(cross_fam_cnt) / max(1, n), device=device)

    out = {
        "intraview_loss": intraview,
        "crossview_loss": torch.zeros((), device=device),
        "total_loss": total,
        "multiview_objective": "token_phase_relational",
        # Token-rel diagnostics
        "token_rel_loss": token_rel_out["token_rel_loss"].detach(),
        "token_rel_top1_with_hard": token_rel_out["token_rel_top1_with_hard"],
        "token_rel_pos_sim_mean": token_rel_out["token_rel_pos_sim_mean"],
        "token_rel_hard_sim_mean": token_rel_out["token_rel_hard_sim_mean"],
        "token_rel_batch_neg_sim_mean": token_rel_out["token_rel_batch_neg_sim_mean"],
        "token_rel_pos_minus_hard_gap": token_rel_out["token_rel_pos_minus_hard_gap"],
        "token_rel_pos_minus_batch_gap": token_rel_out["token_rel_pos_minus_batch_gap"],
        "token_rel_logits_std": token_rel_out["token_rel_logits_std"],
        "token_rel_q_var": token_rel_out["token_rel_q_var"],
        "token_rel_y_var": token_rel_out["token_rel_y_var"],
        "token_rel_valid_rows": token_rel_out["token_rel_valid_rows"],
        "token_rel_same_study_masked_count": token_rel_out["token_rel_same_study_masked_count"],
        "token_subsample_k": torch.tensor(float(z_tokens_sub.shape[1]), device=device),
        # Pool-rel safety diagnostics
        "pool_rel_loss": L_pool_rel.detach(),
        "pool_rel_top1_with_hard": pool_rel_top1,
        "pool_rel_pos_minus_hard_gap": pool_rel_gap,
        # Motion-delta diagnostics
        "delta_loss": delta_out["delta_loss"].detach() if isinstance(delta_out["delta_loss"], torch.Tensor) else L_delta.detach(),
        "delta_l1": delta_out["delta_l1"],
        "delta_nce": delta_out["delta_nce"],
        "delta_valid_rows": delta_out["delta_valid_rows"],
        "delta_pos_sim_mean": delta_out["delta_pos_sim_mean"],
        "delta_hard_sim_mean": delta_out["delta_hard_sim_mean"],
        "delta_pos_minus_hard_gap": delta_out["delta_pos_minus_hard_gap"],
        "delta_q_var": delta_out["delta_q_var"],
        "delta_target_var": delta_out["delta_target_var"],
        # Sampler bucket fractions
        "same_view_row_fraction": same_view_frac,
        "same_family_row_fraction": same_fam_frac,
        "cross_family_row_fraction": cross_fam_frac,
    }
    return out


# --------------------------------------------------------------------------- #
# Privileged-multiview forward (EchoJEPA-MV2SV)
# --------------------------------------------------------------------------- #


def _paired_shared_ntxent(
    z_src: torch.Tensor,
    t_tgt: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paired NT-Xent: each anchor row's positive is its own target row;
    other rows' target vectors serve as negatives. One positive per
    sample, guaranteed — no dependence on batch containing same-study
    duplicates.

    Inputs must already be L2-normalized. Caller is responsible for
    detaching ``t_tgt`` if it must not carry grad back to the teacher
    pathway.

    Returns (loss, diagnostics). Loss is a scalar tensor.
    """
    if z_src.dim() != 2 or t_tgt.dim() != 2:
        raise ValueError(f"z_src/t_tgt must be [B, D]; got {tuple(z_src.shape)} / {tuple(t_tgt.shape)}")
    if z_src.shape != t_tgt.shape:
        raise ValueError(f"z_src {tuple(z_src.shape)} and t_tgt {tuple(t_tgt.shape)} " f"must have identical shape")
    B, _ = z_src.shape
    device = z_src.device
    # logits[i, j] = sim(z_src_i, t_tgt_j) / tau
    logits = (z_src @ t_tgt.t()) / tau  # [B, B]
    labels = torch.arange(B, device=device, dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        top1 = (logits.argmax(dim=1) == labels).float().mean()
        pos_sim = logits.diag().mean() * tau
    diag = {
        "paired_shared_top1": top1.detach(),
        "paired_shared_pos_sim": pos_sim.detach(),
    }
    return loss, diag


def _view_nce_loss(
    q_view: torch.Tensor,
    t_view: torch.Tensor,
    study_hashes: torch.Tensor,
    target_view_ids: torch.Tensor,
    tau_view: float,
    mask_same_study_batch_negatives: bool = True,
    same_target_view_required: bool = True,
    family_fallback: bool = True,
    min_valid_neg: int = 1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cross-view retrieval contrastive (v5): negatives are other rows'
    target-view latents at the SAME target_view_id (or same family as a
    fallback).

    v5 changes over v4:
      * Default ``same_target_view_required=True``. A row's negatives
        are other batch rows i↦j with j's target_view_id == i's AND
        j's study != i's study.
      * When a row has fewer than ``min_valid_neg`` same-target-view
        negatives AND ``family_fallback=True``, the row expands its
        negative pool to same-family target views. The expansion is
        row-wise, and ``view_nce_fallback_fraction`` records what
        fraction of rows needed the fallback.
      * Positive (diagonal) is always preserved; self-row is masked.

    Inputs are expected L2-normalized. ``target_view_ids`` is [B] long
    with entries from VIEW_ID_MAP. Caller must detach ``t_view`` if it
    should not carry grad back through the teacher head.

    Returns (loss, diag). Diagnostics include view_nce_top1,
    view_nce_valid_neg_count_{mean,min}, view_nce_same_target_view_fraction,
    view_nce_fallback_fraction, and per-target-view top1.
    """
    from app.vjepa_multiview.phase_relational_head import VIEW_FAMILY_ID

    if q_view.dim() != 2 or t_view.dim() != 2:
        raise ValueError(f"q_view/t_view must be [B, D]; got {tuple(q_view.shape)} / {tuple(t_view.shape)}")
    if q_view.shape != t_view.shape:
        raise ValueError(f"q_view {tuple(q_view.shape)} != t_view {tuple(t_view.shape)}")
    B, _ = q_view.shape
    device = q_view.device
    logits = (q_view @ t_view.t()) / tau_view  # [B, B]
    labels = torch.arange(B, device=device, dtype=torch.long)
    eye = torch.eye(B, dtype=torch.bool, device=device)
    neg_inf = torch.finfo(logits.dtype).min

    # --- Build the legal-negative mask (excluding self-diag) --------- #
    same_study = study_hashes.unsqueeze(0).eq(study_hashes.unsqueeze(1))
    if target_view_ids is None:
        # Legacy path: only same-study masking (no target-view constraint).
        legal = ~eye
        if mask_same_study_batch_negatives:
            legal = legal & ~same_study
        # No fallback, no same-target-view fraction.
        valid_neg_per_row = legal.sum(dim=1).float()
        fallback_used = torch.zeros(B, dtype=torch.bool, device=device)
        same_tv_mask = torch.zeros_like(legal)
    else:
        tvid = target_view_ids.long()
        same_tv = tvid.unsqueeze(0).eq(tvid.unsqueeze(1))  # [B, B]
        same_tv_mask = same_tv & ~eye
        # Strict same-target-view: negatives must match target_view AND
        # differ in study.
        strict_legal = same_tv_mask
        if mask_same_study_batch_negatives:
            strict_legal = strict_legal & ~same_study

        if same_target_view_required:
            strict_count = strict_legal.sum(dim=1)
            rows_need_fallback = strict_count < min_valid_neg
            if family_fallback and rows_need_fallback.any():
                # Expand insufficient rows to same-family (and optionally
                # across-study). Build a family matrix from VIEW_FAMILY_ID.
                fam_ids_host = [VIEW_FAMILY_ID.get(int(v), 4) for v in tvid.tolist()]
                fam_ids = torch.tensor(fam_ids_host, device=device, dtype=torch.long)
                same_fam = fam_ids.unsqueeze(0).eq(fam_ids.unsqueeze(1))
                fam_legal = same_fam & ~eye
                if mask_same_study_batch_negatives:
                    fam_legal = fam_legal & ~same_study
                # Row-wise: keep strict mask for rows that had enough;
                # use fam_legal for rows that didn't.
                legal = torch.where(rows_need_fallback.unsqueeze(-1), fam_legal, strict_legal)
                fallback_used = rows_need_fallback
            else:
                legal = strict_legal
                fallback_used = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            legal = ~eye
            if mask_same_study_batch_negatives:
                legal = legal & ~same_study
            fallback_used = torch.zeros(B, dtype=torch.bool, device=device)
        valid_neg_per_row = legal.sum(dim=1).float()

    # --- Mask illegal entries in logits (keep positive diagonal) ----- #
    # Row i's j≠i entries that are NOT in ``legal`` are forbidden
    # negatives → set to -inf. The diagonal stays live (positive).
    row_allowed = legal | eye  # diagonal always allowed
    masked_logits = logits.masked_fill(~row_allowed, neg_inf)

    # Rows with zero valid negatives still have their positive; CE is
    # finite (logsumexp over a single element). But they contribute no
    # contrastive signal, so we log their count.
    loss = F.cross_entropy(masked_logits, labels)

    with torch.no_grad():
        top1 = (masked_logits.argmax(dim=1) == labels).float().mean()
        pos_sim = (logits.diag() * tau_view).mean()
        # Off-diag mean over the legal mask.
        if legal.any():
            # Use un-masked logits so the diagnostic reflects the true
            # cosine distribution, not the -inf'd one.
            neg_sim = ((logits * tau_view)[legal]).mean()
        else:
            neg_sim = torch.zeros((), device=device)
        # Per-target-view top1.
        per_view: dict[int, torch.Tensor] = {}
        if target_view_ids is not None:
            hits = masked_logits.argmax(dim=1) == labels
            uniq = target_view_ids.unique()
            for vid in uniq.tolist():
                sel = target_view_ids == int(vid)
                if sel.any():
                    per_view[int(vid)] = hits[sel].float().mean().detach()
        # v5 diagnostics.
        same_tv_pair_frac = (
            same_tv_mask.float().sum() / max(1.0, float(B * (B - 1)))
            if target_view_ids is not None
            else torch.zeros((), device=device)
        )
        fallback_frac = fallback_used.float().mean()

    diag = {
        "view_nce_top1": top1.detach(),
        "view_nce_pos_sim_mean": pos_sim.detach(),
        "view_nce_neg_sim_mean": neg_sim.detach(),
        "view_nce_valid_neg_count_mean": valid_neg_per_row.mean().detach(),
        "view_nce_valid_neg_count_min": valid_neg_per_row.min().detach(),
        "view_nce_same_target_view_fraction": same_tv_pair_frac.detach(),
        "view_nce_fallback_fraction": fallback_frac.detach(),
        "view_nce_top1_by_view": per_view,
    }
    return loss, diag


def _same_study_ntxent(
    z: torch.Tensor,
    study_hashes: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """NT-Xent with same-study batch rows as positives. Retained for
    backward compatibility of the v1 integration test; the v2 forward
    uses ``_paired_shared_ntxent`` instead so positives are guaranteed.
    """
    if z.dim() != 2:
        raise ValueError(f"z must be [B, D]; got {tuple(z.shape)}")
    B, _ = z.shape
    device = z.device
    logits = (z @ z.t()) / tau
    eye = torch.eye(B, dtype=torch.bool, device=device)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(eye, neg_inf)
    same_study = study_hashes.unsqueeze(0).eq(study_hashes.unsqueeze(1))
    pos_mask = same_study & ~eye
    any_pos = pos_mask.any(dim=1)
    log_denom = torch.logsumexp(logits, dim=1)
    pos_counts = pos_mask.float().sum(dim=1).clamp(min=1.0)
    pos_logits_sum = logits.masked_fill(~pos_mask, 0.0).sum(dim=1)
    per_row = -(pos_logits_sum / pos_counts - log_denom)
    if any_pos.any():
        loss = per_row[any_pos].mean()
    else:
        loss = torch.zeros((), device=device)
    diag = {
        "static_ntxent_pos_rows": any_pos.float().sum().detach(),
        "static_ntxent_batch_rows": torch.tensor(float(B), device=device),
    }
    return loss, diag


def ddp_synced_bernoulli(p: float, global_step: int, seed_salt: int = 0xA1B2C3D4) -> bool:
    """Deterministic per-step Bernoulli gate identical across all DDP ranks.

    Seeded by ``hash((global_step, seed_salt))`` using a CPU generator
    so the draw is bit-identical on every rank regardless of GPU
    state. The caller is responsible for broadcasting this from rank 0
    if there's any concern about state drift — but with the seed
    fully determined by (global_step, seed_salt), a pure local draw
    is sufficient.

    ``global_step`` is expected to be the same on every rank (the
    training loop increments it after all ranks finish the step).
    """
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    g = torch.Generator(device="cpu").manual_seed(int(global_step) ^ int(seed_salt))
    return bool(torch.rand((), generator=g).item() < float(p))


def momentum_update_ema_(online: torch.nn.Module, ema: torch.nn.Module, m: float) -> None:
    """In-place EMA update: ema_p <- m * ema_p + (1-m) * online_p.

    Applied after each optimizer step. ``online`` is the trained
    module (may be DDP-wrapped); ``ema`` is the detached target copy
    (must NOT be DDP-wrapped, its params have requires_grad=False).
    """

    def _strip_ddp(mod: torch.nn.Module) -> torch.nn.Module:
        return mod.module if hasattr(mod, "module") else mod

    with torch.no_grad():
        src = _strip_ddp(online)
        for p_src, p_ema in zip(src.parameters(), ema.parameters()):
            p_ema.mul_(m).add_(p_src.data, alpha=1.0 - m)
        for b_src, b_ema in zip(src.buffers(), ema.buffers()):
            # Buffers (e.g. Fourier freqs) are usually identical constants;
            # copy rather than lerp to avoid subtle dtype drift.
            b_ema.copy_(b_src)


def _mean_shared_fused_target(
    pooled_nv: torch.Tensor,
    valid_mask: torch.Tensor,
    factorized_head_ema: torch.nn.Module,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Deterministic fused-teacher target for the ``mean_shared`` mode.

    For each clip in the fused pool, apply the EMA factorized head to
    its detached teacher-pool vector and extract the ``z_shared``
    slot. Take a masked mean over the N fused clips per row.

    This is the default (and only safe) fused-target generator while
    no independent objective trains ``mv_teacher_fusion``. It never
    involves ``mv_teacher_fusion`` and never produces gradient into
    ``factorized_head_ema`` (every call is inside ``no_grad`` and the
    returned tensor is explicitly detached).

    Inputs:
      pooled_nv     : [B, N, D] detached teacher pooled latents
      valid_mask    : [B, N]    bool, True = valid fused clip
      factorized_head_ema : EMA factorized head, requires_grad=False

    Returns:
      (fused, diag) where ``fused`` is [B, shared_dim] detached, and
      ``diag`` exposes fused_valid_views counts + target norms.

    Fallback: if a row has zero valid fused clips, ``denom`` is
    clamped to 1.0 so the division is safe (the row's fused target
    is all-zero vector, and that row's contribution to
    ``F.smooth_l1_loss`` will be against a zero target — benign).
    The caller can optionally mask such rows out of the loss via
    ``valid_mask.any(dim=1)``.
    """
    if pooled_nv.dim() != 3:
        raise ValueError(f"pooled_nv must be [B, N, D]; got {tuple(pooled_nv.shape)}")
    B, N, D = pooled_nv.shape
    if valid_mask.shape != (B, N):
        raise ValueError(f"valid_mask must be [B, N]={(B, N)}; got {tuple(valid_mask.shape)}")
    device = pooled_nv.device
    flat = pooled_nv.reshape(B * N, D)
    with torch.no_grad():
        slots = factorized_head_ema(flat)
        shared = slots["z_shared"].reshape(B, N, -1)
    mask = valid_mask.to(shared.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    fused = (shared * mask).sum(dim=1) / denom
    fused = fused.detach()
    with torch.no_grad():
        valid_f = valid_mask.float()
        valid_count_per_row = valid_f.sum(dim=1)  # [B]
        diag = {
            "fused_valid_views_mean": valid_count_per_row.mean(),
            "fused_valid_views_min": valid_count_per_row.min(),
            "fused_shared_target_norm": fused.norm(dim=-1).mean(),
            "fused_any_row_invalid": (valid_count_per_row == 0).float().sum(),
        }
    return fused, diag


def forward_privileged_multiview_v1_legacy(
    pair: PairBatch,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    factorized_head: torch.nn.Module,
    view_predictor: torch.nn.Module,
    mv_teacher_fusion: torch.nn.Module,
    shared_projector: torch.nn.Module,
    *,
    meta_list: list[dict],
    lambda_pair: float = 0.25,
    lambda_fused: float = 0.10,
    lambda_shared: float = 0.05,
    lambda_phase: float = 0.025,
    p_fused: float = 0.25,
    tau_static: float = 0.10,
    tau_phase: float = 0.10,
    loss_exp: float = 1.0,
    mask_same_study_batch_negatives: bool = True,
) -> dict:
    """Privileged Multi-View EchoJEPA forward.

    Single-view student + multi-view privileged teacher. The student
    sees clip_a; the teacher sees clip_a (for intraview), clip_b (the
    pairwise target), and clip_b_neg (the same-study wrong-phase hard
    negative used by L_phase_rel and — when p_fused fires — as the
    second entry in the fused-teacher pool).

    Loss decomposition:
        L_total = L_intra
                + lambda_pair   · L_pair_view_pred
                + lambda_fused  · L_fused          · Bernoulli(p_fused)
                + lambda_shared · L_same_study_align
                + lambda_phase  · L_phase_rel

    Slot assignment (see plan):
        z_shared : L_same_study_align + pair predictor input + L_fused
        z_phase  : L_phase_rel + pair predictor input
        z_view   : unused in loss currently (present for future use;
                   receives grad via factorized_head init only — see
                   DDP note below).

    DDP note:
        Default config keeps all of lambda_{pair, shared, phase} > 0,
        so on every non-Bernoulli step z_shared + z_phase heads see
        gradient. The z_view head does *not* receive gradient on any
        step here — callers must either set `find_unused_parameters=True`
        on the factorized_head DDP wrap, or freeze the z_view head,
        or add a downstream loss that consumes it. The config flag
        `ddp.find_unused_parameters` in the YAML governs this.
    """
    from app.vjepa_multiview.phase_relational_head import pool_tokens

    if pair.clip_b_neg is None:
        raise ValueError(
            "forward_privileged_multiview requires pair.clip_b_neg "
            "(reused as the fused-pool second entry and as the "
            "L_phase_rel hard negative). Run with group_size=3."
        )
    B = pair.clip_a[0].size(0)
    device = pair.clip_a[0].device

    # --- Teacher: one concat forward on [clip_a, clip_b, clip_b_neg] ---
    # clip_b is the pairwise target; clip_b_neg is the wrong-phase hard neg.
    concat_fpc = [
        torch.cat([pair.clip_a[0], pair.clip_b[0], pair.clip_b_neg[0]], dim=0),
    ]
    with torch.no_grad():
        h_concat = target_encoder(concat_fpc)
        h_concat = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_concat]
    h_a = [hi[:B] for hi in h_concat]
    h_tgt = [hi[B : 2 * B] for hi in h_concat]
    h_neg = [hi[2 * B :] for hi in h_concat]

    # --- Student forward + L_intra (unchanged V-JEPA semantics) ---
    z_ctx = encoder(pair.clip_a, pair.masks_enc)
    assert isinstance(z_ctx, list) and len(z_ctx) >= 1
    assert isinstance(z_ctx[0], list) and len(z_ctx[0]) >= 1
    z_pred = predictor(z_ctx, pair.masks_enc, pair.masks_pred, delta_phi=None)
    intraview = _jepa_loss_fn(z_pred, h_a, pair.masks_pred, loss_exp=loss_exp)

    # --- Factorized student slots from pooled context tokens ---
    c_a_pool = pool_tokens(z_ctx[0][0])
    slots = factorized_head(c_a_pool)
    z_shared = slots["z_shared"]
    z_phase = slots["z_phase"]
    # z_view is produced but not currently consumed by a loss term.

    # Teacher pooled latents (detached). Target = clip_b pooled; neg = clip_b_neg pooled.
    y_tgt_pool = pool_tokens(h_tgt[0]).detach()
    y_neg_pool = pool_tokens(h_neg[0]).detach()

    # Predictor inputs: source/target view ids + Δφ + study hashes.
    # We reuse _build_predictor_inputs: it returns
    #   (view_a_ids, view_b_pos_ids, delta_phase_pos, study_hashes)
    # where view_b_pos_ids == target view id and delta_phase_pos == Δφ
    # (source -> target). Naming is a legacy — the content is exactly
    # what MV2SV needs.
    src_view_ids, tgt_view_ids, delta_phase, study_hashes = _build_predictor_inputs(
        meta_list,
        device=device,
    )

    # Compute teacher-side factorized slots once (one forward per teacher
    # pool). `no_grad` is structural — y_tgt_pool / y_neg_pool are
    # already detached, and we want no gradient into the factorized head
    # from the target side of any loss. The student-side slots above
    # (`z_shared`, `z_phase`) still carry grad; this only freezes the
    # target-side path.
    with torch.no_grad():
        t_tgt_slots = factorized_head(y_tgt_pool)
        t_neg_slots = factorized_head(y_neg_pool)
    t_tgt_shared = t_tgt_slots["z_shared"].detach()
    t_tgt_phase = t_tgt_slots["z_phase"].detach()
    t_neg_phase = t_neg_slots["z_phase"].detach()

    # --- L_pair_view_pred: student predicts teacher's target-clip shared slot.
    q_pair = view_predictor(
        z_shared,
        z_phase,
        src_view_ids,
        tgt_view_ids,
        delta_phase,
    )
    pair_loss = F.smooth_l1_loss(q_pair, t_tgt_shared)

    # --- L_same_study_align: NT-Xent on normalized z_shared.
    z_shared_norm = F.normalize(z_shared.float(), dim=-1)
    shared_loss, shared_diag = _same_study_ntxent(
        z_shared_norm,
        study_hashes,
        tau=tau_static,
    )

    # --- L_phase_rel on z_phase: candidate-set InfoNCE reused verbatim.
    # Positive = teacher z_phase on target clip; hard neg = teacher
    # z_phase on wrong-phase clip.
    z_phase_norm = F.normalize(z_phase.float(), dim=-1)
    y_pos_phase_norm = F.normalize(t_tgt_phase.float(), dim=-1)
    y_hard_phase_norm = F.normalize(t_neg_phase.float(), dim=-1)
    phase_rel_out = _relational_infonce_with_hard_neg(
        q=z_phase_norm,
        y_pos=y_pos_phase_norm,
        y_hard=y_hard_phase_norm,
        study_hashes=study_hashes,
        tau=tau_phase,
        mask_same_study_batch_negatives=mask_same_study_batch_negatives,
        disable_hard_negative=False,
    )

    # --- L_fused (sparse): gated by Bernoulli(p_fused).
    # Pool = [teacher_pool_on_clip_b, teacher_pool_on_clip_b_neg]  (N=2).
    # view_ids and phase values come from meta_list.
    # The gate is per-step (scalar Bernoulli), not per-sample, so every
    # rank in DDP takes the same branch each step — we draw from a
    # deterministic generator keyed on the global_step via Python random
    # at the call site. Here we accept a `p_fused` scalar and draw
    # locally; determinism across ranks is the caller's responsibility
    # (pass the same seeded generator or use p_fused=0 in unit tests).
    fused_active = torch.rand((), device=device) < float(p_fused)
    if fused_active:
        # Build N=2 teacher pool. view ids and phases come from meta.
        def _phase_val(m, k):
            v = m.get(k)
            if v is None or (isinstance(v, float) and v != v):
                return 0.0
            return float(v)

        from app.vjepa_multiview.phase_relational_head import view_to_id

        tgt_view_list = [view_to_id(m.get("clip_b_view")) for m in meta_list]
        tgt_phase_list = [(_phase_val(m, "target_phi_b") - _phase_val(m, "target_phi_a")) % 1.0 for m in meta_list]
        neg_view_list = [view_to_id(m.get("clip_b_view")) for m in meta_list]
        neg_phase_list = [(_phase_val(m, "target_phi_b_neg") - _phase_val(m, "target_phi_a")) % 1.0 for m in meta_list]
        view_ids_nv = torch.tensor(
            list(zip(tgt_view_list, neg_view_list)),
            device=device,
            dtype=torch.long,
        )  # [B, 2]
        phase_nv = torch.tensor(
            list(zip(tgt_phase_list, neg_phase_list)),
            device=device,
            dtype=torch.float32,
        )  # [B, 2]
        pooled_nv = torch.stack([y_tgt_pool, y_neg_pool], dim=1)  # [B, 2, D]
        t_fused_shared = mv_teacher_fusion(pooled_nv, view_ids_nv, phase_nv)
        # Student's shared projection -> fused space.
        q_fused = shared_projector(z_shared)
        # Stop-grad on target side (BYOL-style) — fusion's own weights
        # still train through q_fused -> shared_projector path? No: the
        # only student-side gradient is through q_fused. t_fused_shared
        # carries grad into mv_teacher_fusion's weights if we don't
        # detach it. We *do* want mv_teacher_fusion to train — so keep
        # t_fused_shared live. Signal: the caller's optimizer includes
        # both mv_teacher_fusion and shared_projector.
        fused_loss = F.smooth_l1_loss(q_fused, t_fused_shared)
    else:
        # Need a zero loss of the same dtype / device that is still
        # attached to the relevant parameters so DDP sees grad flow —
        # otherwise find_unused_parameters=True is required on the
        # fusion and shared_projector wraps. We construct a "zero * p"
        # term that touches both modules' parameters via a cheap
        # forward, then multiplies by zero. This keeps the DDP reducer
        # happy with static_graph=False and without
        # find_unused_parameters.
        dummy_pooled = torch.stack([y_tgt_pool, y_neg_pool], dim=1)  # [B, 2, D]
        dummy_view_ids = torch.zeros(B, 2, device=device, dtype=torch.long)
        dummy_phase = torch.zeros(B, 2, device=device, dtype=torch.float32)
        dummy_fused = mv_teacher_fusion(dummy_pooled, dummy_view_ids, dummy_phase)
        dummy_q = shared_projector(z_shared)
        fused_loss = 0.0 * (dummy_fused.sum() + dummy_q.sum())

    # --- Diagnostics ---
    with torch.no_grad():
        pair_q_norm = q_pair.norm(dim=-1).mean()
        pair_target_norm = t_tgt_shared.norm(dim=-1).mean()
        z_shared_z_phase_cos = F.cosine_similarity(z_shared, z_phase, dim=-1).mean()

    out = {
        "intraview_loss": intraview,
        # "crossview_loss" is kept at 0 for schema compatibility with
        # the existing CSV logger.
        "crossview_loss": torch.zeros((), device=device),
        "pair_loss": pair_loss,
        "fused_loss": fused_loss,
        "fused_active": fused_active.float(),
        "shared_loss": shared_loss,
        "phase_rel_loss": phase_rel_out["rel_loss"],
        "phase_rel_top1": phase_rel_out["rel_top1_with_hard"],
        "phase_rel_pos_minus_hard_gap": phase_rel_out["rel_pos_minus_hard_gap"],
        "same_study_masked_count": phase_rel_out["same_study_masked_count"],
        "diag_pair_q_norm": pair_q_norm,
        "diag_pair_target_norm": pair_target_norm,
        "diag_z_shared_vs_z_phase_cos": z_shared_z_phase_cos,
        "static_ntxent_pos_rows": shared_diag["static_ntxent_pos_rows"],
        "multiview_objective": "privileged_multiview",
    }

    # Total loss; caller decides whether to apply warmup scalars on
    # the non-intra components. We emit a default total here for
    # the common case where warmup is linear and externally scaled.
    total = (
        intraview
        + lambda_pair * pair_loss
        + lambda_fused * fused_loss
        + lambda_shared * shared_loss
        + lambda_phase * phase_rel_out["rel_loss"]
    )
    out["total_loss"] = total
    return out


def forward_privileged_multiview(
    pair: PairBatch,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    factorized_head: torch.nn.Module,
    view_predictor: torch.nn.Module,
    pair_shared_projector: torch.nn.Module,
    fused_shared_projector: torch.nn.Module,
    phase_query_head: torch.nn.Module,
    factorized_head_ema: torch.nn.Module,
    *,
    meta_list: list[dict],
    fused_active: bool,
    fused_target_mode: str = "mean_shared",
    mv_teacher_fusion: torch.nn.Module | None = None,
    mv_teacher_fusion_ema: torch.nn.Module | None = None,
    lambda_pair_shared: float = 0.05,
    lambda_pair_view: float = 0.10,
    lambda_view_nce: float = 0.025,
    lambda_shared: float = 0.05,
    lambda_phase: float = 0.0,
    lambda_fused: float = 0.0,
    lambda_local_motion: float = 0.0,
    tau_shared: float = 0.10,
    tau_phase: float = 0.10,
    tau_view: float = 0.10,
    loss_exp: float = 1.0,
    mask_same_study_batch_negatives: bool = True,
    use_z_view: bool = True,
    allow_provisional_clip_b_fallback: bool = False,
) -> dict:
    """v5 Privileged Multi-View EchoJEPA forward (dataloader-complete).

    Signal hierarchy (v4 revision):
      Primary       : L_pair_view  (student (z_shared, z_phase, z_view,
                      src, tgt, Δφ) → teacher z_view slot on target clip)
      Primary       : L_view_nce   (contrastive retrieval of target-view
                      latents across batch)
      Stabilizer    : L_pair_shared (demoted from v3 default)
      Auxiliary     : L_shared     (paired NT-Xent on z_shared)
      Sparse auxil. : L_fused      (Bernoulli-active; mean_shared mode)
      Off by default: L_phase_rel  (enable after pair_view is stable)
      Off by default: L_local_motion (TAPSE-oriented, future work)

    Fixes over the v1-legacy forward:
      2. L_phase_rel uses a conditional ``PhaseQueryHead`` on z_phase
         with (src_view, tgt_view, Δφ) conditioning — not ``q=z_phase``.
      3. L_shared is paired NT-Xent (z_shared_src ↔ t_shared_target),
         other-study targets as negatives — not batch-hash-only.
      4. L_pair is split into L_pair_shared (z_shared → t_shared) and
         L_pair_view (z_shared + z_phase + z_view → t_view). z_view
         receives gradient through L_pair_view, so the third slot is
         no longer a dead branch.
      5. Teacher-side factorized slots come from an EMA copy of
         ``factorized_head``, under ``no_grad``, with ``.detach()`` on
         the outputs. Fused-teacher target uses ``mv_teacher_fusion_ema``.
         No gradient flows into the EMA heads via any target branch.
      6. Bernoulli(p_fused) gate is computed upstream (``fused_active``
         parameter) so all ranks take the same branch. When inactive,
         the fused-teacher forward is skipped entirely — DDP wrappers
         must use find_unused_parameters=True on mv_teacher_fusion /
         shared_projector to avoid a reducer mismatch.

    Target semantics for L_pair:
      - ``t_shared_target`` = EMA factorized_head(teacher_pool_on_target).z_shared
      - ``t_view_target``   = EMA factorized_head(teacher_pool_on_target).z_view
      These are both detached. Predicting t_shared teaches
      view-invariant content; predicting t_view teaches view-specific
      residual that the student has to hallucinate from the source.

    Reuse of 3-clip sampler:
      For this v2 path we still consume ``pair.clip_b`` as the
      pairwise TARGET and ``pair.clip_b_neg`` as the phase-rel hard
      negative. The real cross-view target (Fix 1, different target
      views per study) will land with the sampler extension;
      this function is correct today for the single-target-view-per-
      step mode that the current sampler produces.
    """
    from app.vjepa_multiview.phase_relational_head import pool_tokens

    if pair.clip_b_neg is None:
        raise ValueError(
            "forward_privileged_multiview requires pair.clip_b_neg (3-clip sampler path). " "Run with group_size>=3."
        )
    B = pair.clip_a[0].size(0)
    device = pair.clip_a[0].device

    # --- v5 Fix 1e(d): fail-loud guards on real target_clip + fused_clips.
    #
    # When pair_view / view_nce are active, the scientific MV2SV path
    # requires pair.target_clip from the sampler — NOT the provisional
    # clip_b reuse. Falling back silently would reproduce the
    # positive-only same-study failure mode. Explicit opt-in is required
    # via ``allow_provisional_clip_b_fallback=True`` (only appropriate
    # for pre-sampler wiring smokes).
    mv2sv_pair_active = (lambda_pair_view > 0.0) or (lambda_view_nce > 0.0)
    used_clip_b_fallback = False
    if mv2sv_pair_active:
        if pair.target_clip is None:
            if not allow_provisional_clip_b_fallback:
                raise ValueError(
                    "MV2SV pair loss is active (lambda_pair_view > 0 or "
                    "lambda_view_nce > 0) but pair.target_clip is None. "
                    "The scientific run requires real target_clip tensors "
                    "from the MV2SV sampler; refusing to silently fall "
                    "back to pair.clip_b. Either (a) enable the sampler "
                    "extension by setting phase_multiview.privileged_multiview."
                    "mv2sv_sampler.enabled=true, or (b) explicitly opt into "
                    "the provisional fallback with "
                    "phase_multiview.privileged_multiview."
                    "allow_provisional_clip_b_fallback=true."
                )
            used_clip_b_fallback = True
        if pair.target_views is None and pair.target_clip is not None:
            raise ValueError(
                "pair.target_clip present but pair.target_views is None — " "sampler/collator wiring mismatch."
            )
        if pair.target_delta_phase is None and pair.target_clip is not None:
            raise ValueError("pair.target_clip present but pair.target_delta_phase is None.")

    # Fused guard: if fused is enabled (by gate OR by non-zero lambda),
    # real fused_clips + valid_mask must be present with enough valid
    # per-row views. Otherwise raise.
    mv2sv_fused_active = fused_active and (lambda_fused > 0.0)
    if mv2sv_fused_active:
        if pair.fused_clips is None or pair.fused_valid_mask is None:
            if not allow_provisional_clip_b_fallback:
                raise ValueError(
                    "MV2SV fused loss is active (fused_active=True, "
                    "lambda_fused > 0) but pair.fused_clips / "
                    "fused_valid_mask are None. Enable the fused sampler "
                    "pool or set lambda_fused=0 until Fix 1 is complete."
                )
        elif pair.fused_valid_mask is not None:
            mean_valid = pair.fused_valid_mask.float().sum(dim=1).mean().item()
            if mean_valid < 2.0:
                raise ValueError(
                    f"fused_valid_mask mean valid views = {mean_valid:.2f} "
                    f"< 2. The fused pool is too sparse to run L_fused. "
                    f"Either increase sampler's n_fused_min or set "
                    f"lambda_fused=0."
                )

    # --- Pick the target-clip tensor. Use pair.target_clip when
    # populated by the MV2SV sampler; otherwise fall back to clip_b
    # (phase_relational legacy path) iff the guard above allowed it.
    if pair.target_clip is not None:
        target_clip_tensor = pair.target_clip[0]
    else:
        target_clip_tensor = pair.clip_b[0]

    # --- Teacher: one concat forward on [clip_a, target_clip, clip_b_neg] ---
    concat_fpc = [
        torch.cat([pair.clip_a[0], target_clip_tensor, pair.clip_b_neg[0]], dim=0),
    ]
    with torch.no_grad():
        h_concat = target_encoder(concat_fpc)
        h_concat = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_concat]
    h_a = [hi[:B] for hi in h_concat]
    h_tgt = [hi[B : 2 * B] for hi in h_concat]
    h_neg = [hi[2 * B :] for hi in h_concat]

    # --- Student forward + L_intra ---
    z_ctx = encoder(pair.clip_a, pair.masks_enc)
    assert isinstance(z_ctx, list) and len(z_ctx) >= 1
    assert isinstance(z_ctx[0], list) and len(z_ctx[0]) >= 1
    z_pred = predictor(z_ctx, pair.masks_enc, pair.masks_pred, delta_phi=None)
    intraview = _jepa_loss_fn(z_pred, h_a, pair.masks_pred, loss_exp=loss_exp)

    # --- Student-side factorized slots (carries grad) ---
    c_a_pool = pool_tokens(z_ctx[0][0])
    s_slots = factorized_head(c_a_pool)
    z_shared = s_slots["z_shared"]
    z_phase = s_slots["z_phase"]
    z_view = s_slots["z_view"]

    # --- Teacher-side factorized slots from EMA head (no grad, detached) ---
    y_tgt_pool = pool_tokens(h_tgt[0]).detach()
    y_neg_pool = pool_tokens(h_neg[0]).detach()
    with torch.no_grad():
        t_tgt_slots = factorized_head_ema(y_tgt_pool)
        t_neg_slots = factorized_head_ema(y_neg_pool)
    t_tgt_shared = t_tgt_slots["z_shared"].detach()
    t_tgt_phase = t_tgt_slots["z_phase"].detach()
    t_tgt_view = t_tgt_slots["z_view"].detach()
    t_neg_phase = t_neg_slots["z_phase"].detach()

    # --- Predictor conditioning ---
    # Base (legacy): source_view=clip_a_view, target_view=clip_b_view,
    # Δφ=(target_phi_b - target_phi_a) mod 1.
    # MV2SV (when target_clip present): override target_view and Δφ with
    # the real target_clip metadata from the sampler.
    src_view_ids, tgt_view_ids, delta_phase, study_hashes = _build_predictor_inputs(
        meta_list,
        device=device,
    )
    if pair.target_clip is not None and pair.target_views is not None:
        from app.vjepa_multiview.phase_relational_head import view_to_id

        tgt_view_ids = torch.tensor(
            [view_to_id(v) for v in pair.target_views],
            device=device,
            dtype=torch.long,
        )
    if pair.target_clip is not None and pair.target_delta_phase is not None:
        delta_phase = pair.target_delta_phase.to(device).float()
        # Guard against NaN → treat as 0.0 (same-phase) rather than
        # propagate garbage through the Fourier encoder.
        delta_phase = torch.nan_to_num(delta_phase, nan=0.0)

    # --- Fix 4a: L_pair_shared — student z_shared -> teacher t_shared ---
    q_pair_shared = pair_shared_projector(z_shared)
    pair_shared_loss = F.smooth_l1_loss(q_pair_shared, t_tgt_shared)

    # --- Fix 4b: L_pair_view — student (z_shared, z_phase, z_view, views, Δφ) -> teacher t_view ---
    if use_z_view:
        q_pair_view = view_predictor(
            z_shared,
            z_phase,
            src_view_ids,
            tgt_view_ids,
            delta_phase,
            z_view=z_view,
        )
    else:
        q_pair_view = view_predictor(
            z_shared,
            z_phase,
            src_view_ids,
            tgt_view_ids,
            delta_phase,
        )
    pair_view_loss = F.smooth_l1_loss(q_pair_view, t_tgt_view)

    # --- v4 primary: L_view_nce — cross-view retrieval contrastive.
    # Student's predicted target-view latent q_pair_view must retrieve
    # its own target (diagonal) against other rows' target-view latents.
    # Same-study off-diagonals are masked. Operates on L2-normalized
    # projections so the loss is scale-invariant.
    q_view_norm = F.normalize(q_pair_view.float(), dim=-1)
    t_view_norm = F.normalize(t_tgt_view.float(), dim=-1)
    view_nce_loss, view_nce_diag = _view_nce_loss(
        q_view_norm,
        t_view_norm,
        study_hashes=study_hashes,
        target_view_ids=tgt_view_ids,
        tau_view=tau_view,
        mask_same_study_batch_negatives=mask_same_study_batch_negatives,
    )

    # --- Fix 3: L_shared paired NT-Xent (z_shared_src ↔ t_shared_target) ---
    z_shared_norm = F.normalize(z_shared.float(), dim=-1)
    t_shared_norm = F.normalize(t_tgt_shared.float(), dim=-1)
    shared_loss, shared_diag = _paired_shared_ntxent(
        z_shared_norm,
        t_shared_norm,
        tau=tau_shared,
    )

    # --- Fix 2: L_phase_rel via conditional PhaseQueryHead on z_phase ---
    q_phase_pre, y_pos_phase_pre, y_hard_phase_pre = phase_query_head(
        z_phase,
        src_view_ids,
        tgt_view_ids,
        delta_phase,
        t_tgt_phase,
        t_neg_phase,
    )
    q_phase_norm = F.normalize(q_phase_pre.float(), dim=-1)
    y_pos_phase_norm = F.normalize(y_pos_phase_pre.float(), dim=-1)
    y_hard_phase_norm = F.normalize(y_hard_phase_pre.float(), dim=-1)
    phase_rel_out = _relational_infonce_with_hard_neg(
        q=q_phase_norm,
        y_pos=y_pos_phase_norm,
        y_hard=y_hard_phase_norm,
        study_hashes=study_hashes,
        tau=tau_phase,
        mask_same_study_batch_negatives=mask_same_study_batch_negatives,
        disable_hard_negative=False,
    )

    # --- L_fused (sparse): Bernoulli-gated auxiliary.
    #
    # Two modes supported:
    #   mean_shared  (default, safe): fused target = masked-mean of
    #       per-clip z_shared slots produced by factorized_head_ema.
    #       No mv_teacher_fusion involved — deterministic, no random
    #       EMA-of-untrained-attention problem.
    #   attention_ema (experimental): fused target = mv_teacher_fusion_ema.
    #       Disabled by default; caller must set allow_untrained_attention_fusion
    #       externally (main() verifies the guard).
    #
    # The pool assembled here is the current v1-era provisional pool
    # [target_clip, hard_neg_clip] — this is NOT a true cross-view pool.
    # Once Fix 1 sampler lands, the pool will be per-clip same-study
    # different-view clips; this function's interface doesn't change.
    fused_valid_views_mean = torch.zeros((), device=device)
    fused_valid_views_min = torch.zeros((), device=device)
    fused_shared_target_norm = torch.zeros((), device=device)
    fused_shared_q_norm = torch.zeros((), device=device)
    fused_shared_cos_q_target = torch.zeros((), device=device)
    if fused_active:
        # Fused pool assembly.
        #
        # v5 preferred: real MV2SV fused_clips tensor from the sampler
        # (shape [B, N_fused, C, T, H, W]) + explicit valid_mask.
        # Legacy fallback (v1-era provisional path): reuse target_clip
        # + hard_neg as a 2-clip pool. Fail-loud guard above prevented
        # this when lambda_fused > 0 and allow_provisional_clip_b_fallback
        # is False, so this fallback only fires in deliberately
        # provisional smokes.
        if pair.fused_clips is not None and pair.fused_valid_mask is not None:
            fused_video = pair.fused_clips[0]  # [B, N_fused, C, T, H, W]
            N_fused = fused_video.size(1)
            # Flatten N_fused into batch for one teacher forward.
            C_, T_, H_, W_ = fused_video.size(2), fused_video.size(3), fused_video.size(4), fused_video.size(5)
            flat = fused_video.reshape(B * N_fused, C_, T_, H_, W_)
            with torch.no_grad():
                h_fused = target_encoder([flat])
                h_fused = [F.layer_norm(hi, (hi.size(-1),)) for hi in h_fused]
            # Pool teacher tokens → [B*N_fused, D].
            pooled_flat = pool_tokens(h_fused[0]).detach()
            D_enc = pooled_flat.size(-1)
            pooled_nv = pooled_flat.reshape(B, N_fused, D_enc)
            valid_mask = pair.fused_valid_mask.to(device)
        else:
            # Provisional fallback: same-view-different-phase pool.
            pooled_nv = torch.stack([y_tgt_pool, y_neg_pool], dim=1)
            valid_mask = torch.ones(B, pooled_nv.size(1), dtype=torch.bool, device=device)

        if fused_target_mode == "mean_shared":
            t_fused_shared, fused_diag = _mean_shared_fused_target(
                pooled_nv,
                valid_mask,
                factorized_head_ema,
            )
            fused_valid_views_mean = fused_diag["fused_valid_views_mean"]
            fused_valid_views_min = fused_diag["fused_valid_views_min"]
            fused_shared_target_norm = fused_diag["fused_shared_target_norm"]
        elif fused_target_mode == "attention_ema":
            if mv_teacher_fusion_ema is None:
                raise ValueError(
                    "fused_target_mode='attention_ema' requires "
                    "mv_teacher_fusion_ema, but it was not passed in. "
                    "Check main() gated construction."
                )
            from app.vjepa_multiview.phase_relational_head import view_to_id

            def _phase_val(m, k):
                v = m.get(k)
                if v is None or (isinstance(v, float) and v != v):
                    return 0.0
                return float(v)

            tgt_view_list = [view_to_id(m.get("clip_b_view")) for m in meta_list]
            tgt_phase_list = [(_phase_val(m, "target_phi_b") - _phase_val(m, "target_phi_a")) % 1.0 for m in meta_list]
            neg_view_list = [view_to_id(m.get("clip_b_view")) for m in meta_list]
            neg_phase_list = [
                (_phase_val(m, "target_phi_b_neg") - _phase_val(m, "target_phi_a")) % 1.0 for m in meta_list
            ]
            view_ids_nv = torch.tensor(
                list(zip(tgt_view_list, neg_view_list)),
                device=device,
                dtype=torch.long,
            )
            phase_nv = torch.tensor(
                list(zip(tgt_phase_list, neg_phase_list)),
                device=device,
                dtype=torch.float32,
            )
            with torch.no_grad():
                t_fused_shared = mv_teacher_fusion_ema(pooled_nv, view_ids_nv, phase_nv)
            t_fused_shared = t_fused_shared.detach()
            with torch.no_grad():
                fused_valid_views_mean = valid_mask.float().sum(dim=1).mean()
                fused_valid_views_min = valid_mask.float().sum(dim=1).min()
                fused_shared_target_norm = t_fused_shared.norm(dim=-1).mean()
        else:
            raise ValueError(f"fused_target_mode={fused_target_mode!r}; " f"want 'mean_shared' or 'attention_ema'")

        q_fused = fused_shared_projector(z_shared)
        fused_loss = F.smooth_l1_loss(q_fused, t_fused_shared)
        with torch.no_grad():
            fused_shared_q_norm = q_fused.norm(dim=-1).mean()
            q_n = F.normalize(q_fused.float(), dim=-1)
            t_n = F.normalize(t_fused_shared.float(), dim=-1)
            fused_shared_cos_q_target = (q_n * t_n).sum(dim=-1).mean()
        fused_active_flag = torch.ones((), device=device)
    else:
        # When inactive, skip the fused-teacher forward entirely.
        # In mean_shared mode there is no mv_teacher_fusion in the DDP
        # graph, so no dummy forward is required. Both projectors are
        # still used every step (pair_shared is live; fused_shared is
        # wrapped with find_unused_parameters=True in main() so its
        # grad can legally be None on inactive steps).
        fused_loss = torch.zeros((), device=device)
        fused_active_flag = torch.zeros((), device=device)

    # --- Local-motion auxiliary (v4 scaffolding; disabled by default) ---
    # When ``lambda_local_motion > 0``, a per-token or per-region motion
    # contrast between the student's A4C context and a teacher-side
    # same-view different-phase target would be computed here. The real
    # implementation needs the sampler to emit a same-view Δφ-displaced
    # clip, which Fix 1 does not yet provide. For now this is a
    # placeholder that raises if enabled — preventing accidental use.
    if lambda_local_motion > 0.0:
        raise NotImplementedError(
            "lambda_local_motion > 0 requires the local-motion sampler "
            "extension (not yet implemented). Keep lambda_local_motion=0 "
            "until the token-level same-view Δφ clip path lands."
        )
    local_motion_loss = torch.zeros((), device=device)

    # --- Diagnostics ---
    with torch.no_grad():
        pair_q_shared_norm = q_pair_shared.norm(dim=-1).mean()
        pair_q_view_norm = q_pair_view.norm(dim=-1).mean()
        z_shared_vs_z_phase_cos = F.cosine_similarity(z_shared, z_phase, dim=-1).mean()
        z_shared_vs_z_view_cos = F.cosine_similarity(z_shared, z_view, dim=-1).mean()
        # Per-batch source/target view counts for the on-run sanity log.
        # These are small dicts the training loop can serialize.
        src_view_counts: dict[int, int] = {}
        tgt_view_counts: dict[int, int] = {}
        src_tgt_pair_counts: dict[tuple[int, int], int] = {}
        for sv, tv in zip(src_view_ids.tolist(), tgt_view_ids.tolist()):
            src_view_counts[int(sv)] = src_view_counts.get(int(sv), 0) + 1
            tgt_view_counts[int(tv)] = tgt_view_counts.get(int(tv), 0) + 1
            key = (int(sv), int(tv))
            src_tgt_pair_counts[key] = src_tgt_pair_counts.get(key, 0) + 1
        # Variance of each slot (across batch dim), averaged over the
        # feature dim. Useful as a collapse sentinel — if a slot's var
        # drops near zero, the head is emitting a constant.
        z_shared_var = z_shared.var(dim=0).mean()
        z_phase_var = z_phase.var(dim=0).mean()
        z_view_var = z_view.var(dim=0).mean()
        # Cosine between student query and target for each pair loss.
        # Tracks calibration: a good student's cosine should climb
        # from ~0 toward ~1 as training progresses.
        pair_shared_cos_q_target = (
            (F.normalize(q_pair_shared.float(), dim=-1) * F.normalize(t_tgt_shared.float(), dim=-1)).sum(dim=-1).mean()
        )
        pair_view_cos_q_target = (
            (F.normalize(q_pair_view.float(), dim=-1) * F.normalize(t_tgt_view.float(), dim=-1)).sum(dim=-1).mean()
        )

        # Per-target-view pair_view cos + retrieval top1 for a small
        # set of clinically relevant target views. These are
        # heart-muscle- and great-vessel-centric; MV2SV v4 is
        # hypothesized to help most on RVSP/MR/AS via view
        # hallucination of non-A4C targets.
        from app.vjepa_multiview.phase_relational_head import VIEW_ID_MAP

        view_nce_diag_by_view = view_nce_diag.get("view_nce_top1_by_view", {})
        _RETRIEVAL_TARGETS = ("PLAX", "A5C", "A3C", "A2C")
        pair_view_cos_by_view: dict[str, torch.Tensor] = {}
        view_nce_top1_by_view: dict[str, torch.Tensor] = {}
        for view_name in _RETRIEVAL_TARGETS:
            vid = VIEW_ID_MAP.get(view_name)
            if vid is None:
                continue
            sel = tgt_view_ids == int(vid)
            if sel.any():
                q_sel = F.normalize(q_pair_view[sel].float(), dim=-1)
                t_sel = F.normalize(t_tgt_view[sel].float(), dim=-1)
                pair_view_cos_by_view[view_name] = (q_sel * t_sel).sum(dim=-1).mean().detach()
            if int(vid) in view_nce_diag_by_view:
                view_nce_top1_by_view[view_name] = view_nce_diag_by_view[int(vid)]

    out = {
        "intraview_loss": intraview,
        "crossview_loss": torch.zeros((), device=device),
        "pair_shared_loss": pair_shared_loss,
        "pair_view_loss": pair_view_loss,
        "view_nce_loss": view_nce_loss,
        "shared_loss": shared_loss,
        "phase_rel_loss": phase_rel_out["rel_loss"],
        "phase_rel_top1": phase_rel_out["rel_top1_with_hard"],
        "phase_rel_pos_minus_hard_gap": phase_rel_out["rel_pos_minus_hard_gap"],
        "fused_loss": fused_loss,
        "fused_active": fused_active_flag,
        "fused_target_mode": fused_target_mode,
        # v5 sanity / scientific-run safety: was the legacy clip_b
        # fallback used in place of the sampler's target_clip?
        "used_clip_b_fallback": torch.tensor(float(used_clip_b_fallback), device=device),
        # pct_target_clip_present: fraction of batch rows where the
        # sampler delivered a real target_clip. Scientific runs must
        # have this == 1.0.
        "pct_target_clip_present": (
            pair.target_clip_present.float().mean().detach()
            if getattr(pair, "target_clip_present", None) is not None
            else torch.tensor(0.0, device=device)
        ),
        # pct_fused_clips_present: 1 if pair.fused_clips was provided,
        # else 0.
        "pct_fused_clips_present": torch.tensor(
            1.0 if pair.fused_clips is not None else 0.0,
            device=device,
        ),
        "fused_valid_views_mean": fused_valid_views_mean,
        "fused_valid_views_min": fused_valid_views_min,
        "fused_shared_target_norm": fused_shared_target_norm,
        "fused_shared_q_norm": fused_shared_q_norm,
        "fused_shared_cos_q_target": fused_shared_cos_q_target,
        "local_motion_loss": local_motion_loss,
        "paired_shared_top1": shared_diag["paired_shared_top1"],
        "paired_shared_pos_sim": shared_diag["paired_shared_pos_sim"],
        "view_nce_top1": view_nce_diag["view_nce_top1"],
        "view_nce_pos_sim_mean": view_nce_diag["view_nce_pos_sim_mean"],
        "view_nce_neg_sim_mean": view_nce_diag["view_nce_neg_sim_mean"],
        "view_nce_valid_neg_count_mean": view_nce_diag["view_nce_valid_neg_count_mean"],
        "view_nce_valid_neg_count_min": view_nce_diag["view_nce_valid_neg_count_min"],
        "view_nce_same_target_view_fraction": view_nce_diag["view_nce_same_target_view_fraction"],
        "view_nce_fallback_fraction": view_nce_diag["view_nce_fallback_fraction"],
        "diag_pair_q_shared_norm": pair_q_shared_norm,
        "diag_pair_q_view_norm": pair_q_view_norm,
        "diag_pair_shared_cos_q_target": pair_shared_cos_q_target,
        "diag_pair_view_cos_q_target": pair_view_cos_q_target,
        "diag_pair_view_cos_by_view": pair_view_cos_by_view,
        "diag_view_nce_top1_by_view": view_nce_top1_by_view,
        # v5 on-run sanity diagnostics (Fix 1e / task 207). Dict-valued
        # — serialized by the training loop every log_every_steps.
        "src_view_counts": src_view_counts,
        "tgt_view_counts": tgt_view_counts,
        "src_tgt_pair_counts": src_tgt_pair_counts,
        "diag_z_shared_vs_z_phase_cos": z_shared_vs_z_phase_cos,
        "diag_z_shared_vs_z_view_cos": z_shared_vs_z_view_cos,
        "diag_z_shared_var": z_shared_var,
        "diag_z_phase_var": z_phase_var,
        "diag_z_view_var": z_view_var,
        "same_study_masked_count": phase_rel_out["same_study_masked_count"],
        "multiview_objective": "privileged_multiview",
    }

    total = (
        intraview
        + lambda_pair_shared * pair_shared_loss
        + lambda_pair_view * pair_view_loss
        + lambda_view_nce * view_nce_loss
        + lambda_shared * shared_loss
        + lambda_phase * phase_rel_out["rel_loss"]
        + lambda_fused * fused_loss
        + lambda_local_motion * local_motion_loss
    )
    out["total_loss"] = total
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
    placeholder_csv = cfgs_data.get("placeholder_csv", "/tmp/phase_multiview_placeholder.csv")

    # phase_multiview block drives the phase_matched data_manager branch.
    cfg_pmv = args.get("phase_multiview", {}) or {}
    if not cfg_pmv.get("enabled", False):
        raise ValueError("vjepa_multiview app requires phase_multiview.enabled=true in config")
    sampler_type = cfg_pmv.get("sampler_type", "phase_matched")
    lambda_crossview = float(cfg_pmv.get("lambda_crossview", 0.25))
    use_intraview_loss = bool(cfg_pmv.get("use_intraview_loss", True))
    use_crossview_loss = bool(cfg_pmv.get("use_crossview_loss", True))
    log_every_steps = int(cfg_pmv.get("log_every_steps", 10))
    debug_verify_frame_count = bool(cfg_pmv.get("debug_verify_frame_count", False))
    debug_verify_n = int(cfg_pmv.get("debug_verify_n", 8))

    # --- New: multiview_objective dispatch + relational config --- #
    multiview_objective = str(cfg_pmv.get("multiview_objective", "smooth_l1"))
    if multiview_objective not in (
        "smooth_l1",
        "intraview_only",
        "phase_relational",
        "privileged_multiview",
        "token_phase_relational",
        "mcc_jepa",
    ):
        raise ValueError(
            f"multiview_objective={multiview_objective!r}; want "
            f"'smooth_l1' | 'intraview_only' | 'phase_relational' | "
            f"'privileged_multiview' | 'token_phase_relational' | 'mcc_jepa'"
        )
    # In intraview_only mode the crossview loss is disabled regardless of
    # what the YAML says; force-disable to avoid a confusing config.
    if multiview_objective == "intraview_only":
        use_crossview_loss = False
    # privileged_multiview: the crossview scalar in the legacy CSV stays
    # at 0 by convention (pair/fused/shared/phase losses are logged
    # separately). Force disable the legacy crossview path.
    if multiview_objective == "privileged_multiview":
        use_crossview_loss = False
    if multiview_objective == "token_phase_relational":
        use_crossview_loss = False
    # MCC-JEPA (target-anchored or pure) drives its own loss via
    # app/vjepa_multiview/mcc_jepa_forward.py; the legacy crossview path
    # is irrelevant.
    if multiview_objective == "mcc_jepa":
        use_crossview_loss = False
    # ---- MCC-JEPA config block ----
    mcc_cfg = cfg_pmv.get("mcc_adapter", {}) or {}
    mcc_mode = str(cfg_pmv.get("mcc_mode", "target_anchored"))
    if mcc_mode not in ("pure", "target_anchored"):
        raise ValueError(f"mcc_mode must be 'pure' or 'target_anchored'; got {mcc_mode!r}")
    mcc_lambda_mcc = float(cfg_pmv.get("lambda_mcc", 0.2))
    mcc_lambda_vjepa_self = float(cfg_pmv.get("lambda_vjepa", 1.0))
    mcc_adapter_num_heads = int(mcc_cfg.get("num_heads", 8))
    mcc_adapter_gamma_init = float(mcc_cfg.get("gamma_init", 0.0))
    mcc_adapter_source_proj_dim = mcc_cfg.get("source_proj_dim", None)

    # --- privileged_multiview (MV2SV) config block ---
    privview_cfg = cfg_pmv.get("privileged_multiview", {}) or {}
    # v4 defaults: pair_view is the primary signal; pair_shared is a
    # stabilizer at 0.05-0.10; view_nce retrieval contrastive adds
    # cross-view discrimination.
    pmv_lambda_pair_legacy = float(privview_cfg.get("lambda_pair", 0.05))
    pmv_lambda_pair_shared = float(privview_cfg.get("lambda_pair_shared", pmv_lambda_pair_legacy))
    pmv_lambda_pair_view = float(privview_cfg.get("lambda_pair_view", 0.10))
    pmv_lambda_view_nce = float(privview_cfg.get("lambda_view_nce", 0.025))
    pmv_lambda_fused = float(privview_cfg.get("lambda_fused", 0.0))
    pmv_lambda_shared = float(privview_cfg.get("lambda_shared", 0.05))
    pmv_lambda_phase = float(privview_cfg.get("lambda_phase", 0.0))
    pmv_lambda_local_motion = float(privview_cfg.get("lambda_local_motion", 0.0))
    pmv_p_fused = float(privview_cfg.get("p_fused", 0.0))
    pmv_tau_static = float(privview_cfg.get("tau_static", 0.10))
    pmv_tau_phase = float(privview_cfg.get("tau_phase", 0.10))
    pmv_tau_view = float(privview_cfg.get("tau_view", 0.10))
    pmv_warmup_epochs = float(privview_cfg.get("warmup_epochs", 5.0))
    pmv_shared_dim = int(privview_cfg.get("shared_dim", 256))
    pmv_phase_dim = int(privview_cfg.get("phase_dim", 256))
    pmv_view_dim = int(privview_cfg.get("view_dim", 256))
    pmv_head_hidden = int(privview_cfg.get("head_hidden_dim", 1024))
    pmv_view_embed_dim = int(privview_cfg.get("view_embedding_dim", 64))
    pmv_n_phase_freqs = int(privview_cfg.get("n_phase_freqs", 4))
    pmv_embed_dim = int(privview_cfg.get("embed_dim", 1024))
    pmv_fusion_heads = int(privview_cfg.get("fusion_num_heads", 8))
    pmv_use_z_view = bool(privview_cfg.get("use_z_view", True))
    # Fused-target generator mode. "mean_shared" (default, safe) uses
    # factorized_head_ema on per-clip teacher pools and averages the
    # z_shared slots. "attention_ema" uses mv_teacher_fusion_ema, which
    # is an EMA of an untrained online module unless an independent
    # teacher-side objective is added — hence the guard below.
    pmv_fused_target_mode = str(privview_cfg.get("fused_target_mode", "mean_shared"))
    if pmv_fused_target_mode not in ("mean_shared", "attention_ema"):
        raise ValueError(f"fused_target_mode={pmv_fused_target_mode!r}; " f"want 'mean_shared' or 'attention_ema'")
    pmv_allow_untrained_attention_fusion = bool(privview_cfg.get("allow_untrained_attention_fusion", False))
    if pmv_fused_target_mode == "attention_ema" and not pmv_allow_untrained_attention_fusion:
        raise ValueError(
            "fused_target_mode='attention_ema' is disabled by default: "
            "mv_teacher_fusion is not trained by an independent objective, "
            "so an EMA of it is an EMA of a random attention fuser. "
            "Set allow_untrained_attention_fusion=true only after adding "
            "a teacher-side training loss for mv_teacher_fusion, or keep "
            "the default fused_target_mode='mean_shared'."
        )
    # --- MV2SV sampler-side config (Fix 1e) ---
    # Passed through to the phase_matched sampler so it emits target_clip
    # and (optionally) fused_clips on every MatchRecord. Dataloader side
    # then loads those as additional views on the collated batch.
    pmv_mv2sv_sampler_cfg = privview_cfg.get("mv2sv_sampler", {}) or {}
    pmv_mv2sv_enabled = bool(pmv_mv2sv_sampler_cfg.get("enabled", False))
    pmv_fused_n_min = int((pmv_mv2sv_sampler_cfg.get("fused_pool", {}) or {}).get("n_fused_min", 2))
    pmv_fused_n_max = int((pmv_mv2sv_sampler_cfg.get("fused_pool", {}) or {}).get("n_fused_max", 2))
    pmv_mv2sv_fused_enabled = bool((pmv_mv2sv_sampler_cfg.get("fused_pool", {}) or {}).get("enabled", False))
    # Fail-loud guard overrides (Fix 1d). Default disallow the v1/v3
    # provisional clip_b fallback — scientific runs must use real
    # target_clip from the sampler. Set True only during the legacy
    # "reuse clip_b as target" wiring smoke.
    pmv_allow_provisional_clip_b_fallback = bool(privview_cfg.get("allow_provisional_clip_b_fallback", False))
    rel_cfg = cfg_pmv.get("relational", {}) or {}
    lambda_rel = float(rel_cfg.get("lambda_rel", 0.05))
    rel_warmup_epochs = float(rel_cfg.get("rel_warmup_epochs", 5))
    rel_temperature = float(rel_cfg.get("rel_temperature", 0.10))
    rel_mask_same_study_batch_negatives = bool(rel_cfg.get("rel_mask_same_study_batch_negatives", True))
    target_projector_trainable = bool(rel_cfg.get("target_projector_trainable", True))
    # Negative-set composition. Default is the hardneg method. The
    # "no_hardneg" ablation masks column 1 of the InfoNCE logits to -inf
    # so the same-study wrong-phase hard negative no longer contributes
    # to the loss gradient, while keeping the 3-clip sampler, teacher
    # concat forward, and relational head DDP invariant identical to the
    # method run. Data path parity is preserved end-to-end.
    rel_negative_mode = str(rel_cfg.get("rel_negative_mode", "hard_plus_batch"))
    if rel_negative_mode not in ("hard_plus_batch", "no_hardneg"):
        raise ValueError(
            f"rel_negative_mode={rel_negative_mode!r}; want " f"'hard_plus_batch' (default) or 'no_hardneg' (ablation)"
        )
    disable_hard_negative = rel_negative_mode == "no_hardneg"

    # --- token_phase_relational config block ---
    tok_cfg = cfg_pmv.get("token_relational", {}) or {}
    tok_enabled = bool(tok_cfg.get("enabled", multiview_objective == "token_phase_relational"))
    tok_embed_dim = int(tok_cfg.get("embed_dim", 1024))
    tok_rel_dim = int(tok_cfg.get("rel_dim", 256))
    tok_hidden_dim = int(tok_cfg.get("hidden_dim", 1024))
    tok_view_embed_dim = int(tok_cfg.get("view_embedding_dim", 64))
    tok_n_phase_freqs = int(tok_cfg.get("n_phase_freqs", 4))
    tok_subsample_k = int(tok_cfg.get("token_subsample_k", 64))
    tok_tau_token = float(tok_cfg.get("tau_token", 0.10))
    tok_lambda_token_rel = float(tok_cfg.get("lambda_token_rel", 0.02))
    tok_lambda_pool_rel = float(tok_cfg.get("lambda_pool_rel", 0.005))
    tok_lambda_delta = float(tok_cfg.get("lambda_delta", 0.0))
    tok_warmup_epochs = float(tok_cfg.get("warmup_epochs", 5.0))
    tok_mask_same_study = bool(tok_cfg.get("mask_same_study_batch_negatives", True))
    tok_pool_rel_temperature = float(tok_cfg.get("pool_rel_temperature", 0.10))
    # --- motion_delta config block ---
    md_cfg = cfg_pmv.get("motion_delta", {}) or {}
    md_enabled = bool(md_cfg.get("enabled", False))
    md_delta_dim = int(md_cfg.get("delta_dim", 256))
    md_hidden_dim = int(md_cfg.get("hidden_dim", 1024))
    md_tau_delta = float(md_cfg.get("tau_delta", 0.10))
    md_lambda_l1 = float(md_cfg.get("lambda_delta_l1", 1.0))
    md_lambda_nce = float(md_cfg.get("lambda_delta_nce", 1.0))
    md_same_view_only = bool(md_cfg.get("same_view_only", True))
    # If motion_delta.enabled and lambda_delta from token_relational is 0,
    # allow the motion_delta block itself to override. This lets us gate
    # the presence of the delta heads via motion_delta.enabled and the
    # loss weight via token_relational.lambda_delta (or md_cfg.lambda_delta).
    md_lambda_delta_override = md_cfg.get("lambda_delta", None)
    if md_lambda_delta_override is not None:
        tok_lambda_delta = float(md_lambda_delta_override)
    if md_enabled and not md_same_view_only:
        raise ValueError(
            "motion_delta.same_view_only=false is not implemented — "
            "cross-view token delta would require an assignment/transport "
            "target. Set same_view_only=true or keep motion_delta disabled."
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
                ("%d", "epoch"),
                ("%d", "itr"),
                ("%.6f", "loss"),
                ("%.6f", "intraview"),
                ("%.6f", "crossview"),
                ("%d", "iter-time(ms)"),
                ("%d", "data-time(ms)"),
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
                ("%d", "rel_target_projector_trainable"),
                ("%d", "same_study_masked_count"),
            )
        elif multiview_objective == "privileged_multiview":
            # Privileged MV v4 schema. First 7 columns stay byte-identical
            # to legacy; everything after is MV2SV-specific. v4 adds
            # view_nce + per-target-view retrieval diagnostics.
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"),
                ("%d", "itr"),
                ("%.6f", "loss"),
                ("%.6f", "intraview"),
                ("%.6f", "crossview"),
                ("%d", "iter-time(ms)"),
                ("%d", "data-time(ms)"),
                ("%.6f", "pair_shared_loss"),
                ("%.6f", "pair_view_loss"),
                ("%.6f", "view_nce_loss"),
                ("%.6f", "shared_loss"),
                ("%.6f", "phase_rel_loss"),
                ("%.6f", "phase_rel_top1"),
                ("%.6f", "phase_rel_pos_minus_hard_gap"),
                ("%.6f", "fused_loss"),
                ("%.6f", "fused_active"),
                ("%.6f", "local_motion_loss"),
                ("%.6f", "effective_lambda_pair_shared"),
                ("%.6f", "effective_lambda_pair_view"),
                ("%.6f", "effective_lambda_view_nce"),
                ("%.6f", "effective_lambda_fused"),
                ("%.6f", "paired_shared_top1"),
                ("%.6f", "paired_shared_pos_sim"),
                ("%.6f", "view_nce_top1"),
                ("%.6f", "view_nce_pos_sim_mean"),
                ("%.6f", "view_nce_neg_sim_mean"),
                ("%.6f", "view_nce_valid_neg_count_mean"),
                ("%.6f", "view_nce_valid_neg_count_min"),
                ("%.6f", "view_nce_same_target_view_fraction"),
                ("%.6f", "view_nce_fallback_fraction"),
                ("%.6f", "used_clip_b_fallback"),
                ("%.6f", "pct_target_clip_present"),
                ("%.6f", "pct_fused_clips_present"),
                ("%.6f", "fused_valid_views_mean"),
                ("%.6f", "fused_valid_views_min"),
                ("%.6f", "fused_shared_target_norm"),
                ("%.6f", "fused_shared_q_norm"),
                ("%.6f", "fused_shared_cos_q_target"),
                ("%.6f", "diag_pair_shared_cos_q_target"),
                ("%.6f", "diag_pair_view_cos_q_target"),
                # Per-target-view retrieval / cosine (A4C source → ...).
                # Missing target-view in batch → 0.0 sentinel.
                ("%.6f", "diag_pair_view_cos_PLAX"),
                ("%.6f", "diag_pair_view_cos_A5C"),
                ("%.6f", "diag_pair_view_cos_A3C"),
                ("%.6f", "diag_pair_view_cos_A2C"),
                ("%.6f", "diag_view_nce_top1_PLAX"),
                ("%.6f", "diag_view_nce_top1_A5C"),
                ("%.6f", "diag_view_nce_top1_A3C"),
                ("%.6f", "diag_view_nce_top1_A2C"),
                ("%.6f", "diag_z_shared_vs_z_phase_cos"),
                ("%.6f", "diag_z_shared_vs_z_view_cos"),
                ("%.6f", "diag_pair_q_shared_norm"),
                ("%.6f", "diag_pair_q_view_norm"),
                ("%.6f", "diag_z_shared_var"),
                ("%.6f", "diag_z_phase_var"),
                ("%.6f", "diag_z_view_var"),
                ("%d", "same_study_masked_count"),
            )
        elif multiview_objective == "token_phase_relational":
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"),
                ("%d", "itr"),
                ("%.6f", "loss"),
                ("%.6f", "intraview"),
                ("%.6f", "crossview"),
                ("%d", "iter-time(ms)"),
                ("%d", "data-time(ms)"),
                ("%.6f", "token_rel_loss"),
                ("%.6f", "token_rel_top1_with_hard"),
                ("%.6f", "token_rel_pos_sim_mean"),
                ("%.6f", "token_rel_hard_sim_mean"),
                ("%.6f", "token_rel_batch_neg_sim_mean"),
                ("%.6f", "token_rel_pos_minus_hard_gap"),
                ("%.6f", "token_rel_pos_minus_batch_gap"),
                ("%.6f", "token_rel_logits_std"),
                ("%.6f", "token_rel_q_var"),
                ("%.6f", "token_rel_y_var"),
                ("%.6f", "token_rel_valid_rows"),
                ("%.6f", "token_subsample_k"),
                ("%.6f", "pool_rel_loss"),
                ("%.6f", "pool_rel_top1_with_hard"),
                ("%.6f", "pool_rel_pos_minus_hard_gap"),
                ("%.6f", "delta_loss"),
                ("%.6f", "delta_l1"),
                ("%.6f", "delta_nce"),
                ("%.6f", "delta_valid_rows"),
                ("%.6f", "delta_pos_sim_mean"),
                ("%.6f", "delta_hard_sim_mean"),
                ("%.6f", "delta_pos_minus_hard_gap"),
                ("%.6f", "delta_q_var"),
                ("%.6f", "delta_target_var"),
                ("%.6f", "effective_lambda_token_rel"),
                ("%.6f", "effective_lambda_pool_rel"),
                ("%.6f", "effective_lambda_delta"),
                ("%.6f", "same_view_row_fraction"),
                ("%.6f", "same_family_row_fraction"),
                ("%.6f", "cross_family_row_fraction"),
                ("%d", "token_rel_same_study_masked_count"),
            )
        else:
            csv_logger = CSVLogger(
                log_file,
                ("%d", "epoch"),
                ("%d", "itr"),
                ("%.6f", "loss"),
                ("%.6f", "intraview"),
                ("%.6f", "crossview"),
                ("%d", "iter-time(ms)"),
                ("%d", "data-time(ms)"),
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

        pd.DataFrame({"view_0": ["x"], "view_1": ["y"], "label": [0.0]}).to_csv(placeholder_csv, index=False)
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
    pmv_dispatch_cfg.setdefault("parquet_path", pmv_dispatch_cfg.get("phase_annotations_path"))
    pmv_dispatch_cfg.setdefault("sampler_dir", pmv_dispatch_cfg.get("sampler_dir"))
    pmv_dispatch_cfg.setdefault("quality_tiers", pmv_dispatch_cfg.get("quality_tiers", ["high"]))
    pmv_dispatch_cfg.setdefault("rr_filter_mode", pmv_dispatch_cfg.get("rr_filter_mode", "strict"))
    pmv_dispatch_cfg.setdefault("sampling_mode", pmv_dispatch_cfg.get("sampling_mode", "uniform_phase"))
    pmv_dispatch_cfg.setdefault("phase_tolerance", pmv_dispatch_cfg.get("phase_tolerance", 0.15))
    pmv_dispatch_cfg.setdefault("frames_per_clip", pmv_dispatch_cfg.get("frames_per_clip", max_num_frames))
    pmv_dispatch_cfg.setdefault("frame_step", pmv_dispatch_cfg.get("frame_step", 1))
    pmv_dispatch_cfg.setdefault("pairs_per_study", pmv_dispatch_cfg.get("pairs_per_study", 1))
    pmv_dispatch_cfg.setdefault("allow_frame_step_gt1", pmv_dispatch_cfg.get("allow_frame_step_gt1", False))

    # --- MV2SV sampler config passthrough (Fix 1e) ---
    # When mv2sv_sampler.enabled is True, the phase_matched sampler
    # populates target_clip + fused_clips on every MatchRecord. The
    # pair dataframe gains view_3 + view_4..view_{3+n_fused_max-1}.
    if pmv_mv2sv_enabled:
        pmv_dispatch_cfg["mv2sv_config"] = pmv_mv2sv_sampler_cfg

    # Compute dataloader group_size based on objective + MV2SV config.
    # Base:
    #   smooth_l1                 → 2 (clip_a, clip_b)
    #   intraview_only / phase_relational / privileged_multiview → 3
    #     (adds clip_b_neg)
    # MV2SV additions:
    #   + 1  for target_clip                         (view_3)
    #   + (n_fused_max - 1) extra for fused_clips[1..] (view_4..)
    base_num_clips = (
        3
        if multiview_objective in (
            "intraview_only",
            "phase_relational",
            "privileged_multiview",
            "token_phase_relational",
        )
        else 2
    )
    mv2sv_extra_clips = 0
    if multiview_objective == "privileged_multiview" and pmv_mv2sv_enabled:
        mv2sv_extra_clips = 1  # target_clip
        if pmv_mv2sv_fused_enabled and pmv_fused_n_max > 1:
            mv2sv_extra_clips += pmv_fused_n_max - 1
    dataloader_num_clips = base_num_clips + mv2sv_extra_clips
    if multiview_objective == "privileged_multiview":
        log.info(
            f"privileged_multiview dataloader: group_size={dataloader_num_clips} "
            f"(base={base_num_clips} + mv2sv_extra={mv2sv_extra_clips}; "
            f"mv2sv_enabled={pmv_mv2sv_enabled}, "
            f"fused_enabled={pmv_mv2sv_fused_enabled}, "
            f"n_fused_max={pmv_fused_n_max})"
        )

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
        num_clips=dataloader_num_clips,
        num_clips_per_video=1,
        img_size=crop_size,
        sampler_type=sampler_type,
        phase_matched_config=pmv_dispatch_cfg,
    )

    if not hasattr(dist_sampler, "builder"):
        raise RuntimeError("phase_matched data_manager did not attach a builder to the sampler")

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
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
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
            p
            for n, p in relational_head.named_parameters()
            if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
        ]
        head_nowd_params = [
            p
            for n, p in relational_head.named_parameters()
            if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
        ]
        if head_wd_params:
            optimizer.add_param_group(
                {
                    "params": head_wd_params,
                    "lr": lr,
                    "weight_decay": wd,
                }
            )
        if head_nowd_params:
            optimizer.add_param_group(
                {
                    "params": head_nowd_params,
                    "lr": lr,
                    "weight_decay": 0.0,
                    "WD_exclude": True,
                }
            )
        log.info(
            f"phase_relational: relational_head has "
            f"{sum(p.numel() for p in relational_head.parameters()):,} params "
            f"(trainable_wd={len(head_wd_params)} trainable_nowd={len(head_nowd_params)})"
        )

    # --- Token-rel heads (token_phase_relational only) ----------------- #
    token_rel_head = None
    token_rel_pool_safety = None
    motion_delta_head = None
    delta_target_projector = None
    if multiview_objective == "token_phase_relational":
        from app.vjepa_multiview.phase_relational_head import PhaseRelationalHead
        from app.vjepa_multiview.token_relational_head import (
            DeltaTargetProjector,
            MotionDeltaHead,
            TokenRelationalHead,
        )

        token_rel_head = TokenRelationalHead(
            embed_dim=tok_embed_dim,
            rel_dim=tok_rel_dim,
            hidden_dim=tok_hidden_dim,
            num_views=14,
            view_embedding_dim=tok_view_embed_dim,
            n_phase_freqs=tok_n_phase_freqs,
        ).to(device)
        # Pool safety is V4's pooled phase-relational head, reused at a
        # small weight to preserve the LVEF-positive pooled signal. Only
        # built if lambda_pool_rel > 0.
        if tok_lambda_pool_rel > 0.0:
            token_rel_pool_safety = PhaseRelationalHead(
                embed_dim=tok_embed_dim,
                rel_dim=tok_rel_dim,
                hidden_dim=tok_hidden_dim,
                num_views=14,
                view_embedding_dim=tok_view_embed_dim,
                n_phase_freqs=tok_n_phase_freqs,
            ).to(device)
        if md_enabled:
            motion_delta_head = MotionDeltaHead(
                embed_dim=tok_embed_dim,
                delta_dim=md_delta_dim,
                hidden_dim=md_hidden_dim,
                num_views=14,
                view_embedding_dim=tok_view_embed_dim,
                n_phase_freqs=tok_n_phase_freqs,
            ).to(device)
            delta_target_projector = DeltaTargetProjector(
                embed_dim=tok_embed_dim,
                delta_dim=md_delta_dim,
                hidden_dim=md_hidden_dim,
            ).to(device)

        # Add heads to optimizer (wd + no-wd groups, matching V4 conv).
        tok_modules = {"token_rel_head": token_rel_head}
        if token_rel_pool_safety is not None:
            tok_modules["token_rel_pool_safety"] = token_rel_pool_safety
        if motion_delta_head is not None:
            tok_modules["motion_delta_head"] = motion_delta_head
        if delta_target_projector is not None:
            tok_modules["delta_target_projector"] = delta_target_projector
        total_tok_params = 0
        for mname, mod in tok_modules.items():
            wd_params = [
                p for n, p in mod.named_parameters() if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
            ]
            nowd_params = [
                p for n, p in mod.named_parameters() if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
            ]
            if wd_params:
                optimizer.add_param_group(
                    {"params": wd_params, "lr": lr, "weight_decay": wd}
                )
            if nowd_params:
                optimizer.add_param_group(
                    {"params": nowd_params, "lr": lr, "weight_decay": 0.0, "WD_exclude": True}
                )
            total_tok_params += sum(p.numel() for p in mod.parameters())
        log.info(
            f"token_phase_relational: total trainable head params = "
            f"{total_tok_params:,} across {list(tok_modules.keys())} "
            f"(lambdas: token_rel={tok_lambda_token_rel}, pool_rel={tok_lambda_pool_rel}, "
            f"delta={tok_lambda_delta}; k={tok_subsample_k}, tau_token={tok_tau_token})"
        )

    # --- MCC-JEPA adapter (target-anchored only; pure mode uses no adapter) --- #
    mcc_adapter = None
    if multiview_objective == "mcc_jepa" and mcc_mode == "target_anchored":
        from src.models.mcc_jepa import CrossClipAdapter

        # Predictor output dim = encoder embed dim (predictor_proj restores it).
        encoder_embed_dim = int(getattr(encoder, "embed_dim", 1024))
        proj_dim = None
        if mcc_adapter_source_proj_dim not in (None, "null", ""):
            proj_dim = int(mcc_adapter_source_proj_dim)
        mcc_adapter = CrossClipAdapter(
            embed_dim=encoder_embed_dim,
            num_heads=mcc_adapter_num_heads,
            source_proj_dim=proj_dim,
            gamma_init=mcc_adapter_gamma_init,
        ).to(device)
        wd_params = [
            p for n, p in mcc_adapter.named_parameters() if p.requires_grad and "bias" not in n and len(p.shape) > 1
        ]
        nowd_params = [
            p for n, p in mcc_adapter.named_parameters() if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
        ]
        if wd_params:
            optimizer.add_param_group({"params": wd_params, "lr": lr, "weight_decay": wd})
        if nowd_params:
            optimizer.add_param_group(
                {"params": nowd_params, "lr": lr, "weight_decay": 0.0, "WD_exclude": True}
            )
        log.info(
            f"mcc_jepa target_anchored: adapter params = "
            f"{sum(p.numel() for p in mcc_adapter.parameters()):,} "
            f"(lambda_mcc={mcc_lambda_mcc}, lambda_vjepa={mcc_lambda_vjepa_self}, "
            f"gamma_init={mcc_adapter_gamma_init})"
        )

    # --- MV2SV heads (privileged_multiview only) ----------------------- #
    factorized_head = None
    view_predictor_mod = None
    mv_teacher_fusion = None
    pair_shared_projector = None
    fused_shared_projector = None
    phase_query_head = None
    factorized_head_ema = None
    mv_teacher_fusion_ema = None
    if multiview_objective == "privileged_multiview":
        from app.vjepa_multiview.factorized_head import FactorizedProjectionHead
        from app.vjepa_multiview.phase_query_head import PhaseQueryHead
        from app.vjepa_multiview.shared_projector import SharedProjector
        from app.vjepa_multiview.view_predictor import ConditionalViewPredictor

        factorized_head = FactorizedProjectionHead(
            embed_dim=pmv_embed_dim,
            hidden_dim=pmv_head_hidden,
            shared_dim=pmv_shared_dim,
            phase_dim=pmv_phase_dim,
            view_dim=pmv_view_dim,
        ).to(device)
        view_predictor_mod = ConditionalViewPredictor(
            shared_dim=pmv_shared_dim,
            phase_dim=pmv_phase_dim,
            view_dim=pmv_view_dim,
            target_dim=pmv_view_dim,  # matches FactorizedProjectionHead.z_view
            hidden_dim=pmv_head_hidden // 2,
            num_views=14,
            view_embedding_dim=pmv_view_embed_dim,
            n_phase_freqs=pmv_n_phase_freqs,
            use_z_view=True,
        ).to(device)
        # Split projectors: pair_shared handles L_pair_shared every step;
        # fused_shared handles L_fused on Bernoulli-active steps. They
        # target different distributions (single-view vs fused-mean)
        # but push gradient into the same z_shared encoder slot.
        pair_shared_projector = SharedProjector(
            shared_dim=pmv_shared_dim,
            fused_dim=pmv_shared_dim,
            hidden_dim=pmv_head_hidden // 2,
        ).to(device)
        fused_shared_projector = SharedProjector(
            shared_dim=pmv_shared_dim,
            fused_dim=pmv_shared_dim,
            hidden_dim=pmv_head_hidden // 2,
        ).to(device)
        phase_query_head = PhaseQueryHead(
            phase_dim=pmv_phase_dim,
            rel_dim=pmv_phase_dim,
            hidden_dim=pmv_head_hidden // 2,
            num_views=14,
            view_embedding_dim=pmv_view_embed_dim,
            n_phase_freqs=pmv_n_phase_freqs,
        ).to(device)

        # --- EMA target copies of factorized_head ---
        factorized_head_ema = copy.deepcopy(factorized_head).to(device)
        for p in factorized_head_ema.parameters():
            p.requires_grad = False

        # --- mv_teacher_fusion is attention_ema-only. Skipped in
        # mean_shared mode entirely. Even in attention_ema mode we
        # only build the frozen ``mv_teacher_fusion_ema`` target — the
        # "online" copy is redundant while no independent teacher-side
        # objective trains it. This keeps the module out of the
        # optimizer and DDP graph in every mode. ---
        if pmv_fused_target_mode == "attention_ema":
            from app.vjepa_multiview.mv_teacher_fusion import MultiViewTeacherFusion

            log.warning(
                "fused_target_mode='attention_ema' with "
                "allow_untrained_attention_fusion=True: target is a "
                "frozen random attention fuser. This mode is for "
                "debugging only — results are not scientifically valid."
            )
            mv_teacher_fusion_ema = MultiViewTeacherFusion(
                embed_dim=pmv_embed_dim,
                fused_dim=pmv_shared_dim,
                hidden_dim=pmv_head_hidden,
                num_views=14,
                view_embedding_dim=pmv_view_embed_dim,
                n_phase_freqs=pmv_n_phase_freqs,
                num_heads=pmv_fusion_heads,
            ).to(device)
            for p in mv_teacher_fusion_ema.parameters():
                p.requires_grad = False

        # Add trainable modules' params to the optimizer in wd / no-wd groups.
        # EMA heads are NOT added — they're not trained.
        # mv_teacher_fusion is NOT added in any mode (not trained here).
        pmv_modules = {
            "factorized_head": factorized_head,
            "view_predictor": view_predictor_mod,
            "pair_shared_projector": pair_shared_projector,
            "fused_shared_projector": fused_shared_projector,
            "phase_query_head": phase_query_head,
        }
        total_pmv_params = 0
        for mname, mod in pmv_modules.items():
            wd_params = [
                p for n, p in mod.named_parameters() if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
            ]
            nowd_params = [
                p for n, p in mod.named_parameters() if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
            ]
            if wd_params:
                optimizer.add_param_group(
                    {
                        "params": wd_params,
                        "lr": lr,
                        "weight_decay": wd,
                    }
                )
            if nowd_params:
                optimizer.add_param_group(
                    {
                        "params": nowd_params,
                        "lr": lr,
                        "weight_decay": 0.0,
                        "WD_exclude": True,
                    }
                )
            total_pmv_params += sum(p.numel() for p in mod.parameters())
        log.info(
            f"privileged_multiview (v2): total trainable MV2SV head params = "
            f"{total_pmv_params:,} across "
            f"{list(pmv_modules.keys())}"
        )
        _fh_ema_np = sum(p.numel() for p in factorized_head_ema.parameters())
        _mv_ema_np = (
            sum(p.numel() for p in mv_teacher_fusion_ema.parameters())
            if mv_teacher_fusion_ema is not None
            else 0
        )
        log.info(
            f"privileged_multiview (v2): EMA target heads built for "
            f"factorized_head ({_fh_ema_np:,} params) "
            f"and mv_teacher_fusion ({_mv_ema_np:,} params; 0 = mean_shared mode, no mv_teacher_fusion)"
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
            pretrained = {
                k.replace("module.", ""): v
                for k, v in pretrained.items()
                if not k.startswith(("head.", "fc_norm.", "module.head.", "module.fc_norm."))
            }
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
            encoder, predictor, target_encoder, optimizer, scaler, start_epoch, start_itr = load_checkpoint(
                r_path=latest,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler,
            )
            # If this is a privileged_multiview resume, restore MV2SV heads.
            # Follows the same "start-fresh if missing" policy as the
            # relational-head branch below — a vanilla e100/e125 checkpoint
            # has none of these keys, so all four modules start from init.
            if factorized_head is not None:
                try:
                    _ckpt_peek = torch.load(latest, map_location="cpu", weights_only=False)
                except TypeError:
                    _ckpt_peek = torch.load(latest, map_location="cpu")
                _loaded_any_pmv = False
                for mname, mod in (
                    ("factorized_head", factorized_head),
                    ("view_predictor", view_predictor_mod),
                    ("pair_shared_projector", pair_shared_projector),
                    ("fused_shared_projector", fused_shared_projector),
                    ("phase_query_head", phase_query_head),
                    ("factorized_head_ema", factorized_head_ema),
                    ("mv_teacher_fusion_ema", mv_teacher_fusion_ema),
                ):
                    if mod is None:
                        continue
                    if mname in _ckpt_peek:
                        state = {k.replace("module.", ""): v for k, v in _ckpt_peek[mname].items()}
                        msg = mod.load_state_dict(state, strict=False)
                        log.info(f"Loaded {mname}: {msg}")
                        _loaded_any_pmv = True
                # Backward-compat: old checkpoints saved a single
                # ``shared_projector`` key. If neither of the split keys
                # above was found but shared_projector is present, load
                # it into BOTH projectors.
                if (
                    "shared_projector" in _ckpt_peek
                    and "pair_shared_projector" not in _ckpt_peek
                    and "fused_shared_projector" not in _ckpt_peek
                ):
                    log.warning(
                        "Old checkpoint has 'shared_projector' but not the "
                        "split 'pair_shared_projector' / 'fused_shared_projector' "
                        "keys. Loading shared_projector into both — downstream "
                        "training will diverge them."
                    )
                    legacy_state = {k.replace("module.", ""): v for k, v in _ckpt_peek["shared_projector"].items()}
                    if pair_shared_projector is not None:
                        msg = pair_shared_projector.load_state_dict(
                            legacy_state,
                            strict=False,
                        )
                        log.info(f"Loaded legacy shared_projector -> pair_shared_projector: {msg}")
                    if fused_shared_projector is not None:
                        msg = fused_shared_projector.load_state_dict(
                            legacy_state,
                            strict=False,
                        )
                        log.info(f"Loaded legacy shared_projector -> fused_shared_projector: {msg}")
                    _loaded_any_pmv = True
                if not _loaded_any_pmv:
                    log.warning(
                        "Resuming a privileged_multiview run from a "
                        "checkpoint without any MV2SV head keys — all "
                        "four heads start from init. Optimizer state may "
                        "not align; resetting start_epoch=0 to avoid a "
                        "stale LR-schedule replay against a newly-sized "
                        "param-group list."
                    )
                    start_epoch = 0
                    completed_steps = 0
                del _ckpt_peek

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
                    rh_state = {k.replace("module.", ""): v for k, v in _ckpt_peek["relational_head"].items()}
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
            if token_rel_head is not None:
                try:
                    _ckpt_peek = torch.load(latest, map_location="cpu", weights_only=False)
                except TypeError:
                    _ckpt_peek = torch.load(latest, map_location="cpu")
                _loaded_any_tok = False
                for mname, mod in (
                    ("token_rel_head", token_rel_head),
                    ("token_rel_pool_safety", token_rel_pool_safety),
                    ("motion_delta_head", motion_delta_head),
                    ("delta_target_projector", delta_target_projector),
                ):
                    if mod is None:
                        continue
                    if mname in _ckpt_peek:
                        state = {k.replace("module.", ""): v for k, v in _ckpt_peek[mname].items()}
                        msg = mod.load_state_dict(state, strict=False)
                        log.info(f"Loaded {mname}: {msg}")
                        _loaded_any_tok = True
                if not _loaded_any_tok:
                    log.warning(
                        "Resuming a token_phase_relational run from a checkpoint "
                        "without any token-rel head keys — all heads start from init. "
                        "Optimizer state may not align; resetting start_epoch=0."
                    )
                    start_epoch = 0
                    completed_steps = 0
                del _ckpt_peek
            completed_steps = start_epoch * ipe + start_itr
            for _ in range(completed_steps):
                scheduler.step()
                wd_scheduler.step()
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
                relational_head,
                static_graph=False,
            )
        if factorized_head is not None:
            # v2 forward: z_view is consumed via L_pair_view, so the
            # factorized_head wrap no longer needs find_unused_parameters.
            factorized_head = DistributedDataParallel(
                factorized_head,
                static_graph=False,
            )
            view_predictor_mod = DistributedDataParallel(
                view_predictor_mod,
                static_graph=False,
            )
            # pair_shared_projector is used every step by L_pair_shared,
            # so it doesn't need find_unused_parameters.
            pair_shared_projector = DistributedDataParallel(
                pair_shared_projector,
                static_graph=False,
            )
            # fused_shared_projector is used only on Bernoulli-active
            # steps. Its grads are legally None when fused_active=False,
            # so find_unused_parameters=True is required.
            fused_shared_projector = DistributedDataParallel(
                fused_shared_projector,
                static_graph=False,
                find_unused_parameters=True,
            )
            phase_query_head = DistributedDataParallel(
                phase_query_head,
                static_graph=False,
            )
            # mv_teacher_fusion only exists in attention_ema mode. When
            # present, it is used only on Bernoulli-active steps (and
            # even then only to EMA-update mv_teacher_fusion_ema under
            # no_grad). No gradient ever flows into it, so there's no
            # DDP reducer involvement — we don't wrap it.
            # EMA copies are NOT DDP-wrapped either — no grad.
        if token_rel_head is not None:
            token_rel_head = DistributedDataParallel(
                token_rel_head,
                static_graph=False,
            )
        if token_rel_pool_safety is not None:
            token_rel_pool_safety = DistributedDataParallel(
                token_rel_pool_safety,
                static_graph=False,
            )
        if motion_delta_head is not None:
            # With lambda_delta=0 OR no eligible same-view rows in a step,
            # the motion_delta_head / delta_target_projector get zero-loss
            # proxy gradients. DDP reducer still sees every parameter but
            # the gradient can be None if the dummy branch isn't executed
            # consistently. Use find_unused_parameters=True to be safe.
            motion_delta_head = DistributedDataParallel(
                motion_delta_head,
                static_graph=False,
                find_unused_parameters=True,
            )
        if delta_target_projector is not None:
            delta_target_projector = DistributedDataParallel(
                delta_target_projector,
                static_graph=False,
                find_unused_parameters=True,
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
            "epoch": epoch,
            "itr": itr,
            "loss": loss_avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
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
                "rel_mask_same_study_batch_negatives": rel_mask_same_study_batch_negatives,
                "rel_negative_mode": rel_negative_mode,
            }
        if factorized_head is not None:
            save_dict["factorized_head"] = factorized_head.state_dict()
            save_dict["view_predictor"] = view_predictor_mod.state_dict()
            save_dict["pair_shared_projector"] = pair_shared_projector.state_dict()
            save_dict["fused_shared_projector"] = fused_shared_projector.state_dict()
            save_dict["phase_query_head"] = phase_query_head.state_dict()
            save_dict["factorized_head_ema"] = factorized_head_ema.state_dict()
            if mv_teacher_fusion_ema is not None:
                save_dict["mv_teacher_fusion_ema"] = mv_teacher_fusion_ema.state_dict()
            save_dict["pmv_config"] = {
                "fused_target_mode": pmv_fused_target_mode,
                "lambda_pair_shared": pmv_lambda_pair_shared,
                "lambda_pair_view": pmv_lambda_pair_view,
                "lambda_view_nce": pmv_lambda_view_nce,
                "lambda_fused": pmv_lambda_fused,
                "lambda_shared": pmv_lambda_shared,
                "lambda_phase": pmv_lambda_phase,
                "lambda_local_motion": pmv_lambda_local_motion,
                "p_fused": pmv_p_fused,
                "tau_static": pmv_tau_static,
                "tau_phase": pmv_tau_phase,
                "tau_view": pmv_tau_view,
                "warmup_epochs": pmv_warmup_epochs,
                "shared_dim": pmv_shared_dim,
                "phase_dim": pmv_phase_dim,
                "view_dim": pmv_view_dim,
                "use_z_view": pmv_use_z_view,
            }
        if mcc_adapter is not None:
            save_dict["mcc_adapter"] = mcc_adapter.state_dict()
            save_dict["mcc_config"] = {
                "mcc_mode": mcc_mode,
                "lambda_mcc": mcc_lambda_mcc,
                "lambda_vjepa": mcc_lambda_vjepa_self,
                "adapter_num_heads": mcc_adapter_num_heads,
                "gamma_init": mcc_adapter_gamma_init,
            }
        if token_rel_head is not None:
            save_dict["token_rel_head"] = token_rel_head.state_dict()
            if token_rel_pool_safety is not None:
                save_dict["token_rel_pool_safety"] = token_rel_pool_safety.state_dict()
            if motion_delta_head is not None:
                save_dict["motion_delta_head"] = motion_delta_head.state_dict()
            if delta_target_projector is not None:
                save_dict["delta_target_projector"] = delta_target_projector.state_dict()
            save_dict["token_motion_config"] = {
                "lambda_token_rel": tok_lambda_token_rel,
                "lambda_pool_rel": tok_lambda_pool_rel,
                "lambda_delta": tok_lambda_delta,
                "tau_token": tok_tau_token,
                "tau_delta": md_tau_delta,
                "token_subsample_k": tok_subsample_k,
                "warmup_epochs": tok_warmup_epochs,
                "md_enabled": md_enabled,
                "md_delta_dim": md_delta_dim,
                "md_same_view_only": md_same_view_only,
                "md_lambda_l1": md_lambda_l1,
                "md_lambda_nce": md_lambda_nce,
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
            _t_refresh0 = time.time()
            dist_sampler.builder.refresh_epoch(epoch)
            refresh_epoch_seconds = time.time() - _t_refresh0
            guard.mark_refreshed(epoch)
            n_pair = len(dist_sampler.builder.last_pair_df)
            log.info(
                f"[epoch {epoch}] refreshed pairs: rank={rank} n={n_pair} "
                f"mode={cfg_pmv.get('sampling_mode')} "
                f"refresh_epoch_seconds={refresh_epoch_seconds:.1f}"
            )

            # Frame-count guard: open the first N MP4 URIs and compare
            # len(VideoReader) against clip_{a,b}_n_frames from the pair_df.
            # Fails loud; default off so this is only paid in sanity runs.
            if debug_verify_frame_count and epoch == start_epoch and rank == 0:
                _run_frame_count_guard(dist_sampler.builder.last_pair_df, n=debug_verify_n, log=log)

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

                masks_enc_d = [[m.to(device) for m in masks_enc]]
                masks_pred_d = [[m.to(device) for m in masks_pred]]
                meta_list = extract_pair_metadata(collated_batch)

                if multiview_objective == "privileged_multiview":
                    # MV2SV-aware extraction surfaces target_clip and
                    # (optionally) fused_clips.
                    _expected_n_fused_extra = max(0, pmv_fused_n_max - 1)
                    mv2sv_tensors = _extract_mv2sv_clips(
                        collated_batch,
                        meta_list=meta_list,
                        device=device,
                        expected_n_fused=_expected_n_fused_extra,
                    )
                    pair = PairBatch(
                        clip_a=mv2sv_tensors["clip_a"],
                        clip_b=mv2sv_tensors["clip_b"],
                        masks_enc=masks_enc_d,
                        masks_pred=masks_pred_d,
                        phase_metadata=meta_list,
                        clip_b_neg=mv2sv_tensors["clip_b_neg"],
                        target_clip=mv2sv_tensors["target_clip"],
                        target_views=mv2sv_tensors["target_views"],
                        target_delta_phase=mv2sv_tensors["target_delta_phase"],
                        target_clip_present=mv2sv_tensors["target_clip_present"],
                        fused_clips=mv2sv_tensors["fused_clips"],
                        fused_views=mv2sv_tensors["fused_views"],
                        fused_phases=mv2sv_tensors["fused_phases"],
                        fused_valid_mask=mv2sv_tensors["fused_valid_mask"],
                    )
                else:
                    clip_a, clip_b, clip_b_neg = _extract_multiview_clips(
                        collated_batch,
                        device=device,
                        objective=multiview_objective,
                    )
                    pair = PairBatch(
                        clip_a=clip_a,
                        clip_b=clip_b,
                        masks_enc=masks_enc_d,
                        masks_pred=masks_pred_d,
                        phase_metadata=meta_list,
                        clip_b_neg=clip_b_neg,
                    )

                def _step():
                    new_lr = scheduler.step()
                    new_wd = wd_scheduler.step()
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        if multiview_objective == "phase_relational":
                            out = forward_phase_relational(
                                pair,
                                encoder,
                                target_encoder,
                                predictor,
                                relational_head,
                                meta_list=meta_list,
                                tau=rel_temperature,
                                loss_exp=loss_exp,
                                mask_same_study_batch_negatives=rel_mask_same_study_batch_negatives,
                                disable_hard_negative=disable_hard_negative,
                            )
                            # λ_rel warmup scalar (linear, capped at 1.0).
                            progress = float(epoch) + (float(itr) / max(1, ipe))
                            warmup_frac = min(
                                1.0,
                                progress / max(1e-6, rel_warmup_epochs),
                            )
                            effective_lambda_rel = lambda_rel * warmup_frac
                            total_loss = out["intraview_loss"] + effective_lambda_rel * out["rel_loss"]
                            out["effective_lambda_rel"] = torch.tensor(
                                effective_lambda_rel,
                                device=device,
                            )
                            out["total_loss"] = total_loss
                        elif multiview_objective == "intraview_only":
                            out = forward_intraview_only(
                                pair,
                                encoder,
                                target_encoder,
                                predictor,
                                loss_exp=loss_exp,
                            )
                            total_loss = out["total_loss"]
                        elif multiview_objective == "mcc_jepa":
                            from app.vjepa_multiview.mcc_jepa_forward import forward_mcc_jepa

                            out = forward_mcc_jepa(
                                pair,
                                encoder,
                                target_encoder,
                                predictor,
                                mcc_adapter,
                                mode=mcc_mode,
                                lambda_mcc=mcc_lambda_mcc,
                                lambda_vjepa=mcc_lambda_vjepa_self,
                                loss_exp=loss_exp,
                            )
                            total_loss = out["total_loss"]
                            # Expose diagnostics for CSV logging.
                            out["intraview_loss"] = out["loss_vjepa_self"]
                            out["crossview_loss"] = out["loss_mcc"]
                            if mcc_adapter is not None:
                                out["mcc_gamma"] = mcc_adapter.gamma.detach().clone()
                        elif multiview_objective == "token_phase_relational":
                            progress = float(epoch) + (float(itr) / max(1, ipe))
                            warmup_frac = min(
                                1.0,
                                progress / max(1e-6, tok_warmup_epochs),
                            )
                            eff_l_token_rel = tok_lambda_token_rel * warmup_frac
                            eff_l_pool_rel = tok_lambda_pool_rel * warmup_frac
                            eff_l_delta = tok_lambda_delta * warmup_frac
                            out = forward_token_phase_relational(
                                pair,
                                encoder,
                                target_encoder,
                                predictor,
                                token_rel_head,
                                token_rel_pool_safety,
                                motion_delta_head,
                                delta_target_projector,
                                meta_list=meta_list,
                                token_subsample_k=tok_subsample_k,
                                tau_token=tok_tau_token,
                                tau_delta=md_tau_delta,
                                loss_exp=loss_exp,
                                lambda_token_rel=eff_l_token_rel,
                                lambda_pool_rel=eff_l_pool_rel,
                                lambda_delta=eff_l_delta,
                                lambda_delta_l1=md_lambda_l1,
                                lambda_delta_nce=md_lambda_nce,
                                mask_same_study_batch_negatives=tok_mask_same_study,
                                pool_rel_temperature=tok_pool_rel_temperature,
                            )
                            out["effective_lambda_token_rel"] = torch.tensor(
                                eff_l_token_rel, device=device
                            )
                            out["effective_lambda_pool_rel"] = torch.tensor(
                                eff_l_pool_rel, device=device
                            )
                            out["effective_lambda_delta"] = torch.tensor(
                                eff_l_delta, device=device
                            )
                            total_loss = out["total_loss"]
                        elif multiview_objective == "privileged_multiview":
                            # v4 linear warmup; Fix 6 DDP-synced bernoulli
                            progress = float(epoch) + (float(itr) / max(1, ipe))
                            warmup_frac = min(
                                1.0,
                                progress / max(1e-6, pmv_warmup_epochs),
                            )
                            eff_l_pair_shared = pmv_lambda_pair_shared * warmup_frac
                            eff_l_pair_view = pmv_lambda_pair_view * warmup_frac
                            eff_l_view_nce = pmv_lambda_view_nce * warmup_frac
                            eff_l_shared = pmv_lambda_shared * warmup_frac
                            eff_l_phase = pmv_lambda_phase * warmup_frac
                            eff_l_fused = pmv_lambda_fused * warmup_frac
                            eff_l_local_motion = pmv_lambda_local_motion * warmup_frac
                            # Fix 6: synced Bernoulli — same draw on every rank
                            fused_active = ddp_synced_bernoulli(
                                pmv_p_fused,
                                global_step=global_step,
                            )
                            out = forward_privileged_multiview(
                                pair,
                                encoder,
                                target_encoder,
                                predictor,
                                factorized_head,
                                view_predictor_mod,
                                pair_shared_projector,
                                fused_shared_projector,
                                phase_query_head,
                                factorized_head_ema,
                                meta_list=meta_list,
                                fused_active=fused_active,
                                fused_target_mode=pmv_fused_target_mode,
                                mv_teacher_fusion=mv_teacher_fusion,
                                mv_teacher_fusion_ema=mv_teacher_fusion_ema,
                                lambda_pair_shared=eff_l_pair_shared,
                                lambda_pair_view=eff_l_pair_view,
                                lambda_view_nce=eff_l_view_nce,
                                lambda_shared=eff_l_shared,
                                lambda_phase=eff_l_phase,
                                lambda_fused=eff_l_fused,
                                lambda_local_motion=eff_l_local_motion,
                                tau_shared=pmv_tau_static,
                                tau_phase=pmv_tau_phase,
                                tau_view=pmv_tau_view,
                                loss_exp=loss_exp,
                                use_z_view=pmv_use_z_view,
                                allow_provisional_clip_b_fallback=pmv_allow_provisional_clip_b_fallback,
                            )
                            out["effective_lambda_pair_shared"] = torch.tensor(
                                eff_l_pair_shared,
                                device=device,
                            )
                            out["effective_lambda_pair_view"] = torch.tensor(
                                eff_l_pair_view,
                                device=device,
                            )
                            out["effective_lambda_view_nce"] = torch.tensor(
                                eff_l_view_nce,
                                device=device,
                            )
                            out["effective_lambda_fused"] = torch.tensor(
                                eff_l_fused,
                                device=device,
                            )
                            total_loss = out["total_loss"]
                        else:  # "smooth_l1" — unchanged path
                            out = forward_intraview_and_crossview(
                                pair,
                                encoder,
                                target_encoder,
                                predictor,
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
                    # EMA — encoder (unchanged)
                    m = next(momentum_scheduler)
                    with torch.no_grad():
                        for pq, pk in zip(encoder.parameters(), target_encoder.parameters()):
                            pk.mul_(m).add_(pq, alpha=1 - m)
                    # EMA updates for MV2SV target heads.
                    # factorized_head_ema tracks the trained factorized_head.
                    # mv_teacher_fusion_ema (attention_ema mode only) is a
                    # frozen random target — never updated (no online copy
                    # exists to track).
                    if multiview_objective == "privileged_multiview":
                        momentum_update_ema_(factorized_head, factorized_head_ema, m)
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
                    _mcc_tag = ""
                    if mcc_adapter is not None:
                        _g = float(mcc_adapter.gamma.detach().item())
                        _pd = float(out.get("pred_delta_from_A", torch.zeros(())).detach().item())
                        _mcc_tag = f" mcc[gamma={_g:+.4f} pred_delta={_pd:+.5f}]"
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
                        f"{_mcc_tag}"
                    )
                    if global_step < 3:
                        log.info(
                            f"  mask: enc{tuple(masks_enc_d[0][0].shape)} "
                            f"pred{tuple(masks_pred_d[0][0].shape)} "
                            f"enc_range=[{int(masks_enc_d[0][0].min())},{int(masks_enc_d[0][0].max())}] "
                            f"pred_range=[{int(masks_pred_d[0][0].min())},{int(masks_pred_d[0][0].max())}]"
                        )
                    # MV2SV on-run sanity line (Fix 1e / task 207).
                    if multiview_objective == "privileged_multiview":
                        from app.vjepa_multiview.phase_relational_head import VIEW_ID_MAP as _VIEW_ID_MAP

                        _id_to_view = {v: k for k, v in _VIEW_ID_MAP.items()}
                        _src_c = out.get("src_view_counts", {}) or {}
                        _tgt_c = out.get("tgt_view_counts", {}) or {}
                        _sp_c = out.get("src_tgt_pair_counts", {}) or {}
                        _src_str = ",".join(f"{_id_to_view.get(k, k)}={v}" for k, v in sorted(_src_c.items()))
                        _tgt_str = ",".join(f"{_id_to_view.get(k, k)}={v}" for k, v in sorted(_tgt_c.items()))
                        _fallback = float(out.get("used_clip_b_fallback", 0.0))
                        _pct_tgt = float(out.get("pct_target_clip_present", 0.0))
                        _pct_fused = float(out.get("pct_fused_clips_present", 0.0))
                        _fused_vv = float(out.get("fused_valid_views_mean", 0.0))
                        log.info(
                            f"  mv2sv: src[{_src_str}] tgt[{_tgt_str}] "
                            f"pairs={len(_sp_c)} "
                            f"pct_tgt={_pct_tgt:.2f} pct_fused={_pct_fused:.2f} "
                            f"fused_vv_mean={_fused_vv:.2f} "
                            f"clip_b_fallback={_fallback:.0f}"
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
                            return total**0.5

                        te_grad_l2 = _param_l2(target_encoder)
                        rh = relational_head
                        if rh is not None and hasattr(rh, "module"):
                            # DDP wrapper
                            tp = getattr(rh.module, "target_proj", None)
                        else:
                            tp = getattr(rh, "target_proj", None) if rh is not None else None
                        tp_grad_l2 = _param_l2(tp)
                        csv_logger.log(
                            epoch + 1,
                            itr,
                            total_val,
                            intra_val,
                            cross_val,
                            int(itr_ms),
                            int(data_ms),
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
                    elif multiview_objective == "token_phase_relational":
                        csv_logger.log(
                            epoch + 1,
                            itr,
                            total_val,
                            intra_val,
                            cross_val,
                            int(itr_ms),
                            int(data_ms),
                            float(out.get("token_rel_loss", 0.0)),
                            float(out.get("token_rel_top1_with_hard", 0.0)),
                            float(out.get("token_rel_pos_sim_mean", 0.0)),
                            float(out.get("token_rel_hard_sim_mean", 0.0)),
                            float(out.get("token_rel_batch_neg_sim_mean", 0.0)),
                            float(out.get("token_rel_pos_minus_hard_gap", 0.0)),
                            float(out.get("token_rel_pos_minus_batch_gap", 0.0)),
                            float(out.get("token_rel_logits_std", 0.0)),
                            float(out.get("token_rel_q_var", 0.0)),
                            float(out.get("token_rel_y_var", 0.0)),
                            float(out.get("token_rel_valid_rows", 0.0)),
                            float(out.get("token_subsample_k", 0.0)),
                            float(out.get("pool_rel_loss", 0.0)),
                            float(out.get("pool_rel_top1_with_hard", 0.0)),
                            float(out.get("pool_rel_pos_minus_hard_gap", 0.0)),
                            float(out.get("delta_loss", 0.0)),
                            float(out.get("delta_l1", 0.0)),
                            float(out.get("delta_nce", 0.0)),
                            float(out.get("delta_valid_rows", 0.0)),
                            float(out.get("delta_pos_sim_mean", 0.0)),
                            float(out.get("delta_hard_sim_mean", 0.0)),
                            float(out.get("delta_pos_minus_hard_gap", 0.0)),
                            float(out.get("delta_q_var", 0.0)),
                            float(out.get("delta_target_var", 0.0)),
                            float(out.get("effective_lambda_token_rel", 0.0)),
                            float(out.get("effective_lambda_pool_rel", 0.0)),
                            float(out.get("effective_lambda_delta", 0.0)),
                            float(out.get("same_view_row_fraction", 0.0)),
                            float(out.get("same_family_row_fraction", 0.0)),
                            float(out.get("cross_family_row_fraction", 0.0)),
                            int(out.get("token_rel_same_study_masked_count", 0)),
                        )
                    elif multiview_objective == "privileged_multiview":
                        _pv_by = out.get("diag_pair_view_cos_by_view", {}) or {}
                        _nce_by = out.get("diag_view_nce_top1_by_view", {}) or {}
                        _pv = lambda v: float(_pv_by[v].item()) if v in _pv_by else 0.0  # noqa: E731
                        _nc = lambda v: float(_nce_by[v].item()) if v in _nce_by else 0.0  # noqa: E731
                        csv_logger.log(
                            epoch + 1,
                            itr,
                            total_val,
                            intra_val,
                            cross_val,
                            int(itr_ms),
                            int(data_ms),
                            float(out.get("pair_shared_loss", 0.0)),
                            float(out.get("pair_view_loss", 0.0)),
                            float(out.get("view_nce_loss", 0.0)),
                            float(out.get("shared_loss", 0.0)),
                            float(out.get("phase_rel_loss", 0.0)),
                            float(out.get("phase_rel_top1", 0.0)),
                            float(out.get("phase_rel_pos_minus_hard_gap", 0.0)),
                            float(out.get("fused_loss", 0.0)),
                            float(out.get("fused_active", 0.0)),
                            float(out.get("local_motion_loss", 0.0)),
                            float(out.get("effective_lambda_pair_shared", 0.0)),
                            float(out.get("effective_lambda_pair_view", 0.0)),
                            float(out.get("effective_lambda_view_nce", 0.0)),
                            float(out.get("effective_lambda_fused", 0.0)),
                            float(out.get("paired_shared_top1", 0.0)),
                            float(out.get("paired_shared_pos_sim", 0.0)),
                            float(out.get("view_nce_top1", 0.0)),
                            float(out.get("view_nce_pos_sim_mean", 0.0)),
                            float(out.get("view_nce_neg_sim_mean", 0.0)),
                            float(out.get("view_nce_valid_neg_count_mean", 0.0)),
                            float(out.get("view_nce_valid_neg_count_min", 0.0)),
                            float(out.get("view_nce_same_target_view_fraction", 0.0)),
                            float(out.get("view_nce_fallback_fraction", 0.0)),
                            float(out.get("used_clip_b_fallback", 0.0)),
                            float(out.get("pct_target_clip_present", 0.0)),
                            float(out.get("pct_fused_clips_present", 0.0)),
                            float(out.get("fused_valid_views_mean", 0.0)),
                            float(out.get("fused_valid_views_min", 0.0)),
                            float(out.get("fused_shared_target_norm", 0.0)),
                            float(out.get("fused_shared_q_norm", 0.0)),
                            float(out.get("fused_shared_cos_q_target", 0.0)),
                            float(out.get("diag_pair_shared_cos_q_target", 0.0)),
                            float(out.get("diag_pair_view_cos_q_target", 0.0)),
                            _pv("PLAX"),
                            _pv("A5C"),
                            _pv("A3C"),
                            _pv("A2C"),
                            _nc("PLAX"),
                            _nc("A5C"),
                            _nc("A3C"),
                            _nc("A2C"),
                            float(out.get("diag_z_shared_vs_z_phase_cos", 0.0)),
                            float(out.get("diag_z_shared_vs_z_view_cos", 0.0)),
                            float(out.get("diag_pair_q_shared_norm", 0.0)),
                            float(out.get("diag_pair_q_view_norm", 0.0)),
                            float(out.get("diag_z_shared_var", 0.0)),
                            float(out.get("diag_z_phase_var", 0.0)),
                            float(out.get("diag_z_view_var", 0.0)),
                            int(out.get("same_study_masked_count", 0)),
                        )
                    else:
                        csv_logger.log(
                            epoch + 1,
                            itr,
                            total_val,
                            intra_val,
                            cross_val,
                            int(itr_ms),
                            int(data_ms),
                        )

                assert np.isfinite(total_val), f"total_loss non-finite at step {global_step}: {total_val}"

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
                save_checkpoint(epoch + 1, 0, os.path.join(folder, f"e{epoch + 1}.pt"), loss_avg=loss_meter.avg)

        if folder and (save_at_end or (exit_reason is not None)):
            save_checkpoint(epoch, global_step, os.path.join(folder, "latest.pt"), loss_avg=loss_meter.avg)
    finally:
        if exit_reason:
            log.info(f"exit: {exit_reason}")
        if use_ddp:
            try:
                dist.barrier()
            except Exception:
                pass
