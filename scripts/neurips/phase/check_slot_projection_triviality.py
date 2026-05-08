#!/usr/bin/env python3
"""H1 sanity: is the factorized head's slot projection near-trivial?

The doc's interpretability claim assumes ``z_phase`` carries different
information than ``encoder_pool``. If the factorized head's phase_mlp is
near-linear and didn't move far from init, ``z_phase`` is literally
``encoder_pool @ W + b`` with W nearly frozen — which means the slot is
a change of basis, not a content decomposition.

This script encodes a held-out batch through pilot 655 e5, projects each
pooled feature through each slot head, and reports the pairwise cosine
similarities between:
  - encoder_pool direction
  - z_shared slot direction
  - z_phase slot direction
  - z_view slot direction

Because encoder_pool is 1024-D and slots are 256-D, we project
encoder_pool through a random Gaussian down to 256-D before comparing
(or: we linear-regress slot onto encoder_pool and report the fitted R²).
Latter is the more interpretable test: how much variance in each slot is
linearly explainable by the encoder pool?

Usage:
    python scripts/neurips/phase/check_slot_projection_triviality.py \
        --checkpoint /opt/.../pilot_e5.pt \
        --num-studies 100 --out /tmp/h1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "classifier" / "phase" / "sampler"))

from scripts.neurips.phase.run_cross_view_retrieval_diag import (  # noqa: E402
    build_sampler, load_encoder_and_head, _load_clip, _load_test_study_ids,
    _slot_from_pooled,
)


def encode_pooled(clip: torch.Tensor, encoder: torch.nn.Module) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = clip.unsqueeze(0).cuda()
        tokens = encoder(x)
        pooled = tokens.mean(dim=1).float().squeeze(0)
    return pooled.cpu()


def linreg_r2(X: torch.Tensor, y: torch.Tensor, ridge: float = 1e-3) -> float:
    """Closed-form linear regression of each column of y on X. Returns
    mean R² across columns of y on a random 80/20 split.

    X: [N, D_in]  y: [N, D_out]
    """
    N, D = X.shape
    Do = y.shape[1]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    split = int(0.8 * N)
    tr, va = perm[:split], perm[split:]
    Xt = X[tr]; yt = y[tr]
    # Add bias column
    Xt_b = torch.cat([Xt, torch.ones(len(Xt), 1)], dim=1)
    A = Xt_b.T @ Xt_b + ridge * torch.eye(D + 1)
    B = Xt_b.T @ yt
    W = torch.linalg.solve(A, B)  # [D+1, Do]
    Xv_b = torch.cat([X[va], torch.ones(len(va), 1)], dim=1)
    yv_pred = Xv_b @ W
    yv_true = y[va]
    # Per-column R²
    ss_res = ((yv_true - yv_pred) ** 2).sum(dim=0)
    ss_tot = ((yv_true - yv_true.mean(dim=0)) ** 2).sum(dim=0).clamp(min=1e-9)
    r2 = 1 - ss_res / ss_tot
    return float(r2.mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-studies", type=int, default=100)
    parser.add_argument("--phase-annotations", type=Path,
                        default=Path("/opt/dlami/nvme/probe/phase_annotations.parquet"))
    parser.add_argument("--view-labels", type=Path,
                        default=Path("/opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv"))
    parser.add_argument("--splits-csv", type=Path,
                        default=REPO_ROOT / "classifier" / "phase" / "splits" / "dicoms_split.csv")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    # Load pilot with head
    encoder, head, _ = load_encoder_and_head(args.checkpoint, "z_phase")
    if head is None:
        print("[ERROR] this script requires an MV2SV ckpt (factorized head). Aborting.")
        return 1

    test_studies = _load_test_study_ids(args.splits_csv)
    view_pair_policy = {
        "enabled": True, "same_view_prob": 0.05, "same_family_prob": 0.35,
        "cross_family_prob": 0.60, "require_different_dicom": True,
        "allow_same_view": True, "resample_attempts": 8,
    }
    mv2sv_cfg = {
        "enabled": True,
        "target_view_sampling": {
            "stage": "stage2", "a4c_source_targets": ["A2C", "A5C"],
            "allowed_targets": ["A2C", "A5C", "A3C", "PLAX", "PSAX-MV"],
            "target_dropout": 0.0, "require_different_view": True,
        },
        "fused_pool": {"enabled": False, "n_fused_min": 2, "n_fused_max": 2},
    }
    sampler = build_sampler(args.phase_annotations, mv2sv_cfg, view_pair_policy,
                            args.view_labels, pairs_per_study=2)
    sampler.epoch = 0
    records = sampler.build_records()
    _sdf = getattr(sampler, "_df", None)
    if _sdf is None:
        _sdf = getattr(sampler, "df", None)
    row_to_uri = dict(zip(_sdf.index.astype(int), _sdf["s3_uri"].astype(str)))

    def _uri(clip):
        u = row_to_uri.get(int(clip.row_idx), "")
        if u.startswith("s3://echodata25/mimic-raw-staging"):
            u = u.replace("s3://echodata25/mimic-raw-staging", "s3://echodata25/mimic-echo-224px", 1)
            if u.endswith(".dcm"):
                u = u[:-4] + ".mp4"
        return u

    def _window(clip, fpc=16, fs=1):
        start = int(clip.anchor_frame) - (fpc // 2) * fs
        return [max(0, start + i * fs) for i in range(fpc)]

    kept = []
    for r in records:
        if r.target_clip is None: continue
        if str(r.study_id) not in test_studies: continue
        kept.append(r)
        if len(kept) >= args.num_studies: break
    print(f"[sampler] kept {len(kept)}")

    pool_feats = []
    z_shared = []
    z_phase_ = []
    z_view_ = []
    for k, r in enumerate(kept):
        try:
            u = _uri(r.target_clip)
            if not u: continue
            clip = _load_clip(u, _window(r.target_clip, r.frames_per_clip, r.frame_step))
            p = encode_pooled(clip, encoder)
            pool_feats.append(p)
            # Slots
            with torch.no_grad():
                slots = head(p.unsqueeze(0).cuda().float())
            z_shared.append(slots["z_shared"].squeeze(0).cpu())
            z_phase_.append(slots["z_phase"].squeeze(0).cpu())
            z_view_.append(slots["z_view"].squeeze(0).cpu())
        except Exception as e:
            print(f"[warn] row {k}: {e}")
            continue
        if (k + 1) % 25 == 0:
            print(f"  encoded {len(pool_feats)}/{len(kept)}")
    print(f"[encode] {len(pool_feats)} pooled + 3 slots each")
    if len(pool_feats) < 20:
        print("[ERROR] too few"); return 1

    X_pool = torch.stack(pool_feats).float()       # [N, D_enc]
    Y_shared = torch.stack(z_shared).float()       # [N, 256]
    Y_phase = torch.stack(z_phase_).float()
    Y_view = torch.stack(z_view_).float()

    # H1 test: linear regress each slot on encoder_pool. If R² ≈ 1.0,
    # the slot is a near-linear projection; if R² ≪ 1.0, the MLP in
    # phase_mlp has introduced nonlinearity that matters.
    r2_shared = linreg_r2(X_pool, Y_shared)
    r2_phase = linreg_r2(X_pool, Y_phase)
    r2_view = linreg_r2(X_pool, Y_view)

    # Also report cosines between the L2-normed mean direction of each
    # representation across the batch (gives a rough sense of how
    # different the "typical" direction is per slot).
    def _mean_dir(X):
        Xn = F.normalize(X, dim=-1)
        m = Xn.mean(dim=0)
        return F.normalize(m, dim=0)
    # slots are in different dims than encoder_pool, so for this we only
    # compare pair-wise slot directions.
    d_shared = _mean_dir(Y_shared)
    d_phase = _mean_dir(Y_phase)
    d_view = _mean_dir(Y_view)
    cos_sp = float((d_shared * d_phase).sum())
    cos_sv = float((d_shared * d_view).sum())
    cos_pv = float((d_phase * d_view).sum())

    # Slot-variance stats for reference
    var_pool = float(X_pool.var(dim=0).mean())
    var_shared = float(Y_shared.var(dim=0).mean())
    var_phase = float(Y_phase.var(dim=0).mean())
    var_view = float(Y_view.var(dim=0).mean())

    out = {
        "checkpoint": str(args.checkpoint),
        "n_encoded": len(pool_feats),
        "slot_r2_vs_encoder_pool": {
            "z_shared": r2_shared,
            "z_phase": r2_phase,
            "z_view": r2_view,
            "note": (
                "Held-out R² of linear regression from encoder_pool (1024-D) to each slot "
                "(256-D). R² ~1.0 = slot is a near-linear projection of the pool (H1 true, "
                "factorization is trivial). R² <0.9 = the slot's MLP has introduced "
                "nonlinearity the encoder_pool doesn't linearly capture."
            ),
        },
        "slot_mean_dir_cosines": {
            "shared_vs_phase": cos_sp,
            "shared_vs_view": cos_sv,
            "phase_vs_view": cos_pv,
        },
        "per_rep_mean_variance": {
            "encoder_pool": var_pool,
            "z_shared": var_shared,
            "z_phase": var_phase,
            "z_view": var_view,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
