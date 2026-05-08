#!/usr/bin/env python3
"""F/G/H diagnostics for frozen encoder checkpoints.

F. z_phase phase-decodability probe (MV2SV ckpts only, factorized head
   present). Linear probe predicts [sin(2πφ), cos(2πφ)] from the target
   clip's z_phase slot. Reports circular MAE + per-axis MAE.

G. Frame-shuffle sensitivity. Encode each clip twice — once with the
   natural frame order, once with a fixed random permutation. Reports
   mean cosine(clean, shuffled) and mean ΔR² style gap. Lower cos =
   more temporal structure in the encoder.

H. Intra-clip temporal dissimilarity. Encode the first 8 frames and
   the last 8 frames of the 16-frame clip separately. Reports mean
   cosine(half_a, half_b). High cos (≈1) means the encoder collapses
   temporal dynamics into a study-static summary.

Usage:
    python scripts/neurips/phase/run_temporal_phase_diagnostics.py \
        --checkpoint /opt/.../ckpt.pt \
        --out /opt/.../diag.json \
        [--feature-mode encoder_pool|z_shared|z_phase|z_view|concat_all] \
        [--num-studies 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "classifier" / "phase" / "sampler"))

# Import the debugged helpers from the retrieval diag so we stay in sync.
from scripts.neurips.phase.run_cross_view_retrieval_diag import (  # noqa: E402
    build_sampler,
    load_encoder_and_head,
    _load_clip,
    _load_test_study_ids,
    _slot_from_pooled,
)


def encode_pooled(clip: torch.Tensor, encoder: torch.nn.Module) -> torch.Tensor:
    """clip: [C, T, H, W] -> pooled encoder token [D]."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = clip.unsqueeze(0).cuda()
        tokens = encoder(x)  # [1, N_tok, D]
        pooled = tokens.mean(dim=1).float().squeeze(0)  # [D]
    return pooled.cpu()


def head_slot(pooled: torch.Tensor, head: torch.nn.Module | None, mode: str) -> torch.Tensor:
    if head is None or mode == "encoder_pool":
        return pooled
    with torch.no_grad():
        slots = head(pooled.unsqueeze(0).cuda().float())
    if mode == "z_shared":
        v = slots["z_shared"]
    elif mode == "z_phase":
        v = slots["z_phase"]
    elif mode == "z_view":
        v = slots["z_view"]
    elif mode == "concat_shared_phase":
        v = torch.cat([slots["z_shared"], slots["z_phase"]], dim=-1)
    elif mode == "concat_all":
        v = torch.cat([slots["z_shared"], slots["z_phase"], slots["z_view"]], dim=-1)
    else:
        raise ValueError(mode)
    return v.squeeze(0).float().cpu()


# --------------------------------------------------------------------------- #
# F: z_phase linear probe for sin/cos(φ)
# --------------------------------------------------------------------------- #


def _train_linear_regressor(
    X: torch.Tensor, y: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor,
    epochs: int = 200, lr: float = 1e-3, wd: float = 1e-3,
) -> tuple[torch.Tensor, torch.nn.Module]:
    D = X.shape[1]
    T = y.shape[1]
    head = torch.nn.Linear(D, T)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i : i + 256]
            opt.zero_grad()
            pred = head(X[idx])
            loss = torch.nn.functional.mse_loss(pred, y[idx])
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        pred_val = head(X_val)
    return pred_val, head


def phase_decodability_probe(
    feats_z_phase: list[torch.Tensor],
    phases: list[float],
    studies: list[str],
) -> dict:
    """Train linear probes for sin(2πφ) and cos(2πφ) from z_phase slot
    features. Study-disjoint 80/20 split within the test-set sample.
    Circular MAE = min(|Δφ|, 1-|Δφ|) after inverting predicted (sin,cos)
    back to φ via arctan2, in cycle fractions. Reports in degrees."""
    X = torch.stack(feats_z_phase).float()
    sin = torch.tensor([np.sin(2 * np.pi * p) for p in phases], dtype=torch.float32)
    cos = torch.tensor([np.cos(2 * np.pi * p) for p in phases], dtype=torch.float32)
    y = torch.stack([sin, cos], dim=1)

    uniq = sorted(set(studies))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(uniq))
    n_train = int(0.8 * len(uniq))
    train_studies = set(uniq[i] for i in perm[:n_train])
    tr_mask = torch.tensor([s in train_studies for s in studies])
    va_mask = ~tr_mask
    if tr_mask.sum() < 20 or va_mask.sum() < 10:
        return {"note": "too few samples", "n_train": int(tr_mask.sum()),
                "n_val": int(va_mask.sum())}

    mean = X[tr_mask].mean(dim=0, keepdim=True)
    std = X[tr_mask].std(dim=0, keepdim=True).clamp(min=1e-6)
    Xn = (X - mean) / std

    pred_val, _ = _train_linear_regressor(Xn[tr_mask], y[tr_mask], Xn[va_mask], y[va_mask])
    sin_p, cos_p = pred_val[:, 0].numpy(), pred_val[:, 1].numpy()
    sin_t, cos_t = y[va_mask, 0].numpy(), y[va_mask, 1].numpy()
    phase_pred = np.arctan2(sin_p, cos_p) / (2 * np.pi) % 1.0
    phase_true = np.arctan2(sin_t, cos_t) / (2 * np.pi) % 1.0
    dist = np.abs(phase_pred - phase_true)
    circ = np.minimum(dist, 1.0 - dist)
    circ_mae_deg = float(np.mean(circ) * 360)
    sin_mae = float(np.mean(np.abs(sin_p - sin_t)))
    cos_mae = float(np.mean(np.abs(cos_p - cos_t)))
    # baselines
    baseline_sin = float(np.mean(np.abs(sin_t - sin_t.mean())))
    baseline_cos = float(np.mean(np.abs(cos_t - cos_t.mean())))
    return {
        "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum()),
        "circ_mae_deg": circ_mae_deg,
        "sin_mae": sin_mae, "cos_mae": cos_mae,
        "sin_mae_mean_baseline": baseline_sin, "cos_mae_mean_baseline": baseline_cos,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--feature-mode",
        choices=["encoder_pool", "z_shared", "z_phase", "z_view",
                 "concat_shared_phase", "concat_all"],
        default="encoder_pool",
        help=(
            "For F: set to z_phase on MV2SV ckpts to probe the z_phase slot "
            "specifically; encoder_pool for all others. For G/H this sets the "
            "feature used to compute cosines."
        ),
    )
    parser.add_argument("--num-studies", type=int, default=200)
    parser.add_argument("--phase-annotations", type=Path, default=Path("/opt/dlami/nvme/probe/phase_annotations.parquet"))
    parser.add_argument("--view-labels", type=Path, default=Path("/opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv"))
    parser.add_argument("--splits-csv", type=Path, default=REPO_ROOT / "classifier" / "phase" / "splits" / "dicoms_split.csv")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--also-phase-probe", action="store_true",
                        help="Run F (requires MV2SV ckpt with factorized head).")
    args = parser.parse_args()

    encoder, head, embed_dim = load_encoder_and_head(args.checkpoint, args.feature_mode)
    has_head = head is not None

    view_pair_policy = {
        "enabled": True, "same_view_prob": 0.05, "same_family_prob": 0.35,
        "cross_family_prob": 0.60, "require_different_dicom": True,
        "allow_same_view": True, "resample_attempts": 8,
    }
    mv2sv_cfg = {
        "enabled": True,
        "target_view_sampling": {
            "stage": "stage2",
            "a4c_source_targets": ["A2C", "A5C"],
            "allowed_targets": ["A2C", "A5C", "A3C", "PLAX", "PSAX-MV"],
            "target_dropout": 0.0, "require_different_view": True,
        },
        "fused_pool": {"enabled": False, "n_fused_min": 2, "n_fused_max": 2},
    }
    test_studies = _load_test_study_ids(args.splits_csv)
    sampler = build_sampler(args.phase_annotations, mv2sv_cfg, view_pair_policy,
                            args.view_labels, pairs_per_study=4, seed=args.seed)
    sampler.epoch = 0
    all_records = sampler.build_records()
    _sdf = getattr(sampler, "_df", None)
    if _sdf is None:
        _sdf = getattr(sampler, "df", None)
    row_to_uri = dict(zip(_sdf.index.astype(int), _sdf["s3_uri"].astype(str)))

    def _uri_for(clip) -> str:
        raw = row_to_uri.get(int(clip.row_idx), "")
        if raw.startswith("s3://echodata25/mimic-raw-staging"):
            raw = raw.replace("s3://echodata25/mimic-raw-staging", "s3://echodata25/mimic-echo-224px", 1)
            if raw.endswith(".dcm"):
                raw = raw[:-4] + ".mp4"
        return raw

    def _frame_window(clip, frames_per_clip: int = 16, frame_step: int = 1) -> list[int]:
        start = int(clip.anchor_frame) - (frames_per_clip // 2) * frame_step
        return [max(0, start + i * frame_step) for i in range(frames_per_clip)]

    # Filter: any record with a target_clip that falls in the test split.
    kept = []
    for r in all_records:
        if r.target_clip is None:
            continue
        if str(r.study_id) not in test_studies:
            continue
        kept.append(r)
        if len(kept) >= args.num_studies:
            break
    print(f"[sampler] kept {len(kept)} test-split records with target_clip")

    # For each record we encode the TARGET clip three ways:
    #   tgt_clean      (natural frame order)
    #   tgt_shuffled   (same clip, fixed random permutation of 16 frames)
    #   tgt_half_a     (frames [0..7])
    #   tgt_half_b     (frames [8..15])
    # Plus phase at the target anchor frame for probe F.

    rng_g = np.random.default_rng(args.seed)
    shuffle_perm = list(rng_g.permutation(16))

    tgt_feats_clean = []
    tgt_feats_shuf = []
    tgt_feats_half_a = []
    tgt_feats_half_b = []
    tgt_z_phase_feats = []  # populated only if has_head
    phases = []
    studies = []
    target_views = []

    for k, r in enumerate(kept):
        try:
            tgt_uri = _uri_for(r.target_clip)
            if not tgt_uri:
                continue
            win16 = _frame_window(r.target_clip, 16, r.frame_step)
            clip16 = _load_clip(tgt_uri, win16)  # [C, 16, H, W]
            # clean: pooled + slot if requested
            pooled_clean = encode_pooled(clip16, encoder)
            feat_clean = head_slot(pooled_clean, head, args.feature_mode)
            # shuffled: permute time axis then encode
            clip_shuf = clip16[:, shuffle_perm, :, :].contiguous()
            pooled_shuf = encode_pooled(clip_shuf, encoder)
            feat_shuf = head_slot(pooled_shuf, head, args.feature_mode)
            # halves: 8 frames each, but encoder expects 16 -> duplicate to match.
            # Most V-JEPA ViT encoders want T==frames_per_clip; we just pad by
            # repeating frames to fill 16 (keeps tubelet alignment).
            half_a = clip16[:, :8, :, :]
            half_a_pad = torch.cat([half_a, half_a], dim=1)  # [C, 16, H, W]
            half_b = clip16[:, 8:, :, :]
            half_b_pad = torch.cat([half_b, half_b], dim=1)
            pooled_a = encode_pooled(half_a_pad, encoder)
            pooled_b = encode_pooled(half_b_pad, encoder)
            feat_a = head_slot(pooled_a, head, args.feature_mode)
            feat_b = head_slot(pooled_b, head, args.feature_mode)

            tgt_feats_clean.append(feat_clean)
            tgt_feats_shuf.append(feat_shuf)
            tgt_feats_half_a.append(feat_a)
            tgt_feats_half_b.append(feat_b)
            if args.also_phase_probe and has_head:
                tgt_z_phase_feats.append(head_slot(pooled_clean, head, "z_phase"))
            phases.append(float(r.target_phi_b))
            studies.append(str(r.study_id))
            target_views.append(str(r.target_view))
        except Exception as e:
            print(f"[warn] skip row {k}: {e}")
            continue
        if (k + 1) % 25 == 0:
            print(f"  encoded {len(tgt_feats_clean)}/{len(kept)}")

    n = len(tgt_feats_clean)
    print(f"[encode] {n} target clips encoded successfully (4 encodes each)")
    if n < 10:
        print("[ERROR] too few; skipping metrics")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"n": n, "note": "too few successful encodes"}, indent=2))
        return 1

    Cl = F.normalize(torch.stack(tgt_feats_clean), dim=-1)
    Sh = F.normalize(torch.stack(tgt_feats_shuf), dim=-1)
    Ha = F.normalize(torch.stack(tgt_feats_half_a), dim=-1)
    Hb = F.normalize(torch.stack(tgt_feats_half_b), dim=-1)

    # G: frame-shuffle sensitivity
    cos_clean_shuf = (Cl * Sh).sum(dim=-1)
    # H: intra-clip temporal (first vs second half)
    cos_half_ab = (Ha * Hb).sum(dim=-1)

    metrics: dict[str, Any] = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "feature_mode": args.feature_mode,
            "num_encoded": n,
        },
        "G_frame_shuffle_sensitivity": {
            "cos_clean_shuf_mean": float(cos_clean_shuf.mean()),
            "cos_clean_shuf_std": float(cos_clean_shuf.std()),
            "cos_clean_shuf_p05": float(np.percentile(cos_clean_shuf.numpy(), 5)),
            "cos_clean_shuf_p95": float(np.percentile(cos_clean_shuf.numpy(), 95)),
            "note": "Lower mean cos = encoder changes more under frame shuffle = stronger temporal structure",
        },
        "H_intra_clip_temporal": {
            "cos_half_ab_mean": float(cos_half_ab.mean()),
            "cos_half_ab_std": float(cos_half_ab.std()),
            "cos_half_ab_p05": float(np.percentile(cos_half_ab.numpy(), 5)),
            "cos_half_ab_p95": float(np.percentile(cos_half_ab.numpy(), 95)),
            "note": "Close to 1.0 = encoder collapses intra-clip temporal dynamics",
        },
    }
    if args.also_phase_probe and tgt_z_phase_feats:
        metrics["F_z_phase_probe"] = phase_decodability_probe(
            tgt_z_phase_feats, phases, studies
        )
    elif args.also_phase_probe:
        # For non-MV2SV ckpts we probe the encoder pool instead — that's
        # the finalbudget protocol and gives a comparable baseline.
        metrics["F_z_phase_probe"] = phase_decodability_probe(
            tgt_feats_clean, phases, studies
        )
        metrics["F_z_phase_probe"]["note"] = "Ran on encoder_pool (no factorized head available)"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    print(f"[out] wrote {args.out}")
    print(f"[summary] G cos(clean,shuf)={metrics['G_frame_shuffle_sensitivity']['cos_clean_shuf_mean']:.4f}  "
          f"H cos(half_a,half_b)={metrics['H_intra_clip_temporal']['cos_half_ab_mean']:.4f}")
    if "F_z_phase_probe" in metrics and "circ_mae_deg" in metrics["F_z_phase_probe"]:
        print(f"[summary] F circ_mae_deg={metrics['F_z_phase_probe']['circ_mae_deg']:.1f}  ({metrics['F_z_phase_probe']['note'] if 'note' in metrics['F_z_phase_probe'] else ''})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
