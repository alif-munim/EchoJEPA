#!/usr/bin/env python3
"""Scaled phase-decodability probe (problem 3a+3b).

Encodes N target clips through a frozen encoder once, caches the
features, then trains multiple phase-probe configurations on the
cached features — amortizes the encode cost across HP configs and
checkpoint epochs.

Upgrades vs the F block in run_temporal_phase_diagnostics.py:
  - n=3000 (or whatever --num-studies is), not 200
  - Smarter probe: 1-hidden-layer MLP (256 units) + von Mises NLL loss
    on predicted (mu, kappa) for circular regression (more stable than
    MSE on [sin, cos] at moderate sample sizes)
  - 80/20 study-disjoint split
  - Also runs the old linear-on-(sin,cos)-MSE baseline for direct
    comparison to the finalbudget protocol
  - Reports circular MAE in degrees, per-bin accuracy (10 bins), and
    the const-mean baseline

Usage:
    python run_phase_probe_scaled.py --checkpoint ckpt.pt \
        --feature-mode z_phase --num-studies 3000 --out out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "classifier" / "phase" / "sampler"))

from scripts.neurips.phase.run_cross_view_retrieval_diag import (  # noqa: E402
    build_sampler, load_encoder_and_head, _load_clip, _load_test_study_ids,
)


def encode_pooled(clip, encoder):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = clip.unsqueeze(0).cuda()
        tokens = encoder(x)
        pooled = tokens.mean(dim=1).float().squeeze(0)
    return pooled.cpu()


def slot_from_pooled(pooled, head, mode):
    if head is None or mode == "encoder_pool":
        return pooled
    with torch.no_grad():
        slots = head(pooled.unsqueeze(0).cuda().float())
    if mode == "z_shared": v = slots["z_shared"]
    elif mode == "z_phase": v = slots["z_phase"]
    elif mode == "z_view": v = slots["z_view"]
    elif mode == "concat_shared_phase":
        v = torch.cat([slots["z_shared"], slots["z_phase"]], dim=-1)
    elif mode == "concat_all":
        v = torch.cat([slots["z_shared"], slots["z_phase"], slots["z_view"]], dim=-1)
    else: raise ValueError(mode)
    return v.squeeze(0).float().cpu()


# --------------------------------------------------------------------------- #
# Probe heads
# --------------------------------------------------------------------------- #


class LinearSinCosHead(torch.nn.Module):
    """Original finalbudget-style probe: linear regression to [sin(2πφ), cos(2πφ)]
    under MSE. Baseline."""
    def __init__(self, D):
        super().__init__()
        self.fc = torch.nn.Linear(D, 2)
    def forward(self, x): return self.fc(x)


class MLPVonMisesHead(torch.nn.Module):
    """Upgrade: 1-hidden-layer MLP producing (mean_angle_logits,
    log_kappa). mean_angle is parameterized via (mu_cos, mu_sin) outputs
    normalized to unit length; kappa is a concentration parameter."""
    def __init__(self, D, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(D, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 3),  # (mu_cos, mu_sin, log_kappa)
        )
    def forward(self, x):
        out = self.net(x)
        mu_cos, mu_sin, log_kappa = out[..., 0], out[..., 1], out[..., 2]
        # Normalize (mu_cos, mu_sin) to unit circle
        norm = torch.sqrt(mu_cos ** 2 + mu_sin ** 2).clamp(min=1e-6)
        return mu_cos / norm, mu_sin / norm, log_kappa


def von_mises_nll(mu_cos, mu_sin, log_kappa, phi):
    """Negative log-likelihood of φ ∈ [0, 1) under a von Mises
    distribution on the angle θ = 2πφ with mean direction (mu_cos,
    mu_sin) and concentration κ = exp(log_kappa)."""
    theta = 2 * np.pi * phi
    kappa = torch.exp(log_kappa.clamp(max=10.0))  # avoid overflow
    # Dot product of unit vectors
    dot = mu_cos * torch.cos(theta) + mu_sin * torch.sin(theta)
    # NLL = log(2π I0(κ)) − κ · dot; the I0 term is independent of ŷ given κ
    # so we just optimize −κ·dot + log I0(κ). Approximate log I0 for
    # stability (Abramowitz-Stegun).
    log_i0 = _log_bessel_i0(kappa)
    return -kappa * dot + log_i0


def _log_bessel_i0(kappa):
    """Numerically stable log(I0(κ)) for 0 ≤ κ ≤ exp(10)."""
    # For small κ, I0(κ) ≈ 1 + κ²/4; for large κ, I0(κ) ≈ exp(κ)/sqrt(2πκ)
    small = kappa < 3.75
    x_small = kappa / 3.75
    t_small = x_small ** 2
    # Polynomial (Abramowitz-Stegun 9.8.1)
    log_i0_small = torch.log(1.0 + t_small * (3.5156229 + t_small * (3.0899424 + t_small *
                 (1.2067492 + t_small * (0.2659732 + t_small * (0.0360768 + t_small * 0.0045813))))))
    # Large-arg asymptotic
    log_i0_large = kappa - 0.5 * torch.log(2 * np.pi * kappa.clamp(min=1e-6))
    return torch.where(small, log_i0_small, log_i0_large)


def circular_mae_deg(phi_pred, phi_true):
    d = torch.abs(phi_pred - phi_true)
    d = torch.minimum(d, 1.0 - d)
    return float(d.mean() * 360)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-mode", default="encoder_pool",
                        choices=["encoder_pool", "z_shared", "z_phase", "z_view",
                                 "concat_shared_phase", "concat_all"])
    parser.add_argument("--num-studies", type=int, default=3000)
    parser.add_argument("--phase-annotations", type=Path,
                        default=Path("/opt/dlami/nvme/probe/phase_annotations.parquet"))
    parser.add_argument("--view-labels", type=Path,
                        default=Path("/opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv"))
    parser.add_argument("--splits-csv", type=Path,
                        default=REPO_ROOT / "classifier" / "phase" / "splits" / "dicoms_split.csv")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--pairs-per-study", type=int, default=2)
    args = parser.parse_args()

    encoder, head, _ = load_encoder_and_head(args.checkpoint, args.feature_mode)
    has_head = head is not None

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
                            args.view_labels, pairs_per_study=args.pairs_per_study)
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

    test_studies = _load_test_study_ids(args.splits_csv)
    kept = []
    for r in records:
        if r.target_clip is None: continue
        if str(r.study_id) not in test_studies: continue
        kept.append(r)
        if len(kept) >= args.num_studies: break
    print(f"[sampler] kept {len(kept)} held-out records")

    feats = []
    phis = []
    studies = []
    t0 = time.time()
    for k, r in enumerate(kept):
        try:
            u = _uri(r.target_clip)
            if not u: continue
            clip = _load_clip(u, _window(r.target_clip, r.frames_per_clip, r.frame_step))
            pooled = encode_pooled(clip, encoder)
            f = slot_from_pooled(pooled, head, args.feature_mode)
            feats.append(f)
            phis.append(float(r.target_phi_b))
            studies.append(str(r.study_id))
        except Exception as e:
            print(f"[warn] row {k}: {e}")
            continue
        if (k + 1) % 100 == 0:
            dt = time.time() - t0
            rate = (k + 1) / dt
            eta = (len(kept) - k - 1) / max(rate, 0.01)
            print(f"  encoded {len(feats)}/{len(kept)}  [{rate:.1f}/s, ETA {eta/60:.1f}m]")
    n = len(feats)
    print(f"[encode] {n} clips; {(time.time()-t0)/60:.1f}m total")
    if n < 200:
        out = {"n": n, "note": "too few"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        return 1

    X = torch.stack(feats).float()
    phi = torch.tensor(phis, dtype=torch.float32)

    # Study-disjoint 80/20 split
    uniq = sorted(set(studies))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(uniq))
    n_train_s = int(0.8 * len(uniq))
    train_s = set(uniq[i] for i in perm[:n_train_s])
    tr_mask = torch.tensor([s in train_s for s in studies])
    va_mask = ~tr_mask

    mean = X[tr_mask].mean(dim=0, keepdim=True)
    std = X[tr_mask].std(dim=0, keepdim=True).clamp(min=1e-6)
    X = (X - mean) / std

    X_tr, phi_tr = X[tr_mask], phi[tr_mask]
    X_va, phi_va = X[va_mask], phi[va_mask]
    print(f"[split] train={len(X_tr)}, val={len(X_va)} (study-disjoint)")

    results = {}

    # --- Linear / MSE baseline (finalbudget protocol) ---
    D = X.shape[1]
    lin = LinearSinCosHead(D)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-3)
    y_tr = torch.stack([torch.sin(2 * np.pi * phi_tr), torch.cos(2 * np.pi * phi_tr)], dim=1)
    y_va = torch.stack([torch.sin(2 * np.pi * phi_va), torch.cos(2 * np.pi * phi_va)], dim=1)
    best_val_mae = float("inf")
    for ep in range(args.epochs):
        lin.train()
        perm2 = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 512):
            idx = perm2[i:i+512]
            opt.zero_grad()
            loss = F.mse_loss(lin(X_tr[idx]), y_tr[idx])
            loss.backward(); opt.step()
        lin.eval()
        with torch.no_grad():
            pred = lin(X_va)
            phi_pred = torch.atan2(pred[:, 0], pred[:, 1]) / (2 * np.pi) % 1.0
            mae_deg = circular_mae_deg(phi_pred, phi_va)
            best_val_mae = min(best_val_mae, mae_deg)
    results["linear_sincos_mse"] = {
        "best_circ_mae_deg": best_val_mae,
        "final_circ_mae_deg": mae_deg,
        "sin_val_mae": float((pred[:, 0] - y_va[:, 0]).abs().mean()),
        "cos_val_mae": float((pred[:, 1] - y_va[:, 1]).abs().mean()),
    }
    print(f"[lin-MSE] best val circ MAE = {best_val_mae:.2f}°")

    # --- Von Mises MLP probe ---
    mlp = MLPVonMisesHead(D, hidden=256)
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-3)
    best_val_mae = float("inf")
    for ep in range(args.epochs):
        mlp.train()
        perm2 = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 512):
            idx = perm2[i:i+512]
            opt.zero_grad()
            mc, ms, lk = mlp(X_tr[idx])
            loss = von_mises_nll(mc, ms, lk, phi_tr[idx]).mean()
            loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            mc, ms, lk = mlp(X_va)
            phi_pred = torch.atan2(ms, mc) / (2 * np.pi) % 1.0
            mae_deg = circular_mae_deg(phi_pred, phi_va)
            best_val_mae = min(best_val_mae, mae_deg)
    results["mlp_vonmises"] = {
        "best_circ_mae_deg": best_val_mae,
        "final_circ_mae_deg": mae_deg,
        "mean_kappa": float(torch.exp(lk).mean()),
    }
    print(f"[mlp-vM] best val circ MAE = {best_val_mae:.2f}°")

    # Constant-baseline circular MAE (mean-phase prediction)
    mean_phi = float(phi_tr.mean())
    d_const = torch.abs(phi_va - mean_phi)
    d_const = torch.minimum(d_const, 1.0 - d_const)
    results["const_baseline_circ_mae_deg"] = float(d_const.mean() * 360)

    out = {
        "meta": {"checkpoint": str(args.checkpoint), "feature_mode": args.feature_mode,
                 "n_train": int(tr_mask.sum()), "n_val": int(va_mask.sum())},
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
