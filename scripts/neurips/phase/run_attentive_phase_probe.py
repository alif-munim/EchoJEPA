#!/usr/bin/env python3
"""Attentive phase probe — token-level readout.

Same sampler / encoder / split as run_phase_probe_scaled.py, but instead
of mean-pooling the encoder's [B, N_tok, D] output to a 1024-D vector,
the probe runs a learnable cross-attention query over the full token
sequence. Reuses src/models/AttentivePooler (d=1, num_queries=1, 16
heads).

Why attentive instead of linear-on-pool:
  - Phase is a per-frame property. V-JEPA's encoder emits 1568 tokens
    per 16-frame clip (8 temporal × 196 spatial). Mean-pooling mixes
    tokens from different phase windows together, so a linear probe on
    the pool can only recover phase if it's linearly-in-average, which
    it generally isn't.
  - An attentive probe with a learnable query can attend to the
    temporal tokens closest to the anchor frame, matching finalbudget's
    protocol that achieved 42° circular MAE vs the ~88° floor we see
    with linear-on-pool.

Protocol:
  - Encode 3000 held-out-test-split clips once; cache the [N, N_tok, D]
    tensor to CPU memory.
  - Train an AttentivePooler(num_queries=1, embed_dim=D, num_heads=16,
    depth=1) + Linear(D, 2) head for [sin(2πφ), cos(2πφ)].
  - MSE loss; AdamW lr 1e-3 wd 1e-3; 200 epochs.
  - Study-disjoint 80/20 split.

Only supports encoder-token feature mode (not slot modes — slots are
post-pool singletons by construction and don't need an attentive probe).

Usage:
    python run_attentive_phase_probe.py --checkpoint ckpt.pt \
        --num-studies 3000 --out out.json
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


def encode_tokens(clip, encoder):
    """Return [N_tok, D] — full token sequence, not mean-pooled."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = clip.unsqueeze(0).cuda()
        tokens = encoder(x)  # [1, N_tok, D]
    return tokens.float().squeeze(0).cpu()


def circular_mae_deg(phi_pred, phi_true):
    d = torch.abs(phi_pred - phi_true)
    d = torch.minimum(d, 1.0 - d)
    return float(d.mean() * 360)


def train_attentive_probe(
    tokens: torch.Tensor,   # [N, N_tok, D], CPU
    phi: torch.Tensor,       # [N], CPU
    studies: list[str],
    embed_dim: int,
    n_heads: int = 16,
    depth: int = 1,
    epochs: int = 100,
    lr: float = 1e-3,
    wd: float = 1e-3,
    batch_size: int = 64,
) -> dict:
    from src.models.attentive_pooler import AttentivePooler

    N, N_tok, D = tokens.shape
    # Study-disjoint split
    uniq = sorted(set(studies))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(uniq))
    n_train_s = int(0.8 * len(uniq))
    train_s = set(uniq[i] for i in perm[:n_train_s])
    tr_mask = torch.tensor([s in train_s for s in studies])
    va_mask = ~tr_mask
    n_tr = int(tr_mask.sum()); n_va = int(va_mask.sum())
    print(f"[attn-probe] train={n_tr} val={n_va} tokens/clip={N_tok} D={D}")

    # Targets
    sin = torch.sin(2 * np.pi * phi)
    cos = torch.cos(2 * np.pi * phi)
    y = torch.stack([sin, cos], dim=1)  # [N, 2]

    pooler = AttentivePooler(
        num_queries=1, embed_dim=embed_dim, num_heads=n_heads,
        mlp_ratio=4.0, depth=depth, complete_block=True,
    ).cuda()
    head = torch.nn.Linear(embed_dim, 2).cuda()
    params = list(pooler.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    tokens_tr = tokens[tr_mask]
    y_tr = y[tr_mask]
    tokens_va = tokens[va_mask]
    y_va = y[va_mask]
    phi_va = phi[va_mask]

    best_val_mae = float("inf")
    best_ep = -1
    t0 = time.time()
    for ep in range(epochs):
        pooler.train(); head.train()
        perm2 = torch.randperm(n_tr)
        running_loss = 0.0; n_batches = 0
        for i in range(0, n_tr, batch_size):
            idx = perm2[i:i + batch_size]
            x_batch = tokens_tr[idx].cuda(non_blocking=True)
            y_batch = y_tr[idx].cuda(non_blocking=True)
            opt.zero_grad()
            q = pooler(x_batch)  # [B, 1, D]
            pred = head(q.squeeze(1))  # [B, 2]
            loss = F.mse_loss(pred, y_batch)
            loss.backward(); opt.step()
            running_loss += float(loss); n_batches += 1
        # Eval
        pooler.eval(); head.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, n_va, batch_size):
                x_batch = tokens_va[i:i + batch_size].cuda(non_blocking=True)
                q = pooler(x_batch)
                p = head(q.squeeze(1))
                preds.append(p.cpu())
        pred_va = torch.cat(preds, dim=0)  # [n_va, 2]
        phi_pred = torch.atan2(pred_va[:, 0], pred_va[:, 1]) / (2 * np.pi) % 1.0
        mae_deg = circular_mae_deg(phi_pred, phi_va)
        if mae_deg < best_val_mae:
            best_val_mae = mae_deg
            best_ep = ep
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  ep{ep+1:3d}  train_mse={running_loss/n_batches:.4f}  val_circ_MAE={mae_deg:.2f}°  best={best_val_mae:.2f}° (@ep{best_ep+1})")

    # Const baseline
    mean_phi = float(phi[tr_mask].mean())
    d_const = torch.abs(phi_va - mean_phi)
    d_const = torch.minimum(d_const, 1.0 - d_const)
    const_mae = float(d_const.mean() * 360)
    return {
        "n_train": n_tr, "n_val": n_va,
        "best_val_circ_mae_deg": best_val_mae,
        "best_epoch": best_ep + 1,
        "final_val_circ_mae_deg": mae_deg,
        "const_baseline_circ_mae_deg": const_mae,
        "training_seconds": time.time() - t0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-studies", type=int, default=3000)
    parser.add_argument("--phase-annotations", type=Path,
                        default=Path("/opt/dlami/nvme/probe/phase_annotations.parquet"))
    parser.add_argument("--view-labels", type=Path,
                        default=Path("/opt/dlami/nvme/data/view_labels/mimic_view_predictions.csv"))
    parser.add_argument("--splits-csv", type=Path,
                        default=REPO_ROOT / "classifier" / "phase" / "splits" / "dicoms_split.csv")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pairs-per-study", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()

    # Load encoder (encoder_pool mode just skips head construction)
    encoder, _, _ = load_encoder_and_head(args.checkpoint, "encoder_pool")

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

    tok_feats = []
    phis = []
    studies = []
    t0 = time.time()
    for k, r in enumerate(kept):
        try:
            u = _uri(r.target_clip)
            if not u: continue
            clip = _load_clip(u, _window(r.target_clip, r.frames_per_clip, r.frame_step))
            tokens = encode_tokens(clip, encoder)  # [N_tok, D]
            tok_feats.append(tokens)
            phis.append(float(r.target_phi_b))
            studies.append(str(r.study_id))
        except Exception as e:
            print(f"[warn] row {k}: {e}")
            continue
        if (k + 1) % 100 == 0:
            dt = time.time() - t0
            rate = (k + 1) / max(dt, 0.01)
            eta = (len(kept) - k - 1) / max(rate, 0.01)
            print(f"  encoded {len(tok_feats)}/{len(kept)}  [{rate:.1f}/s, ETA {eta/60:.1f}m]")
    n = len(tok_feats)
    print(f"[encode] {n} clips; {(time.time()-t0)/60:.1f}m total")
    if n < 200:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"n": n, "note": "too few"}, indent=2))
        return 1

    # Stack to [N, N_tok, D] — memory-critical
    X = torch.stack(tok_feats)
    print(f"[tokens] shape={tuple(X.shape)}  dtype={X.dtype}  "
          f"mem_GB={X.element_size()*X.numel()/1e9:.2f}")
    phi = torch.tensor(phis, dtype=torch.float32)

    results = train_attentive_probe(
        X, phi, studies, embed_dim=X.shape[-1],
        n_heads=args.num_heads, depth=args.depth,
        epochs=args.epochs,
    )
    out = {
        "meta": {"checkpoint": str(args.checkpoint), "num_heads": args.num_heads,
                 "depth": args.depth, "n_tok": X.shape[1], "embed_dim": X.shape[-1]},
        "attentive_probe": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
