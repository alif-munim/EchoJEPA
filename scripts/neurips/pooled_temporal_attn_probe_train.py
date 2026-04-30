"""Pooled temporal-attention probe on cached [N, S, T, D] features.

Bridging probe between linear-A raw (`[T,D] flatten -> linear`) and the
full-token attentive probe (cross-attention over the full spatial×temporal
token grid). This script operates ONLY on the spatially-pooled [T=8, D]
sequence and adds a lightweight learned-query cross-attention pooler +
small MLP head.

Matches the cache format, train/val split convention, seed handling, and
per-clip pred-averaging of `scripts/neurips/diff_probe_train.py` exactly.

Usage:
  python -m scripts.neurips.pooled_temporal_attn_probe_train \
    --train-cache features/diff_probe/jepa_e100_train.pt \
    --test-cache  features/diff_probe/jepa_e100_test.pt \
    --model jepa_e100 \
    --out-json results/pooled_temporal_attn/jepa_e100.json \
    --seeds 0 1 2 3 4 --lrs 1e-4 3e-4 1e-3 --wds 1e-4 1e-2
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------- Probe architecture ----------------------------


class CrossAttnPool(nn.Module):
    """Learned query cross-attention over [T, D] pooled sequence.

    Architecture:
      input [B, T, D] -> LayerNorm
      learned query [1, 1, D]
      cross-attn (query -> keys/values from input), n_heads heads
      residual + LayerNorm
      MLP(D -> 4D -> D) -> residual + LayerNorm
      head Linear(D, 1)
    """

    def __init__(self, d, n_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        # Find a valid head count <= n_heads that divides d
        while n_heads > 1 and d % n_heads != 0:
            n_heads -= 1
        self.n_heads = n_heads
        self.ln_in = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.attn = nn.MultiheadAttention(d, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d, d),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):  # x: [B, T, D]
        x = self.ln_in(x)
        q = self.q.expand(x.shape[0], -1, -1)   # [B, 1, D]
        z, _ = self.attn(q, x, x, need_weights=False)   # [B, 1, D]
        z = self.ln1(z + q)
        z = self.ln2(z + self.mlp(z))
        return self.head(z.squeeze(1)).squeeze(-1)   # [B]


# ---------------------------- Data prep helpers ------------------------------


def load_cache(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    feats = d["features"].float()   # [N, S, T, D]
    labels = d["labels"].float()    # [N]
    paths = list(d.get("paths", [f"clip_{i}" for i in range(feats.shape[0])]))
    return feats, labels, paths


def to_segments_raw(feats, labels):
    """[N, S, T, D] -> ([N*S, T, D], [N*S], clip_idx[N*S]).  RAW pooled features."""
    N, S, T, D = feats.shape
    x = feats.reshape(N * S, T, D)
    y = labels.unsqueeze(1).expand(N, S).reshape(-1)
    clip_idx = torch.arange(N).unsqueeze(1).expand(N, S).reshape(-1)
    return x, y, clip_idx


def stratified_split(labels_np, frac_val=0.1, seed=0, n_bins=5):
    rng = np.random.default_rng(seed)
    n = len(labels_np)
    q = np.quantile(labels_np, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(labels_np, q)
    train_idx, val_idx = [], []
    for b in range(n_bins):
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac_val)))
        val_idx.extend(idx[:k].tolist())
        train_idx.extend(idx[k:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.std(y_true) < 1e-9 or np.std(y_pred) < 1e-9:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


# -------------------------------- Trainer -----------------------------------


def _eval_on_test(probe, test_x, test_y, test_clip_idx, label_mean, label_std, device):
    probe.eval()
    test_loader = DataLoader(TensorDataset(test_x), batch_size=256, shuffle=False)
    preds_segment = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=True)
            preds_segment.append(probe(xb).float().cpu().numpy())
    preds_segment = np.concatenate(preds_segment)
    preds_segment = preds_segment * label_std + label_mean

    tidx = test_clip_idx.numpy()
    n_clips = int(tidx.max()) + 1
    ps = np.zeros(n_clips, dtype=np.float64)
    pc = np.zeros(n_clips, dtype=np.int64)
    ct = np.full(n_clips, np.nan, dtype=np.float64)
    ty = test_y.numpy()
    for p, c, y in zip(preds_segment, tidx, ty):
        ps[c] += p
        pc[c] += 1
        ct[c] = y
    pr = ps / np.maximum(pc, 1)
    mask = pc > 0
    y_true, y_pred = ct[mask], pr[mask]
    return {
        "test_r2": r2_score(y_true, y_pred),
        "test_pearson": pearson_r(y_true, y_pred),
        "test_mae": float(np.mean(np.abs(y_true - y_pred))),
        "n_test_clips": int(mask.sum()),
    }


def train_one(
    train_x, train_y,
    val_x, val_y,
    test_x, test_y, test_clip_idx,
    label_mean, label_std,
    lr, wd,
    n_heads=8, mlp_ratio=4, dropout=0.1,
    batch_size=64, max_epochs=50,
    patience=7, min_epochs=15, min_delta=0.002,
    device="cuda", seed=0,
):
    torch.manual_seed(seed)
    t_prime, d = train_x.shape[1], train_x.shape[2]
    probe = CrossAttnPool(d, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    def norm_y(y): return (y - label_mean) / label_std

    train_ds = TensorDataset(train_x, norm_y(train_y))
    val_ds = TensorDataset(val_x, norm_y(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_r2 = -np.inf
    best_val_mse = np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        probe.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = probe(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        probe.eval()
        vp, vy = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                vp.append(probe(xb).float().cpu().numpy())
                vy.append(yb.numpy())
        vp = np.concatenate(vp)
        vy = np.concatenate(vy)
        vr2 = r2_score(vy, vp)
        vmse = float(np.mean((vy - vp) ** 2))

        if vr2 > best_val_r2 + min_delta:
            best_val_r2 = vr2
            best_val_mse = vmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)

    test_metrics = _eval_on_test(probe, test_x, test_y, test_clip_idx, label_mean, label_std, device)
    return {
        "val_r2": float(best_val_r2),
        "val_mse": float(best_val_mse),
        "best_epoch": int(best_epoch),
        "n_epochs_run": int(epoch),
        **test_metrics,
    }


# --------------------------------- Main ------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    ap.add_argument("--wds", type=float, nargs="+", default=[1e-4, 1e-2])
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--min-epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pta] device={device}", flush=True)
    print(f"[pta] loading train: {args.train_cache}", flush=True)
    tr_feats, tr_labels, _ = load_cache(args.train_cache)
    print(f"[pta] loading test:  {args.test_cache}", flush=True)
    te_feats, te_labels, _ = load_cache(args.test_cache)
    print(f"[pta] train feats {tuple(tr_feats.shape)}  test feats {tuple(te_feats.shape)}", flush=True)

    label_mean = float(tr_labels.mean())
    label_std = float(tr_labels.std().clamp(min=1e-6))
    print(f"[pta] label mean/std (train) = {label_mean:.4f} / {label_std:.4f}", flush=True)

    # Pre-compute segmented tensors (raw pooled sequence)
    tr_x_all, tr_y_all, _ = to_segments_raw(tr_feats, tr_labels)
    te_x, te_y, te_clip_idx = to_segments_raw(te_feats, te_labels)
    N_tr, S = tr_feats.shape[0], tr_feats.shape[1]
    print(f"[pta] segments: tr_x_all {tuple(tr_x_all.shape)}  te_x {tuple(te_x.shape)}", flush=True)

    results = []
    for seed in args.seeds:
        train_idx, val_idx = stratified_split(tr_labels.numpy(), frac_val=0.1, seed=seed)
        seg_train_mask = np.isin(np.repeat(np.arange(N_tr), S), train_idx)
        seg_val_mask = np.isin(np.repeat(np.arange(N_tr), S), val_idx)
        tr_x = tr_x_all[seg_train_mask]
        tr_y = tr_y_all[seg_train_mask]
        vl_x = tr_x_all[seg_val_mask]
        vl_y = tr_y_all[seg_val_mask]

        for lr in args.lrs:
            for wd in args.wds:
                print(f"[pta] seed={seed} lr={lr:g} wd={wd:g}", flush=True)
                out = train_one(
                    tr_x, tr_y, vl_x, vl_y, te_x, te_y, te_clip_idx,
                    label_mean, label_std,
                    lr=lr, wd=wd,
                    n_heads=args.n_heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout,
                    batch_size=args.batch_size, max_epochs=args.max_epochs,
                    patience=args.patience, min_epochs=args.min_epochs,
                    device=device, seed=seed,
                )
                out.update({
                    "model": args.model,
                    "train_cache": args.train_cache,
                    "test_cache": args.test_cache,
                    "arch": "pooled-temporal-attn",
                    "input": "raw",
                    "seed": seed,
                    "lr": lr,
                    "wd": wd,
                    "n_heads": args.n_heads,
                    "mlp_ratio": args.mlp_ratio,
                    "dropout": args.dropout,
                    "label_mean": label_mean,
                    "label_std": label_std,
                })
                results.append(out)
                print(f"[pta]   -> test_r2={out['test_r2']:.4f}  val_r2={out['val_r2']:.4f}  best_epoch={out['best_epoch']}", flush=True)

    # Best-HP selection: for each seed, pick the (lr, wd) with highest val R²; report the corresponding test R²
    by_seed = {}
    for r in results:
        by_seed.setdefault(r["seed"], []).append(r)
    per_seed_best = []
    for seed, rows in by_seed.items():
        best = max(rows, key=lambda x: x["val_r2"])
        per_seed_best.append(best)

    test_r2s = np.array([r["test_r2"] for r in per_seed_best])
    test_maes = np.array([r["test_mae"] for r in per_seed_best])
    test_rs = np.array([r["test_pearson"] for r in per_seed_best])

    summary = {
        "model": args.model,
        "n_seeds": len(per_seed_best),
        "best_per_seed": per_seed_best,
        "test_r2_mean": float(test_r2s.mean()),
        "test_r2_std":  float(test_r2s.std()),
        "test_mae_mean": float(test_maes.mean()),
        "test_mae_std":  float(test_maes.std()),
        "test_pearson_mean": float(test_rs.mean()),
        "test_pearson_std":  float(test_rs.std()),
    }

    Path(os.path.dirname(args.out_json)).mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "args": vars(args),
            "results": results,
            "summary": summary,
        }, f, indent=2)

    print(f"\n[pta] {args.model}  test R² (best-HP-per-seed, n={len(per_seed_best)})  "
          f"mean={summary['test_r2_mean']:.3f}  std={summary['test_r2_std']:.3f}")
    print(f"[pta] wrote {args.out_json}")


if __name__ == "__main__":
    main()
