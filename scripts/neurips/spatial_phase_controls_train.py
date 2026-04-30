"""Spatial / phase / temporal-information controls on cached pooled features.

Reuses the [N, S=2, T=8, D=1024] caches used by diff_probe_train.py and the
pooled temporal-attn bridge. Runs a battery of linear-A probes with different
input transforms plus a re-training of the pooled temporal-attn probe on
duplicated/mean-repeated sequences.

All experiments follow the same convention as diff_probe_train.py:
  - stratified 90/10 train/val split by EF quintile per seed
  - 5 seeds
  - best epoch by val R²
  - best HP per seed by val R²
  - per-video pred-averaging across S=2 segments
  - test metrics on EchoNet-Dynamic test (1,277 videos)

Variants (selectable via --variants):

  Linear-A probes (LR=1e-3, WD sweep {1e-4, 1e-2}, max 30 epochs):
    raw          - input [T, D]                                    (baseline; matches job 418)
    diff         - input [T-1, D], z_{t+1} - z_t                   (baseline; matches job 423)
    mean         - input [1, D], time-mean z̄ = (1/T) Σ z_t         (control A)
    single_t0..single_t7   - input [1, D], single token z_t        (control B per-token)
    single_best_val        - best-token chosen by val R²           (control B oracle)

  Pooled temporal-attn probes (LR sweep {1e-4, 3e-4, 1e-3}, WD sweep {1e-4, 1e-2}, max 50 epochs):
    pta_raw          - input [T, D]                                (baseline; matches job 424)
    pta_mean_repeat  - input [T, D] with mean-token repeated T times (control D-1)
    pta_best_repeat  - input [T, D] with val-selected best token repeated T times (control D-2)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------- Probes ----------------------------


class LinearA(nn.Module):
    def __init__(self, t_prime, d):
        super().__init__()
        self.fc = nn.Linear(t_prime * d, 1)

    def forward(self, x):
        return self.fc(x.flatten(1)).squeeze(-1)


class CrossAttnPool(nn.Module):
    """Same as pooled_temporal_attn_probe_train.py."""

    def __init__(self, d, n_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
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

    def forward(self, x):
        x = self.ln_in(x)
        q = self.q.expand(x.shape[0], -1, -1)
        z, _ = self.attn(q, x, x, need_weights=False)
        z = self.ln1(z + q)
        z = self.ln2(z + self.mlp(z))
        return self.head(z.squeeze(1)).squeeze(-1)


# ---------------------------- Data ----------------------------


def load_cache(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["features"].float(), d["labels"].float()


def stratified_split(labels_np, frac_val=0.1, seed=0, n_bins=5):
    rng = np.random.default_rng(seed)
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


def to_segments(feats, labels):
    """[N, S, T, D] -> ([N*S, T, D], [N*S], clip_idx[N*S])."""
    N, S, T, D = feats.shape
    x = feats.reshape(N * S, T, D)
    y = labels.unsqueeze(1).expand(N, S).reshape(-1)
    clip_idx = torch.arange(N).unsqueeze(1).expand(N, S).reshape(-1)
    return x, y, clip_idx


def apply_variant(x_TD, variant, token_idx=None):
    """x_TD: [B, T, D]. Returns transformed input per variant."""
    B, T, D = x_TD.shape
    if variant == "raw":
        return x_TD
    if variant == "diff":
        return x_TD[:, 1:, :] - x_TD[:, :-1, :]
    if variant == "mean":
        return x_TD.mean(dim=1, keepdim=True)  # [B, 1, D]
    if variant.startswith("single_t"):
        t = int(variant.replace("single_t", ""))
        return x_TD[:, t:t + 1, :]
    if variant == "mean_repeat":
        mean = x_TD.mean(dim=1, keepdim=True)
        return mean.expand(B, T, D).contiguous()
    if variant == "best_repeat":
        assert token_idx is not None, "best_repeat needs token_idx"
        tok = x_TD[:, token_idx:token_idx + 1, :]
        return tok.expand(B, T, D).contiguous()
    raise ValueError(f"unknown variant {variant}")


# ---------------------------- Metrics ----------------------------


def r2_score(yt, yp):
    yt = np.asarray(yt, dtype=np.float64)
    yp = np.asarray(yp, dtype=np.float64)
    sr = float(np.sum((yt - yp) ** 2))
    st = float(np.sum((yt - yt.mean()) ** 2))
    return 0.0 if st <= 0 else 1.0 - sr / st


def pearson_r(yt, yp):
    if np.std(yt) < 1e-9 or np.std(yp) < 1e-9:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def eval_on_test(probe, test_x, test_y, test_clip_idx, label_mean, label_std, device):
    probe.eval()
    loader = DataLoader(TensorDataset(test_x), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(probe(xb).float().cpu().numpy())
    preds = np.concatenate(preds) * label_std + label_mean
    tidx = test_clip_idx.numpy()
    n_clips = int(tidx.max()) + 1
    ps = np.zeros(n_clips, dtype=np.float64)
    pc = np.zeros(n_clips, dtype=np.int64)
    ct = np.full(n_clips, np.nan, dtype=np.float64)
    ty = test_y.numpy()
    for p, c, y in zip(preds, tidx, ty):
        ps[c] += p
        pc[c] += 1
        ct[c] = y
    pr = ps / np.maximum(pc, 1)
    m = pc > 0
    yt, yp = ct[m], pr[m]
    return {
        "test_r2": r2_score(yt, yp),
        "test_pearson": pearson_r(yt, yp),
        "test_mae": float(np.mean(np.abs(yt - yp))),
        "n_test_clips": int(m.sum()),
    }


# ---------------------------- Training ----------------------------


def train_linear(
    train_x, train_y, val_x, val_y, test_x, test_y, test_clip_idx,
    label_mean, label_std,
    lr=1e-3, wd=1e-4,
    batch_size=64, max_epochs=30, patience=5, min_epochs=10, min_delta=0.005,
    device="cuda", seed=0,
):
    torch.manual_seed(seed)
    t_prime, d = train_x.shape[1], train_x.shape[2]
    probe = LinearA(t_prime, d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    def norm_y(y): return (y - label_mean) / label_std

    train_ds = TensorDataset(train_x, norm_y(train_y))
    val_ds = TensorDataset(val_x, norm_y(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_r2 = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        probe.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            loss = nn.functional.mse_loss(probe(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        probe.eval()
        vp, vy = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                vp.append(probe(xb).float().cpu().numpy())
                vy.append(yb.numpy())
        vp = np.concatenate(vp); vy = np.concatenate(vy)
        vr2 = r2_score(vy, vp)
        if vr2 > best_val_r2 + min_delta:
            best_val_r2 = vr2
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)
    tm = eval_on_test(probe, test_x, test_y, test_clip_idx, label_mean, label_std, device)
    tm.update({"val_r2": float(best_val_r2), "best_epoch": int(best_epoch), "n_epochs_run": int(epoch)})
    return tm


def train_pta(
    train_x, train_y, val_x, val_y, test_x, test_y, test_clip_idx,
    label_mean, label_std,
    lr=1e-4, wd=1e-4,
    n_heads=8, mlp_ratio=4, dropout=0.1,
    batch_size=64, max_epochs=50, patience=7, min_epochs=15, min_delta=0.002,
    device="cuda", seed=0,
):
    torch.manual_seed(seed)
    d = train_x.shape[2]
    probe = CrossAttnPool(d, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    def norm_y(y): return (y - label_mean) / label_std

    train_ds = TensorDataset(train_x, norm_y(train_y))
    val_ds = TensorDataset(val_x, norm_y(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_r2 = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        probe.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            loss = nn.functional.mse_loss(probe(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        probe.eval()
        vp, vy = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                vp.append(probe(xb).float().cpu().numpy())
                vy.append(yb.numpy())
        vp = np.concatenate(vp); vy = np.concatenate(vy)
        vr2 = r2_score(vy, vp)
        if vr2 > best_val_r2 + min_delta:
            best_val_r2 = vr2
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)
    tm = eval_on_test(probe, test_x, test_y, test_clip_idx, label_mean, label_std, device)
    tm.update({"val_r2": float(best_val_r2), "best_epoch": int(best_epoch), "n_epochs_run": int(epoch)})
    return tm


# ---------------------------- Main ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[controls] device={device}", flush=True)
    print(f"[controls] train={args.train_cache}", flush=True)
    tr_feats, tr_labels = load_cache(args.train_cache)
    print(f"[controls] test={args.test_cache}", flush=True)
    te_feats, te_labels = load_cache(args.test_cache)
    print(f"[controls] train feats {tuple(tr_feats.shape)}  test feats {tuple(te_feats.shape)}", flush=True)

    label_mean = float(tr_labels.mean())
    label_std = float(tr_labels.std().clamp(min=1e-6))

    T = tr_feats.shape[2]

    # Pre-segment the raw TD tensors
    tr_TD_all, tr_y_all, _ = to_segments(tr_feats, tr_labels)
    te_TD, te_y, te_clip_idx = to_segments(te_feats, te_labels)
    N_tr, S = tr_feats.shape[0], tr_feats.shape[1]

    results = []
    per_token = []   # list of dicts for per-token table

    LINEAR_LR = 1e-3
    LINEAR_WDS = [1e-4, 1e-2]
    PTA_LRS = [1e-4, 3e-4, 1e-3]
    PTA_WDS = [1e-4, 1e-2]

    # Linear variants (applied per-input-transform)
    LIN_VARIANTS = ["raw", "diff", "mean"] + [f"single_t{t}" for t in range(T)]

    for seed in args.seeds:
        train_idx, val_idx = stratified_split(tr_labels.numpy(), frac_val=0.1, seed=seed)
        seg_train_mask = np.isin(np.repeat(np.arange(N_tr), S), train_idx)
        seg_val_mask = np.isin(np.repeat(np.arange(N_tr), S), val_idx)

        # ------ Linear probes on each variant ------
        for variant in LIN_VARIANTS:
            tr_x_v = apply_variant(tr_TD_all, variant)
            te_x_v = apply_variant(te_TD, variant)
            tr_x = tr_x_v[seg_train_mask]
            tr_y = tr_y_all[seg_train_mask]
            vl_x = tr_x_v[seg_val_mask]
            vl_y = tr_y_all[seg_val_mask]

            for wd in LINEAR_WDS:
                print(f"[controls] seed={seed} linear {variant} wd={wd}", flush=True)
                out = train_linear(
                    tr_x, tr_y, vl_x, vl_y, te_x_v, te_y, te_clip_idx,
                    label_mean, label_std,
                    lr=LINEAR_LR, wd=wd,
                    batch_size=args.batch_size, device=device, seed=seed,
                )
                out.update({
                    "model": args.model, "arch": "linear-A", "variant": variant,
                    "seed": seed, "lr": LINEAR_LR, "wd": wd,
                })
                results.append(out)
                if variant.startswith("single_t"):
                    per_token.append({
                        "model": args.model, "seed": seed, "wd": wd,
                        "token_index": int(variant.replace("single_t", "")),
                        "test_r2": out["test_r2"], "test_pearson": out["test_pearson"],
                        "test_mae": out["test_mae"], "val_r2": out["val_r2"],
                    })

        # ------ Pooled temporal-attn on duplicated/mean-repeat ------
        # (we already have raw pooled-temporal-attn from job 424 for the comparison table;
        #  these are the NEW controls)
        for variant in ["mean_repeat"]:
            tr_x_v = apply_variant(tr_TD_all, variant)
            te_x_v = apply_variant(te_TD, variant)
            tr_x = tr_x_v[seg_train_mask]
            tr_y = tr_y_all[seg_train_mask]
            vl_x = tr_x_v[seg_val_mask]
            vl_y = tr_y_all[seg_val_mask]
            for lr in PTA_LRS:
                for wd in PTA_WDS:
                    print(f"[controls] seed={seed} pta {variant} lr={lr} wd={wd}", flush=True)
                    out = train_pta(
                        tr_x, tr_y, vl_x, vl_y, te_x_v, te_y, te_clip_idx,
                        label_mean, label_std,
                        lr=lr, wd=wd,
                        batch_size=args.batch_size, device=device, seed=seed,
                    )
                    out.update({
                        "model": args.model, "arch": "pooled-temporal-attn", "variant": variant,
                        "seed": seed, "lr": lr, "wd": wd,
                    })
                    results.append(out)

    # ---- best_repeat for pooled temporal-attn: use val-selected best token per seed ----
    # For each seed, scan per-token val R² from single-token linear results, pick best token by val.
    best_token_by_seed = {}
    for seed in args.seeds:
        candidates = [r for r in results
                      if r["arch"] == "linear-A" and r["variant"].startswith("single_t")
                      and r["seed"] == seed]
        # pick WD by val, then best token
        best = max(candidates, key=lambda r: r["val_r2"])
        best_token_by_seed[seed] = int(best["variant"].replace("single_t", ""))
    print(f"[controls] best token by seed (val-selected): {best_token_by_seed}", flush=True)

    # Run pta best_repeat using that token index
    for seed in args.seeds:
        tok = best_token_by_seed[seed]
        tr_x_v = apply_variant(tr_TD_all, "best_repeat", token_idx=tok)
        te_x_v = apply_variant(te_TD, "best_repeat", token_idx=tok)
        train_idx, val_idx = stratified_split(tr_labels.numpy(), frac_val=0.1, seed=seed)
        seg_train_mask = np.isin(np.repeat(np.arange(N_tr), S), train_idx)
        seg_val_mask = np.isin(np.repeat(np.arange(N_tr), S), val_idx)
        tr_x = tr_x_v[seg_train_mask]; tr_y = tr_y_all[seg_train_mask]
        vl_x = tr_x_v[seg_val_mask]; vl_y = tr_y_all[seg_val_mask]
        for lr in PTA_LRS:
            for wd in PTA_WDS:
                print(f"[controls] seed={seed} pta best_repeat tok={tok} lr={lr} wd={wd}", flush=True)
                out = train_pta(
                    tr_x, tr_y, vl_x, vl_y, te_x_v, te_y, te_clip_idx,
                    label_mean, label_std,
                    lr=lr, wd=wd,
                    batch_size=args.batch_size, device=device, seed=seed,
                )
                out.update({
                    "model": args.model, "arch": "pooled-temporal-attn", "variant": "best_repeat",
                    "token_index": tok,
                    "seed": seed, "lr": lr, "wd": wd,
                })
                results.append(out)

    # ---- Summaries: best HP per seed per (arch, variant), then 5-seed mean±std ----
    def best_per_seed_summary(arch, variant):
        rows = [r for r in results if r["arch"] == arch and r["variant"] == variant]
        if not rows:
            return None
        by_seed = {}
        for r in rows:
            by_seed.setdefault(r["seed"], []).append(r)
        bests = [max(vs, key=lambda x: x["val_r2"]) for vs in by_seed.values()]
        r2 = np.array([b["test_r2"] for b in bests])
        mae = np.array([b["test_mae"] for b in bests])
        pr = np.array([b["test_pearson"] for b in bests])
        return {
            "arch": arch, "variant": variant,
            "n_seeds": len(bests),
            "test_r2_mean": float(r2.mean()), "test_r2_std": float(r2.std()),
            "test_mae_mean": float(mae.mean()), "test_mae_std": float(mae.std()),
            "test_pearson_mean": float(pr.mean()), "test_pearson_std": float(pr.std()),
            "best_per_seed": bests,
        }

    # For single_t_mean: mean R² across tokens (flattening over all single_t* variants)
    # For single_t_best_val: per seed, pick the token with highest val R² (already done for best_token_by_seed)
    single_t_per_token_summary = []
    for t in range(T):
        s = best_per_seed_summary("linear-A", f"single_t{t}")
        if s is not None:
            s["token_index"] = t
            single_t_per_token_summary.append(s)

    # mean-across-tokens of per-token mean R² (aggregate for interpretation, not for HP selection)
    per_token_r2_means = np.array([s["test_r2_mean"] for s in single_t_per_token_summary])
    single_t_avg = {
        "arch": "linear-A", "variant": "single_t_avg",
        "test_r2_mean_of_tokens": float(per_token_r2_means.mean()),
        "test_r2_std_of_tokens": float(per_token_r2_means.std()),
        "n_tokens": T,
    }

    # single_t_best_val: for each seed, best token by val, then test R² at that token
    # We reuse best_token_by_seed but need the corresponding TEST rows (best WD per seed).
    single_t_best_val_test_r2 = []
    single_t_best_val_mae = []
    single_t_best_val_pearson = []
    for seed, tok in best_token_by_seed.items():
        rows = [r for r in results if r["arch"] == "linear-A"
                and r["variant"] == f"single_t{tok}" and r["seed"] == seed]
        best = max(rows, key=lambda r: r["val_r2"])
        single_t_best_val_test_r2.append(best["test_r2"])
        single_t_best_val_mae.append(best["test_mae"])
        single_t_best_val_pearson.append(best["test_pearson"])
    single_t_best_val_summary = {
        "arch": "linear-A", "variant": "single_t_best_val",
        "n_seeds": len(single_t_best_val_test_r2),
        "test_r2_mean": float(np.mean(single_t_best_val_test_r2)),
        "test_r2_std": float(np.std(single_t_best_val_test_r2)),
        "test_mae_mean": float(np.mean(single_t_best_val_mae)),
        "test_pearson_mean": float(np.mean(single_t_best_val_pearson)),
        "best_token_by_seed": best_token_by_seed,
    }

    summaries = {
        "linear_raw": best_per_seed_summary("linear-A", "raw"),
        "linear_diff": best_per_seed_summary("linear-A", "diff"),
        "linear_mean": best_per_seed_summary("linear-A", "mean"),
        "linear_single_t_per_token": single_t_per_token_summary,
        "linear_single_t_avg": single_t_avg,
        "linear_single_t_best_val": single_t_best_val_summary,
        "pta_mean_repeat": best_per_seed_summary("pooled-temporal-attn", "mean_repeat"),
        "pta_best_repeat": best_per_seed_summary("pooled-temporal-attn", "best_repeat"),
    }

    Path(os.path.dirname(args.out_json)).mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "args": vars(args),
            "label_mean": label_mean, "label_std": label_std,
            "summaries": summaries,
            "results": results,
        }, f, indent=2)

    # Compact print
    print(f"\n===== {args.model} SUMMARY =====")
    for k, s in summaries.items():
        if s is None: continue
        if k == "linear_single_t_per_token":
            print(f"\n  per-token R² (5-seed mean ± std per token, best WD per seed):")
            for row in s:
                print(f"    t={row['token_index']}: {row['test_r2_mean']:.3f} ± {row['test_r2_std']:.3f}")
            continue
        if k == "linear_single_t_avg":
            print(f"\n  single_t_avg (mean over tokens): {s['test_r2_mean_of_tokens']:.3f}")
            continue
        if k == "linear_single_t_best_val":
            print(f"  single_t_best_val (per-seed val-selected token): "
                  f"{s['test_r2_mean']:.3f} ± {s['test_r2_std']:.3f}")
            print(f"    best tokens by seed: {s['best_token_by_seed']}")
            continue
        print(f"  {k}: R²={s['test_r2_mean']:.3f} ± {s['test_r2_std']:.3f}   "
              f"MAE={s['test_mae_mean']:.3f}   Pearson={s['test_pearson_mean']:.3f}")
    print(f"\n[controls] wrote {args.out_json}")


if __name__ == "__main__":
    main()
