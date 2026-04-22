"""Phase 3: train linear-A / linear-B / MLP probes on cached pre-pool features.

Loads `[N, S=2, T=8, D]` cached tensors (from feature_extraction_pre_pool) for
train/val/test splits of EchoNet-Dynamic and trains LVEF regression probes on
either adjacent-temporal differences (`z_{t+1} - z_t`, T'=7) or raw features.

Each segment is a separate training example (same label); per-clip inference
averages the two segment predictions. Matches the job 216/220 convention.

Protocol: claude/neurips/experiments/diff-probe-analysis.md

Outputs a single results JSON per (model, ckpt, probe_variant, seed) for
downstream aggregation.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------- Probe architectures ----------------------------


class LinearA(nn.Module):
    """Flatten [T', D] -> [T'*D] -> Linear(T'*D, 1)."""

    def __init__(self, t_prime, d):
        super().__init__()
        self.fc = nn.Linear(t_prime * d, 1)

    def forward(self, x):  # x: [B, T', D]
        return self.fc(x.flatten(1)).squeeze(-1)


class LinearB(nn.Module):
    """Mean over T' -> Linear(D, 1)."""

    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, x):  # x: [B, T', D]
        return self.fc(x.mean(dim=1)).squeeze(-1)


class MLP(nn.Module):
    """Flatten [T', D] -> [T'*D] -> Linear(-, 256) -> GELU -> Drop -> Linear(256, 1)."""

    def __init__(self, t_prime, d, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_prime * d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: [B, T', D]
        return self.net(x.flatten(1)).squeeze(-1)


def build_probe(arch, t_prime, d):
    if arch == "linear-A":
        return LinearA(t_prime, d)
    if arch == "linear-B":
        return LinearB(d)
    if arch == "mlp":
        return MLP(t_prime, d)
    raise ValueError(f"unknown arch {arch}")


# ---------------------------- Data prep helpers ------------------------------


def load_cache(path):
    """Load {features [N,S,T,D], labels [N], paths [N]}. Returns fp32 features."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    feats = d["features"].float()  # [N, S, T, D]
    labels = d["labels"].float()   # [N]
    paths = list(d.get("paths", [f"clip_{i}" for i in range(feats.shape[0])]))
    return feats, labels, paths


def to_segments(feats, labels, use_diff, shuffle_T=False, rng=None):
    """Return ([N*S, T', D], [N*S], clip_idx[N*S]) — one row per (clip, segment).

    shuffle_T=True permutes the T axis independently per (clip, segment) with
    a fresh permutation sampled from rng, then (if use_diff) diffs. Used for
    the temporal-shuffle control.
    """
    N, S, T, D = feats.shape
    x = feats
    if shuffle_T:
        assert rng is not None
        perms = torch.from_numpy(
            np.stack([rng.permutation(T) for _ in range(N * S)]).reshape(N, S, T)
        ).long()
        # gather along T axis
        idx = perms.unsqueeze(-1).expand(N, S, T, D)
        x = torch.gather(x, 2, idx)
    if use_diff:
        x = x[:, :, 1:, :] - x[:, :, :-1, :]  # [N, S, T-1, D]
    # flatten segment dim into batch
    Tp = x.shape[2]
    x = x.reshape(N * S, Tp, D)
    y = labels.unsqueeze(1).expand(N, S).reshape(-1)
    clip_idx = torch.arange(N).unsqueeze(1).expand(N, S).reshape(-1)
    return x, y, clip_idx


def stratified_split(labels_np, frac_val=0.1, seed=0, n_bins=5):
    """Stratified train/val split by EF quintile. Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    n = len(labels_np)
    # bin by quantile
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


def bootstrap_r2(y_true, y_pred, n_resamples=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_resamples, dtype=np.float64)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        vals[i] = r2_score(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


# -------------------------------- Trainer -----------------------------------


def _eval_probe_on_test(probe, test_x, test_y, test_clip_idx, label_mean, label_std, device):
    """Run trained `probe` over `test_x`, denormalize, and aggregate per clip."""
    probe.eval()
    test_loader = DataLoader(TensorDataset(test_x), batch_size=256, shuffle=False)
    preds_segment = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=True)
            preds_segment.append(probe(xb).float().cpu().numpy())
    preds_segment = np.concatenate(preds_segment)
    preds_segment = preds_segment * label_std + label_mean

    test_clip_idx_np = test_clip_idx.numpy()
    n_clips = int(test_clip_idx_np.max()) + 1
    clip_pred_sum = np.zeros(n_clips, dtype=np.float64)
    clip_pred_cnt = np.zeros(n_clips, dtype=np.int64)
    clip_truth = np.full(n_clips, np.nan, dtype=np.float64)
    test_y_np = test_y.numpy()
    for p, c, y in zip(preds_segment, test_clip_idx_np, test_y_np):
        clip_pred_sum[c] += p
        clip_pred_cnt[c] += 1
        clip_truth[c] = y
    clip_pred = clip_pred_sum / np.maximum(clip_pred_cnt, 1)
    mask = clip_pred_cnt > 0
    test_r2 = r2_score(clip_truth[mask], clip_pred[mask])
    return {
        "test_r2": float(test_r2),
        "n_test_clips": int(mask.sum()),
        "clip_truth": clip_truth[mask].tolist(),
        "clip_pred": clip_pred[mask].tolist(),
    }


def train_one_probe(
    arch,
    train_x, train_y,
    val_x, val_y,
    test_x, test_y, test_clip_idx,
    label_mean, label_std,
    lr=1e-3, wd=1e-4,
    batch_size=64, max_epochs=30,
    patience=5, min_epochs=10, min_delta=0.005,
    device="cuda", seed=0,
    extra_test_sets=None,
):
    torch.manual_seed(seed)
    t_prime, d = train_x.shape[1], train_x.shape[2]
    probe = build_probe(arch, t_prime, d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    # z-score labels using train-set stats (already computed outside but
    # double-check here on the normalized input)
    def norm_y(y):
        return (y - label_mean) / label_std

    def denorm(y_hat):
        return y_hat * label_std + label_mean

    train_ds = TensorDataset(train_x, norm_y(train_y))
    val_ds = TensorDataset(val_x, norm_y(val_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_r2 = -np.inf
    best_val_mse = np.inf
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
        val_preds, val_ys = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                val_preds.append(probe(xb).float().cpu().numpy())
                val_ys.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_ys = np.concatenate(val_ys)
        vr2 = r2_score(val_ys, val_preds)
        vmse = float(np.mean((val_ys - val_preds) ** 2))

        if vr2 > best_val_r2 + min_delta:
            best_val_r2 = vr2
            best_val_mse = vmse
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)

    # Primary test eval (clean test features as passed in)
    primary = _eval_probe_on_test(probe, test_x, test_y, test_clip_idx, label_mean, label_std, device)

    extra_results = {}
    if extra_test_sets:
        for name, (ex_x, ex_y, ex_idx) in extra_test_sets.items():
            extra_results[name] = _eval_probe_on_test(probe, ex_x, ex_y, ex_idx, label_mean, label_std, device)

    return {
        "test_r2": primary["test_r2"],
        "val_r2": float(best_val_r2),
        "val_mse": float(best_val_mse),
        "n_test_clips": primary["n_test_clips"],
        "clip_truth": primary["clip_truth"],
        "clip_pred": primary["clip_pred"],
        "extra": extra_results,
    }


# --------------------------------- Main ------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True, help="features.pt for EchoNet train CSV")
    ap.add_argument("--test-cache", required=True, help="features.pt for EchoNet test CSV")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--model", required=True, help="e.g. mae_e99")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--archs", nargs="+", default=["linear-A", "linear-B", "mlp"])
    ap.add_argument("--inputs", nargs="+", default=["diff", "raw"])
    ap.add_argument("--wd-sweep", type=float, nargs="+", default=[1e-4],
                    help="extra weight decays for linear-A (baseline always 1e-4)")
    ap.add_argument("--shuffle-control", action="store_true",
                    help="Temporal-shuffle permutes T axis per seed; only linear-A-diff is trained")
    ap.add_argument("--inference-mf", action="store_true",
                    help="Also evaluate on T-axis-permuted test features (representation-space matched-frame). "
                         "Adds test_r2_mf and CI columns to each result row.")
    ap.add_argument("--bootstrap-resamples", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=30)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[diff_probe] loading train cache: {args.train_cache}", flush=True)
    tr_feats, tr_labels, tr_paths = load_cache(args.train_cache)
    print(f"[diff_probe] loading test cache:  {args.test_cache}", flush=True)
    te_feats, te_labels, te_paths = load_cache(args.test_cache)
    print(f"[diff_probe] train feats {tuple(tr_feats.shape)}  test feats {tuple(te_feats.shape)}", flush=True)

    label_mean = float(tr_labels.mean())
    label_std = float(tr_labels.std().clamp(min=1e-6))
    print(f"[diff_probe] label mean/std (train) = {label_mean:.4f} / {label_std:.4f}", flush=True)

    results = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        train_idx, val_idx = stratified_split(tr_labels.numpy(), frac_val=0.1, seed=seed)

        for use_diff_name in args.inputs:
            use_diff = (use_diff_name == "diff")
            tr_x_all, tr_y_all, _ = to_segments(
                tr_feats, tr_labels, use_diff=use_diff,
                shuffle_T=args.shuffle_control, rng=rng,
            )
            te_x, te_y, te_clip_idx = to_segments(
                te_feats, te_labels, use_diff=use_diff,
                shuffle_T=args.shuffle_control, rng=rng,
            )
            # Representation-space matched-frame test set: same probe, shuffled test features.
            # Use a fresh deterministic RNG per (seed, input) so the MF permutation is
            # independent of the shuffle-control RNG above.
            mf_test_sets = None
            if args.inference_mf and not args.shuffle_control:
                mf_rng = np.random.default_rng(10_000 + seed)
                te_x_mf, te_y_mf, te_clip_idx_mf = to_segments(
                    te_feats, te_labels, use_diff=use_diff,
                    shuffle_T=True, rng=mf_rng,
                )
                mf_test_sets = {"mf": (te_x_mf, te_y_mf, te_clip_idx_mf)}
            N_tr, S = tr_feats.shape[0], tr_feats.shape[1]
            seg_train_mask = np.isin(
                np.repeat(np.arange(N_tr), S), train_idx
            )
            seg_val_mask = np.isin(
                np.repeat(np.arange(N_tr), S), val_idx
            )
            tr_x = tr_x_all[seg_train_mask]
            tr_y = tr_y_all[seg_train_mask]
            vl_x = tr_x_all[seg_val_mask]
            vl_y = tr_y_all[seg_val_mask]

            archs = args.archs
            if args.shuffle_control:
                archs = ["linear-A"]  # only linear-A-diff for control

            for arch in archs:
                wds = [1e-4]
                if arch == "linear-A" and not args.shuffle_control:
                    wds = sorted(set([1e-4] + list(args.wd_sweep)))
                for wd in wds:
                    tag = f"{args.model}|seed={seed}|arch={arch}|input={use_diff_name}|wd={wd}"
                    if args.shuffle_control:
                        tag += "|SHUFFLE"
                    print(f"[diff_probe] >>> {tag}", flush=True)
                    out = train_one_probe(
                        arch=arch,
                        train_x=tr_x, train_y=tr_y,
                        val_x=vl_x, val_y=vl_y,
                        test_x=te_x, test_y=te_y, test_clip_idx=te_clip_idx,
                        label_mean=label_mean, label_std=label_std,
                        lr=args.lr, wd=wd,
                        batch_size=args.batch_size, max_epochs=args.max_epochs,
                        device=device, seed=seed,
                        extra_test_sets=mf_test_sets,
                    )
                    lo, hi = bootstrap_r2(
                        np.asarray(out["clip_truth"]),
                        np.asarray(out["clip_pred"]),
                        n_resamples=args.bootstrap_resamples, seed=seed,
                    )
                    row = {
                        "model": args.model,
                        "seed": seed,
                        "arch": arch,
                        "input": use_diff_name,
                        "wd": wd,
                        "shuffle_control": bool(args.shuffle_control),
                        "test_r2": out["test_r2"],
                        "val_r2": out["val_r2"],
                        "val_mse": out["val_mse"],
                        "n_test_clips": out["n_test_clips"],
                        "r2_ci_lo": lo,
                        "r2_ci_hi": hi,
                    }
                    mf_result = out.get("extra", {}).get("mf")
                    if mf_result is not None:
                        mf_lo, mf_hi = bootstrap_r2(
                            np.asarray(mf_result["clip_truth"]),
                            np.asarray(mf_result["clip_pred"]),
                            n_resamples=args.bootstrap_resamples, seed=seed,
                        )
                        row["test_r2_mf"] = mf_result["test_r2"]
                        row["r2_mf_ci_lo"] = mf_lo
                        row["r2_mf_ci_hi"] = mf_hi
                        row["temporal_delta"] = mf_result["test_r2"] - out["test_r2"]
                    print(f"[diff_probe] <<< {tag}  test_r2={out['test_r2']:.4f}  "
                          f"CI=[{lo:.4f},{hi:.4f}]" +
                          (f"  mf={mf_result['test_r2']:.4f}" if mf_result is not None else ""),
                          flush=True)
                    results.append(row)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "train_cache": args.train_cache,
                "test_cache": args.test_cache,
                "label_mean": label_mean,
                "label_std": label_std,
                "seeds": args.seeds,
                "archs": args.archs,
                "inputs": args.inputs,
                "wd_sweep": args.wd_sweep,
                "shuffle_control": bool(args.shuffle_control),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[diff_probe] wrote {out_path} ({len(results)} rows)", flush=True)


if __name__ == "__main__":
    main()
