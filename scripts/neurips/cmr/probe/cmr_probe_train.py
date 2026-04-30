"""Unified CMR probe-family trainer for ACDC LVEF (regression) + 5-class DX.

Consumes a pre-pool feature cache with shape [N, S, T, D] (S=1 for the CMR
single-clip protocol, T=8 tubelets for 16 frames). Trains one of the probe
variants listed below and writes a per-run JSON with all configuration +
clip-level + per-patient-averaged test metrics.

Variants (match the diagnostic protocol discussed for echo):
    raw            linear over [T, D] flattened
    diff           linear over [T-1, D] adjacent differences
    mean           linear over time-mean [D]
    single_t{k}    linear over one time-index [D] (k = 0..T-1)
    tattn_raw      temporal-attention bridge probe over [T, D]
    tattn_mean_rep tattn over [T, D] but every t is the time-mean (control)
    tattn_best_rep tattn over [T, D] but every t is a chosen-by-val token

For tattn_best_rep we pick the token index from the companion single_t
runs' validation metric (so pass --best_token_idx from the orchestrator).

Output: single JSON per (model, epoch, task, variant) aggregating 5 seeds.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
# Probe architectures
# -----------------------------------------------------------------------------


class LinearHead(nn.Module):
    """Flatten [T', D] → Linear(T'*D, out_dim)."""

    def __init__(self, t_prime: int, d: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(t_prime * d, out_dim)

    def forward(self, x):  # [B, T', D] → [B, out_dim]
        return self.fc(x.flatten(1))


class TAttnBridge(nn.Module):
    """Pooled temporal-attention bridge probe.

    LN -> learned-query cross-attention over T -> residual/LN -> MLP -> head.
    Architecture mirrors pooled_temporal_attn_probe_train from echo, with
    head count auto-picked to divide D.
    """

    def __init__(self, d: int, out_dim: int, num_heads_pref: int = 8,
                 mlp_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        # Pick the largest head count ≤ preferred that divides D (min 1).
        heads = num_heads_pref
        while heads > 1 and d % heads != 0:
            heads -= 1
        self.ln_pre = nn.LayerNorm(d)
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=heads,
                                          batch_first=True, dropout=dropout)
        self.ln_post = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * mlp_mult, d),
        )
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):  # [B, T, D] → [B, out_dim]
        B = x.shape[0]
        x = self.ln_pre(x)
        q = self.query.expand(B, -1, -1)
        attn_out, _ = self.attn(q, x, x, need_weights=False)
        h = self.ln_post(attn_out.squeeze(1))
        h = h + self.mlp(h)
        return self.head(h)


# -----------------------------------------------------------------------------
# Input transforms per variant
# -----------------------------------------------------------------------------


def transform_input(
    feats: torch.Tensor,  # [N, S, T, D]
    variant: str,
    best_token_idx: Optional[int] = None,
) -> Tuple[torch.Tensor, bool]:
    """Return `[N*S, T', D]` and is_sequence flag."""
    N, S, T, D = feats.shape
    x = feats.reshape(N * S, T, D)

    if variant == "raw":
        return x, True
    if variant == "diff":
        return x[:, 1:, :] - x[:, :-1, :], True
    if variant == "mean":
        return x.mean(dim=1, keepdim=True), True  # [N*S, 1, D]
    m = re.fullmatch(r"single_t(\d+)", variant)
    if m:
        k = int(m.group(1))
        assert 0 <= k < T, f"single_t{k} out of range (T={T})"
        return x[:, k : k + 1, :], True  # [N*S, 1, D]
    if variant == "tattn_raw":
        return x, True
    if variant == "tattn_mean_rep":
        mean = x.mean(dim=1, keepdim=True)  # [B,1,D]
        return mean.expand(-1, T, -1).contiguous(), True
    if variant == "tattn_best_rep":
        assert best_token_idx is not None, "best_token_idx must be provided for tattn_best_rep"
        assert 0 <= best_token_idx < T
        tok = x[:, best_token_idx : best_token_idx + 1, :]
        return tok.expand(-1, T, -1).contiguous(), True
    raise ValueError(f"unknown variant {variant}")


# -----------------------------------------------------------------------------
# Patient ID + split helpers
# -----------------------------------------------------------------------------


_PATIENT_RE = re.compile(r"acdc_patient(\d+)(?:_|\.)")


def patient_id_from_path(path: str) -> str:
    m = _PATIENT_RE.search(path)
    if m:
        return f"acdc_patient{m.group(1)}"
    return path  # fallback — unlikely on ACDC


def patient_split(paths: List[str], frac_val: float, seed: int,
                  labels: Optional[np.ndarray] = None,
                  stratify_bins: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) as arrays into the clip list.

    Splits by **patient** so no patient leaks between train/val. If
    stratify_bins > 0 and labels provided, stratify the patient-level split
    by label bin (for LVEF) or by label class (if labels are int).
    """
    rng = np.random.default_rng(seed)
    pids = np.array([patient_id_from_path(p) for p in paths])
    unique_pids = np.unique(pids)
    # Patient-level label (mean for regression, mode for classification)
    if labels is not None and stratify_bins > 0:
        pid_to_label = {}
        for pid in unique_pids:
            m = (pids == pid)
            lab = labels[m]
            if lab.dtype.kind in "if":
                pid_to_label[pid] = float(lab.mean())
            else:
                pid_to_label[pid] = int(np.bincount(lab.astype(int)).argmax())
        pid_labels = np.array([pid_to_label[p] for p in unique_pids])
        if pid_labels.dtype.kind == "f":
            q = np.quantile(pid_labels, np.linspace(0, 1, stratify_bins + 1)[1:-1])
            bins = np.digitize(pid_labels, q)
        else:
            bins = pid_labels.astype(int)
        val_pids = []
        for b in np.unique(bins):
            idx = np.where(bins == b)[0]
            rng.shuffle(idx)
            k = max(1, int(round(len(idx) * frac_val)))
            val_pids.extend(unique_pids[idx[:k]].tolist())
        val_pids = set(val_pids)
    else:
        perm = unique_pids.copy()
        rng.shuffle(perm)
        k = max(1, int(round(len(perm) * frac_val)))
        val_pids = set(perm[:k].tolist())

    val_mask = np.array([p in val_pids for p in pids])
    return np.where(~val_mask)[0], np.where(val_mask)[0]


# -----------------------------------------------------------------------------
# Cache loading
# -----------------------------------------------------------------------------


def load_cache(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    feats = d["features"].float()  # [N, S, T, D]
    labels = d["labels"]            # [N] raw
    paths = list(d["paths"])
    meta = d.get("meta", {})
    if feats.dim() == 3:  # [N, T, D] → add S axis
        feats = feats.unsqueeze(1)
    return feats, labels, paths, meta


# -----------------------------------------------------------------------------
# Regression / classification heads
# -----------------------------------------------------------------------------


def build_probe(variant: str, t_prime: int, d: int, out_dim: int) -> nn.Module:
    if variant.startswith("tattn"):
        return TAttnBridge(d=d, out_dim=out_dim)
    return LinearHead(t_prime=t_prime, d=d, out_dim=out_dim)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


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
    if y_true.std() == 0 or y_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def macro_auroc(y_true, y_probs):
    from sklearn.metrics import roc_auc_score
    # One-vs-rest macro
    try:
        return float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# -----------------------------------------------------------------------------
# Per-patient aggregation
# -----------------------------------------------------------------------------


def aggregate_per_patient_reg(paths: List[str], preds: np.ndarray, y: np.ndarray):
    pid = [patient_id_from_path(p) for p in paths]
    order = {}
    for p in pid:
        order.setdefault(p, len(order))
    npat = len(order)
    sum_pred = np.zeros(npat); cnt = np.zeros(npat); truth = np.full(npat, np.nan)
    for i, p in enumerate(pid):
        o = order[p]
        sum_pred[o] += preds[i]
        cnt[o] += 1
        truth[o] = y[i]
    return truth, sum_pred / np.maximum(cnt, 1)


def aggregate_per_patient_cls(paths: List[str], probs: np.ndarray, y: np.ndarray):
    """Average softmax probs within patient; take argmax for per-patient class."""
    pid = [patient_id_from_path(p) for p in paths]
    order = {}
    for p in pid:
        order.setdefault(p, len(order))
    C = probs.shape[1]
    npat = len(order)
    sum_prob = np.zeros((npat, C)); cnt = np.zeros(npat); truth = np.full(npat, -1, dtype=int)
    for i, p in enumerate(pid):
        o = order[p]
        sum_prob[o] += probs[i]
        cnt[o] += 1
        truth[o] = int(y[i])
    prob_avg = sum_prob / np.maximum(cnt, 1)[:, None]
    return truth, prob_avg, prob_avg.argmax(axis=1)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def train_probe_once(
    task: str,
    variant: str,
    train_x: torch.Tensor, train_y: torch.Tensor,
    val_x: torch.Tensor, val_y: torch.Tensor,
    test_x: torch.Tensor, test_y: torch.Tensor,
    num_classes: int,
    label_mean: float, label_std: float,
    lr: float, wd: float,
    batch_size: int, max_epochs: int,
    patience: int, min_epochs: int, min_delta: float,
    device: str, seed: int,
):
    torch.manual_seed(seed)
    t_prime, d = train_x.shape[1], train_x.shape[2]
    out_dim = 1 if task == "lvef" else num_classes
    probe = build_probe(variant, t_prime, d, out_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    def norm_y(y):
        return (y - label_mean) / label_std

    if task == "lvef":
        train_ds = TensorDataset(train_x, norm_y(train_y.float()))
        val_ds = TensorDataset(val_x, norm_y(val_y.float()))
        loss_fn = nn.MSELoss()
    else:  # dx
        train_ds = TensorDataset(train_x, train_y.long())
        val_ds = TensorDataset(val_x, val_y.long())
        loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_val_metric = -np.inf
    best_state = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        probe.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = probe(xb)
            if task == "lvef":
                pred = pred.squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        probe.eval()
        val_preds, val_ys = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                p = probe(xb)
                if task == "lvef":
                    p = p.squeeze(-1)
                val_preds.append(p.float().cpu().numpy())
                val_ys.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_ys = np.concatenate(val_ys)

        if task == "lvef":
            vmetric = r2_score(val_ys, val_preds)
        else:
            probs = torch.softmax(torch.from_numpy(val_preds), dim=-1).numpy()
            # If only 1 class in val (tiny), fall back to -cross-entropy
            uniq = np.unique(val_ys)
            if len(uniq) >= 2:
                vmetric = macro_auroc(val_ys, probs)
            else:
                ce = -np.mean(np.log(probs[np.arange(len(val_ys)), val_ys.astype(int)] + 1e-12))
                vmetric = -ce  # maximize

        if vmetric > best_val_metric + min_delta:
            best_val_metric = vmetric
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    if best_state is not None:
        probe.load_state_dict(best_state)

    # Test eval
    probe.eval()
    test_loader = DataLoader(TensorDataset(test_x), batch_size=256, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device, non_blocking=True)
            p = probe(xb)
            if task == "lvef":
                p = p.squeeze(-1)
            preds.append(p.float().cpu().numpy())
    preds = np.concatenate(preds)
    return preds, float(best_val_metric)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--model", required=True, help="tag for output, e.g. jepa_e30")
    ap.add_argument("--task", required=True, choices=["lvef", "dx"])
    ap.add_argument("--variant", required=True,
                    help="raw | diff | mean | single_t{k} | tattn_raw | tattn_mean_rep | tattn_best_rep")
    ap.add_argument("--num-classes", type=int, default=5, help="only used for dx")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lr-sweep", type=float, nargs="+", default=[1e-3])
    ap.add_argument("--wd-sweep", type=float, nargs="+", default=[1e-4, 1e-2])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--frac-val", type=float, default=0.1)
    ap.add_argument("--best-token-idx", type=int, default=None,
                    help="Required for tattn_best_rep")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[probe] loading train: {args.train_cache}", flush=True)
    tr_feats, tr_labels, tr_paths, tr_meta = load_cache(args.train_cache)
    print(f"[probe] loading test:  {args.test_cache}", flush=True)
    te_feats, te_labels, te_paths, te_meta = load_cache(args.test_cache)
    print(f"[probe] train {tuple(tr_feats.shape)}  test {tuple(te_feats.shape)}", flush=True)

    N, S, T, D = tr_feats.shape
    assert te_feats.shape[-1] == D, "train/test D mismatch"

    # Build per-variant inputs (clip-level → flattened with S-axis folded)
    tr_x, _ = transform_input(tr_feats, args.variant, best_token_idx=args.best_token_idx)
    te_x, _ = transform_input(te_feats, args.variant, best_token_idx=args.best_token_idx)

    # y per (clip, segment) — same as clip label since S=1 typical
    tr_y = tr_labels.unsqueeze(1).expand(N, S).reshape(-1)
    te_y = te_labels.unsqueeze(1).expand(te_feats.shape[0], S).reshape(-1)

    # Paths per (clip, segment) for patient-aggregation
    tr_paths_seg = [p for p in tr_paths for _ in range(S)]
    te_paths_seg = [p for p in te_paths for _ in range(S)]

    # Label normalization (regression only)
    if args.task == "lvef":
        lm = float(tr_labels.float().mean())
        ls = float(tr_labels.float().std().clamp(min=1e-6))
    else:
        lm, ls = 0.0, 1.0

    print(f"[probe] variant={args.variant}  task={args.task}  D={D} T'={tr_x.shape[1]} "
          f"lr_sweep={args.lr_sweep} wd_sweep={args.wd_sweep}", flush=True)

    # Per-seed loop
    rows = []
    for seed in args.seeds:
        train_idx_clip, val_idx_clip = patient_split(
            tr_paths, frac_val=args.frac_val, seed=seed,
            labels=tr_labels.numpy() if args.task == "dx" else tr_labels.float().numpy(),
            stratify_bins=5 if args.task == "lvef" else args.num_classes,
        )
        # Lift clip-indices to (clip, segment) indices
        seg_train_mask = np.isin(np.repeat(np.arange(N), S), train_idx_clip)
        seg_val_mask = np.isin(np.repeat(np.arange(N), S), val_idx_clip)
        tr_x_s = tr_x[seg_train_mask]
        tr_y_s = tr_y[seg_train_mask]
        vl_x_s = tr_x[seg_val_mask]
        vl_y_s = tr_y[seg_val_mask]

        best = None
        for lr in args.lr_sweep:
            for wd in args.wd_sweep:
                preds, val_metric = train_probe_once(
                    task=args.task, variant=args.variant,
                    train_x=tr_x_s, train_y=tr_y_s,
                    val_x=vl_x_s, val_y=vl_y_s,
                    test_x=te_x, test_y=te_y,
                    num_classes=args.num_classes,
                    label_mean=lm, label_std=ls,
                    lr=lr, wd=wd,
                    batch_size=args.batch_size, max_epochs=args.max_epochs,
                    patience=args.patience, min_epochs=args.min_epochs,
                    min_delta=args.min_delta,
                    device=device, seed=seed,
                )
                if (best is None) or (val_metric > best["val_metric"]):
                    best = {"val_metric": val_metric, "lr": lr, "wd": wd, "preds": preds}

        # Denormalize / aggregate
        if args.task == "lvef":
            preds_clip = best["preds"] * ls + lm
            truth_pat, pred_pat = aggregate_per_patient_reg(te_paths_seg, preds_clip, te_y.numpy())
            r2 = r2_score(truth_pat, pred_pat)
            prs = pearson_r(truth_pat, pred_pat)
            mae = float(np.mean(np.abs(truth_pat - pred_pat)))
            clip_r2 = r2_score(te_y.numpy(), preds_clip)
            clip_prs = pearson_r(te_y.numpy(), preds_clip)
            clip_mae = float(np.mean(np.abs(te_y.numpy() - preds_clip)))
            rows.append({
                "seed": seed, "variant": args.variant, "task": args.task,
                "lr": best["lr"], "wd": best["wd"], "val_metric": best["val_metric"],
                "test_r2_clip": clip_r2, "test_pearson_clip": clip_prs, "test_mae_clip": clip_mae,
                "test_r2_patient": float(r2), "test_pearson_patient": float(prs),
                "test_mae_patient": mae,
                "n_test_clips": int(len(te_y)), "n_test_patients": int(len(truth_pat)),
            })
        else:
            logits = best["preds"]  # [N_test*S, num_classes]
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            pred_class_clip = probs.argmax(axis=1)
            y_np = te_y.numpy().astype(int)
            clip_acc = float((pred_class_clip == y_np).mean())
            clip_auroc = macro_auroc(y_np, probs)
            clip_f1 = macro_f1(y_np, pred_class_clip)
            truth_pat, prob_avg_pat, pred_class_pat = aggregate_per_patient_cls(te_paths_seg, probs, y_np)
            pat_acc = float((pred_class_pat == truth_pat).mean())
            pat_auroc = macro_auroc(truth_pat, prob_avg_pat)
            pat_f1 = macro_f1(truth_pat, pred_class_pat)
            rows.append({
                "seed": seed, "variant": args.variant, "task": args.task,
                "lr": best["lr"], "wd": best["wd"], "val_metric": best["val_metric"],
                "test_acc_clip": clip_acc, "test_auroc_clip": clip_auroc, "test_f1_clip": clip_f1,
                "test_acc_patient": pat_acc, "test_auroc_patient": pat_auroc, "test_f1_patient": pat_f1,
                "n_test_clips": int(len(te_y)), "n_test_patients": int(len(truth_pat)),
            })

        print(f"[probe] seed={seed} best lr={best['lr']} wd={best['wd']} val={best['val_metric']:.4f}  "
              + (f"test_r2_pat={rows[-1]['test_r2_patient']:.4f}" if args.task == "lvef"
                 else f"test_auroc_pat={rows[-1]['test_auroc_patient']:.4f}"), flush=True)

    # Aggregate across seeds
    summary = {}
    if rows:
        keys = [k for k in rows[0] if k.startswith("test_")]
        for k in keys:
            vals = np.array([r[k] for r in rows if not np.isnan(r.get(k, np.nan))])
            if len(vals) > 0:
                summary[k + "_mean"] = float(vals.mean())
                summary[k + "_std"] = float(vals.std(ddof=0))

    out = {
        "model": args.model,
        "task": args.task,
        "variant": args.variant,
        "best_token_idx": args.best_token_idx,
        "train_cache": args.train_cache,
        "test_cache": args.test_cache,
        "D": int(D),
        "T": int(T),
        "S": int(S),
        "label_mean": lm,
        "label_std": ls,
        "tr_meta": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in tr_meta.items()},
        "seeds": args.seeds,
        "lr_sweep": args.lr_sweep,
        "wd_sweep": args.wd_sweep,
        "rows": rows,
        "summary": summary,
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
