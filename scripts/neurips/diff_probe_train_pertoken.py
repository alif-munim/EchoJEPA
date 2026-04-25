"""Per-token diff-probe training for the spatial-location caveat experiment.

Loads [N, S=2, T=8, 196, D=1024] fp16 caches (FEATURE_KEEP_SPATIAL=1 extraction).
Trains two probe variants on per-spatial-location temporal differences:
  (b) linear-spatial-mean: mean over T' -> flatten (S_spatial * D) -> Linear(., 1)
  (c-A) attention-A: content-independent learned alpha[spatial] -> sum -> Linear(D, 1)
  (c-B) attention-B: content-dependent alpha = softmax(w . d_t) -> sum -> Linear(D, 1)

Controls (gate: R^2 < 0.05):
  temporal_shuffle: permute T per (clip, segment) BEFORE diff.
  spatial_shuffle:  permute spatial axis independently per T position BEFORE diff
                    (breaks spatial correspondence between consecutive frames).

For variant (c), learned attention maps are saved as 14x14 fp32 tensors per
(model, seed, formulation) for the main-text figure.

Loads caches with torch.load(..., mmap=True, weights_only=False) so the 44GB
train shard never materializes in RAM twice. Per-batch diffs computed on GPU.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------- Probe architectures ----------------------------


class LinearSpatialMean(nn.Module):
    """(b) mean over T' -> flatten [spatial*D] -> Linear(spatial*D, 1)."""

    def __init__(self, spatial, d):
        super().__init__()
        self.fc = nn.Linear(spatial * d, 1)

    def forward(self, d):  # d: [B, T', spatial, D]
        x = d.mean(dim=1)            # [B, spatial, D]
        x = x.flatten(1)             # [B, spatial*D]
        return self.fc(x).squeeze(-1)


class AttnA(nn.Module):
    """(c-A) content-independent: alpha[spatial] learned, softmax over spatial.

    Steps: d [B, T', spatial, D] -> mean T -> d_t [B, spatial, D]
           pooled = sum_sp softmax(alpha)[sp] * d_t[:, sp, :]   -> [B, D]
           Linear(D, 1)
    """

    def __init__(self, spatial, d):
        super().__init__()
        self.alpha_logits = nn.Parameter(torch.zeros(spatial))
        self.fc = nn.Linear(d, 1)

    def forward(self, d):  # d: [B, T', spatial, D]
        d_t = d.mean(dim=1)                               # [B, spatial, D]
        a = F.softmax(self.alpha_logits, dim=0)           # [spatial]
        pooled = torch.einsum("s,bsd->bd", a, d_t)        # [B, D]
        return self.fc(pooled).squeeze(-1)

    def attention_map(self):
        return F.softmax(self.alpha_logits, dim=0).detach().cpu()  # [spatial]


class AttnB(nn.Module):
    """(c-B) content-dependent: alpha[B, spatial] = softmax(w . d_t)."""

    def __init__(self, spatial, d):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.w, std=1.0 / (d ** 0.5))
        self.fc = nn.Linear(d, 1)

    def forward(self, d):  # d: [B, T', spatial, D]
        d_t = d.mean(dim=1)                              # [B, spatial, D]
        scores = torch.einsum("d,bsd->bs", self.w, d_t)  # [B, spatial]
        a = F.softmax(scores, dim=1)                     # [B, spatial]
        pooled = torch.einsum("bs,bsd->bd", a, d_t)      # [B, D]
        return self.fc(pooled).squeeze(-1)

    def attention_map_batch(self, d):
        d_t = d.mean(dim=1)
        scores = torch.einsum("d,bsd->bs", self.w, d_t)
        return F.softmax(scores, dim=1).detach().cpu()  # [B, spatial]


def build_probe(arch, spatial, d):
    if arch == "linear_spatial":
        return LinearSpatialMean(spatial, d)
    if arch == "attn_a":
        return AttnA(spatial, d)
    if arch == "attn_b":
        return AttnB(spatial, d)
    raise ValueError(arch)


# ----------------------------- Data helpers ---------------------------------


def load_cache(path):
    """Memory-mapped load. Returns (features [N,S,T,spatial,D] fp16, labels, paths)."""
    d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return d["features"], d["labels"].float(), list(d.get("paths", []))


def apply_shuffles(z, shuffle_mode, rng):
    """z [N, S, T, spatial, D] fp16 -> optionally shuffled copy (returns fp16).

    shuffle_mode in {"none", "temporal", "spatial"}:
      none     - return z unchanged
      temporal - permute T axis with a fresh perm per (clip, segment)
      spatial  - permute spatial axis with fresh perm per (clip, segment, T)
    """
    if shuffle_mode == "none":
        return z
    N, S, T, SP, D = z.shape
    if shuffle_mode == "temporal":
        perms = np.stack([rng.permutation(T) for _ in range(N * S)]).reshape(N, S, T)
        perms_t = torch.from_numpy(perms).long()
        idx = perms_t.view(N, S, T, 1, 1).expand(N, S, T, SP, D)
        return torch.gather(z, 2, idx)
    if shuffle_mode == "spatial":
        perms = np.stack([rng.permutation(SP) for _ in range(N * S * T)]).reshape(N, S, T, SP)
        perms_t = torch.from_numpy(perms).long()
        idx = perms_t.view(N, S, T, SP, 1).expand(N, S, T, SP, D)
        return torch.gather(z, 3, idx)
    raise ValueError(shuffle_mode)


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


# ------------------------------ Per-batch diff --------------------------------


def diff_on_gpu(batch_z, device):
    """batch_z [B, S, T, sp, D] fp16 -> diff [B*S, T-1, sp, D] fp32 on device.

    Flattens segment dim into batch (matches job 314 convention).
    """
    z = batch_z.to(device, non_blocking=True).float()       # [B, S, T, sp, D]
    diff = z[:, :, 1:, :, :] - z[:, :, :-1, :, :]           # [B, S, T-1, sp, D]
    B, S, Tp, SP, D = diff.shape
    return diff.reshape(B * S, Tp, SP, D)


# -------------------------------- Trainer -----------------------------------


def make_clip_indices(N, S):
    return torch.arange(N).unsqueeze(1).expand(N, S).reshape(-1)


def iter_batches(z, y, idx_pool, batch_size, rng=None, shuffle=True):
    """Yield batched (z_slice [B_clips, S, T, sp, D], y_slice [B_clips]).

    idx_pool: 1-D numpy array of clip indices into z.
    """
    idx = idx_pool.copy()
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        sel = idx[i:i + batch_size]
        yield z[torch.from_numpy(sel).long()], y[torch.from_numpy(sel).long()], sel


def train_probe(
    arch, tr_z, tr_y, tr_idx, vl_idx,
    te_z, te_y,
    label_mean, label_std,
    lr=1e-3, wd=1e-4,
    batch_size=32, max_epochs=30,
    patience=5, min_epochs=10, min_delta=0.005,
    device="cuda", seed=0,
):
    torch.manual_seed(seed)
    rng = np.random.default_rng(10000 + seed)

    # Infer shape
    spatial = tr_z.shape[3]
    d = tr_z.shape[4]
    probe = build_probe(arch, spatial, d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    def norm_y(y):
        return (y - label_mean) / label_std

    tr_y_norm = norm_y(tr_y)

    best_val_r2 = -np.inf
    best_val_mse = np.inf
    best_state = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        probe.train()
        for zb, yb_raw, _sel in iter_batches(tr_z, tr_y_norm, tr_idx, batch_size, rng=rng, shuffle=True):
            # zb: [B_clips, S, T, sp, D]; yb_raw: [B_clips]
            diff = diff_on_gpu(zb, device)                           # [B*S, T-1, sp, D]
            # Expand labels: each clip contributes S segment rows.
            B_clips, S = zb.shape[0], zb.shape[1]
            yb = yb_raw.to(device).repeat_interleave(S).float()
            pred = probe(diff)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # Val
        probe.eval()
        v_preds, v_ys = [], []
        with torch.no_grad():
            for zb, yb_raw, _sel in iter_batches(tr_z, tr_y_norm, vl_idx, 64, shuffle=False):
                diff = diff_on_gpu(zb, device)
                B_clips, S = zb.shape[0], zb.shape[1]
                yb = yb_raw.to(device).repeat_interleave(S).float()
                v_preds.append(probe(diff).float().cpu().numpy())
                v_ys.append(yb.cpu().numpy())
        v_preds = np.concatenate(v_preds)
        v_ys = np.concatenate(v_ys)
        vr2 = r2_score(v_ys, v_preds)
        vmse = float(np.mean((v_ys - v_preds) ** 2))

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

    # Test: segment-level predictions -> average per clip
    probe.eval()
    N_test = te_z.shape[0]
    S = te_z.shape[1]
    preds_seg = np.zeros(N_test * S, dtype=np.float64)
    clip_ids = make_clip_indices(N_test, S).numpy()
    y_seg = te_y.numpy().repeat(S)
    row = 0
    attn_maps_b = []
    with torch.no_grad():
        for start in range(0, N_test, 64):
            zb = te_z[start:start + 64]
            diff = diff_on_gpu(zb, device)
            out = probe(diff).float().cpu().numpy()
            preds_seg[row:row + out.shape[0]] = out * label_std + label_mean
            row += out.shape[0]
            if arch == "attn_b":
                attn_maps_b.append(probe.attention_map_batch(diff).numpy())

    n_clips = N_test
    clip_pred = np.zeros(n_clips, dtype=np.float64)
    clip_cnt = np.zeros(n_clips, dtype=np.int64)
    clip_truth = np.full(n_clips, np.nan, dtype=np.float64)
    for p, c, y in zip(preds_seg, clip_ids, y_seg):
        clip_pred[c] += p
        clip_cnt[c] += 1
        clip_truth[c] = y
    clip_pred = clip_pred / np.maximum(clip_cnt, 1)
    mask = clip_cnt > 0
    test_r2 = r2_score(clip_truth[mask], clip_pred[mask])

    # Attention maps (test)
    attn_payload = {}
    if arch == "attn_a":
        attn_payload["a_map"] = probe.attention_map().numpy()  # [196]
    elif arch == "attn_b":
        all_b = np.concatenate(attn_maps_b, axis=0)  # [N_test*S, 196]
        attn_payload["b_map_mean"] = all_b.mean(axis=0)
        attn_payload["b_map_std"] = all_b.std(axis=0)

    return {
        "test_r2": float(test_r2),
        "val_r2": float(best_val_r2),
        "val_mse": float(best_val_mse),
        "clip_truth": clip_truth[mask].tolist(),
        "clip_pred": clip_pred[mask].tolist(),
        "attn": attn_payload,
    }


# --------------------------------- Main ------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--test-cache", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--attn-out", default=None,
                    help="optional .npz path to save attention maps per (arch, seed, shuffle)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--archs", nargs="+", default=["linear_spatial", "attn_a", "attn_b"])
    ap.add_argument("--shuffles", nargs="+", default=["none", "temporal", "spatial"])
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[pertoken] load train {args.train_cache}", flush=True)
    tr_z, tr_y, _ = load_cache(args.train_cache)
    print(f"[pertoken] load test  {args.test_cache}", flush=True)
    te_z, te_y, _ = load_cache(args.test_cache)
    print(f"[pertoken] tr {tuple(tr_z.shape)} te {tuple(te_z.shape)}", flush=True)

    label_mean = float(tr_y.mean())
    label_std = float(tr_y.std().clamp(min=1e-6))
    print(f"[pertoken] label mean/std = {label_mean:.4f} / {label_std:.4f}", flush=True)

    results = []
    attn_store = {}

    for seed in args.seeds:
        train_idx, val_idx = stratified_split(tr_y.numpy(), frac_val=0.1, seed=seed)

        for shuffle_mode in args.shuffles:
            # Apply shuffles deterministically per (seed, mode) once per pass.
            # Controls: shuffle_mode != "none" fuzzes TRAIN and TEST identically so
            # that passing the gate (R^2 near 0) means the shuffle removed the signal
            # the probe uses.
            sh_rng_tr = np.random.default_rng(20000 + seed * 3 + hash(shuffle_mode) % 7)
            sh_rng_te = np.random.default_rng(40000 + seed * 3 + hash(shuffle_mode) % 7)
            tr_z_s = apply_shuffles(tr_z, shuffle_mode, sh_rng_tr) if shuffle_mode != "none" else tr_z
            te_z_s = apply_shuffles(te_z, shuffle_mode, sh_rng_te) if shuffle_mode != "none" else te_z

            for arch in args.archs:
                tag = f"{args.model}|seed={seed}|arch={arch}|shuffle={shuffle_mode}|wd={args.wd}"
                print(f"[pertoken] >>> {tag}", flush=True)
                out = train_probe(
                    arch=arch,
                    tr_z=tr_z_s, tr_y=tr_y, tr_idx=train_idx, vl_idx=val_idx,
                    te_z=te_z_s, te_y=te_y,
                    label_mean=label_mean, label_std=label_std,
                    lr=args.lr, wd=args.wd,
                    batch_size=args.batch_size, max_epochs=args.max_epochs,
                    device=device, seed=seed,
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
                    "shuffle": shuffle_mode,
                    "wd": args.wd,
                    "test_r2": out["test_r2"],
                    "val_r2": out["val_r2"],
                    "val_mse": out["val_mse"],
                    "r2_ci_lo": lo,
                    "r2_ci_hi": hi,
                }
                print(f"[pertoken] <<< {tag}  test_r2={out['test_r2']:.4f}  CI=[{lo:.4f},{hi:.4f}]", flush=True)
                results.append(row)

                if out["attn"]:
                    key = f"{arch}__seed{seed}__{shuffle_mode}"
                    for k, v in out["attn"].items():
                        attn_store[f"{key}__{k}"] = v

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
                "shuffles": args.shuffles,
                "wd": args.wd,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[pertoken] wrote {out_path} ({len(results)} rows)", flush=True)

    if args.attn_out and attn_store:
        Path(args.attn_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.attn_out, **attn_store)
        print(f"[pertoken] wrote {args.attn_out} ({len(attn_store)} tensors)", flush=True)


if __name__ == "__main__":
    main()
