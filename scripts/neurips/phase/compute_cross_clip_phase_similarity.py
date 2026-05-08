#!/usr/bin/env python3
"""Cross-clip phase-aligned similarity.

For pairs (A, B) of clips drawn from the same MIMIC study, we encode
both clips with a frozen ViT-L/16, spatial-mean-pool each tubelet to
a 1024-dim vector, read each tubelet's cardiac phase from the
per-frame phase JSON in the phase_annotations parquet, and compute
cosine similarity as a function of phase offset Δφ in [-0.5, 0.5).

Output per encoder: a histogram-style array of mean cosine
similarity binned by Δφ, plus per-pair counts. Shape [n_bins],
float32.

If state-sync (V4) worked, we expect the Δφ = 0 bin to be highest
(matched-phase positive), with a drop-off for larger offsets. MAE
should show flatter phase dependence (pixel/view-local features).
BYOL should be uniformly high with weak phase contrast. JEPA e100
should be somewhere in the middle. This test is the specific
phase-matched structure the state-sync objective is trained for.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase-parquet", type=Path,
                   help="Path to phase_annotations.parquet (per-frame phase JSON per clip). "
                        "Required for shard mode; unused in --aggregate mode.")
    p.add_argument("--out-dir", required=True, type=Path)
    # Trajectory + cross-encoder checkpoints (any subset).
    p.add_argument("--jepa-e100-ckpt", type=Path)
    p.add_argument("--jepa-e125-ckpt", type=Path)
    p.add_argument("--v4-e25-ckpt",    type=Path)
    p.add_argument("--mae-ckpt",       type=Path)
    p.add_argument("--byol-ckpt",      type=Path)
    p.add_argument("--salt-ckpt",      type=Path)

    p.add_argument("--n-studies",      type=int, default=400,
                   help="Number of distinct MIMIC studies to sample.")
    p.add_argument("--pairs-per-study", type=int, default=2,
                   help="Number of (A, B) clip pairs per study.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--frames-per-clip", type=int, default=16)
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--n-bins", type=int, default=16,
                   help="Δφ histogram bins across [-0.5, 0.5).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--min-quality", choices=("any", "medium", "high"), default="medium",
                   help="Minimum quality_tier for included clips.")
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate_clips_only(batch):
    tensors = []
    for s in batch:
        buf = s[0]
        while isinstance(buf, (list, tuple)):
            buf = buf[0]
        tensors.append(buf)
    return torch.stack(tensors, dim=0)


def _build_video_loader(csv_path: Path, frames_per_clip: int, frame_step: int,
                       resolution: int, batch_size: int, num_workers: int):
    from src.datasets.video_dataset import VideoDataset
    from evals.video_classification_frozen.utils import make_transforms
    transform = make_transforms(training=False, crop_size=resolution,
                                num_views_per_clip=1)
    ds = VideoDataset(
        data_paths=[str(csv_path)], frames_per_clip=frames_per_clip,
        frame_step=frame_step, num_clips=1, transform=transform,
        random_clip_sampling=False, allow_clip_overlap=True,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False, collate_fn=_collate_clips_only,
    )


# -------------------- Encoder loaders (reused from sibling script) --------------------

def _load_vjepa(ckpt_path: Path, resolution: int, frames: int, tubelet_size: int,
                 ckpt_key: str, name: str) -> torch.nn.Module:
    import src.models.vision_transformer as vit
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt[ckpt_key]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    model = vit.vit_large(
        img_size=resolution, num_frames=frames, patch_size=16,
        tubelet_size=tubelet_size, uniform_power=True, use_rope=True,
    )
    for k, v in model.state_dict().items():
        if k in sd and sd[k].shape != v.shape:
            sd[k] = v
    msg = model.load_state_dict(sd, strict=False)
    print(f"[{name}] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}", flush=True)
    model.eval().requires_grad_(False)
    return model


def _load_mae(ckpt_path: Path, resolution: int, frames: int, tubelet_size: int) -> torch.nn.Module:
    from evals.video_classification_frozen.modelcustom.videomae_encoder import (
        _import_modeling_finetune, _convert_pretrain_to_finetune_state_dict,
    )
    mf = _import_modeling_finetune()
    model = mf.vit_large_patch16_224(
        pretrained=False, num_classes=1000,
        all_frames=frames, tubelet_size=tubelet_size, use_mean_pooling=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("model") or ckpt.get("module") or ckpt
    sd = _convert_pretrain_to_finetune_state_dict(raw, model.state_dict())
    msg = model.load_state_dict(sd, strict=False)
    print(f"[mae] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}", flush=True)
    model.eval().requires_grad_(False)
    return model


def _vjepa_forward(model, clip):
    return model(clip)


def _mae_forward(model, clip):
    x = model.patch_embed(clip)
    B, _, _ = x.size()
    if model.pos_embed is not None:
        pe = model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pe = model._get_matched_pos_embed(pe, x.size(1))
        x = x + pe
    x = model.pos_drop(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)
    expected = 8 * 14 * 14
    if x.shape[1] != expected:
        raise RuntimeError(f"MAE returned N={x.shape[1]} tokens; expected {expected}")
    return x


# -------------------- Pair sampling --------------------

def _rewrite_dicom_to_mp4(uri: str) -> str:
    """MIMIC DICOMs at mimic-raw-staging/...dcm → mp4 mirror at mimic-echo-224px."""
    return (uri
            .replace("mimic-raw-staging", "mimic-echo-224px")
            .replace(".dcm", ".mp4"))


def _sample_study_pairs(phase_df: pd.DataFrame, n_studies: int, pairs_per_study: int,
                        seed: int) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Sample (clip_a_uri, clip_b_uri, phase_a_per_frame, phase_b_per_frame) tuples.

    Draws pairs within a study so A != B. Per-frame phase arrays are decoded
    from the parquet's per_frame_phase_json column.
    """
    # Group by study, keep only studies with >= 2 clips.
    grp = phase_df.groupby("study_id")
    studies_with_pairs = [sid for sid, g in grp if len(g) >= 2]
    rng = random.Random(seed)
    rng.shuffle(studies_with_pairs)
    selected = studies_with_pairs[:n_studies]

    pairs = []
    for sid in selected:
        g = grp.get_group(sid).reset_index(drop=True)
        n = len(g)
        local_rng = random.Random(seed + hash(str(sid)) % (2**31))
        for _ in range(pairs_per_study):
            i, j = local_rng.sample(range(n), 2)
            row_a = g.iloc[i]
            row_b = g.iloc[j]
            try:
                phase_a = np.asarray(json.loads(row_a["per_frame_phase_json"]), dtype=np.float32)
                phase_b = np.asarray(json.loads(row_b["per_frame_phase_json"]), dtype=np.float32)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if phase_a.size < 16 or phase_b.size < 16:
                continue
            # Skip pairs with NaN phase everywhere.
            if np.isnan(phase_a).all() or np.isnan(phase_b).all():
                continue
            uri_a = _rewrite_dicom_to_mp4(str(row_a["s3_uri"]))
            uri_b = _rewrite_dicom_to_mp4(str(row_b["s3_uri"]))
            pairs.append((uri_a, uri_b, phase_a, phase_b))
    return pairs


def _tubelet_phases(per_frame_phase: np.ndarray, frames_per_clip: int,
                    frame_step: int, tubelet_size: int) -> np.ndarray:
    """Given per-frame phase and the dataloader's sampling settings, return
    per-tubelet mean phase (length T = frames_per_clip // tubelet_size).

    The clip sampler takes `frames_per_clip` frames at stride `frame_step`.
    VideoDataset samples a window of length `frames_per_clip * frame_step`
    that starts somewhere in the video; without knowing the exact offset,
    we approximate by averaging the **evenly-spaced frame-stride sample**
    starting at a centered offset. The result is a usable-enough Δφ for
    binning purposes.
    """
    total = per_frame_phase.size
    window = frames_per_clip * frame_step
    if total <= window:
        offset = 0
        idx = np.linspace(0, total - 1, frames_per_clip).astype(int)
    else:
        offset = (total - window) // 2
        idx = offset + np.arange(0, window, frame_step)[:frames_per_clip]
    frame_phases = per_frame_phase[idx]
    # Reshape [T, tubelet_size] and average inside each tubelet.
    # Circular averaging via sin/cos so phase=0.99 and phase=0.01 average near 0.0 / 1.0.
    angles = 2 * np.pi * frame_phases
    tub = angles.reshape(-1, tubelet_size)
    mean_sin = np.nanmean(np.sin(tub), axis=1)
    mean_cos = np.nanmean(np.cos(tub), axis=1)
    tub_phase = (np.arctan2(mean_sin, mean_cos) / (2 * np.pi)) % 1.0
    return tub_phase.astype(np.float32)


# -------------------- Core computation --------------------

def _pool_tubelets(feats_BND: torch.Tensor, T: int = 8, S: int = 196) -> torch.Tensor:
    B, N, D = feats_BND.shape
    assert N == T * S
    pooled = feats_BND.view(B, T, S, D).mean(dim=2)
    return F.normalize(pooled, dim=-1)  # [B, T, D]


def _signed_phase_diff(phi_a: float, phi_b: float) -> float:
    """Smallest signed phase difference in [-0.5, 0.5)."""
    d = (phi_b - phi_a) % 1.0
    if d > 0.5:
        d -= 1.0
    return d


def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(-0.5, 0.5, n_bins + 1)


def _accumulate_phase_curve(
    sims: np.ndarray, deltas: np.ndarray, edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin per-tubelet-pair (Δφ, cos) into the shared edges; return (sum, count)."""
    bins = np.clip(np.digitize(deltas, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(bins, minlength=len(edges) - 1)
    sums = np.bincount(bins, weights=sims.astype(np.float64), minlength=len(edges) - 1)
    return sums, counts


def _run_shard(args: argparse.Namespace) -> int:
    _seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load phase metadata, filter, sample pairs.
    print(f"[rank {args.rank}] loading phase parquet...", flush=True)
    df = pd.read_parquet(args.phase_parquet, columns=[
        "s3_uri", "study_id", "per_frame_phase_json", "quality_tier",
    ])
    if args.min_quality != "any":
        tiers = {"medium": {"medium", "high"}, "high": {"high"}}[args.min_quality]
        df = df[df["quality_tier"].isin(tiers)].reset_index(drop=True)
    # Drop rows without phase JSON.
    df = df[df["per_frame_phase_json"].notna()].reset_index(drop=True)
    print(f"[rank {args.rank}] {len(df):,} phase-annotated clips, "
          f"{df['study_id'].nunique():,} studies", flush=True)

    all_pairs = _sample_study_pairs(df, args.n_studies, args.pairs_per_study, args.seed)
    my_pairs = all_pairs[args.rank::args.world_size]
    print(f"[rank {args.rank}/{args.world_size}] shard: {len(my_pairs)} pairs", flush=True)

    # Write clip-list CSV for this rank (two clips per pair → one row each).
    clip_uris = []
    for (ua, ub, pa, pb) in my_pairs:
        clip_uris.append(ua); clip_uris.append(ub)
    shard_csv = args.out_dir / f"rank{args.rank}_clips.csv"
    with shard_csv.open("w") as fh:
        for uri in clip_uris:
            fh.write(f"{uri} 0\n")

    if not my_pairs:
        print(f"[rank {args.rank}] no pairs", flush=True)
        return 0

    # Build loader; results come out in the same order we wrote them.
    loader = _build_video_loader(
        shard_csv, args.frames_per_clip, args.frame_step,
        args.resolution, args.batch_size, num_workers=8,
    )

    device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # Pre-compute per-clip tubelet phases using the same sampling convention
    # as the loader (center-window + frame-step). Index aligned with
    # clip_uris by construction.
    T = args.frames_per_clip // args.tubelet_size
    tub_phases = []  # list of [T] arrays aligned with clip_uris
    for (ua, ub, pa, pb) in my_pairs:
        tub_phases.append(_tubelet_phases(pa, args.frames_per_clip,
                                          args.frame_step, args.tubelet_size))
        tub_phases.append(_tubelet_phases(pb, args.frames_per_clip,
                                          args.frame_step, args.tubelet_size))
    tub_phases = np.stack(tub_phases, axis=0)  # [2*P, T]

    # Build encoder specs list.
    specs = []
    if args.jepa_e100_ckpt is not None: specs.append(("jepa_e100", args.jepa_e100_ckpt, _load_vjepa, _vjepa_forward, "target_encoder"))
    if args.jepa_e125_ckpt is not None: specs.append(("jepa_e125", args.jepa_e125_ckpt, _load_vjepa, _vjepa_forward, "target_encoder"))
    if args.v4_e25_ckpt    is not None: specs.append(("v4_e25",    args.v4_e25_ckpt,    _load_vjepa, _vjepa_forward, "target_encoder"))
    if args.mae_ckpt       is not None: specs.append(("mae",       args.mae_ckpt,       None,        _mae_forward,   None))
    if args.byol_ckpt      is not None: specs.append(("byol",      args.byol_ckpt,      _load_vjepa, _vjepa_forward, "target_encoder"))
    if args.salt_ckpt      is not None: specs.append(("salt",      args.salt_ckpt,      _load_vjepa, _vjepa_forward, "encoder"))

    edges = _bin_edges(args.n_bins)

    for tag, ckpt, loader_fn, forward_fn, ckpt_key in specs:
        if loader_fn is None:  # MAE
            enc = _load_mae(ckpt, args.resolution, args.frames_per_clip, args.tubelet_size)
        else:
            enc = _load_vjepa(ckpt, args.resolution, args.frames_per_clip,
                               args.tubelet_size, ckpt_key=ckpt_key, name=tag)
        enc = enc.to(device)

        # Pass 1: compute pooled [2*P, T, D] for this encoder.
        all_pool = []
        for bi, clips in enumerate(loader):
            if not torch.is_tensor(clips) or clips.dim() != 5:
                raise RuntimeError(f"Bad clips shape: {getattr(clips, 'shape', '?')}")
            clips = clips.to(device, non_blocking=True).float()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                feats = forward_fn(enc, clips)
            feats = feats.float()
            pooled = _pool_tubelets(feats, T=T, S=196)  # [B, T, D]
            all_pool.append(pooled.detach().cpu())
            if bi % 20 == 0:
                print(f"[{tag}-r{args.rank}] batch {bi:4d}  done", flush=True)
        all_pool = torch.cat(all_pool, dim=0).numpy()  # [2*P, T, D]

        # Pass 2: pair up (A, B) → compute cosine between every (t_A, t_B)
        # tubelet combo, binned by Δφ (1D) and by (phase_A, phase_B) (2D matrix).
        # Both accumulators are updated from the same cos/phase arrays.
        P = len(my_pairs)
        sums = np.zeros(args.n_bins, dtype=np.float64)
        counts = np.zeros(args.n_bins, dtype=np.int64)
        # 2D: rows = phase_A bin in [0, 1), cols = phase_B bin in [0, 1).
        abs_edges = np.linspace(0.0, 1.0, args.n_bins + 1)
        mat_sum = np.zeros((args.n_bins, args.n_bins), dtype=np.float64)
        mat_cnt = np.zeros((args.n_bins, args.n_bins), dtype=np.int64)
        for p in range(P):
            poolA = all_pool[2 * p]      # [T, D]
            poolB = all_pool[2 * p + 1]  # [T, D]
            phiA = tub_phases[2 * p]      # [T]
            phiB = tub_phases[2 * p + 1]  # [T]
            cos = poolA @ poolB.T         # [T, T]
            deltas = np.zeros((T, T), dtype=np.float32)
            for ti in range(T):
                for tj in range(T):
                    deltas[ti, tj] = _signed_phase_diff(float(phiA[ti]), float(phiB[tj]))
            mask = np.isfinite(deltas) & np.isfinite(cos)
            s, c = _accumulate_phase_curve(cos[mask], deltas[mask], edges)
            sums += s; counts += c
            # 2D accumulation: bin each (i, j) cell into (phase_A, phase_B).
            phiA_mat = np.broadcast_to(phiA[:, None] % 1.0, (T, T))
            phiB_mat = np.broadcast_to(phiB[None, :] % 1.0, (T, T))
            a_bins = np.clip(np.digitize(phiA_mat[mask], abs_edges) - 1, 0, args.n_bins - 1)
            b_bins = np.clip(np.digitize(phiB_mat[mask], abs_edges) - 1, 0, args.n_bins - 1)
            flat = a_bins * args.n_bins + b_bins
            s_flat = np.bincount(flat, weights=cos[mask].astype(np.float64),
                                 minlength=args.n_bins * args.n_bins)
            c_flat = np.bincount(flat, minlength=args.n_bins * args.n_bins)
            mat_sum += s_flat.reshape(args.n_bins, args.n_bins)
            mat_cnt += c_flat.reshape(args.n_bins, args.n_bins)

        np.save(args.out_dir / f"rank{args.rank}_{tag}_phase_sum.npy", sums)
        np.save(args.out_dir / f"rank{args.rank}_{tag}_phase_count.npy", counts)
        np.save(args.out_dir / f"rank{args.rank}_{tag}_phaseAB_sum.npy", mat_sum)
        np.save(args.out_dir / f"rank{args.rank}_{tag}_phaseAB_count.npy", mat_cnt)
        print(f"[{tag}-r{args.rank}] done — {counts.sum()} tubelet-pair samples", flush=True)
        del enc
        torch.cuda.empty_cache()

    (args.out_dir / f"rank{args.rank}_bin_edges.npy").write_bytes(
        edges.astype(np.float32).tobytes()
    )
    np.save(args.out_dir / f"rank{args.rank}_bin_edges.npy", edges.astype(np.float32))

    return 0


def _run_aggregate(args: argparse.Namespace) -> int:
    edges = _bin_edges(args.n_bins)
    np.save(args.out_dir / "bin_edges.npy", edges.astype(np.float32))
    abs_edges = np.linspace(0.0, 1.0, args.n_bins + 1)
    np.save(args.out_dir / "abs_bin_edges.npy", abs_edges.astype(np.float32))

    for tag in ("jepa_e100", "jepa_e125", "v4_e25", "mae", "byol", "salt"):
        sums = None
        counts = None
        mat_sum = None
        mat_cnt = None
        partials = 0
        for r in range(args.world_size):
            sum_f = args.out_dir / f"rank{r}_{tag}_phase_sum.npy"
            cnt_f = args.out_dir / f"rank{r}_{tag}_phase_count.npy"
            mat_sum_f = args.out_dir / f"rank{r}_{tag}_phaseAB_sum.npy"
            mat_cnt_f = args.out_dir / f"rank{r}_{tag}_phaseAB_count.npy"
            if not sum_f.exists() or not cnt_f.exists():
                continue
            s = np.load(sum_f).astype(np.float64)
            c = np.load(cnt_f).astype(np.int64)
            sums = s if sums is None else sums + s
            counts = c if counts is None else counts + c
            if mat_sum_f.exists() and mat_cnt_f.exists():
                ms = np.load(mat_sum_f).astype(np.float64)
                mc = np.load(mat_cnt_f).astype(np.int64)
                mat_sum = ms if mat_sum is None else mat_sum + ms
                mat_cnt = mc if mat_cnt is None else mat_cnt + mc
            partials += 1
        if sums is None or counts is None or counts.sum() == 0:
            print(f"[aggregate] no data for {tag}", flush=True)
            continue
        if mat_sum is not None and mat_cnt is not None and mat_cnt.sum() > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                mat_mean = mat_sum / np.maximum(mat_cnt, 1)
            np.save(args.out_dir / f"{tag}_phaseAB_similarity.npy", mat_mean.astype(np.float32))
            np.save(args.out_dir / f"{tag}_phaseAB_counts.npy", mat_cnt)
            print(f"[aggregate] {tag} 2D matrix: shape={mat_mean.shape} "
                  f"diagonal mean={np.diag(mat_mean).mean():.3f} "
                  f"off-diagonal mean={(mat_mean.sum() - mat_mean.trace()) / (mat_mean.size - mat_mean.shape[0]):.3f}",
                  flush=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_cos = sums / np.maximum(counts, 1)
        out_mean = args.out_dir / f"{tag}_phase_similarity.npy"
        out_cnt = args.out_dir / f"{tag}_phase_counts.npy"
        np.save(out_mean, mean_cos.astype(np.float32))
        np.save(out_cnt, counts)
        print(f"[aggregate] {tag}: {partials} partials, {counts.sum():,} samples", flush=True)
        print(f"  mean cosine by Δφ bin: {np.array2string(mean_cos, precision=3)}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.aggregate:
        return _run_aggregate(args)
    if args.phase_parquet is None:
        print("Shard mode requires --phase-parquet", file=sys.stderr)
        return 2
    return _run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
