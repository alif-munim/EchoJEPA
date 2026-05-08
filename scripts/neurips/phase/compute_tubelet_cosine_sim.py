#!/usr/bin/env python3
"""Compute tubelet-to-tubelet cosine-similarity matrices for two encoders.

Sharded across GPUs: each rank processes clips[rank::world_size] and
writes a partial-sum .npy + per-encoder clip count. A final aggregation
pass (run with --aggregate) sums the partials and divides to produce
the final 8x8 mean matrices.

For each encoder (JEPA IN21K e100 and MAE IN21K e99), per rank we:
  1. Load the encoder frozen on that rank's GPU.
  2. For each clip in this rank's shard, extract patch-token features
     [1568, D] (ViT-L/16, tubelet_size=2, 16 frames, 224x224 ->
     8 temporal tubelets x 14x14=196 spatial patches).
  3. Reshape to [8, 196, D], mean-pool over spatial -> [8, D].
  4. Compute the clip's 8x8 cosine similarity matrix.
  5. Accumulate the 8x8 sum and clip count for this rank.

Outputs under --out-dir (per rank):
  rank{r}_jepa_sum.npy, rank{r}_jepa_count.txt
  rank{r}_mae_sum.npy,  rank{r}_mae_count.txt

After all ranks finish, --aggregate mode reads rank{0..N-1}_*.{npy,txt}
and writes jepa_tubelet_cos.npy + mae_tubelet_cos.npy.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jepa-ckpt", type=Path)
    p.add_argument("--mae-ckpt",  type=Path)
    p.add_argument("--byol-ckpt", type=Path)
    p.add_argument("--salt-ckpt", type=Path)
    # V-JEPA trajectory variants (same architecture as --jepa-ckpt, different
    # checkpoint labels). All use ckpt_key=target_encoder.
    p.add_argument("--jepa-e100-ckpt", type=Path)
    p.add_argument("--jepa-e125-ckpt", type=Path)
    p.add_argument("--v4-e25-ckpt",    type=Path)
    p.add_argument("--test-csv",  type=Path)
    p.add_argument("--out-dir",   required=True, type=Path)
    p.add_argument("--n-clips",   type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--frames-per-clip", type=int, default=16)
    p.add_argument("--frame-step", type=int, default=2)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--tubelet-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--aggregate", action="store_true",
                   help="Aggregate partial files instead of running inference.")
    return p.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_test_paths(csv_path: Path, n: int, seed: int) -> list[str]:
    paths: list[str] = []
    with csv_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                paths.append(parts[0])
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n]


def _collate_clips_only(batch):
    """VideoDataset returns (buffer, label, clip_indices, uri, phase_meta).
    default_collate trips on `None`/str fields; we only need the tensor.

    Layer shape: buffer = list-over-num_clips of list-over-num_views of
    [C, T, H, W] tensors. With num_clips=1, num_views=1, this is
    ``[[tensor]]`` per sample → index as ``s[0][0][0]``.
    """
    tensors = []
    for s in batch:
        buf = s[0]
        # Unwrap nested lists until we reach the tensor.
        while isinstance(buf, (list, tuple)):
            buf = buf[0]
        tensors.append(buf)
    return torch.stack(tensors, dim=0)  # [B, C, T, H, W]


def _build_video_loader(
    csv_path: Path, frames_per_clip: int, frame_step: int, resolution: int,
    batch_size: int, num_workers: int,
):
    """Use the repo's own eval transform. VideoDataset + decord emit numpy
    arrays, and torchvision transforms don't accept those; the repo's
    video_transforms pipeline (ClipToTensor + Resize/CenterCrop/Normalize)
    is the supported path. Custom collate sidesteps default_collate's
    intolerance for None / str fields in VideoDataset's return tuple.
    """
    from src.datasets.video_dataset import VideoDataset
    from evals.video_classification_frozen.utils import make_transforms

    transform = make_transforms(
        training=False,
        crop_size=resolution,
        num_views_per_clip=1,
    )

    ds = VideoDataset(
        data_paths=[str(csv_path)],
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=1,
        transform=transform,
        random_clip_sampling=False,
        allow_clip_overlap=True,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False,
        collate_fn=_collate_clips_only,
    )
    return loader


def _load_vjepa_style_encoder(
    ckpt_path: Path, resolution: int, frames: int, tubelet_size: int,
    ckpt_key: str, name: str,
) -> torch.nn.Module:
    """Load a V-JEPA-style ViT-L/16 encoder from a pretrain checkpoint.
    Used for JEPA (key=target_encoder), BYOL (key=target_encoder), and
    SALT (key=encoder). All share the same backbone architecture.
    """
    import src.models.vision_transformer as vit
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt[ckpt_key]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}

    model = vit.vit_large(
        img_size=resolution, num_frames=frames,
        patch_size=16, tubelet_size=tubelet_size,
        uniform_power=True, use_rope=True,
    )
    for k, v in model.state_dict().items():
        if k in sd and sd[k].shape != v.shape:
            sd[k] = v
    msg = model.load_state_dict(sd, strict=False)
    print(f"[{name}] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}", flush=True)
    model.eval().requires_grad_(False)
    return model


def _load_jepa_encoder(ckpt_path: Path, resolution: int, frames: int,
                       tubelet_size: int) -> torch.nn.Module:
    return _load_vjepa_style_encoder(
        ckpt_path, resolution, frames, tubelet_size,
        ckpt_key="target_encoder", name="jepa",
    )


def _load_byol_encoder(ckpt_path: Path, resolution: int, frames: int,
                       tubelet_size: int) -> torch.nn.Module:
    return _load_vjepa_style_encoder(
        ckpt_path, resolution, frames, tubelet_size,
        ckpt_key="target_encoder", name="byol",
    )


def _load_salt_encoder(ckpt_path: Path, resolution: int, frames: int,
                       tubelet_size: int) -> torch.nn.Module:
    return _load_vjepa_style_encoder(
        ckpt_path, resolution, frames, tubelet_size,
        ckpt_key="encoder", name="salt",
    )


def _load_mae_encoder(ckpt_path: Path, resolution: int, frames: int,
                      tubelet_size: int) -> torch.nn.Module:
    from evals.video_classification_frozen.modelcustom.videomae_encoder import (
        _import_modeling_finetune, _convert_pretrain_to_finetune_state_dict,
    )
    mf = _import_modeling_finetune()
    # num_classes=0 sets head=Identity but constructor still does
    # trunc_normal_(self.head.weight, ...) → AttributeError. Use a
    # nonzero num_classes (head is ignored; we call forward_features).
    model = mf.vit_large_patch16_224(
        pretrained=False, num_classes=1000,
        all_frames=frames, tubelet_size=tubelet_size,
        use_mean_pooling=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("model") or ckpt.get("module") or ckpt
    sd = _convert_pretrain_to_finetune_state_dict(raw, model.state_dict())
    msg = model.load_state_dict(sd, strict=False)
    print(f"[mae] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}", flush=True)
    model.eval().requires_grad_(False)
    return model


def _jepa_forward(model: torch.nn.Module, clip: torch.Tensor) -> torch.Tensor:
    return model(clip)


def _mae_forward(model: torch.nn.Module, clip: torch.Tensor) -> torch.Tensor:
    """VideoMAE's forward_features always pools (mean or cls token). We
    reimplement the tokenwise path: patch_embed -> pos -> blocks -> norm.
    Mirrors modeling_finetune.VisionTransformer.forward_features up to
    (but not including) the mean-pool / cls-slice step.
    """
    x = model.patch_embed(clip)
    B, _, _ = x.size()
    if model.pos_embed is not None:
        pe = model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pe = model._get_matched_pos_embed(pe, x.size(1))
        x = x + pe
    x = model.pos_drop(x)
    for blk in model.blocks:
        x = blk(x)
    x = model.norm(x)  # nn.Identity when use_mean_pooling=True; LayerNorm otherwise
    expected = 8 * 14 * 14
    if x.shape[1] != expected:
        raise RuntimeError(
            f"MAE returned N={x.shape[1]} tokens; expected {expected}"
        )
    return x


def _accumulate_cos_8x8(feats_BND: torch.Tensor, T: int = 8, S: int = 196) -> torch.Tensor:
    """Spatial-mean-pool per tubelet, then 8x8 cosine similarity per clip.
    Returns [B, T, T].
    """
    B, N, D = feats_BND.shape
    assert N == T * S, f"N={N} != {T*S}"
    pooled = feats_BND.view(B, T, S, D).mean(dim=2)
    pooled = F.normalize(pooled, dim=-1)
    return torch.einsum("btd,bsd->bts", pooled, pooled)


def _accumulate_cos_8x8_token_level(feats_BND: torch.Tensor, T: int = 8, S: int = 196) -> torch.Tensor:
    """Per-position cosine similarity, averaged across spatial positions.

    For each spatial position p in 1..S, take the 1024-dim token at
    (tubelet_i, p) and (tubelet_j, p) for i,j in 1..T, compute cosine,
    then average over p. This keeps the same 8x8 structure but skips
    the spatial mean-pool step, so we can verify the "JEPA globally
    contextualised / MAE tubelet-local" pattern is a property of the
    features, not of the pooling.

    Returns [B, T, T].
    """
    B, N, D = feats_BND.shape
    assert N == T * S, f"N={N} != {T*S}"
    # [B, T, S, D] -> normalise per token -> [B, S, T, D]
    x = feats_BND.view(B, T, S, D)
    x = F.normalize(x, dim=-1)
    x = x.permute(0, 2, 1, 3).contiguous()  # [B, S, T, D]
    # For each spatial position p: cos(t_i, t_j) = <x[b,p,i,:], x[b,p,j,:]>
    # einsum sums over D to get [B, S, T, T], mean over S for [B, T, T].
    cos_BSTT = torch.einsum("bpid,bpjd->bpij", x, x)
    return cos_BSTT.mean(dim=1)


def _extract_and_accumulate(
    name: str, model: torch.nn.Module, forward_fn, loader,
    frames: int, tubelet_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (pooled_sum, token_level_sum, seen). Both sums are [T, T]."""
    T = frames // tubelet_size
    device = next(model.parameters()).device
    pooled_sum = torch.zeros((T, T), device=device, dtype=torch.float64)
    token_sum = torch.zeros((T, T), device=device, dtype=torch.float64)
    seen = 0
    for batch_idx, clips in enumerate(loader):
        # _collate_clips_only yields [B, C, T, H, W] directly.
        if not torch.is_tensor(clips) or clips.dim() != 5:
            raise RuntimeError(f"Expected [B,C,T,H,W] tensor, got {type(clips)} shape={getattr(clips, 'shape', '?')}")
        clips = clips.to(device, non_blocking=True).float()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            feats = forward_fn(model, clips)
        feats = feats.float()
        if feats.shape[1] != T * 196:
            print(f"[{name}] WARN batch {batch_idx}: got N={feats.shape[1]}; skipping", flush=True)
            continue
        pooled_BTT = _accumulate_cos_8x8(feats, T=T, S=196)
        token_BTT = _accumulate_cos_8x8_token_level(feats, T=T, S=196)
        pooled_sum += pooled_BTT.sum(dim=0).double()
        token_sum += token_BTT.sum(dim=0).double()
        seen += pooled_BTT.shape[0]
        if batch_idx % 10 == 0:
            print(f"[{name}] batch {batch_idx:4d}  clips_seen={seen}", flush=True)

    pooled_np = pooled_sum.detach().cpu().numpy().astype(np.float64)
    token_np = token_sum.detach().cpu().numpy().astype(np.float64)
    print(f"[{name}] shard done — {seen} clips", flush=True)
    return pooled_np, token_np, seen


def _run_shard(args: argparse.Namespace) -> int:
    _seed_all(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_paths = _load_test_paths(args.test_csv, args.n_clips, args.seed)
    my_paths = all_paths[args.rank :: args.world_size]
    print(f"[rank {args.rank}/{args.world_size}] shard size = {len(my_paths)}", flush=True)

    shard_csv = args.out_dir / f"rank{args.rank}_clips.csv"
    with shard_csv.open("w") as fh:
        for p in my_paths:
            fh.write(f"{p} 0\n")

    loader = _build_video_loader(
        shard_csv, args.frames_per_clip, args.frame_step,
        args.resolution, args.batch_size, num_workers=8,
    )

    device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # Run each requested encoder sequentially on this rank's shard.
    # Skip any encoder whose --ckpt path was not supplied.
    encoder_specs = []
    if args.jepa_ckpt is not None:
        encoder_specs.append(("jepa", args.jepa_ckpt, _load_jepa_encoder, _jepa_forward))
    if args.mae_ckpt is not None:
        encoder_specs.append(("mae",  args.mae_ckpt,  _load_mae_encoder,  _mae_forward))
    if args.byol_ckpt is not None:
        encoder_specs.append(("byol", args.byol_ckpt, _load_byol_encoder, _jepa_forward))
    if args.salt_ckpt is not None:
        encoder_specs.append(("salt", args.salt_ckpt, _load_salt_encoder, _jepa_forward))
    # Trajectory variants: same loader as JEPA (target_encoder key).
    if args.jepa_e100_ckpt is not None:
        encoder_specs.append(("jepa_e100", args.jepa_e100_ckpt, _load_jepa_encoder, _jepa_forward))
    if args.jepa_e125_ckpt is not None:
        encoder_specs.append(("jepa_e125", args.jepa_e125_ckpt, _load_jepa_encoder, _jepa_forward))
    if args.v4_e25_ckpt is not None:
        encoder_specs.append(("v4_e25",    args.v4_e25_ckpt,    _load_jepa_encoder, _jepa_forward))

    counts = {}
    for tag, ckpt, loader_fn, forward_fn in encoder_specs:
        enc = loader_fn(ckpt, args.resolution, args.frames_per_clip, args.tubelet_size).to(device)
        pool_sum, tok_sum, cnt = _extract_and_accumulate(
            f"{tag}-r{args.rank}", enc, forward_fn, loader,
            args.frames_per_clip, args.tubelet_size,
        )
        np.save(args.out_dir / f"rank{args.rank}_{tag}_sum.npy", pool_sum)
        np.save(args.out_dir / f"rank{args.rank}_{tag}_tokenlevel_sum.npy", tok_sum)
        (args.out_dir / f"rank{args.rank}_{tag}_count.txt").write_text(str(cnt))
        counts[tag] = cnt
        del enc
        torch.cuda.empty_cache()

    print(f"[rank {args.rank}] done. counts={counts}", flush=True)
    return 0


def _run_aggregate(args: argparse.Namespace) -> int:
    # Aggregate two variants per encoder: pooled (mean-pool spatial)
    # and tokenlevel (no pool; per-position cosine averaged).
    for encoder in ("jepa", "mae", "byol", "salt",
                    "jepa_e100", "jepa_e125", "v4_e25"):
        # Read the count once per encoder.
        total_cnt = 0
        cnts_per_rank = {}
        for r in range(args.world_size):
            cnt_f = args.out_dir / f"rank{r}_{encoder}_count.txt"
            if not cnt_f.exists():
                print(f"[aggregate] MISSING rank {r} {encoder} count — skipping", flush=True)
                continue
            c = int(cnt_f.read_text().strip())
            cnts_per_rank[r] = c
            total_cnt += c

        for variant, partial_tag, out_tag in [
            ("pooled",     "sum",            "tubelet_cos"),
            ("tokenlevel", "tokenlevel_sum", "tubelet_cos_tokenlevel"),
        ]:
            total_sum = None
            partials_used = 0
            for r in range(args.world_size):
                sum_f = args.out_dir / f"rank{r}_{encoder}_{partial_tag}.npy"
                if not sum_f.exists() or r not in cnts_per_rank:
                    continue
                s = np.load(sum_f).astype(np.float64)
                total_sum = s if total_sum is None else total_sum + s
                partials_used += 1
            if total_sum is None or total_cnt == 0:
                print(f"[aggregate] NO PARTIALS for {encoder}/{variant}", flush=True)
                continue
            mean = (total_sum / total_cnt).astype(np.float32)
            out = args.out_dir / f"{encoder}_{out_tag}.npy"
            np.save(out, mean)
            print(f"[aggregate] {encoder}/{variant}: merged {partials_used} partials, total_cnt={total_cnt} -> {out}", flush=True)
            print(f"[aggregate] {encoder}/{variant} mean matrix:\n{np.array2string(mean, precision=3, suppress_small=True)}", flush=True)
    return 0


def main() -> int:
    args = _parse_args()
    if args.aggregate:
        return _run_aggregate(args)
    if args.test_csv is None:
        print("Shard mode requires --test-csv", file=sys.stderr)
        return 2
    any_ckpt = any(c is not None for c in (
        args.jepa_ckpt, args.mae_ckpt, args.byol_ckpt, args.salt_ckpt,
        args.jepa_e100_ckpt, args.jepa_e125_ckpt, args.v4_e25_ckpt,
    ))
    if not any_ckpt:
        print("Shard mode requires at least one --{jepa,mae,byol,salt,"
              "jepa-e100,jepa-e125,v4-e25}-ckpt", file=sys.stderr)
        return 2
    return _run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
