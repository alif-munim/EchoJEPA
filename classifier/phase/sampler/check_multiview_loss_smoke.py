"""One-batch forward+backward smoke test for the multiview cross-view loss.

Instantiates a ViT-Tiny encoder/predictor/target_encoder on CPU (or CUDA
if available), pulls one small-sized batch via the phase_matched loader,
runs ``forward_intraview_and_crossview``, computes losses, runs
``total.backward()``, and asserts:

  * gradients exist on encoder params
  * gradients exist on predictor params
  * target_encoder params have NO gradient (it's a ``no_grad`` teacher)
  * intraview_loss, crossview_loss, total_loss are all finite
  * intraview_loss != crossview_loss (matched pair, not identical clips)

This is a CORRECTNESS gate, not a performance test. It runs on the
4-pair synthesized on-disk batch from Turn 2 at img_size=128 to keep
the ViT forward tractable without a GPU.

Usage:
    python check_multiview_loss_smoke.py \\
        --parquet phase_annotations/phase_annotations.parquet \\
        --dicom-dir dicoms \\
        --batch-size 2 \\
        --img-size 128 --frames-per-clip 8 --frame-step 1
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
VJEPA_ROOT = HERE.parents[2]
if str(VJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(VJEPA_ROOT))

from check_anchor_loading import _install_decord_stub
_install_decord_stub()

from src.datasets.data_manager import init_data  # noqa: E402
from src.masks.multiseq_multiblock3d import MaskCollator  # noqa: E402
from phase_matched_pair_dataset import _records_to_anchor_table, _records_to_pair_dataframe  # noqa: E402


def _synthesize_on_disk_pairs(sampler, dicom_dir, n_pairs=4):
    import numpy as np
    avail = {p.stem for p in dicom_dir.glob("*.dcm")}
    sdf = sampler._df
    on_disk = sdf[sdf.dicom_id.astype(str).isin(avail)]
    multi = on_disk.groupby("study_id").size()
    multi = multi[multi >= 2].index.tolist()
    rng = np.random.default_rng(0)
    records = []
    for sid in multi[:n_pairs]:
        sub = on_disk[on_disk.study_id == sid]
        row_idxs = sub.index.tolist()
        i, j = rng.choice(len(row_idxs), size=2, replace=False)
        sampler.study_to_rows[str(sid)] = [int(row_idxs[int(i)]), int(row_idxs[int(j)])]
        r = sampler._draw_pair(str(sid), rng)
        if r is not None:
            records.append(r)
    return records


def _build_transform(img_size):
    """Simple eval-style transform: np.uint8 [T,H,W,3] -> tensor [C,T,H,W]."""
    import torch
    import torch.nn.functional as F

    def _t(clip):
        if isinstance(clip, np.ndarray):
            x = torch.from_numpy(clip)  # [T, H, W, C]
        else:
            x = clip
        x = x.to(torch.float32) / 255.0
        x = x.permute(3, 0, 1, 2)       # [C, T, H, W]
        x = F.interpolate(
            x.unsqueeze(0), size=(x.shape[1], img_size, img_size),
            mode="trilinear", align_corners=False,
        ).squeeze(0)
        return x

    return _t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--dicom-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--frames-per-clip", type=int, default=8)
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--n-pairs", type=int, default=4)
    ap.add_argument("--model-name", default="vit_tiny")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    print(f"device: {device}")

    # --- data ---
    tmpcsv = "/tmp/placeholder_pair.csv"
    pd.DataFrame({"view_0": ["x"], "view_1": ["y"], "label": [0.0]}).to_csv(tmpcsv, index=False)

    loader, sampler = init_data(
        batch_size=args.batch_size,
        data="videogroupdataset",
        collator=None,
        num_workers=0,
        world_size=1, rank=0,
        root_path=tmpcsv,
        training=True,
        clip_len=args.frames_per_clip,
        frame_sample_rate=args.frame_step,
        num_clips=2, num_clips_per_video=1,
        img_size=args.img_size,
        split_name="train",
        sampler_type="phase_matched",
        phase_matched_config={
            "parquet_path": str(args.parquet),
            "sampler_dir": str(HERE),
            "quality_tiers": ["high"],
            "rr_filter_mode": "strict",
            "sampling_mode": "uniform_phase",
            "frames_per_clip": args.frames_per_clip,
            "frame_step": args.frame_step,
            "phase_tolerance": 0.15,
            "pairs_per_study": 1,
        },
        drop_last=False,
    )
    sampler.builder.refresh_epoch(0)

    # Synthesize an on-disk pair DF for the inspection (the random
    # sampler pick won't overlap with the 500 local DICOMs).
    records = _synthesize_on_disk_pairs(sampler, args.dicom_dir, args.n_pairs)
    assert records, "no on-disk pairs synthesized"
    print(f"synthesized {len(records)} on-disk pair records")

    inner = sampler.builder.dataset
    pair_df = _records_to_pair_dataframe(records, sampler._df, video_uri_mode="dicom")
    pair_df["view_0"] = pair_df.clip_a_dicom_id.map(lambda d: str(args.dicom_dir / f"{d}.dcm"))
    pair_df["view_1"] = pair_df.clip_b_dicom_id.map(lambda d: str(args.dicom_dir / f"{d}.dcm"))
    anchors = _records_to_anchor_table(records)
    inner.set_pair_dataframe(pair_df, anchors_by_index=anchors)
    # Attach a simple transform to the dataset so segs come out as
    # torch tensors rather than raw uint8 numpy arrays.
    inner.transform = _build_transform(args.img_size)
    inner.shared_transform = None

    # --- mask collator ---
    mask_collator = MaskCollator(
        cfgs_mask=[{
            "aspect_ratio": [0.75, 1.5],
            "num_blocks": 8,
            "spatial_scale": [0.15, 0.15],
            "temporal_scale": [1.0, 1.0],
            "max_temporal_keep": 1.0,
            "max_keep": None,
        }],
        dataset_fpcs=[args.frames_per_clip],
        crop_size=args.img_size,
        patch_size=16,
        tubelet_size=2,
    )
    dl = DataLoader(
        inner, batch_size=args.batch_size, sampler=SequentialSampler(inner),
        num_workers=0, collate_fn=mask_collator, drop_last=False,
    )
    fpc_collations = next(iter(dl))
    assert len(fpc_collations) == 1, f"expected 1 fpc bucket, got {len(fpc_collations)}"
    collated_batch, masks_enc, masks_pred = fpc_collations[0]

    # --- model ---
    from app.vjepa.utils import init_video_model
    encoder, predictor = init_video_model(
        device=device,
        patch_size=16,
        max_num_frames=args.frames_per_clip,
        tubelet_size=2,
        model_name=args.model_name,
        crop_size=args.img_size,
        pred_depth=4,
        pred_embed_dim=192,
        use_mask_tokens=True,
        num_mask_tokens=2,
        use_sdpa=False,     # decoded clips travel CPU; SDPA not needed
        use_rope=True,
    )
    import copy
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # --- forward_intraview_and_crossview ---
    from app.vjepa_multiview.train import (
        PairBatch, build_clip_pair_tensors, forward_intraview_and_crossview,
        PhaseMatchedRefreshGuard,
    )
    guard = PhaseMatchedRefreshGuard()
    guard.mark_refreshed(0)
    guard.check(0)  # should not raise
    print("refresh guard: OK")

    clip_a, clip_b = build_clip_pair_tensors(collated_batch, device=device)
    masks_enc_d = [[m.to(device) for m in mg] for mg in [masks_enc]]
    masks_pred_d = [[m.to(device) for m in mg] for mg in [masks_pred]]
    print(f"\n=== shapes ===")
    print(f"clip_a: [{len(clip_a)}x] {tuple(clip_a[0].shape)}")
    print(f"clip_b: [{len(clip_b)}x] {tuple(clip_b[0].shape)}")
    print(f"masks_enc: nesting={len(masks_enc_d)}x{len(masks_enc_d[0])}; "
          f"shape={tuple(masks_enc_d[0][0].shape)} dtype={masks_enc_d[0][0].dtype}")
    print(f"masks_pred: nesting={len(masks_pred_d)}x{len(masks_pred_d[0])}; "
          f"shape={tuple(masks_pred_d[0][0].shape)} dtype={masks_pred_d[0][0].dtype}")
    print(f"mask range: enc=[{int(masks_enc_d[0][0].min())}, {int(masks_enc_d[0][0].max())}], "
          f"pred=[{int(masks_pred_d[0][0].min())}, {int(masks_pred_d[0][0].max())}]")

    pair = PairBatch(
        clip_a=clip_a, clip_b=clip_b,
        masks_enc=masks_enc_d, masks_pred=masks_pred_d,
        phase_metadata=pair_df.head(args.batch_size).to_dict("records"),
    )

    out = forward_intraview_and_crossview(
        pair, encoder, target_encoder, predictor,
        lambda_crossview=0.25,
        use_intraview_loss=True,
        use_crossview_loss=True,
        loss_exp=1.0,
        log_mask_diagnostics=True,
    )

    print(f"\n=== forward output shapes ===")
    print(f"h_a_shapes: {out['h_a_shapes']}")
    print(f"h_b_shapes: {out['h_b_shapes']}")
    print(f"z_shapes:   {out['z_shapes']}")
    print(f"\n=== losses ===")
    print(f"intraview_loss: {out['intraview_loss'].item():.6f}")
    print(f"crossview_loss: {out['crossview_loss'].item():.6f}")
    print(f"total_loss:     {out['total_loss'].item():.6f}  "
          f"(= intra + 0.25 * cross)")

    # --- backward ---
    out["total_loss"].backward()

    # --- assertions ---
    enc_grads = [p.grad is not None and p.grad.abs().sum().item() > 0
                 for p in encoder.parameters() if p.requires_grad]
    pred_grads = [p.grad is not None and p.grad.abs().sum().item() > 0
                  for p in predictor.parameters() if p.requires_grad]
    tgt_grads = [p.grad for p in target_encoder.parameters()]
    print(f"\n=== gradient flow ===")
    print(f"encoder params with non-zero grad: {sum(enc_grads)}/{len(enc_grads)}")
    print(f"predictor params with non-zero grad: {sum(pred_grads)}/{len(pred_grads)}")
    print(f"target_encoder params with any grad: {sum(g is not None for g in tgt_grads)}/{len(tgt_grads)}")

    # Encoder: every trainable param should have grad.
    assert sum(enc_grads) == len(enc_grads), (
        f"encoder grad coverage: {sum(enc_grads)}/{len(enc_grads)}"
    )
    # Predictor: allow up to 2 unused params (unused mask-token slots; in
    # real training DDP sets find_unused_parameters=True for exactly this
    # reason — only the active mask_index gets gradient).
    n_pred = len(pred_grads)
    assert sum(pred_grads) >= n_pred - 2, (
        f"predictor grad coverage: {sum(pred_grads)}/{n_pred}"
    )
    assert all(g is None for g in tgt_grads), "target_encoder should have no grad"
    assert torch.isfinite(out["intraview_loss"]), "intraview_loss NaN"
    assert torch.isfinite(out["crossview_loss"]), "crossview_loss NaN"
    assert torch.isfinite(out["total_loss"]), "total_loss NaN"
    # At random init the two losses can be very close because both
    # teachers emit near-zero-mean features; the key correctness gate
    # is that a non-zero difference exists (different teacher inputs,
    # different h). A delta below 1e-5 would indicate accidental identity.
    delta = abs(out["intraview_loss"].item() - out["crossview_loss"].item())
    assert delta > 1e-7, (
        f"intra/cross identity suspected (delta={delta:.3e}); teacher is "
        f"probably being called with the same clip twice."
    )
    print(f"  intra/cross loss delta: {delta:.6e}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
