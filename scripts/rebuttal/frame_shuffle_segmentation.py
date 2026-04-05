"""
P1.5b: CAMUS segmentation under frame shuffling.

Tests whether temporal dependence is task-specific: segmentation (per-frame
spatial task) should be invariant to temporal disruption, unlike LVEF.

Design: Run matched_frame shuffling (most rigorous condition) on CAMUS
segmentation using frozen encoder + trained decoder. Compare clean vs shuffled
Dice scores for all 3 models.

Prediction: All models show minimal Dice degradation (<2%) since segmentation
depends on spatial features at a single temporal position, not frame order.

Usage:
    # All 3 models in parallel on separate GPUs:
    python scripts/rebuttal/frame_shuffle_segmentation.py \
        --encoder_type vjepa \
        --encoder_checkpoint checkpoints/echojepa-l-pt50.pt \
        --encoder_model_name vit_large \
        --decoder_checkpoint results/segmentation/echojepa_l_pt50/lr5e-02_wd1e-04/best_decoder.pt \
        --device cuda:0 --label echojepa_l_pt50 &

    python scripts/rebuttal/frame_shuffle_segmentation.py \
        --encoder_type vjepa \
        --encoder_checkpoint checkpoints/byol_vitl_imagenet_v2_e50.pt \
        --encoder_model_name vit_large \
        --decoder_checkpoint results/segmentation/echobyol_l_pt50/lr5e-02_wd1e-04/best_decoder.pt \
        --device cuda:1 --label echobyol_l_pt50 &

    python scripts/rebuttal/frame_shuffle_segmentation.py \
        --encoder_type videomae \
        --encoder_checkpoint checkpoints/videomae_l_mimic_ep50.pth \
        --decoder_checkpoint results/segmentation/echomae_l_pt50/lr1e-02_wd1e-04/best_decoder.pt \
        --device cuda:2 --label echomae_l_pt50 &
"""

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from evals.segmentation_frozen.camus_dataset import (
    CAMUSSegDataset,
    NUM_CLASSES,
    load_split,
)
from evals.segmentation_frozen.eval import (
    LinearSegDecoder,
    extract_spatial_features,
)

STRUCTURE_NAMES = {1: "LV", 2: "MYO", 3: "LA"}


# --- Encoder loading (reused from noised_segmentation.py) ---


def load_encoder(encoder_type, checkpoint, model_name=None, device="cpu", resolution=224):
    """Load frozen encoder. Returns (encoder, embed_dim)."""
    if encoder_type in ("vjepa", "byol"):
        import src.models.vision_transformer as vit
        if model_name is None:
            model_name = "vit_large"
        model = vit.__dict__[model_name](
            img_size=resolution, num_frames=16, patch_size=16, tubelet_size=2,
            uniform_power=True, use_rope=True,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        key = "target_encoder"
        if key not in state:
            for k in ["encoder", "model", "state_dict"]:
                if k in state:
                    key = k
                    break
            else:
                key = None
        sd = state[key] if key else state
        sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
        model_sd = model.state_dict()
        for k in list(sd.keys()):
            if k in model_sd and sd[k].shape != model_sd[k].shape:
                del sd[k]
        model.load_state_dict(sd, strict=False)
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False
        return model, model.embed_dim

    elif encoder_type == "videomae":
        from evals.segmentation_frozen.eval import load_videomae_encoder
        return load_videomae_encoder(checkpoint, device=device)

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


# --- Metrics ---


def dice_score(pred, target):
    """Per-class Dice (excluding background). Returns {class_idx: dice}."""
    scores = {}
    for c in range(1, NUM_CLASSES):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            scores[c] = 1.0 if target_c.sum() == 0 else 0.0
        else:
            scores[c] = (2 * intersection / union).item()
    return scores


# --- Frame shuffling ---


def shuffle_video_frames(video, seed):
    """
    Apply frame-level shuffle to video tensor.

    Uses a fixed permutation (same for all videos) so the encoder cannot
    exploit positional consistency across the batch.

    Args:
        video: [C, T, H, W] tensor
        seed: int seed for reproducible fixed permutation

    Returns:
        [C, T, H, W] with frames permuted
    """
    T = video.shape[1]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(T)
    return video[:, perm, :, :]


# --- Evaluation ---


@torch.no_grad()
def evaluate_condition(encoder, decoder, dataloader, model_type, device,
                       shuffle=False, shuffle_seed=100):
    """Evaluate segmentation on clean or shuffled condition."""
    all_scores = {
        phase: {c: [] for c in range(1, NUM_CLASSES)}
        for phase in ("ed", "es")
    }

    for batch in dataloader:
        video = batch["video"]       # [B, C, T, H, W]
        ed_mask = batch["ed_mask"]
        es_mask = batch["es_mask"]
        ed_t = batch["ed_temporal_token"]
        es_t = batch["es_temporal_token"]

        # Apply frame shuffling per sample
        if shuffle:
            shuffled_videos = []
            for i in range(video.shape[0]):
                shuffled_videos.append(shuffle_video_frames(video[i], shuffle_seed))
            video = torch.stack(shuffled_videos)

        video = video.to(device)
        ed_mask = ed_mask.to(device)
        es_mask = es_mask.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ed_feat = extract_spatial_features(encoder, video, model_type, ed_t, device)
            es_feat = extract_spatial_features(encoder, video, model_type, es_t, device)

        ed_logits = decoder(ed_feat.float())
        es_logits = decoder(es_feat.float())

        ed_pred = ed_logits.argmax(dim=1)
        es_pred = es_logits.argmax(dim=1)

        for i in range(video.shape[0]):
            ed_dice = dice_score(ed_pred[i], ed_mask[i])
            es_dice = dice_score(es_pred[i], es_mask[i])
            for c in range(1, NUM_CLASSES):
                all_scores["ed"][c].append(ed_dice[c])
                all_scores["es"][c].append(es_dice[c])

    results = {}
    for phase in ("ed", "es"):
        for c in range(1, NUM_CLASSES):
            name = STRUCTURE_NAMES[c]
            vals = all_scores[phase][c]
            results[f"{phase}_{name}_dice"] = float(np.mean(vals)) if vals else 0.0

    for c in range(1, NUM_CLASSES):
        name = STRUCTURE_NAMES[c]
        results[f"mean_{name}_dice"] = (results[f"ed_{name}_dice"] + results[f"es_{name}_dice"]) / 2

    results["mean_dice"] = float(np.mean([
        results[f"mean_{name}_dice"] for name in STRUCTURE_NAMES.values()
    ]))
    return results


def main():
    parser = argparse.ArgumentParser(description="P1.5b: CAMUS segmentation under frame shuffling")
    parser.add_argument("--encoder_type", required=True, choices=["vjepa", "byol", "videomae"])
    parser.add_argument("--encoder_checkpoint", required=True)
    parser.add_argument("--encoder_model_name", default=None)
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--camus_root", default="data/camus/CAMUS_public")
    parser.add_argument("--views", nargs="+", default=["4CH", "2CH"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_seeds", type=int, default=3,
                        help="Number of shuffle seeds for mean±std (default: 3)")
    parser.add_argument("--fix_orientation", action="store_true",
                        help="Rotate CAMUS NIfTI to standard A4C orientation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default="model")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load encoder
    print(f"Loading encoder: {args.encoder_type} from {args.encoder_checkpoint}")
    encoder, embed_dim = load_encoder(
        args.encoder_type, args.encoder_checkpoint, args.encoder_model_name,
        device, args.resolution,
    )
    model_type = "vjepa" if args.encoder_type in ("vjepa", "byol") else args.encoder_type

    # Load decoder
    print(f"Loading decoder: {args.decoder_checkpoint}")
    decoder = LinearSegDecoder(embed_dim, NUM_CLASSES, target_size=args.resolution)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False))
    decoder.eval().to(device)

    # Load CAMUS test data
    print(f"Loading CAMUS test split from {args.camus_root}")
    test_ids = load_split(args.camus_root, "testing")
    dataset = CAMUSSegDataset(
        patient_ids=test_ids,
        camus_root=args.camus_root,
        views=tuple(args.views),
        resolution=args.resolution,
        num_frames=16,
        augment=False,
        fix_orientation=args.fix_orientation,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  {len(dataset)} samples ({len(test_ids)} patients × {len(args.views)} views)")

    # --- Clean baseline ---
    print(f"\n  [Clean — original frame order]")
    clean = evaluate_condition(encoder, decoder, dataloader, model_type, device, shuffle=False)
    print(f"    Mean Dice: {clean['mean_dice']:.4f}  "
          f"(LV={clean['mean_LV_dice']:.3f}, MYO={clean['mean_MYO_dice']:.3f}, LA={clean['mean_LA_dice']:.3f})")

    # --- Shuffled × N seeds ---
    shuffled_results = []
    for seed_idx in range(args.n_seeds):
        seed = seed_idx + 100
        print(f"\n  [Shuffled — seed={seed}]")
        result = evaluate_condition(
            encoder, decoder, dataloader, model_type, device,
            shuffle=True, shuffle_seed=seed,
        )
        shuffled_results.append(result)
        print(f"    Mean Dice: {result['mean_dice']:.4f}  "
              f"(LV={result['mean_LV_dice']:.3f}, MYO={result['mean_MYO_dice']:.3f}, "
              f"LA={result['mean_LA_dice']:.3f})")

    # --- Summary ---
    mean_shuffled_dice = float(np.mean([r["mean_dice"] for r in shuffled_results]))
    std_shuffled_dice = float(np.std([r["mean_dice"] for r in shuffled_results]))
    drop = clean["mean_dice"] - mean_shuffled_dice
    pct = (drop / clean["mean_dice"] * 100) if clean["mean_dice"] > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"P1.5b: {args.label} — CAMUS Segmentation Under Frame Shuffling")
    print(f"{'=' * 80}")
    print(f"  Clean Mean Dice:    {clean['mean_dice']:.4f}")
    print(f"  Shuffled Mean Dice: {mean_shuffled_dice:.4f} ± {std_shuffled_dice:.4f}")
    print(f"  Degradation:        {drop:+.4f} ({pct:+.1f}%)")
    print()

    # Per-structure breakdown
    for name in STRUCTURE_NAMES.values():
        key = f"mean_{name}_dice"
        c_val = clean[key]
        s_vals = [r[key] for r in shuffled_results]
        s_mean = float(np.mean(s_vals))
        s_std = float(np.std(s_vals))
        d = c_val - s_mean
        p = (d / c_val * 100) if c_val > 0 else 0
        print(f"  {name}: {c_val:.4f} → {s_mean:.4f}±{s_std:.4f} ({p:+.1f}%)")

    # Save CSV
    if args.output is None:
        args.output = f"scripts/rebuttal/samples/{args.label}_frame_shuffle_segmentation.csv"
    with open(args.output, "w") as f:
        f.write("condition,seed,mean_dice,LV_dice,MYO_dice,LA_dice,"
                "ed_LV_dice,ed_MYO_dice,ed_LA_dice,es_LV_dice,es_MYO_dice,es_LA_dice\n")
        # Clean
        m = clean
        f.write(f"clean,0,{m['mean_dice']:.6f},"
                f"{m['mean_LV_dice']:.6f},{m['mean_MYO_dice']:.6f},{m['mean_LA_dice']:.6f},"
                f"{m['ed_LV_dice']:.6f},{m['ed_MYO_dice']:.6f},{m['ed_LA_dice']:.6f},"
                f"{m['es_LV_dice']:.6f},{m['es_MYO_dice']:.6f},{m['es_LA_dice']:.6f}\n")
        # Shuffled
        for seed_idx, m in enumerate(shuffled_results):
            seed = seed_idx + 100
            f.write(f"shuffled,{seed},{m['mean_dice']:.6f},"
                    f"{m['mean_LV_dice']:.6f},{m['mean_MYO_dice']:.6f},{m['mean_LA_dice']:.6f},"
                    f"{m['ed_LV_dice']:.6f},{m['ed_MYO_dice']:.6f},{m['ed_LA_dice']:.6f},"
                    f"{m['es_LV_dice']:.6f},{m['es_MYO_dice']:.6f},{m['es_LA_dice']:.6f}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
