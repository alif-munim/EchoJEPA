"""
P0.6 (CAMUS): Noised segmentation inference — Dice degradation under echo perturbations.

Loads a frozen encoder + trained segmentation decoder ONCE, then evaluates on
clean and perturbed CAMUS test videos (3 perturbation types × 3 severities + clean).

Perturbations are applied on-the-fly to the [0,1] video tensor before ImageNet
normalization. Each video gets a deterministic perturbation seeded by patient ID.

Usage:
    TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH python scripts/rebuttal/noised_segmentation.py \
        --encoder_type vjepa \
        --encoder_checkpoint checkpoints/echojepa-l-pt50.pt \
        --encoder_model_name vit_large \
        --decoder_checkpoint results/segmentation/echojepa_l_pt50/lr5e-02_wd1e-04/best_decoder.pt \
        --device cuda:0 \
        --label echojepa_l_pt50

    # Run all 3 pt50 models in parallel:
    CUDA_VISIBLE_DEVICES=0 python scripts/rebuttal/noised_segmentation.py \
        --encoder_type vjepa --encoder_checkpoint checkpoints/echojepa-l-pt50.pt \
        --encoder_model_name vit_large \
        --decoder_checkpoint results/segmentation/echojepa_l_pt50/lr5e-02_wd1e-04/best_decoder.pt \
        --device cuda:0 --label echojepa_l_pt50 &

    CUDA_VISIBLE_DEVICES=1 python scripts/rebuttal/noised_segmentation.py \
        --encoder_type vjepa --encoder_checkpoint checkpoints/byol_vitl_imagenet_v2_e50.pt \
        --encoder_model_name vit_large \
        --decoder_checkpoint results/segmentation/echobyol_l_pt50/lr5e-02_wd1e-04/best_decoder.pt \
        --device cuda:0 --label echobyol_l_pt50 &

    CUDA_VISIBLE_DEVICES=2 python scripts/rebuttal/noised_segmentation.py \
        --encoder_type videomae --encoder_checkpoint checkpoints/videomae_l_mimic_ep50.pth \
        --decoder_checkpoint results/segmentation/echomae_l_pt50/lr1e-02_wd1e-04/best_decoder.pt \
        --device cuda:0 --label echomae_l_pt50 &
"""

import argparse
import hashlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from evals.segmentation_frozen.camus_dataset import (
    CAMUSSegDataset,
    LABEL_NAMES,
    NUM_CLASSES,
    load_split,
)
from evals.segmentation_frozen.eval import (
    LinearSegDecoder,
    extract_spatial_features,
)
from scripts.rebuttal.echo_perturbations import (
    PERTURBATIONS,
    SEVERITY_LEVELS,
    TRANSDUCER_PRESETS,
    apply_perturbation,
    create_scan_mask,
)

# ImageNet normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

STRUCTURE_NAMES = {1: "LV", 2: "MYO", 3: "LA"}


# --- Encoder loading (reused from noised_inference.py) ---

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


# --- Perturbation wrapper for CAMUS ---

def apply_perturbation_to_video(video, perturbation_type, severity, patient_id):
    """
    Apply perturbation to a CAMUS video tensor.

    The CAMUS dataset returns video already ImageNet-normalized as [C, T, H, W].
    We need to un-normalize → perturb in [0,1] space → re-normalize.

    Args:
        video: [C, T, H, W] ImageNet-normalized tensor
        perturbation_type: str
        severity: str
        patient_id: str (used as seed for deterministic perturbation)
    Returns:
        [C, T, H, W] perturbed and re-normalized tensor
    """
    # Un-normalize: pixel = normalized * std + mean
    # IMAGENET_MEAN/STD are [3, 1, 1, 1], broadcast over [C, T, H, W]
    mean = IMAGENET_MEAN.squeeze()  # [3, 1, 1, 1] -> [3]
    std = IMAGENET_STD.squeeze()
    mean = mean.view(3, 1, 1, 1)  # [3, 1, 1, 1] for [C, T, H, W]
    std = std.view(3, 1, 1, 1)
    pixel = video * std + mean  # [C, T, H, W] in ~[0, 1]
    pixel = pixel.clamp(0, 1)

    # Apply perturbation in pixel space
    seed = int(hashlib.md5(patient_id.encode()).hexdigest()[:8], 16)
    mask = create_scan_mask(pixel[:, 0, :, :])
    perturbed = apply_perturbation(
        pixel, perturbation_type, severity, scan_mask=mask, seed=seed,
        transducer_pos=TRANSDUCER_PRESETS["camus"],
    )

    # Re-normalize
    return (perturbed - mean) / std


# --- Main evaluation ---

@torch.no_grad()
def evaluate_condition(encoder, decoder, dataloader, model_type, device,
                       perturbation_type=None, severity=None):
    """
    Evaluate segmentation on one condition (clean or perturbed).

    Returns dict with per-structure, per-phase, and mean Dice scores.
    """
    # Collect per-sample scores: {phase: {class: [dice_values]}}
    all_scores = {
        phase: {c: [] for c in range(1, NUM_CLASSES)}
        for phase in ("ed", "es")
    }

    for batch in dataloader:
        video = batch["video"]       # [B, C, T, H, W]
        ed_mask = batch["ed_mask"]   # [B, H, W]
        es_mask = batch["es_mask"]   # [B, H, W]
        ed_t = batch["ed_temporal_token"]  # [B]
        es_t = batch["es_temporal_token"]  # [B]
        patient_ids = batch["patient_id"]  # list of str

        # Apply perturbation per-sample (different seed per patient)
        if perturbation_type is not None:
            perturbed_videos = []
            for i in range(video.shape[0]):
                pv = apply_perturbation_to_video(
                    video[i], perturbation_type, severity, patient_ids[i]
                )
                perturbed_videos.append(pv)
            video = torch.stack(perturbed_videos)

        video = video.to(device)
        ed_mask = ed_mask.to(device)
        es_mask = es_mask.to(device)

        # Extract spatial features at ED and ES temporal positions
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ed_feat = extract_spatial_features(encoder, video, model_type, ed_t, device)
            es_feat = extract_spatial_features(encoder, video, model_type, es_t, device)

        # Decode
        ed_logits = decoder(ed_feat.float())  # [B, C, H, W]
        es_logits = decoder(es_feat.float())

        ed_pred = ed_logits.argmax(dim=1)  # [B, H, W]
        es_pred = es_logits.argmax(dim=1)

        # Compute Dice per sample
        for i in range(video.shape[0]):
            ed_dice = dice_score(ed_pred[i], ed_mask[i])
            es_dice = dice_score(es_pred[i], es_mask[i])
            for c in range(1, NUM_CLASSES):
                all_scores["ed"][c].append(ed_dice[c])
                all_scores["es"][c].append(es_dice[c])

    # Aggregate
    results = {}
    for phase in ("ed", "es"):
        for c in range(1, NUM_CLASSES):
            name = STRUCTURE_NAMES[c]
            vals = all_scores[phase][c]
            results[f"{phase}_{name}_dice"] = float(np.mean(vals)) if vals else 0.0

    # Mean across phases
    for c in range(1, NUM_CLASSES):
        name = STRUCTURE_NAMES[c]
        results[f"mean_{name}_dice"] = (results[f"ed_{name}_dice"] + results[f"es_{name}_dice"]) / 2

    results["mean_dice"] = float(np.mean([results[f"mean_{name}_dice"] for name in STRUCTURE_NAMES.values()]))

    return results


def main():
    parser = argparse.ArgumentParser(description="P0.6: Noised CAMUS segmentation inference")
    # Encoder
    parser.add_argument("--encoder_type", required=True, choices=["vjepa", "byol", "videomae"])
    parser.add_argument("--encoder_checkpoint", required=True)
    parser.add_argument("--encoder_model_name", default=None,
                        help="vit_large, vit_giant_xformers, etc. (auto-detected if None)")
    # Decoder
    parser.add_argument("--decoder_checkpoint", required=True,
                        help="Path to best_decoder.pt")
    # Data
    parser.add_argument("--camus_root", default="data/camus/CAMUS_public")
    parser.add_argument("--views", nargs="+", default=["4CH", "2CH"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    # Perturbation control
    parser.add_argument("--perturbation_types", nargs="*", default=None,
                        help="Perturbation types (default: all)")
    parser.add_argument("--severity_levels", nargs="*", default=None,
                        help="Severity levels (default: all)")
    parser.add_argument("--skip_clean", action="store_true")
    parser.add_argument("--fix_orientation", action="store_true",
                        help="Rotate CAMUS NIfTI to standard A4C orientation (rot90 CCW + flip-H)")
    # Compute
    parser.add_argument("--device", default="cuda:0")
    # Output
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default="model")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load encoder ---
    print(f"Loading encoder: {args.encoder_type} from {args.encoder_checkpoint}")
    encoder, embed_dim = load_encoder(
        args.encoder_type, args.encoder_checkpoint, args.encoder_model_name,
        device, args.resolution,
    )
    print(f"  embed_dim={embed_dim}")

    # Determine model_type for spatial feature extraction
    model_type = args.encoder_type
    if model_type == "byol":
        model_type = "vjepa"  # same architecture

    # --- Load decoder ---
    print(f"Loading decoder: {args.decoder_checkpoint}")
    decoder = LinearSegDecoder(embed_dim, NUM_CLASSES, target_size=args.resolution)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False))
    decoder.eval().to(device)

    # --- Load CAMUS test data ---
    print(f"Loading CAMUS test split from {args.camus_root}")
    test_ids = load_split(args.camus_root, "testing")
    print(f"  {len(test_ids)} test patients, views: {args.views}")

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

    # --- Build condition list ---
    ptypes = args.perturbation_types if args.perturbation_types else list(PERTURBATIONS.keys())
    severities = args.severity_levels if args.severity_levels else SEVERITY_LEVELS

    conditions = []
    if not args.skip_clean:
        conditions.append(("clean", None, None))
    for ptype in ptypes:
        for severity in severities:
            conditions.append((f"{ptype}/{severity}", ptype, severity))

    print(f"  Conditions: {len(conditions)}")

    # --- Run all conditions ---
    results = {}
    for cond_name, ptype, severity in conditions:
        print(f"\n  [{cond_name}]")
        metrics = evaluate_condition(
            encoder, decoder, dataloader, model_type, device, ptype, severity,
        )
        results[cond_name] = metrics
        print(f"    Mean Dice: {metrics['mean_dice']:.4f}  "
              f"(LV={metrics['mean_LV_dice']:.3f}, MYO={metrics['mean_MYO_dice']:.3f}, LA={metrics['mean_LA_dice']:.3f})")

    # --- Summary table ---
    print(f"\n{'=' * 90}")
    print(f"RESULTS: {args.label} — CAMUS Segmentation")
    print(f"{'=' * 90}")
    header = f"{'Condition':<30} {'Mean Dice':<12} {'LV':<10} {'MYO':<10} {'LA':<10}"
    print(header)
    print("-" * 90)
    for cond_name, metrics in results.items():
        print(f"{cond_name:<30} {metrics['mean_dice']:<12.4f} "
              f"{metrics['mean_LV_dice']:<10.4f} {metrics['mean_MYO_dice']:<10.4f} {metrics['mean_LA_dice']:<10.4f}")

    # Degradation summary
    if "clean" in results:
        clean_dice = results["clean"]["mean_dice"]
        worst_sev = severities[-1]
        has_severe = any(f"{p}/{worst_sev}" in results for p in ptypes)
        if has_severe:
            print(f"\n{'=' * 90}")
            print(f"DEGRADATION (clean → {worst_sev})")
            print(f"{'=' * 90}")
            for ptype in ptypes:
                key = f"{ptype}/{worst_sev}"
                if key in results:
                    severe_dice = results[key]["mean_dice"]
                    drop = clean_dice - severe_dice
                    pct = (drop / clean_dice * 100) if clean_dice > 0 else 0
                    print(f"  {ptype:<25s}: {clean_dice:.4f} → {severe_dice:.4f}  "
                          f"(Δ={drop:+.4f}, {pct:+.1f}%)")

    # Save CSV
    if args.output is None:
        args.output = f"scripts/rebuttal/samples/{args.label}_noised_segmentation.csv"
    with open(args.output, "w") as f:
        f.write("condition,mean_dice,LV_dice,MYO_dice,LA_dice,"
                "ed_LV_dice,ed_MYO_dice,ed_LA_dice,es_LV_dice,es_MYO_dice,es_LA_dice\n")
        for cond_name, m in results.items():
            f.write(f"{cond_name},{m['mean_dice']:.6f},"
                    f"{m['mean_LV_dice']:.6f},{m['mean_MYO_dice']:.6f},{m['mean_LA_dice']:.6f},"
                    f"{m['ed_LV_dice']:.6f},{m['ed_MYO_dice']:.6f},{m['ed_LA_dice']:.6f},"
                    f"{m['es_LV_dice']:.6f},{m['es_MYO_dice']:.6f},{m['es_LA_dice']:.6f}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
