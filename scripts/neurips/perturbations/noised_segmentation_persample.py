"""
CAMUS noised segmentation with per-sample output for bootstrap CIs.

Same pipeline as noised_segmentation.py but outputs one row per sample
per condition (clean + 3 perturbation types × 3 severities = 10 conditions
× 100 samples = 1000 rows).
"""

import argparse
import csv
import hashlib
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
from scripts.neurips.echo_perturbations import (
    PERTURBATIONS,
    SEVERITY_LEVELS,
    TRANSDUCER_PRESETS,
    apply_perturbation,
    create_scan_mask,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
STRUCTURE_NAMES = {1: "LV", 2: "MYO", 3: "LA"}


def load_encoder(encoder_type, checkpoint, model_name=None, device="cpu", resolution=224):
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


def dice_score(pred, target):
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


def apply_perturbation_to_video(video, perturbation_type, severity, patient_id):
    mean = IMAGENET_MEAN.squeeze().view(3, 1, 1, 1)
    std = IMAGENET_STD.squeeze().view(3, 1, 1, 1)
    pixel = (video * std + mean).clamp(0, 1)
    seed = int(hashlib.md5(patient_id.encode()).hexdigest()[:8], 16)
    mask = create_scan_mask(pixel[:, 0, :, :])
    perturbed = apply_perturbation(
        pixel, perturbation_type, severity, scan_mask=mask, seed=seed,
        transducer_pos=TRANSDUCER_PRESETS["camus"],
    )
    return (perturbed - mean) / std


@torch.no_grad()
def evaluate_persample(encoder, decoder, dataloader, model_type, device,
                       perturbation_type=None, severity=None):
    """Evaluate one condition, returning per-sample Dice scores."""
    results = []
    sample_idx = 0

    for batch in dataloader:
        video = batch["video"]
        ed_mask = batch["ed_mask"]
        es_mask = batch["es_mask"]
        ed_t = batch["ed_temporal_token"]
        es_t = batch["es_temporal_token"]
        patient_ids = batch["patient_id"]

        if perturbation_type is not None:
            perturbed = []
            for i in range(video.shape[0]):
                perturbed.append(apply_perturbation_to_video(
                    video[i], perturbation_type, severity, patient_ids[i],
                ))
            video = torch.stack(perturbed)

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
            row = {"sample_idx": sample_idx + i}
            for c in range(1, NUM_CLASSES):
                name = STRUCTURE_NAMES[c]
                row[f"ed_{name}_dice"] = ed_dice[c]
                row[f"es_{name}_dice"] = es_dice[c]
                row[f"mean_{name}_dice"] = (ed_dice[c] + es_dice[c]) / 2
            row["mean_dice"] = np.mean(
                [row[f"mean_{name}_dice"] for name in STRUCTURE_NAMES.values()]
            )
            results.append(row)

        sample_idx += video.shape[0]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", required=True, choices=["vjepa", "videomae"])
    parser.add_argument("--encoder_checkpoint", required=True)
    parser.add_argument("--encoder_model_name", default=None)
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--camus_root", default="data/camus/CAMUS_public")
    parser.add_argument("--views", nargs="+", default=["4CH", "2CH"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fix_orientation", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--label", default="model")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading encoder: {args.encoder_type} from {args.encoder_checkpoint}")
    encoder, embed_dim = load_encoder(
        args.encoder_type, args.encoder_checkpoint, args.encoder_model_name,
        device, args.resolution,
    )
    encoder.eval()

    model_type = args.encoder_type

    print(f"Loading decoder from {args.decoder_checkpoint}")
    decoder = LinearSegDecoder(embed_dim, num_classes=NUM_CLASSES, target_size=args.resolution)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False))
    decoder = decoder.to(device).eval()

    test_ids = load_split(args.camus_root, "testing")
    ds = CAMUSSegDataset(
        patient_ids=test_ids, camus_root=args.camus_root,
        views=tuple(args.views), resolution=args.resolution,
        num_frames=16, augment=False, fix_orientation=args.fix_orientation,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test set: {len(ds)} samples")

    # Build conditions: clean + 3 types × 3 severities
    conditions = [("clean", None, None)]
    for ptype in PERTURBATIONS:
        for sev in SEVERITY_LEVELS:
            conditions.append((f"{ptype}/{sev}", ptype, sev))

    out_path = f"scripts/neurips/samples/{args.label}_noised_seg_persample.csv"
    fieldnames = (
        ["sample_idx", "condition", "mean_dice"]
        + [f"{p}_{s}_dice" for p in ["ed", "es"] for s in STRUCTURE_NAMES.values()]
        + [f"mean_{s}_dice" for s in STRUCTURE_NAMES.values()]
    )

    all_rows = []
    for cond_name, ptype, sev in conditions:
        print(f"  {cond_name} ...", end=" ", flush=True)
        rows = evaluate_persample(encoder, decoder, dl, model_type, device, ptype, sev)
        for r in rows:
            r["condition"] = cond_name
        all_rows.extend(rows)
        mean_dice = np.mean([r["mean_dice"] for r in rows])
        print(f"mean_dice={mean_dice:.4f}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
