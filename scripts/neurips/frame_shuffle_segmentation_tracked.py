"""
CAMUS segmentation frame shuffling — Version B: TRACKED extraction.

After shuffling, extract features at the positions where ED/ES content
actually landed (via inverse permutation), rather than at the original
fixed positions. The encoder still sees the shuffled video (disrupted
temporal attention + mismatched positional encodings), but the decoder
segments the correct cardiac phase content.

Comparing Version A (original positions) vs Version B (tracked positions)
isolates content misalignment from temporal encoding disruption.

Conditions: clean, shuffle, matched_frame only (2 shuffled × 3 seeds).
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

STRUCTURE_NAMES = {1: "LV", 2: "MYO", 3: "LA"}
CONDITIONS = ["clean", "shuffle", "matched_frame"]


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


def apply_shuffle_tracked(video, condition, seed, sample_idx=0):
    """Apply shuffle and return (shuffled_video, frame_permutation).

    Permutation semantics: output[:, i] = input[:, perm[i]].
    So perm[i] = which original frame is now at position i.
    """
    T = video.shape[1]

    if condition == "clean":
        return video, np.arange(T)
    elif condition == "shuffle":
        # Per-video random — same RNG as original experiment
        video_hash = int(hashlib.md5(str(sample_idx).encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(video_hash + seed)
        perm = rng.permutation(T)
    elif condition == "matched_frame":
        # Fixed perm across all videos — same RNG as original experiment
        fixed_rng = np.random.RandomState(seed)
        perm = fixed_rng.permutation(T)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return video[:, perm, :, :], perm


@torch.no_grad()
def evaluate_tracked(encoder, decoder, dataloader, model_type, device,
                     condition, seed):
    """Evaluate with tracked extraction positions.

    After shuffling, compute inverse permutation to find where ED/ES content
    actually landed, and extract features from those new positions.
    """
    results = []
    sample_idx = 0

    for batch in dataloader:
        video = batch["video"]
        ed_mask = batch["ed_mask"]
        es_mask = batch["es_mask"]
        ed_frame = batch["ed_sampled_idx"]   # Frame-level index (0-15)
        es_frame = batch["es_sampled_idx"]

        ablated = []
        tracked_ed_t = []
        tracked_es_t = []

        for i in range(video.shape[0]):
            shuf_vid, perm = apply_shuffle_tracked(
                video[i], condition, seed, sample_idx + i,
            )
            ablated.append(shuf_vid)

            # Inverse perm: inv[j] = position where original frame j ended up
            inv_perm = np.argsort(perm)
            new_ed_frame = int(inv_perm[ed_frame[i].item()])
            new_es_frame = int(inv_perm[es_frame[i].item()])
            tracked_ed_t.append(new_ed_frame // 2)  # Frame idx → tubelet idx
            tracked_es_t.append(new_es_frame // 2)

        video = torch.stack(ablated).to(device)
        ed_mask = ed_mask.to(device)
        es_mask = es_mask.to(device)
        tracked_ed_t = torch.tensor(tracked_ed_t)
        tracked_es_t = torch.tensor(tracked_es_t)

        batch_start = sample_idx
        sample_idx += video.shape[0]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ed_feat = extract_spatial_features(
                encoder, video, model_type, tracked_ed_t, device,
            )
            es_feat = extract_spatial_features(
                encoder, video, model_type, tracked_es_t, device,
            )

        ed_logits = decoder(ed_feat.float())
        es_logits = decoder(es_feat.float())
        ed_pred = ed_logits.argmax(dim=1)
        es_pred = es_logits.argmax(dim=1)

        for i in range(video.shape[0]):
            ed_dice = dice_score(ed_pred[i], ed_mask[i])
            es_dice = dice_score(es_pred[i], es_mask[i])
            row = {
                "sample_idx": batch_start + i,
                "tracked_ed_t": tracked_ed_t[i].item(),
                "tracked_es_t": tracked_es_t[i].item(),
            }
            for c in range(1, NUM_CLASSES):
                name = STRUCTURE_NAMES[c]
                row[f"ed_{name}_dice"] = ed_dice[c]
                row[f"es_{name}_dice"] = es_dice[c]
                row[f"mean_{name}_dice"] = (ed_dice[c] + es_dice[c]) / 2
            row["mean_dice"] = np.mean(
                [row[f"mean_{name}_dice"] for name in STRUCTURE_NAMES.values()]
            )
            results.append(row)

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
    parser.add_argument("--n_seeds", type=int, default=3)
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

    print(f"Loading decoder from {args.decoder_checkpoint}")
    decoder = LinearSegDecoder(embed_dim, num_classes=NUM_CLASSES, target_size=args.resolution)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location="cpu"))
    decoder = decoder.to(device).eval()

    test_ids = load_split(args.camus_root, "testing")
    ds = CAMUSSegDataset(
        patient_ids=test_ids, camus_root=args.camus_root,
        views=tuple(args.views), resolution=args.resolution,
        num_frames=16, augment=False, fix_orientation=args.fix_orientation,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test set: {len(ds)} samples")

    out_path = f"scripts/neurips/samples/{args.label}_camus_tracked_persample.csv"

    fieldnames = (
        ["sample_idx", "condition", "seed", "tracked_ed_t", "tracked_es_t", "mean_dice"]
        + [f"{p}_{s}_dice" for p in ["ed", "es"] for s in STRUCTURE_NAMES.values()]
        + [f"mean_{s}_dice" for s in STRUCTURE_NAMES.values()]
    )

    all_rows = []
    for cond in CONDITIONS:
        seeds = [0] if cond == "clean" else list(range(args.n_seeds))
        for seed in seeds:
            print(f"  {cond} seed={seed} ...", end=" ", flush=True)
            rows = evaluate_tracked(
                encoder, decoder, dl, args.encoder_type, device, cond, seed,
            )
            for r in rows:
                r["condition"] = cond
                r["seed"] = seed
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
