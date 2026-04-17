"""
CAMUS segmentation frame shuffling with per-sample output for bootstrap CIs.

Outputs one row per sample (n=100) per condition, enabling paired bootstrap
over 100 test samples rather than 3 seed-level aggregates.

Modes:
  --mode severity  : fractions [0.0, 0.25, 0.50, 0.75, 1.0] × 3 seeds
  --mode 6cond     : clean, reverse, tubelet, matched, shuffle, matched_frame
"""

import argparse
import hashlib
import json
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
SEVERITY_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
CONDITIONS_6 = ["clean", "reverse", "tubelet", "matched", "shuffle", "matched_frame"]


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


def partial_shuffle(video, fraction, rng):
    T = video.shape[1]
    k = int(round(T * fraction))
    if k < 2:
        return video
    positions = sorted(rng.choice(T, size=k, replace=False).tolist())
    shuffled_values = rng.permutation(positions).tolist()
    indices = list(range(T))
    for orig_pos, new_val in zip(positions, shuffled_values):
        indices[orig_pos] = new_val
    return video[:, indices, :, :]


def apply_temporal_ablation(video, condition, seed, sample_idx=0):
    T = video.shape[1]
    video_hash = int(hashlib.md5(str(sample_idx).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(video_hash + seed)

    if condition == "clean":
        return video
    elif condition == "reverse":
        return video[:, torch.arange(T - 1, -1, -1), :, :]
    elif condition == "tubelet":
        n_tubelets = T // 2
        tubelet_perm = rng.permutation(n_tubelets)
        indices = []
        for ti in tubelet_perm:
            indices.extend([ti * 2, ti * 2 + 1])
        if T % 2 == 1:
            indices.append(T - 1)
        return video[:, indices, :, :]
    elif condition == "matched":
        n_tubelets = T // 2
        fixed_rng = np.random.RandomState(seed)
        tubelet_perm = fixed_rng.permutation(n_tubelets)
        indices = []
        for ti in tubelet_perm:
            indices.extend([ti * 2, ti * 2 + 1])
        if T % 2 == 1:
            indices.append(T - 1)
        return video[:, indices, :, :]
    elif condition == "shuffle":
        perm = rng.permutation(T)
        return video[:, perm, :, :]
    elif condition == "matched_frame":
        fixed_rng = np.random.RandomState(seed)
        frame_perm = fixed_rng.permutation(T)
        return video[:, frame_perm, :, :]
    else:
        raise ValueError(f"Unknown condition: {condition}")


@torch.no_grad()
def evaluate_persample(encoder, decoder, dataloader, model_type, device,
                       ablation_fn):
    """
    Run evaluation, returning per-sample Dice scores.

    ablation_fn: callable(video_i, sample_idx) -> ablated video_i
    Returns list of dicts, one per sample.
    """
    results = []
    sample_idx = 0

    for batch in dataloader:
        video = batch["video"]
        ed_mask = batch["ed_mask"]
        es_mask = batch["es_mask"]
        ed_t = batch["ed_temporal_token"]
        es_t = batch["es_temporal_token"]

        # Apply ablation per sample
        ablated = []
        for i in range(video.shape[0]):
            ablated.append(ablation_fn(video[i], sample_idx + i))
        video = torch.stack(ablated)
        batch_start = sample_idx
        sample_idx += video.shape[0]

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
            row = {"sample_idx": batch_start + i}
            for c in range(1, NUM_CLASSES):
                name = STRUCTURE_NAMES[c]
                row[f"ed_{name}_dice"] = ed_dice[c]
                row[f"es_{name}_dice"] = es_dice[c]
                row[f"mean_{name}_dice"] = (ed_dice[c] + es_dice[c]) / 2
            row["mean_dice"] = np.mean([row[f"mean_{name}_dice"] for name in STRUCTURE_NAMES.values()])
            results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_type", required=True, choices=["vjepa", "byol", "videomae"])
    parser.add_argument("--encoder_checkpoint", required=True)
    parser.add_argument("--encoder_model_name", default=None)
    parser.add_argument("--decoder_checkpoint", required=True)
    parser.add_argument("--mode", required=True, choices=["severity", "6cond"])
    parser.add_argument("--camus_root", default="data/camus/CAMUS_public")
    parser.add_argument("--views", nargs="+", default=["4CH", "2CH"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--fix_orientation", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default="model")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading encoder: {args.encoder_type} from {args.encoder_checkpoint}")
    encoder, embed_dim = load_encoder(
        args.encoder_type, args.encoder_checkpoint, args.encoder_model_name,
        device, args.resolution,
    )
    model_type = "vjepa" if args.encoder_type in ("vjepa", "byol") else args.encoder_type

    print(f"Loading decoder: {args.decoder_checkpoint}")
    decoder = LinearSegDecoder(embed_dim, NUM_CLASSES, target_size=args.resolution)
    decoder.load_state_dict(torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False))
    decoder.eval().to(device)

    print(f"Loading CAMUS test split from {args.camus_root}")
    test_ids = load_split(args.camus_root, "testing")
    dataset = CAMUSSegDataset(
        patient_ids=test_ids, camus_root=args.camus_root,
        views=tuple(args.views), resolution=args.resolution,
        num_frames=16, augment=False, fix_orientation=args.fix_orientation,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    n_samples = len(dataset)
    print(f"  {n_samples} samples ({len(test_ids)} patients × {len(args.views)} views)")

    if args.mode == "severity":
        run_severity_persample(encoder, decoder, dataloader, model_type, device, args, n_samples)
    else:
        run_6cond_persample(encoder, decoder, dataloader, model_type, device, args, n_samples)


def run_severity_persample(encoder, decoder, dataloader, model_type, device, args, n_samples):
    if args.output is None:
        args.output = f"scripts/neurips/samples/{args.label}_camus_severity_persample.csv"

    all_rows = []
    for frac in SEVERITY_FRACTIONS:
        for seed_idx in range(args.n_seeds):
            seed = seed_idx + 100
            rng = np.random.RandomState(seed)

            if frac == 0.0:
                ablation_fn = lambda v, idx: v
            else:
                # Capture frac and seed in closure
                def make_ablation(f, s):
                    _rng = np.random.RandomState(s)
                    def fn(v, idx):
                        sample_rng = np.random.RandomState(_rng.randint(0, 2**31))
                        return partial_shuffle(v, f, sample_rng)
                    return fn
                ablation_fn = make_ablation(frac, seed)

            results = evaluate_persample(encoder, decoder, dataloader, model_type, device, ablation_fn)
            for r in results:
                r["fraction"] = frac
                r["seed"] = seed
            all_rows.extend(results)

            mean_dice = np.mean([r["mean_dice"] for r in results])
            print(f"  frac={frac:.2f} seed={seed}: mean_dice={mean_dice:.4f} (n={len(results)})")

    # Write CSV
    cols = ["fraction", "seed", "sample_idx", "mean_dice",
            "mean_LV_dice", "mean_MYO_dice", "mean_LA_dice",
            "ed_LV_dice", "ed_MYO_dice", "ed_LA_dice",
            "es_LV_dice", "es_MYO_dice", "es_LA_dice"]
    with open(args.output, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"\nSaved {len(all_rows)} rows to {args.output}")


def run_6cond_persample(encoder, decoder, dataloader, model_type, device, args, n_samples):
    if args.output is None:
        args.output = f"scripts/neurips/samples/{args.label}_camus_6cond_persample.csv"

    deterministic = {"clean", "reverse"}
    all_rows = []

    for cond in CONDITIONS_6:
        n_runs = 1 if cond in deterministic else args.n_seeds
        for seed_idx in range(n_runs):
            seed = seed_idx + 100

            if cond == "clean":
                ablation_fn = lambda v, idx: v
            else:
                def make_ablation(c, s):
                    def fn(v, idx):
                        return apply_temporal_ablation(v, c, s, sample_idx=idx)
                    return fn
                ablation_fn = make_ablation(cond, seed)

            results = evaluate_persample(encoder, decoder, dataloader, model_type, device, ablation_fn)
            for r in results:
                r["condition"] = cond
                r["seed"] = seed
            all_rows.extend(results)

            mean_dice = np.mean([r["mean_dice"] for r in results])
            print(f"  {cond} seed={seed}: mean_dice={mean_dice:.4f} (n={len(results)})")

    # Write CSV
    cols = ["condition", "seed", "sample_idx", "mean_dice",
            "mean_LV_dice", "mean_MYO_dice", "mean_LA_dice",
            "ed_LV_dice", "ed_MYO_dice", "ed_LA_dice",
            "es_LV_dice", "es_MYO_dice", "es_LA_dice"]
    with open(args.output, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in all_rows:
            f.write(",".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"\nSaved {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
