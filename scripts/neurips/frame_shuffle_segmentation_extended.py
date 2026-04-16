"""
Extended CAMUS segmentation frame shuffling: severity gradient + 6-condition.

Extends the basic full-shuffle experiment with:
  --mode severity  : partial shuffle at fractions [0.0, 0.25, 0.50, 0.75, 1.0]
  --mode 6cond     : clean, reverse, tubelet, matched, shuffle, matched_frame

Usage:
    python scripts/neurips/frame_shuffle_segmentation_extended.py \
        --encoder_type vjepa --encoder_checkpoint CKPT \
        --encoder_model_name vit_large --decoder_checkpoint DEC \
        --device cuda:0 --label jepa_in21k_e100 --mode severity

    python scripts/neurips/frame_shuffle_segmentation_extended.py \
        --encoder_type vjepa --encoder_checkpoint CKPT \
        --encoder_model_name vit_large --decoder_checkpoint DEC \
        --device cuda:0 --label jepa_in21k_e100 --mode 6cond
"""

import argparse
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
SEVERITY_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
CONDITIONS_6 = ["clean", "reverse", "tubelet", "matched", "shuffle", "matched_frame"]


# --- Encoder loading ---

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


# --- Temporal ablation functions ---

def partial_shuffle(video, fraction, rng):
    """Shuffle a fraction of frames, keeping the rest in original order."""
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
    """Apply one of 6 temporal ablation conditions."""
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


# --- Evaluation ---

@torch.no_grad()
def evaluate_severity(encoder, decoder, dataloader, model_type, device,
                      fraction, seed):
    """Evaluate with partial frame shuffling at given severity fraction."""
    all_scores = {
        phase: {c: [] for c in range(1, NUM_CLASSES)}
        for phase in ("ed", "es")
    }
    rng = np.random.RandomState(seed)

    for batch in dataloader:
        video = batch["video"]
        ed_mask = batch["ed_mask"]
        es_mask = batch["es_mask"]
        ed_t = batch["ed_temporal_token"]
        es_t = batch["es_temporal_token"]

        if fraction > 0:
            shuffled = []
            for i in range(video.shape[0]):
                # Per-sample RNG for reproducibility
                sample_rng = np.random.RandomState(rng.randint(0, 2**31))
                shuffled.append(partial_shuffle(video[i], fraction, sample_rng))
            video = torch.stack(shuffled)

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

    return _aggregate_scores(all_scores)


@torch.no_grad()
def evaluate_condition(encoder, decoder, dataloader, model_type, device,
                       condition, seed):
    """Evaluate with a specific 6-condition ablation."""
    all_scores = {
        phase: {c: [] for c in range(1, NUM_CLASSES)}
        for phase in ("ed", "es")
    }
    sample_counter = 0

    for batch in dataloader:
        video = batch["video"]
        ed_mask = batch["ed_mask"]
        es_mask = batch["es_mask"]
        ed_t = batch["ed_temporal_token"]
        es_t = batch["es_temporal_token"]

        if condition != "clean":
            ablated = []
            for i in range(video.shape[0]):
                ablated.append(apply_temporal_ablation(
                    video[i], condition, seed, sample_idx=sample_counter + i
                ))
            video = torch.stack(ablated)
        sample_counter += video.shape[0]

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

    return _aggregate_scores(all_scores)


def _aggregate_scores(all_scores):
    """Aggregate per-sample scores into mean metrics."""
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
        patient_ids=test_ids, camus_root=args.camus_root,
        views=tuple(args.views), resolution=args.resolution,
        num_frames=16, augment=False, fix_orientation=args.fix_orientation,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"  {len(dataset)} samples ({len(test_ids)} patients × {len(args.views)} views)")

    if args.mode == "severity":
        run_severity(encoder, decoder, dataloader, model_type, device, args)
    else:
        run_6cond(encoder, decoder, dataloader, model_type, device, args)


def run_severity(encoder, decoder, dataloader, model_type, device, args):
    """Severity gradient: partial shuffle at 0/25/50/75/100%."""
    if args.output is None:
        args.output = f"scripts/neurips/samples/{args.label}_camus_severity.csv"

    rows = []
    print(f"\n{'='*70}")
    print(f"Severity gradient: {args.label}")
    print(f"{'='*70}")

    for frac in SEVERITY_FRACTIONS:
        pct = int(frac * 100)
        seed_results = []
        for seed_idx in range(args.n_seeds):
            seed = seed_idx + 100
            result = evaluate_severity(encoder, decoder, dataloader, model_type, device, frac, seed)
            seed_results.append(result)
            rows.append({"fraction": frac, "seed": seed, **result})

        mean_dice = np.mean([r["mean_dice"] for r in seed_results])
        std_dice = np.std([r["mean_dice"] for r in seed_results])
        lv = np.mean([r["mean_LV_dice"] for r in seed_results])
        myo = np.mean([r["mean_MYO_dice"] for r in seed_results])
        la = np.mean([r["mean_LA_dice"] for r in seed_results])
        print(f"  {pct:3d}% shuffled: Dice={mean_dice:.4f}±{std_dice:.4f}  "
              f"(LV={lv:.3f}, MYO={myo:.3f}, LA={la:.3f})")

    # Save CSV
    with open(args.output, "w") as f:
        f.write("fraction,seed,mean_dice,LV_dice,MYO_dice,LA_dice,"
                "ed_LV_dice,ed_MYO_dice,ed_LA_dice,es_LV_dice,es_MYO_dice,es_LA_dice\n")
        for r in rows:
            f.write(f"{r['fraction']:.2f},{r['seed']},{r['mean_dice']:.6f},"
                    f"{r['mean_LV_dice']:.6f},{r['mean_MYO_dice']:.6f},{r['mean_LA_dice']:.6f},"
                    f"{r['ed_LV_dice']:.6f},{r['ed_MYO_dice']:.6f},{r['ed_LA_dice']:.6f},"
                    f"{r['es_LV_dice']:.6f},{r['es_MYO_dice']:.6f},{r['es_LA_dice']:.6f}\n")
    print(f"\nSaved: {args.output}")


def run_6cond(encoder, decoder, dataloader, model_type, device, args):
    """6-condition temporal ablation."""
    if args.output is None:
        args.output = f"scripts/neurips/samples/{args.label}_camus_6cond.csv"

    rows = []
    print(f"\n{'='*70}")
    print(f"6-condition ablation: {args.label}")
    print(f"{'='*70}")

    # Deterministic conditions: single run
    deterministic = {"clean", "reverse"}

    for cond in CONDITIONS_6:
        if cond in deterministic:
            n_runs = 1
        else:
            n_runs = args.n_seeds

        seed_results = []
        for seed_idx in range(n_runs):
            seed = seed_idx + 100
            result = evaluate_condition(encoder, decoder, dataloader, model_type, device, cond, seed)
            seed_results.append(result)
            rows.append({"condition": cond, "seed": seed, **result})

        mean_dice = np.mean([r["mean_dice"] for r in seed_results])
        std_dice = np.std([r["mean_dice"] for r in seed_results]) if n_runs > 1 else 0.0
        lv = np.mean([r["mean_LV_dice"] for r in seed_results])
        myo = np.mean([r["mean_MYO_dice"] for r in seed_results])
        la = np.mean([r["mean_LA_dice"] for r in seed_results])
        std_str = f"±{std_dice:.4f}" if n_runs > 1 else ""
        print(f"  {cond:15s}: Dice={mean_dice:.4f}{std_str}  "
              f"(LV={lv:.3f}, MYO={myo:.3f}, LA={la:.3f})")

    # Save CSV
    with open(args.output, "w") as f:
        f.write("condition,seed,mean_dice,LV_dice,MYO_dice,LA_dice,"
                "ed_LV_dice,ed_MYO_dice,ed_LA_dice,es_LV_dice,es_MYO_dice,es_LA_dice\n")
        for r in rows:
            f.write(f"{r['condition']},{r['seed']},{r['mean_dice']:.6f},"
                    f"{r['mean_LV_dice']:.6f},{r['mean_MYO_dice']:.6f},{r['mean_LA_dice']:.6f},"
                    f"{r['ed_LV_dice']:.6f},{r['ed_MYO_dice']:.6f},{r['ed_LA_dice']:.6f},"
                    f"{r['es_LV_dice']:.6f},{r['es_MYO_dice']:.6f},{r['es_LA_dice']:.6f}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
