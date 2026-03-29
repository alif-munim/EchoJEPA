"""
P0.6: Noised test inference — task degradation under echo-specific perturbations.

Loads a frozen encoder + trained probe ONCE, then runs inference on clean and
perturbed test sets (3 perturbation types × 3 severity levels = 9 conditions + clean).
Reports degradation curves per perturbation type.

Perturbations are applied on-the-fly (no disk I/O). Each video gets a deterministic
perturbation seeded by its path hash, so all models see identical noised inputs.

Usage:
    # LVEF regression (single-view)
    TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH python scripts/rebuttal/noised_inference.py \
        --encoder_type vjepa \
        --encoder_checkpoint checkpoints/echojepa-l-pt50.pt \
        --encoder_model_name vit_large \
        --encoder_key target_encoder \
        --probe_checkpoint evals/vitb/icml/echojepa_l_pt50_lvef/video_classification_frozen/icml-echojepa-l-pt50-lvef-d4/best.pt \
        --task_type regression \
        --test_csv data/csv/rebuttal/lvef/lvef_val_1k.csv \
        --device cuda:0

    # View classification (single-view)
    TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH python scripts/rebuttal/noised_inference.py \
        --encoder_type vjepa \
        --encoder_checkpoint checkpoints/anneal/keep/vitl-pt-210-an25.pt \
        --encoder_model_name vit_large \
        --encoder_key target_encoder \
        --probe_checkpoint path/to/view_probe/best.pt \
        --task_type classification \
        --test_csv data/csv/uhn_views_22k_val.csv \
        --device cuda:0

    # VideoMAE encoder
    TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH python scripts/rebuttal/noised_inference.py \
        --encoder_type videomae \
        --encoder_checkpoint checkpoints/videomae_l_mimic_ep50.pth \
        --probe_checkpoint path/to/mae_probe/best.pt \
        --task_type regression \
        --test_csv data/csv/rebuttal/lvef/lvef_val_1k.csv \
        --device cuda:0

Supported encoder types: vjepa, videomae
Supported task types: regression, classification

Output: prints a table of metrics per condition, saves CSV to --output.
"""

import argparse
import hashlib
import sys

import decord
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, ".")
import src.models.vision_transformer as vit
from scripts.rebuttal.echo_perturbations import (
    PERTURBATIONS,
    SEVERITY_LEVELS,
    apply_perturbation,
    create_scan_mask,
)
from src.models.attentive_pooler import AttentiveClassifier, AttentiveRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader

decord.bridge.set_bridge("torch")

# ImageNet normalization (matches eval pipeline)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


# --- Data Loading ---


def load_clip(video_path, frames=16, frame_step=2, resolution=224):
    """Load a single clip as [C, T, H, W] float32 in [0, 1]."""
    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    needed = frames * frame_step
    start = max(0, (total - needed) // 2)
    indices = list(range(start, min(start + needed, total), frame_step))
    while len(indices) < frames:
        indices.append(indices[-1])
    indices = indices[:frames]
    clip = vr.get_batch(indices)  # [T, H, W, C]
    clip = clip.permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
    clip = TF.resize(clip, [resolution, resolution], antialias=True)
    return clip


def normalize_clip(clip):
    """Apply ImageNet normalization to [C, T, H, W] clip."""
    return (clip - IMAGENET_MEAN) / IMAGENET_STD


def read_csv(csv_path):
    """Read space-delimited CSV: path label."""
    paths, labels = [], []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                paths.append(parts[0])
                labels.append(float(parts[1]))
    return paths, labels


def path_seed(video_path):
    """Deterministic seed from video path (reproducible across runs/models)."""
    return int(hashlib.md5(video_path.encode()).hexdigest()[:8], 16)


# --- Model Loading ---


def load_vjepa_encoder(checkpoint, model_name, checkpoint_key, device, resolution=224, frames=16):
    """Load a V-JEPA encoder (also works for BYOL — same architecture)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = vit.__dict__[model_name](
        img_size=resolution,
        num_frames=frames,
        patch_size=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )
    state = ckpt[checkpoint_key]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    # Handle shape mismatches gracefully
    model_sd = model.state_dict()
    for k in list(state.keys()):
        if k in model_sd and state[k].shape != model_sd[k].shape:
            print(f"  Shape mismatch, skipping: {k}")
            del state[k]
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def load_videomae_encoder(checkpoint, device, resolution=224, frames=16):
    """Load a VideoMAE encoder."""
    from evals.video_classification_frozen.modelcustom.videomae_encoder import (
        _convert_pretrain_to_finetune_state_dict,
        _import_modeling_finetune,
    )

    mf = _import_modeling_finetune()
    model = mf.vit_large_patch16_224(
        img_size=resolution, all_frames=frames, tubelet_size=2, num_classes=1000,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = _convert_pretrain_to_finetune_state_dict(state, model.state_dict())
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def load_probe(probe_checkpoint, task_type, embed_dim, device):
    """
    Load the best probe head from a multi-head checkpoint.

    Returns: (probe_model, best_head_idx, metadata_dict)
    """
    ckpt = robust_checkpoint_loader(probe_checkpoint, map_location="cpu")

    # Find best head
    best_vals = ckpt["best_val_acc_per_head"]
    if task_type == "regression":
        best_idx = best_vals.index(min(best_vals))  # Lower MAE = better
    else:
        best_idx = best_vals.index(max(best_vals))  # Higher acc = better

    # Get probe config from state dict
    state_dict = ckpt["classifiers"][best_idx]
    # Clean DDP prefix
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    # Infer probe shape from weights
    query_tokens = state_dict["pooler.query_tokens"]  # [1, n_queries, embed_dim]
    linear_key = "regressor.weight" if task_type == "regression" else "linear.weight"
    num_outputs = state_dict[linear_key].shape[0]

    # Count depth (number of self-attention blocks)
    sa_keys = [k for k in state_dict if "self_attention_blocks" in k and "norm1.weight" in k]
    depth = len(sa_keys) + 1  # +1 for the cross-attention block

    # Infer num_heads from cross-attention q weight
    q_weight = state_dict["pooler.cross_attention_block.xattn.q.weight"]
    # head_dim is typically 64, num_heads = embed_dim / head_dim
    num_heads = embed_dim // 64

    print(f"  Probe: embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, "
          f"outputs={num_outputs}, best_head={best_idx} (val={best_vals[best_idx]:.4f})")

    if task_type == "regression":
        probe = AttentiveRegressor(
            embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_targets=num_outputs,
        )
    else:
        probe = AttentiveClassifier(
            embed_dim=embed_dim, num_heads=num_heads, depth=depth, num_classes=num_outputs,
        )

    probe.load_state_dict(state_dict)
    probe.eval().to(device)

    metadata = {
        "target_mean": ckpt.get("target_mean"),
        "target_std": ckpt.get("target_std"),
        "task_type": ckpt.get("task_type", task_type),
        "best_head_idx": best_idx,
        "best_val": best_vals[best_idx],
    }
    return probe, best_idx, metadata


# --- Inference ---


@torch.no_grad()
def extract_features(encoder, clip, device, is_videomae=False):
    """Encode a single clip -> [1, N, D] tokens."""
    clip = clip.unsqueeze(0).to(device)  # [1, C, T, H, W]
    if is_videomae:
        out = encoder.forward_features(clip)
        if out.dim() == 2:
            out = out.unsqueeze(0)
        return out  # [1, N, D]
    else:
        return encoder(clip)  # [1, N, D]


@torch.no_grad()
def run_condition(encoder, probe, paths, labels, device, is_videomae,
                  perturbation_type=None, severity=None,
                  resolution=224, frames=16, frame_step=2):
    """
    Run inference for one condition (clean or a specific perturbation).

    Returns: (predictions_array, labels_array, n_skipped)
    """
    all_preds = []
    all_labels = []
    n_skipped = 0

    for i, (video_path, label) in enumerate(zip(paths, labels)):
        try:
            clip = load_clip(video_path, frames, frame_step, resolution)
        except Exception:
            n_skipped += 1
            continue

        # Apply perturbation (before normalization)
        if perturbation_type is not None:
            seed = path_seed(video_path)
            mask = create_scan_mask(clip[:, 0, :, :])
            clip = apply_perturbation(clip, perturbation_type, severity, scan_mask=mask, seed=seed)

        # Normalize and encode
        clip = normalize_clip(clip)
        features = extract_features(encoder, clip, device, is_videomae)
        pred = probe(features)  # [1, num_targets]
        all_preds.append(pred.cpu().squeeze())
        all_labels.append(label)

        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(paths)}")

    preds = torch.stack(all_preds).numpy() if all_preds else np.array([])
    labels_arr = np.array(all_labels)
    return preds, labels_arr, n_skipped


# --- Metrics ---


def compute_metrics(preds, labels, task_type, target_mean=None, target_std=None):
    """Compute task-appropriate metrics."""
    if len(preds) == 0:
        return {}

    if task_type == "regression":
        # Un-normalize predictions if z-scored
        if target_mean is not None and target_std is not None:
            preds = preds * target_std + target_mean

        from scipy.stats import pearsonr
        residuals = labels - preds.squeeze()
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((labels - labels.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae = np.mean(np.abs(residuals))
        r, _ = pearsonr(labels, preds.squeeze()) if len(labels) > 2 else (0.0, 1.0)
        return {"R²": r2, "MAE": mae, "Pearson": r}
    else:
        from sklearn.metrics import accuracy_score, roc_auc_score
        if preds.ndim == 1:
            pred_classes = (preds > 0).astype(int)
        else:
            pred_classes = np.argmax(preds, axis=1)
        acc = accuracy_score(labels.astype(int), pred_classes)
        try:
            if preds.ndim > 1 and preds.shape[1] > 2:
                auc = roc_auc_score(labels.astype(int), preds, multi_class="ovr", average="macro")
            elif preds.ndim > 1 and preds.shape[1] == 2:
                auc = roc_auc_score(labels.astype(int), preds[:, 1])
            else:
                auc = roc_auc_score(labels.astype(int), preds)
        except Exception:
            auc = float("nan")
        return {"Accuracy": acc, "AUROC": auc}


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="P0.6: Noised test inference")
    # Encoder
    parser.add_argument("--encoder_type", required=True, choices=["vjepa", "videomae"])
    parser.add_argument("--encoder_checkpoint", required=True)
    parser.add_argument("--encoder_model_name", default="vit_large",
                        help="vit_large, vit_giant_xformers, etc. (vjepa only)")
    parser.add_argument("--encoder_key", default="target_encoder",
                        help="Checkpoint key for encoder weights (vjepa only)")
    # Probe
    parser.add_argument("--probe_checkpoint", required=True)
    parser.add_argument("--task_type", required=True, choices=["regression", "classification"])
    # Data
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--frame_step", type=int, default=2)
    # Perturbation control
    parser.add_argument("--perturbation_types", nargs="*", default=None,
                        help="Perturbation types to run (default: all). "
                             "Options: depth_attenuation, gaussian_shadow, haze_artifact")
    parser.add_argument("--severity_levels", nargs="*", default=None,
                        help="Severity levels to run (default: all). "
                             "Options: mild, moderate, severe")
    parser.add_argument("--skip_clean", action="store_true",
                        help="Skip clean baseline (useful if already computed)")
    # Compute
    parser.add_argument("--device", default="cuda:0")
    # Output
    parser.add_argument("--output", default=None, help="Output CSV path (default: auto-named)")
    parser.add_argument("--label", default="model", help="Model label for output table")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load encoder ---
    print(f"Loading encoder: {args.encoder_type} from {args.encoder_checkpoint}")
    is_videomae = args.encoder_type == "videomae"
    if is_videomae:
        encoder = load_videomae_encoder(args.encoder_checkpoint, device, args.resolution, args.frames)
        embed_dim = 1024  # ViT-L
    else:
        encoder = load_vjepa_encoder(
            args.encoder_checkpoint, args.encoder_model_name, args.encoder_key,
            device, args.resolution, args.frames,
        )
        # Infer embed_dim from model name
        embed_dims = {"vit_large": 1024, "vit_giant_xformers": 1408, "vit_huge": 1280, "vit_base": 768}
        embed_dim = embed_dims.get(args.encoder_model_name, 1024)

    # --- Load probe ---
    print(f"Loading probe: {args.probe_checkpoint}")
    probe, best_head, metadata = load_probe(args.probe_checkpoint, args.task_type, embed_dim, device)

    # --- Load test data ---
    print(f"Loading test CSV: {args.test_csv}")
    paths, labels = read_csv(args.test_csv)
    print(f"  {len(paths)} videos")

    # --- Build condition list ---
    ptypes = args.perturbation_types if args.perturbation_types else list(PERTURBATIONS.keys())
    severities = args.severity_levels if args.severity_levels else SEVERITY_LEVELS

    # Validate
    for p in ptypes:
        if p not in PERTURBATIONS:
            parser.error(f"Unknown perturbation type: {p}. Choose from {list(PERTURBATIONS.keys())}")
    for s in severities:
        if s not in SEVERITY_LEVELS:
            parser.error(f"Unknown severity: {s}. Choose from {SEVERITY_LEVELS}")

    conditions = []
    if not args.skip_clean:
        conditions.append(("clean", None, None))
    for ptype in ptypes:
        for severity in severities:
            conditions.append((f"{ptype}/{severity}", ptype, severity))

    print(f"  Conditions: {len(conditions)} ({', '.join(c[0] for c in conditions)})")

    results = {}
    for cond_name, ptype, severity in conditions:
        print(f"\n  [{cond_name}]")
        preds, labs, skipped = run_condition(
            encoder, probe, paths, labels, device, is_videomae,
            ptype, severity, args.resolution, args.frames, args.frame_step,
        )
        metrics = compute_metrics(preds, labs, args.task_type,
                                  metadata.get("target_mean"), metadata.get("target_std"))
        results[cond_name] = metrics
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"    {metric_str}  (skipped {skipped})")

    # --- Summary table ---
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {args.label}")
    print(f"{'=' * 80}")

    if args.task_type == "regression":
        metric_keys = ["R²", "MAE", "Pearson"]
    else:
        metric_keys = ["Accuracy", "AUROC"]

    header = f"{'Condition':<30}" + "".join(f"{k:<12}" for k in metric_keys)
    print(header)
    print("-" * 80)
    for cond_name in results:
        row = f"{cond_name:<30}"
        for k in metric_keys:
            row += f"{results[cond_name].get(k, float('nan')):<12.4f}"
        print(row)

    # Degradation summary (clean vs worst severity per type)
    if "clean" in results:
        clean_metrics = results["clean"]
        primary_key = "R²" if args.task_type == "regression" else "AUROC"
        clean_val = clean_metrics[primary_key]
        worst_sev = severities[-1]  # last = most severe

        has_severe = any(f"{p}/{worst_sev}" in results for p in ptypes)
        if has_severe:
            print(f"\n{'=' * 80}")
            print(f"DEGRADATION (clean → {worst_sev})")
            print(f"{'=' * 80}")
            for ptype in ptypes:
                key = f"{ptype}/{worst_sev}"
                if key in results:
                    severe_val = results[key][primary_key]
                    drop = clean_val - severe_val
                    pct = (drop / abs(clean_val) * 100) if clean_val != 0 else 0
                    print(f"  {ptype:<25s}: {clean_val:.4f} → {severe_val:.4f}  (Δ={drop:+.4f}, {pct:+.1f}%)")

    # Save CSV
    if args.output is None:
        args.output = f"scripts/rebuttal/samples/{args.label}_noised_inference.csv"
    with open(args.output, "w") as f:
        f.write("condition," + ",".join(metric_keys) + "\n")
        for cond_name, metrics in results.items():
            vals = ",".join(f"{metrics.get(k, float('nan')):.6f}" for k in metric_keys)
            f.write(f"{cond_name},{vals}\n")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
