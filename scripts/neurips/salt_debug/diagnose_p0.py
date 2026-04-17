"""
Quick diagnostic for P0 experiment validity.

Checks:
1. CKA(clean, clean) = 1.0 for each model (sanity)
2. Feature norms and dimensions
3. Checkpoint key verification
4. Feature variance (near-zero = dead features)
5. Perturbed cache integrity

Usage:
    PYTHONPATH=. TMPDIR=/tmp LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
        python scripts/neurips/diagnose_p0.py \
        --cache scripts/neurips/perturbed_cache.pt \
        --device cuda:0 \
        --models pt50
"""

import argparse
import sys

import torch
import numpy as np

sys.path.insert(0, ".")
import src.models.vision_transformer as vit
from scripts.neurips.model_registry import get_models, add_model_args


def linear_cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = torch.norm(Y.T @ X, p="fro") ** 2
    hsic_xx = torch.norm(X.T @ X, p="fro") ** 2
    hsic_yy = torch.norm(Y.T @ Y, p="fro") ** 2
    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()


def load_encoder(cfg, device, resolution=224, frames=16):
    model_type = cfg.get("type", "vjepa")
    if model_type == "videomae":
        from evals.video_classification_frozen.modelcustom.videomae_encoder import (
            _convert_pretrain_to_finetune_state_dict,
            _import_modeling_finetune,
        )
        mf = _import_modeling_finetune()
        model = mf.vit_large_patch16_224(
            img_size=resolution, all_frames=frames, tubelet_size=2, num_classes=1000,
        )
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        state = _convert_pretrain_to_finetune_state_dict(state, model.state_dict())
        model.load_state_dict(state, strict=False)
    else:
        model = vit.__dict__[cfg["model_name"]](
            img_size=resolution, num_frames=frames, patch_size=16, tubelet_size=2,
            **cfg.get("kwargs", {}),
        )
        ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        state = ckpt[cfg["checkpoint_key"]]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        model_sd = model.state_dict()
        for k in list(state.keys()):
            if k in model_sd and state[k].shape != model_sd[k].shape:
                del state[k]
        model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_features(model, clips, device, is_videomae=False, batch_size=4):
    all_feats = []
    for i in range(0, len(clips), batch_size):
        batch = clips[i:i + batch_size].to(device)
        if is_videomae:
            out = model.forward_features(batch)
            if out.dim() == 3:
                out = out.mean(dim=1)
        else:
            out = model(batch)
            out = out.mean(dim=1)
        all_feats.append(out.cpu())
    return torch.cat(all_feats, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Diagnose P0 experiment validity")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    add_model_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading cache...")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    clean = cache["clean"]
    perturbed = cache["perturbed"]
    ptypes = cache["perturbation_types"]
    severities = cache["severity_levels"]
    print(f"  {clean.shape[0]} videos, shape {clean.shape}")
    print(f"  Perturbation types: {ptypes}")
    print(f"  Severity levels: {severities}")

    # Check cache integrity
    print(f"\n{'='*70}")
    print("1. CACHE INTEGRITY")
    print(f"{'='*70}")
    print(f"  Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"  Clean mean: {clean.mean():.4f}, std: {clean.std():.4f}")
    for pt in ptypes:
        for sev in severities:
            p = perturbed[pt][sev]
            diff = (clean - p).abs().mean().item()
            print(f"  {pt}/{sev}: range [{p.min():.3f}, {p.max():.3f}], diff from clean: {diff:.4f}")

    models = get_models(args.models)
    print(f"\nModels to check: {list(models.keys())}")

    for model_name, cfg in models.items():
        print(f"\n{'='*70}")
        print(f"2. MODEL: {model_name}")
        print(f"{'='*70}")

        # Check checkpoint exists
        import os
        ckpt_path = cfg["checkpoint"]
        if not os.path.exists(ckpt_path):
            print(f"  ✗ CHECKPOINT NOT FOUND: {ckpt_path}")
            continue
        print(f"  ✓ Checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e9:.1f} GB)")

        # Check checkpoint keys
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"  Keys in checkpoint: {list(ckpt.keys())[:10]}")
        if cfg.get("checkpoint_key"):
            key = cfg["checkpoint_key"]
            if key in ckpt:
                sd = ckpt[key]
                print(f"  ✓ checkpoint_key '{key}' found, {len(sd)} params")
                # Check first param shape
                first_key = list(sd.keys())[0]
                print(f"    First param: {first_key} → {sd[first_key].shape}")
            else:
                print(f"  ✗ checkpoint_key '{key}' NOT FOUND. Available: {list(ckpt.keys())}")
                continue
        del ckpt

        # Load encoder
        is_videomae = cfg.get("type") == "videomae"
        try:
            model = load_encoder(cfg, device)
            print(f"  ✓ Encoder loaded")
        except Exception as e:
            print(f"  ✗ Encoder load FAILED: {e}")
            continue

        # Extract clean features
        print(f"  Extracting features (N={clean.shape[0]})...")
        feat_clean = extract_features(model, clean, device, is_videomae, args.batch_size)
        print(f"  Feature shape: {feat_clean.shape}")
        print(f"  Feature norm: mean={feat_clean.norm(dim=-1).mean():.4f}, "
              f"std={feat_clean.norm(dim=-1).std():.4f}")
        print(f"  Feature mean: {feat_clean.mean():.6f}")
        print(f"  Feature std: {feat_clean.std():.6f}")
        print(f"  Feature range: [{feat_clean.min():.4f}, {feat_clean.max():.4f}]")

        # Per-dim variance (check for dead dimensions)
        dim_var = feat_clean.var(dim=0)
        dead_dims = (dim_var < 1e-8).sum().item()
        print(f"  Dead dimensions (var < 1e-8): {dead_dims}/{feat_clean.shape[1]}")

        # SANITY: CKA(clean, clean) should be 1.0
        cka_self = linear_cka(feat_clean, feat_clean)
        print(f"\n  CKA(clean, clean) = {cka_self:.6f}  {'✓ PASS' if cka_self > 0.999 else '✗ FAIL'}")

        # CKA(clean, clean_subset) — split in half, should be high
        n = len(feat_clean)
        half = n // 2
        cka_half = linear_cka(feat_clean[:half], feat_clean[:half])
        print(f"  CKA(clean_half, clean_half) = {cka_half:.6f}")

        # Extract features for one perturbed condition and compute CKA
        severe_key = f"{ptypes[0]}/{severities[-1]}"
        print(f"\n  Extracting features for {severe_key}...")
        feat_perturbed = extract_features(
            model, perturbed[ptypes[0]][severities[-1]], device, is_videomae, args.batch_size
        )
        cka_severe = linear_cka(feat_clean, feat_perturbed)
        print(f"  CKA(clean, {severe_key}) = {cka_severe:.4f}")

        # Feature norm comparison
        norm_clean = feat_clean.norm(dim=-1).mean().item()
        norm_pert = feat_perturbed.norm(dim=-1).mean().item()
        print(f"  Norm ratio (perturbed/clean): {norm_pert/norm_clean:.4f}")

        # Cosine similarity (what frame shuffling uses)
        cos_sim = torch.nn.functional.cosine_similarity(
            feat_clean, feat_perturbed, dim=-1
        ).mean().item()
        print(f"  Mean cosine(clean, {severe_key}) = {cos_sim:.4f}")

        del model, feat_clean, feat_perturbed
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
