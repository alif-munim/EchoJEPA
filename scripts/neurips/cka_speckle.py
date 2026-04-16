"""
CKA perturbation invariance analysis for ICML rebuttal.

Measures representational stability under echo-specific perturbations using
linear CKA (Kornblith et al. 2019). EchoJEPA should maintain high CKA
(perturbation-invariant representations) while EchoMAE CKA drops.

Perturbation types: depth attenuation, acoustic shadow, haze artifact.
Each at three severity levels (mild, moderate, severe).

Requires perturbed_cache.pt from generate_perturbed_videos.py.

Usage:
    python scripts/neurips/cka_speckle.py \
        --cache scripts/neurips/perturbed_cache.pt \
        --device cuda:0
"""

import argparse

import torch

import src.models.vision_transformer as vit
from scripts.neurips.model_registry import get_models, add_model_args


def linear_cka(X, Y):
    """
    Linear CKA (Kornblith et al. 2019).
    X: [N, D1], Y: [N, D2]
    Returns scalar CKA value in [0, 1].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = torch.norm(Y.T @ X, p="fro") ** 2
    hsic_xx = torch.norm(X.T @ X, p="fro") ** 2
    hsic_yy = torch.norm(Y.T @ Y, p="fro") ** 2
    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)).item()


def load_vjepa_encoder(cfg, device, resolution=224, frames=16):
    ckpt = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = vit.__dict__[cfg["model_name"]](
        img_size=resolution, num_frames=frames, patch_size=16, tubelet_size=2, **cfg["kwargs"]
    )
    state = ckpt[cfg["checkpoint_key"]]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    for k, v in model.state_dict().items():
        if k not in state:
            pass
        elif state[k].shape != v.shape:
            state[k] = v
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def load_videomae_encoder(cfg, device, resolution=224, frames=16):
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
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_batch_features(model, clips, device, is_videomae=False, batch_size=8):
    """Extract mean-pooled features for a batch of clips [N, C, T, H, W]."""
    all_feats = []
    for i in range(0, len(clips), batch_size):
        batch = clips[i : i + batch_size].to(device)
        if is_videomae:
            out = model.forward_features(batch)
            if out.dim() == 3:
                out = out.mean(dim=1)
        else:
            out = model(batch)  # [B, N_tok, D]
            out = out.mean(dim=1)  # [B, D]
        all_feats.append(out.cpu())
    return torch.cat(all_feats, dim=0)  # [N, D]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="Path to perturbed_cache.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    add_model_args(parser)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading perturbed cache...")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    clean = cache["clean"]  # [N, C, T, H, W]
    perturbed = cache["perturbed"]  # dict: ptype -> severity -> [N, C, T, H, W]
    perturbation_types = cache["perturbation_types"]
    severity_levels = cache["severity_levels"]
    print(f"  {clean.shape[0]} videos, {len(perturbation_types)} perturbation types, "
          f"{len(severity_levels)} severity levels")

    # results[model][ptype][severity] = cka_value
    results = {}

    models = get_models(args.models)
    for model_name, cfg in models.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        is_videomae = cfg.get("type") == "videomae"
        if is_videomae:
            model = load_videomae_encoder(cfg, device)
        else:
            model = load_vjepa_encoder(cfg, device)

        # Extract clean features
        print("  Extracting clean features...")
        feat_clean = extract_batch_features(model, clean, device, is_videomae, args.batch_size)
        print(f"  Clean features: {feat_clean.shape}")

        results[model_name] = {}
        for ptype in perturbation_types:
            results[model_name][ptype] = {}
            for severity in severity_levels:
                print(f"  Extracting features: {ptype} / {severity}...")
                feat_noisy = extract_batch_features(
                    model, perturbed[ptype][severity], device, is_videomae, args.batch_size
                )
                cka = linear_cka(feat_clean, feat_noisy)
                results[model_name][ptype][severity] = cka
                print(f"    CKA = {cka:.4f}")

        del model
        torch.cuda.empty_cache()

    # Summary tables (one per perturbation type)
    for ptype in perturbation_types:
        print(f"\n{'=' * 60}")
        print(f"CKA: {ptype}")
        print(f"{'=' * 60}")
        header = f"{'Model':<20}" + "".join([f"{s:<12}" for s in severity_levels])
        print(header)
        print("-" * 56)
        for model_name in results:
            row = f"{model_name:<20}"
            for severity in severity_levels:
                row += f"{results[model_name][ptype][severity]:<12.4f}"
            print(row)

    # Combined summary (mean CKA across perturbation types per severity)
    print(f"\n{'=' * 60}")
    print("MEAN CKA (averaged across perturbation types)")
    print(f"{'=' * 60}")
    header = f"{'Model':<20}" + "".join([f"{s:<12}" for s in severity_levels]) + "drop"
    print(header)
    print("-" * 68)
    for model_name in results:
        row = f"{model_name:<20}"
        mean_ckas = []
        for severity in severity_levels:
            mean_cka = sum(
                results[model_name][p][severity] for p in perturbation_types
            ) / len(perturbation_types)
            mean_ckas.append(mean_cka)
            row += f"{mean_cka:<12.4f}"
        drop = mean_ckas[0] - mean_ckas[-1]
        row += f"{drop:.4f}"
        print(row)


if __name__ == "__main__":
    main()
