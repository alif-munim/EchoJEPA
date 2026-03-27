"""
CKA speckle invariance analysis for ICML rebuttal.

Measures representational stability under speckle noise perturbation using
linear CKA (Kornblith et al. 2019). EchoJEPA should maintain high CKA (noise-
invariant latent space) while EchoMAE CKA drops (pixel reconstruction encodes noise).

Requires perturbed_cache.pt from generate_perturbed_videos.py.

Usage:
    python scripts/rebuttal/cka_speckle.py \
        --cache scripts/rebuttal/perturbed_cache.pt \
        --device cuda:0
"""

import argparse

import torch
import torch.nn.functional as F

import src.models.vision_transformer as vit

# Model configs
MODELS = {
    "EchoJEPA-G": {
        "checkpoint": "checkpoints/anneal/keep/pt-280-an81.pt",
        "model_name": "vit_giant_xformers",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
    },
    "EchoJEPA-L": {
        "checkpoint": "checkpoints/anneal/keep/vitl-pt-210-an25.pt",
        "model_name": "vit_large",
        "checkpoint_key": "target_encoder",
        "kwargs": {"uniform_power": True, "use_rope": True},
    },
    "EchoMAE-L": {
        "checkpoint": "checkpoints/videomae-ep163.pth",
        "model_name": None,
        "checkpoint_key": None,
        "kwargs": {},
    },
}


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
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading perturbed cache...")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    clean = cache["clean"]  # [N, C, T, H, W]
    perturbed = cache["perturbed"]  # dict: sigma -> [N, C, T, H, W]
    sigma_levels = cache["sigma_levels"]
    print(f"  {clean.shape[0]} videos, {len(sigma_levels)} noise levels: {sigma_levels}")

    results = {}
    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        is_videomae = model_name == "EchoMAE-L"
        if is_videomae:
            model = load_videomae_encoder(cfg, device)
        else:
            model = load_vjepa_encoder(cfg, device)

        # Extract clean features
        print("  Extracting clean features...")
        feat_clean = extract_batch_features(model, clean, device, is_videomae, args.batch_size)
        print(f"  Clean features: {feat_clean.shape}")

        # Extract features at each noise level and compute CKA
        cka_values = {}
        for sigma in sigma_levels:
            print(f"  Extracting features at sigma={sigma}...")
            feat_noisy = extract_batch_features(
                model, perturbed[sigma], device, is_videomae, args.batch_size
            )
            cka = linear_cka(feat_clean, feat_noisy)
            cka_values[sigma] = cka
            print(f"    CKA(clean, sigma={sigma}) = {cka:.4f}")

        results[model_name] = cka_values
        del model
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: CKA Speckle Invariance")
    print(f"{'='*70}")
    header = f"{'Model':<20}" + "".join([f"{'σ='+str(s):<12}" for s in sigma_levels])
    print(header)
    print("-" * 70)
    for model_name, cka_vals in results.items():
        row = f"{model_name:<20}"
        for sigma in sigma_levels:
            row += f"{cka_vals[sigma]:<12.4f}"
        print(row)

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    for model_name, cka_vals in results.items():
        min_cka = min(cka_vals.values())
        max_cka = max(cka_vals.values())
        drop = max_cka - min_cka
        print(f"  {model_name}: CKA range [{min_cka:.3f}, {max_cka:.3f}], drop = {drop:.3f}")


if __name__ == "__main__":
    main()
