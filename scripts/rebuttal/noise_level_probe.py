"""
Perturbation severity probe for ICML rebuttal.

Can the model predict which perturbation severity was applied? If yes, it
encodes degradation information. If no, the representation is invariant.

Runs one probe per perturbation type (3-class: mild/moderate/severe).

Expected: EchoMAE high accuracy (encodes degradation). EchoJEPA near chance (~33%).

Requires perturbed_cache.pt from generate_perturbed_videos.py.

Usage:
    python scripts/rebuttal/noise_level_probe.py \
        --cache scripts/rebuttal/perturbed_cache.pt \
        --device cuda:0
"""

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import src.models.vision_transformer as vit

# Model configs (same as cka_speckle.py)
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
    all_feats = []
    for i in range(0, len(clips), batch_size):
        batch = clips[i : i + batch_size].to(device)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="Path to perturbed_cache.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    rng = np.random.RandomState(args.seed)

    print("Loading perturbed cache...")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    perturbed = cache["perturbed"]
    perturbation_types = cache["perturbation_types"]
    severity_levels = cache["severity_levels"]
    N = cache["clean"].shape[0]
    n_classes = len(severity_levels)
    print(f"  {N} videos, {len(perturbation_types)} perturbation types, {n_classes} severity levels")

    # Train/test split (by video index)
    n_test = max(1, int(N * args.test_frac))
    indices = rng.permutation(N)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    print(f"  Train: {len(train_idx)} videos, Test: {len(test_idx)} videos")

    # results[model][ptype] = {"train_acc": ..., "test_acc": ...}
    results = {}

    for model_name, cfg in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        is_videomae = model_name == "EchoMAE-L"
        if is_videomae:
            model = load_videomae_encoder(cfg, device)
        else:
            model = load_vjepa_encoder(cfg, device)

        results[model_name] = {}

        for ptype in perturbation_types:
            print(f"\n  Perturbation: {ptype}")
            print(f"  {'-' * 40}")

            # Extract features for each severity level
            all_features = []
            all_labels = []
            for level_idx, severity in enumerate(severity_levels):
                print(f"    Extracting features: {severity}...")
                feats = extract_batch_features(
                    model, perturbed[ptype][severity], device, is_videomae, args.batch_size
                )
                all_features.append(feats.numpy())
                all_labels.append(np.full(N, level_idx))

            X = np.concatenate(all_features, axis=0)  # [N*3, D]
            y = np.concatenate(all_labels, axis=0)  # [N*3]

            # Split using video indices (same video can't be in train and test)
            train_mask = np.concatenate([train_idx + i * N for i in range(n_classes)])
            test_mask = np.concatenate([test_idx + i * N for i in range(n_classes)])
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]

            print(f"    Samples: {len(X_train)} train, {len(X_test)} test, dim={X_train.shape[1]}")

            # Logistic regression
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed)
            clf.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))

            results[model_name][ptype] = {"train_acc": train_acc, "test_acc": test_acc}
            print(f"    Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f} (chance: {1.0/n_classes:.3f})")

        del model
        torch.cuda.empty_cache()

    # Summary tables
    chance = 1.0 / n_classes
    for ptype in perturbation_types:
        print(f"\n{'=' * 60}")
        print(f"SEVERITY PROBE: {ptype} ({n_classes}-class, chance={chance:.1%})")
        print(f"{'=' * 60}")
        print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Interpretation'}")
        print("-" * 64)
        for model_name in results:
            r = results[model_name][ptype]
            interp = "ENCODES DEGRADATION" if r["test_acc"] > chance + 0.1 else "INVARIANT"
            print(f"{model_name:<20} {r['train_acc']:<12.4f} {r['test_acc']:<12.4f} {interp}")

    # Mean across perturbation types
    print(f"\n{'=' * 60}")
    print(f"MEAN TEST ACCURACY (across perturbation types)")
    print(f"{'=' * 60}")
    print(f"{'Model':<20} ", end="")
    for ptype in perturbation_types:
        short = ptype[:12]
        print(f"{short:<14}", end="")
    print("Mean")
    print("-" * 68)
    for model_name in results:
        print(f"{model_name:<20} ", end="")
        accs = []
        for ptype in perturbation_types:
            acc = results[model_name][ptype]["test_acc"]
            accs.append(acc)
            print(f"{acc:<14.4f}", end="")
        print(f"{np.mean(accs):.4f}")


if __name__ == "__main__":
    main()
