"""Quick BYOL learning curve test: extract frozen features from ViT-L encoder
at different BYOL pretraining epochs, then fit a linear probe (sklearn Ridge)
on UHN LVEF regression. Compares representation quality across training."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset

import src.models.vision_transformer as video_vit
from src.datasets.video_dataset import VideoDataset
from src.utils.checkpoint_loader import robust_checkpoint_loader


def load_byol_encoder(checkpoint_path, device, key="target_encoder"):
    """Load frozen ViT-L encoder from a BYOL checkpoint."""
    encoder = video_vit.vit_large(
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        uniform_power=True,
        use_rope=True,
    )

    ckpt = robust_checkpoint_loader(checkpoint_path, map_location="cpu")
    epoch = ckpt["epoch"]
    loss = ckpt.get("loss", float("nan"))

    state_dict = ckpt[key]
    # Strip DDP/wrapper prefixes
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}

    msg = encoder.load_state_dict(state_dict, strict=False)
    missing = [k for k in msg.missing_keys if "pos_embed" not in k]
    if missing:
        print(f"  WARNING: missing keys: {missing}")

    encoder = encoder.to(device).eval()
    del ckpt
    torch.cuda.empty_cache()
    return encoder, epoch, loss


@torch.no_grad()
def extract_features(encoder, dataloader, device, max_samples=2000):
    """Extract mean-pooled features from frozen encoder."""
    all_features = []
    all_labels = []
    n = 0

    for batch in dataloader:
        clips, labels, _, _ = batch
        # clips is a list of tensors; take the first clip
        if isinstance(clips, list):
            x = clips[0]
        else:
            x = clips
        # VideoDataset returns [B, T, H, W, C] uint8; convert to [B, C, T, H, W] float
        x = x.permute(0, 4, 1, 2, 3).to(device).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
        x = (x - mean) / std

        features = encoder(x)  # [B, N, D]
        features = features.mean(dim=1)  # [B, D] — global average pool
        features = F.layer_norm(features, [features.shape[-1]])

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
        n += x.shape[0]

        if n >= max_samples:
            break

    return np.concatenate(all_features)[:max_samples], np.concatenate(all_labels)[:max_samples]


def make_dataloader(csv_path, batch_size=16, num_workers=4, shuffle=False):
    """Create a simple VideoDataset dataloader."""
    dataset = VideoDataset(
        data_paths=[csv_path],
        datasets_weights=None,
        transform=None,
        num_clips=1,
        dataset_fpcs=[16],
        fps=8,
        frame_step=None,
        random_clip_sampling=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader


def main():
    device = torch.device("cuda:0")
    base = "/mnt/custom-file-systems/efs/fs-0049217cdf69186d7_fsap-0fa7145b64eaa046b/vjepa2"

    checkpoints = [
        ("e0", f"{base}/checkpoints/byol_vitl_e0.pt"),
        ("e10", f"{base}/checkpoints/byol_vitl_e10.pt"),
        ("e44", f"{base}/checkpoints/byol_vitl_e44.pt"),
    ]

    train_csv = "/home/sagemaker-user/user-default-efs/vjepa2/data/csv/a4c_b_lvef_train_224px.csv"
    val_csv = "/home/sagemaker-user/user-default-efs/vjepa2/data/csv/a4c_b_lvef_val_224px.csv"

    max_train = 3000
    max_val = 1000

    print(f"Loading data (train: {max_train}, val: {max_val} samples)...")
    train_loader = make_dataloader(train_csv, batch_size=16, shuffle=False)
    val_loader = make_dataloader(val_csv, batch_size=16, shuffle=False)

    # Also test with target_encoder
    for key in ["target_encoder", "encoder"]:
        print(f"\n{'='*60}")
        print(f"Using checkpoint key: {key}")
        print(f"{'='*60}")

        results = []
        for label, ckpt_path in checkpoints:
            print(f"\n--- {label} ({ckpt_path.split('/')[-1]}) ---")
            t0 = time.time()

            encoder, epoch, loss = load_byol_encoder(ckpt_path, device, key=key)
            print(f"  Loaded: epoch={epoch}, byol_loss={loss:.4f}")

            print(f"  Extracting train features...")
            X_train, y_train = extract_features(encoder, train_loader, device, max_samples=max_train)
            print(f"  Extracting val features...")
            X_val, y_val = extract_features(encoder, val_loader, device, max_samples=max_val)

            print(f"  Features: train {X_train.shape}, val {X_val.shape}")
            print(f"  Feature norm: mean={np.linalg.norm(X_train, axis=1).mean():.2f}, "
                  f"std={np.linalg.norm(X_train, axis=1).std():.2f}")

            # Z-score normalize labels
            y_mean, y_std = y_train.mean(), y_train.std()
            y_train_z = (y_train - y_mean) / y_std
            y_val_z = (y_val - y_mean) / y_std

            # Ridge regression with multiple alphas
            best_r2 = -999
            best_alpha = None
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_train, y_train_z)
                pred = ridge.predict(X_val)
                r2 = r2_score(y_val_z, pred)
                if r2 > best_r2:
                    best_r2 = r2
                    best_alpha = alpha

            elapsed = time.time() - t0
            print(f"  Val R2: {best_r2:.4f} (alpha={best_alpha}, {elapsed:.1f}s)")
            results.append((label, epoch, loss, best_r2, best_alpha))

            del encoder
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"Summary ({key}):")
        print(f"{'Label':<8} {'Epoch':<7} {'BYOL Loss':<12} {'Val R2':<10} {'Alpha':<8}")
        for label, epoch, loss, r2, alpha in results:
            print(f"{label:<8} {epoch:<7} {loss:<12.4f} {r2:<10.4f} {alpha:<8}")


if __name__ == "__main__":
    main()
