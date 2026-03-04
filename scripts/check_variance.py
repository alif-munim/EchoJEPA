# check_variance_real.py
import argparse
import io

import boto3
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

import src.models.vision_transformer as video_vit


def _get_vit_ctor(arch: str):
    arch = arch.lower()
    candidates = {
        "large": ["vit_large", "vitl", "vit_large_224"],
        "giant": ["vit_giant", "vitg", "vit_giant_224", "vit_giant_384"],
    }

    if arch not in candidates:
        raise ValueError(f"Unsupported --arch '{arch}'. Use 'large' or 'giant'.")

    # pick the first constructor that exists in your repo
    for name in candidates[arch]:
        ctor = getattr(video_vit, name, None)
        if ctor is not None:
            return ctor, name

    available = [n for n in dir(video_vit) if n.startswith("vit")]
    raise RuntimeError(
        f"Couldn't find a constructor for arch='{arch}'. "
        f"Tried {candidates[arch]}. Available vit* fns: {available}"
    )


def load_vjepa_encoder(checkpoint_path, arch="large", device="cuda", img_size=224, patch_size=16, num_frames=16, tubelet_size=2):
    ctor, ctor_name = _get_vit_ctor(arch)
    print(f"Building encoder: arch={arch} (ctor={ctor_name}) img={img_size} patch={patch_size} T={num_frames} tubelet={tubelet_size}")

    encoder = ctor(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder_state = checkpoint["encoder"]
    encoder_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in encoder_state.items()}

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

    # Basic sanity: fail fast if we obviously instantiated the wrong arch
    total = len(encoder.state_dict())
    loaded = total - len(missing)
    frac = loaded / max(total, 1)
    print(f"Loaded params: {loaded}/{total} ({frac:.1%}) | missing={len(missing)} unexpected={len(unexpected)}")
    if frac < 0.90:
        raise RuntimeError(
            f"Only loaded {loaded}/{total} parameters ({frac:.1%}). "
            "Likely wrong --arch or wrong img/patch/T settings for this checkpoint."
        )

    return encoder.to(device)


def load_video_from_s3(s3_uri, s3_client, num_frames=16):
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    vr = VideoReader(io.BytesIO(data), num_threads=-1, ctx=cpu(0))

    total_frames = len(vr)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(np.int64)
    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]

    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

    frames = F.interpolate(
        frames.unsqueeze(0),
        size=(num_frames, 224, 224),
        mode="trilinear",
        align_corners=False,
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    frames = (frames - mean) / std

    return frames.squeeze(0)  # [C, T, H, W]


def check_variance_real(encoder, csv_path, num_samples=8, device="cuda", num_frames=16):
    encoder.eval()

    with open(csv_path, "r") as f:
        lines = f.readlines()[:num_samples]

    s3_client = boto3.client("s3")

    videos = []
    for line in lines:
        s3_path = line.strip().split()[0]
        print(f"Loading: {s3_path.split('/')[-1]}")
        try:
            video = load_video_from_s3(s3_path, s3_client, num_frames=num_frames)
            videos.append(video)
        except Exception as e:
            print(f"  Failed: {e}")

    if len(videos) < 2:
        print("Need at least 2 videos!")
        return

    batch = torch.stack(videos).to(device)
    print(f"\nBatch shape: {batch.shape}")

    with torch.no_grad():
        features = encoder(batch)
        print(f"Feature shape: {features.shape}")

        feature_std = features.std().item()
        print(f"\nFeature std: {feature_std:.4f}")

        flat = features.mean(dim=1)
        flat_norm = F.normalize(flat, dim=1)
        cos_sim_matrix = flat_norm @ flat_norm.T

        mask = ~torch.eye(len(videos), dtype=bool, device=device)
        off_diag_sim = cos_sim_matrix[mask].mean().item()
        print(f"Off-diagonal cosine sim: {off_diag_sim:.4f}")

        print("\nPairwise similarity matrix:")
        print(cos_sim_matrix.cpu().numpy().round(4))

        print("\n" + "=" * 50)
        if feature_std < 0.1:
            print("⚠️  LOW STD - likely collapse")
        else:
            print("✅ STD looks healthy")

        if off_diag_sim > 0.95:
            print("⚠️  HIGH SIMILARITY - representations too similar")
        else:
            print("✅ Similarity looks healthy")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to V-JEPA2 checkpoint (.pt) containing checkpoint['encoder']")
    p.add_argument("--arch", choices=["large", "giant"], default="large", help="Model architecture to instantiate")
    p.add_argument("--csv", default="/home/sagemaker-user/user-default-efs/vjepa2/data/csv/a4c_b_lvef_train.csv",
                   help="CSV file with S3 mp4 paths in first column")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-frames", type=int, default=16)

    # Optional overrides if you ever need 336/384 etc.
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--tubelet-size", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()
    encoder = load_vjepa_encoder(
        args.ckpt,
        arch=args.arch,
        device=args.device,
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
    )
    check_variance_real(
        encoder,
        args.csv,
        num_samples=args.num_samples,
        device=args.device,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
