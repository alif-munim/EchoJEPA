"""Test SALT LVEF on first 20 videos to check prediction pattern."""
import sys
import torch
import numpy as np
sys.path.insert(0, ".")

import decord
import torchvision.transforms.functional as TF
import src.models.vision_transformer as vit
from src.models.attentive_pooler import AttentiveRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader

decord.bridge.set_bridge("torch")

SALT_CKPT = "/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/e79.pt"
SALT_PROBE = "/opt/dlami/nvme/lvef_probes/salt_v1_e79_best.pt"
TEST_CSV = "/opt/dlami/nvme/echonet_dynamic/test_local.csv"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

# Load encoder
ckpt = torch.load(SALT_CKPT, map_location="cpu", weights_only=False)
model = vit.__dict__["vit_large"](
    img_size=224, num_frames=16, patch_size=16, tubelet_size=2,
    uniform_power=True, use_rope=True,
)
state = ckpt["encoder"]
state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.eval().cuda()

# Load probe
probe_ckpt = robust_checkpoint_loader(SALT_PROBE, map_location="cpu")
best_vals = probe_ckpt["best_val_acc_per_head"]
best_idx = best_vals.index(min(best_vals))
sd = probe_ckpt["classifiers"][best_idx]
if any(k.startswith("module.") for k in sd):
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
sa_blocks = set()
for k in sd:
    if k.startswith("pooler.blocks."):
        sa_blocks.add(int(k.split(".")[2]))
depth = len(sa_blocks) + 1
probe = AttentiveRegressor(embed_dim=1024, num_heads=16, depth=depth, num_targets=1)
probe.load_state_dict(sd)
probe.eval().cuda()

target_mean = probe_ckpt.get("target_mean")
target_std = probe_ckpt.get("target_std")
print(f"target_mean={target_mean}, target_std={target_std}")
print(f"All head vals: {best_vals}")

# Read test data
paths, labels = [], []
with open(TEST_CSV) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            paths.append(parts[0])
            labels.append(float(parts[1]))

# Test first 20 videos
N = 20
preds, labs = [], []
print(f"\n{'video':<60} {'label':>8} {'pred':>8} {'z-score':>8}")
print("-" * 90)

for i in range(N):
    try:
        vr = decord.VideoReader(paths[i], num_threads=1)
        total = len(vr)
        needed = 16 * 2
        start = max(0, (total - needed) // 2)
        indices = list(range(start, min(start + needed, total), 2))
        while len(indices) < 16:
            indices.append(indices[-1])
        indices = indices[:16]
        clip = vr.get_batch(indices).permute(3, 0, 1, 2).float() / 255.0
        clip = TF.resize(clip, [224, 224], antialias=True)
        clip = (clip - MEAN) / STD
        clip = clip.unsqueeze(0).cuda()

        with torch.no_grad():
            features = model(clip)
            z = probe(features).cpu().squeeze().item()

        raw = z * target_std + target_mean if target_mean is not None else z
        preds.append(raw)
        labs.append(labels[i])

        fname = paths[i].split("/")[-1]
        print(f"{fname:<60} {labels[i]:>8.2f} {raw:>8.2f} {z:>8.4f}")
    except Exception as e:
        print(f"Error on {paths[i]}: {e}")

# Compute R²
preds = np.array(preds)
labs = np.array(labs)
ss_res = np.sum((labs - preds) ** 2)
ss_tot = np.sum((labs - labs.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
pearson = np.corrcoef(preds, labs)[0, 1]
mae = np.mean(np.abs(labs - preds))

print(f"\nR² = {r2:.4f}")
print(f"Pearson = {pearson:.4f}")
print(f"MAE = {mae:.2f}")
print(f"Pred range: [{preds.min():.1f}, {preds.max():.1f}], mean={preds.mean():.1f}")
print(f"Label range: [{labs.min():.1f}, {labs.max():.1f}], mean={labs.mean():.1f}")
