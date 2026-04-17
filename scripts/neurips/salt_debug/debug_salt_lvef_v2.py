"""Debug SALT LVEF: test all probe heads, bfloat16, and compare checkpoints."""
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
# Check if alternate checkpoint exists
SALT_CKPT_ALT = "/opt/vjepa2/checkpoints/pretrain/mimic/salt_s2v1_e79.pt"
SALT_PROBE = "/opt/dlami/nvme/lvef_probes/salt_v1_e79_best.pt"
TEST_CSV = "/opt/dlami/nvme/echonet_dynamic/test_local.csv"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

import os
print(f"Primary CKPT exists: {os.path.exists(SALT_CKPT)}")
print(f"Alt CKPT exists: {os.path.exists(SALT_CKPT_ALT)}")

# Check if they're the same file
if os.path.exists(SALT_CKPT) and os.path.exists(SALT_CKPT_ALT):
    s1 = os.path.getsize(SALT_CKPT)
    s2 = os.path.getsize(SALT_CKPT_ALT)
    print(f"Primary size: {s1:,} bytes")
    print(f"Alt size: {s2:,} bytes")
    print(f"Same size: {s1 == s2}")

    # Compare a specific tensor
    ckpt1 = torch.load(SALT_CKPT, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(SALT_CKPT_ALT, map_location="cpu", weights_only=False)
    k = "encoder"
    sd1 = ckpt1[k]
    sd2 = ckpt2[k]
    first_key = list(sd1.keys())[0]
    print(f"\nComparing tensor '{first_key}':")
    print(f"  Equal: {torch.equal(sd1[first_key], sd2[first_key])}")
    print(f"  Max diff: {(sd1[first_key] - sd2[first_key]).abs().max().item()}")
    del ckpt2
    ckpt = ckpt1
else:
    ckpt = torch.load(SALT_CKPT, map_location="cpu", weights_only=False)

# Load encoder
model = vit.__dict__["vit_large"](
    img_size=224, num_frames=16, patch_size=16, tubelet_size=2,
    uniform_power=True, use_rope=True,
)
state = ckpt["encoder"]
state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.eval().cuda()
del ckpt

# Load videos
paths, labels = [], []
with open(TEST_CSV) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            paths.append(parts[0])
            labels.append(float(parts[1]))

def load_clip(path):
    vr = decord.VideoReader(path, num_threads=1)
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
    return clip.unsqueeze(0).cuda()

# Load probe checkpoint (all heads)
probe_ckpt = robust_checkpoint_loader(SALT_PROBE, map_location="cpu")
target_mean = probe_ckpt.get("target_mean")
target_std = probe_ckpt.get("target_std")
best_vals = probe_ckpt["best_val_acc_per_head"]
print(f"\ntarget_mean={target_mean}, target_std={target_std}")
print(f"Head MAEs: {[f'{v:.3f}' for v in best_vals]}")

# Test 50 videos with ALL heads
N = 50
head_preds = {i: [] for i in range(len(best_vals))}
head_preds_bf16 = {i: [] for i in range(len(best_vals))}
labs = []

# Build probes for all heads
probes = {}
for head_idx in range(len(best_vals)):
    sd = probe_ckpt["classifiers"][head_idx]
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
    probes[head_idx] = probe

print(f"\nRunning {N} videos through all {len(probes)} probe heads...")

for i in range(N):
    try:
        clip = load_clip(paths[i])
    except:
        continue

    with torch.no_grad():
        # Float32
        features = model(clip)
        for h, probe in probes.items():
            z = probe(features).cpu().squeeze().item()
            raw = z * target_std + target_mean
            head_preds[h].append(raw)

        # Bfloat16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features_bf16 = model(clip)
        for h, probe in probes.items():
            z = probe(features_bf16.float()).cpu().squeeze().item()
            raw = z * target_std + target_mean
            head_preds_bf16[h].append(raw)

    labs.append(labels[i])

labs = np.array(labs)

print(f"\n{'Head':<8} {'ValMAE':>8} {'R²(f32)':>10} {'R²(bf16)':>10} {'Range(f32)':>15} {'Range(bf16)':>15}")
print("-" * 75)
for h in range(len(best_vals)):
    p32 = np.array(head_preds[h])
    p16 = np.array(head_preds_bf16[h])

    def r2(p):
        ss_res = np.sum((labs - p) ** 2)
        ss_tot = np.sum((labs - labs.mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0

    r2_32 = r2(p32)
    r2_16 = r2(p16)
    rng32 = f"[{p32.min():.1f}, {p32.max():.1f}]"
    rng16 = f"[{p16.min():.1f}, {p16.max():.1f}]"
    print(f"  {h:<6} {best_vals[h]:>8.3f} {r2_32:>10.4f} {r2_16:>10.4f} {rng32:>15} {rng16:>15}")

# Feature comparison fp32 vs bf16
with torch.no_grad():
    clip = load_clip(paths[0])
    f32 = model(clip)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fbf = model(clip)
    diff = (f32 - fbf.float()).abs()
    print(f"\nFeature diff (fp32 vs bf16): mean={diff.mean():.6f}, max={diff.max():.6f}")
    print(f"fp32 stats: mean={f32.mean():.6f}, std={f32.std():.6f}")
    print(f"bf16 stats: mean={fbf.float().mean():.6f}, std={fbf.float().std():.6f}")
