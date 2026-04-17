"""Quick diagnostic: inspect SALT checkpoint and test single-video inference."""
import sys
import torch
sys.path.insert(0, ".")

SALT_CKPT = "/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/e79.pt"
SALT_PROBE = "/opt/dlami/nvme/lvef_probes/salt_v1_e79_best.pt"
TEST_CSV = "/opt/dlami/nvme/echonet_dynamic/test_local.csv"

# 1. Inspect checkpoint top-level keys
print("=== Checkpoint top-level keys ===")
ckpt = torch.load(SALT_CKPT, map_location="cpu", weights_only=False)
for k in sorted(ckpt.keys()):
    v = ckpt[k]
    if isinstance(v, dict):
        print(f"  {k}: dict with {len(v)} keys")
    elif isinstance(v, torch.Tensor):
        print(f"  {k}: tensor {v.shape}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

# 2. Check if target_encoder exists
print(f"\n'target_encoder' in ckpt: {'target_encoder' in ckpt}")
print(f"'encoder' in ckpt: {'encoder' in ckpt}")

# 3. Show first 20 encoder keys (after prefix stripping)
enc_state = ckpt["encoder"]
stripped = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_state.items()}
print(f"\n=== Encoder state dict: {len(stripped)} keys (after stripping) ===")
for k in sorted(stripped.keys())[:20]:
    print(f"  {k}: {stripped[k].shape}")
print("  ...")

# 4. Check for norm-related keys
norm_keys = [k for k in stripped if "norm" in k.lower()]
print(f"\n=== Norm-related keys ({len(norm_keys)}) ===")
for k in sorted(norm_keys):
    print(f"  {k}: {stripped[k].shape}")

# 5. Load into standard ViT and check missing/unexpected
import src.models.vision_transformer as vit
model = vit.__dict__["vit_large"](
    img_size=224, num_frames=16, patch_size=16, tubelet_size=2,
    uniform_power=True, use_rope=True,
)
model_sd = model.state_dict()
missing = [k for k in model_sd if k not in stripped]
unexpected = [k for k in stripped if k not in model_sd]
print(f"\n=== Loading comparison ===")
print(f"Model keys: {len(model_sd)}")
print(f"Checkpoint keys: {len(stripped)}")
print(f"Missing in ckpt: {len(missing)} -> {missing}")
print(f"Unexpected: {len(unexpected)} -> {unexpected[:15]}")

# 6. Load without norms_block mapping and run single video
model.load_state_dict(stripped, strict=False)
model.eval().cuda()

# Load a single video
import decord
import torchvision.transforms.functional as TF
decord.bridge.set_bridge("torch")

with open(TEST_CSV) as f:
    first_line = f.readline().strip().split()
video_path, label = first_line[0], float(first_line[1])
print(f"\n=== Test video: {video_path}, label={label} ===")

vr = decord.VideoReader(video_path, num_threads=1)
total = len(vr)
needed = 16 * 2
start = max(0, (total - needed) // 2)
indices = list(range(start, min(start + needed, total), 2))
while len(indices) < 16:
    indices.append(indices[-1])
indices = indices[:16]
clip = vr.get_batch(indices).permute(3, 0, 1, 2).float() / 255.0
clip = TF.resize(clip, [224, 224], antialias=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
clip = (clip - MEAN) / STD
clip = clip.unsqueeze(0).cuda()

with torch.no_grad():
    features = model(clip)
print(f"Features shape: {features.shape}")
print(f"Features mean: {features.mean():.4f}, std: {features.std():.4f}")
print(f"Features min: {features.min():.4f}, max: {features.max():.4f}")

# 7. Try with norms_block.3 mapping
model2 = vit.__dict__["vit_large"](
    img_size=224, num_frames=16, patch_size=16, tubelet_size=2,
    uniform_power=True, use_rope=True,
)
stripped2 = dict(stripped)
if "norms_block.3.weight" in stripped2:
    stripped2["norm.weight"] = stripped2["norms_block.3.weight"]
    stripped2["norm.bias"] = stripped2["norms_block.3.bias"]
model2.load_state_dict(stripped2, strict=False)
model2.eval().cuda()

with torch.no_grad():
    features2 = model2(clip)
print(f"\nWith norms_block.3 mapping:")
print(f"Features shape: {features2.shape}")
print(f"Features mean: {features2.mean():.4f}, std: {features2.std():.4f}")

# 8. Load probe and test both
from src.models.attentive_pooler import AttentiveRegressor
from src.utils.checkpoint_loader import robust_checkpoint_loader

probe_ckpt = robust_checkpoint_loader(SALT_PROBE, map_location="cpu")
best_vals = probe_ckpt["best_val_acc_per_head"]
best_idx = best_vals.index(min(best_vals))
state_dict = probe_ckpt["classifiers"][best_idx]
if any(k.startswith("module.") for k in state_dict):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

sa_block_indices = set()
for k in state_dict:
    if k.startswith("pooler.blocks."):
        sa_block_indices.add(int(k.split(".")[2]))
depth = len(sa_block_indices) + 1

probe = AttentiveRegressor(embed_dim=1024, num_heads=16, depth=depth, num_targets=1)
probe.load_state_dict(state_dict)
probe.eval().cuda()

target_mean = probe_ckpt.get("target_mean")
target_std = probe_ckpt.get("target_std")
print(f"\nProbe: depth={depth}, target_mean={target_mean}, target_std={target_std}")

with torch.no_grad():
    pred1 = probe(features).cpu().squeeze().item()
    pred2 = probe(features2).cpu().squeeze().item()

if target_mean is not None and target_std is not None:
    pred1_raw = pred1 * target_std + target_mean
    pred2_raw = pred2 * target_std + target_mean
else:
    pred1_raw = pred1
    pred2_raw = pred2

print(f"\nPrediction (no norm mapping):  z={pred1:.4f}, raw={pred1_raw:.2f}")
print(f"Prediction (norms_block.3):    z={pred2:.4f}, raw={pred2_raw:.2f}")
print(f"Label: {label}")
