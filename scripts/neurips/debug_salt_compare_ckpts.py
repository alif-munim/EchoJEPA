"""Compare the two SALT checkpoints to see if encoder weights differ."""
import torch

CKPT1 = "/opt/dlami/nvme/checkpoints/pretrain/mimic/salt_s2_vitl_224px_16f/e79.pt"
CKPT2 = "/opt/vjepa2/checkpoints/pretrain/mimic/salt_s2v1_e79.pt"

c1 = torch.load(CKPT1, map_location="cpu", weights_only=False)
c2 = torch.load(CKPT2, map_location="cpu", weights_only=False)

# Compare metadata
for k in ["epoch", "loss", "lr", "stage", "batch_size", "world_size", "teacher_checkpoint"]:
    v1 = c1.get(k)
    v2 = c2.get(k)
    same = v1 == v2
    print(f"  {k}: {v1} vs {v2}  {'✓' if same else '✗ DIFFERENT'}")

# Compare encoder keys
sd1 = c1["encoder"]
sd2 = c2["encoder"]
print(f"\nEncoder keys: {len(sd1)} vs {len(sd2)}")
keys1 = set(sd1.keys())
keys2 = set(sd2.keys())
if keys1 != keys2:
    print(f"  Extra in ckpt1: {keys1 - keys2}")
    print(f"  Extra in ckpt2: {keys2 - keys1}")
else:
    print("  Same key set ✓")

# Compare encoder weight values
n_different = 0
max_diff = 0.0
different_keys = []
for k in sorted(keys1 & keys2):
    if not torch.equal(sd1[k], sd2[k]):
        n_different += 1
        d = (sd1[k].float() - sd2[k].float()).abs().max().item()
        max_diff = max(max_diff, d)
        different_keys.append((k, d))

print(f"\nDiffering tensors: {n_different}/{len(keys1 & keys2)}")
print(f"Max absolute diff: {max_diff}")
if different_keys:
    print("First 10 differing keys:")
    for k, d in different_keys[:10]:
        print(f"  {k}: max_diff={d:.6f}")
else:
    print("Encoder weights are IDENTICAL ✓")

# Compare predictor
if "predictor" in c1 and "predictor" in c2:
    pd1 = c1["predictor"]
    pd2 = c2["predictor"]
    print(f"\nPredictor keys: {len(pd1)} vs {len(pd2)}")
    n_diff_p = sum(1 for k in pd1 if k in pd2 and not torch.equal(pd1[k], pd2[k]))
    print(f"Differing predictor tensors: {n_diff_p}")
