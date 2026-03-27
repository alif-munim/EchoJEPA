# Bug 014: PyTorch Zipfile Serializer Fails on Checkpoints >4 GB

**Severity:** CRITICAL
**Status:** FIXED (2026-03-27)
**Files:** `app/byol_video/train.py`, `app/byol_video/utils.py`

## Summary

PyTorch's `torch.save()` uses a zipfile-based serializer whose offset tracking uses 32-bit integers. When the checkpoint file exceeds ~4 GB (the 32-bit boundary), `inline_container.cc` raises an "unexpected pos" error, corrupting or failing the save.

BYOL-Video ViT-L checkpoints are ~4.8 GB total (model ~2.4 GB + optimizer ~2.4 GB), consistently triggering this bug.

## Root Cause

PyTorch's zipfile serializer (`torch/_C/inline_container.cc`) tracks file positions with 32-bit offsets internally, despite ZIP64 support at the format level. When writing entries past the ~4.3 GB mark (2^32 bytes), the position calculation overflows, producing "unexpected pos" errors.

This is a known PyTorch issue, not specific to our code. However, our checkpoint files are large enough (ViT-L + AdamW optimizer = ~4.8 GB) to trigger it.

## Misdiagnosis

Initially misdiagnosed as shared tensor storage from DDP/`torch.compile`. Two fixes were attempted before identifying the true root cause:

1. **`_unwrap_state_dict()`** — Unwrapped DDP `.module` and torch.compile `._orig_mod` before calling `state_dict()`. This helped with clean state dict extraction but didn't fix the size issue.
2. **`_clone_for_save()`** — Recursively cloned all tensors with explicit `.clone()` to break shared storage (since `copy.deepcopy` preserves shared storage). Still failed because the combined file was >4 GB regardless of storage sharing.

The giveaway was that the error always occurred at ~4.5 GB, consistently near the 32-bit boundary, regardless of whether tensors shared storage.

## Fix

Split the checkpoint into two files, each under 4 GB:

- `latest.pt` — model weights, schedulers, epoch counter (~2.4 GB)
- `latest_opt.pt` — optimizer state only (~2.4 GB)

Both use atomic writes (save to `.tmp`, then `os.replace`).

```python
# train.py — save
model_dict = {k: v for k, v in save_dict.items() if k != "opt"}
opt_dict = {"opt": save_dict["opt"]}
torch.save(model_dict, tmp_path)
os.replace(tmp_path, local_path)
torch.save(opt_dict, opt_tmp_path)
os.replace(opt_tmp_path, opt_path)
```

```python
# utils.py — load
if "opt" in checkpoint:
    opt.load_state_dict(checkpoint["opt"])
else:
    opt_path = r_path.replace(".pt", "_opt.pt")
    if os.path.exists(opt_path):
        opt_ckpt = robust_checkpoint_loader(opt_path)
        opt.load_state_dict(opt_ckpt["opt"])
```

S3 upload logic also updated to upload both files.

**Secondary fix:** Added `import os` to `utils.py` (was missing, caused `NameError` on first attempt to load split checkpoint).

## Validation

1. Mini-epoch test (ipe=5): both files saved successfully at 2.4 GB each
2. Full training resumed from split checkpoint without errors
3. All model weights loaded with `<All keys matched successfully>`

## Impact

- Blocked BYOL-Video pretraining for ~2 hours (multiple failed checkpoint attempts)
- Affected any model with >4 GB combined checkpoint (ViT-L and above with optimizer state)
- V-JEPA 2.1 `train.py` also updated with `_clone_for_save()` but does not yet use split save (checkpoint may be smaller)

## Lessons

1. **When errors occur at consistent file size boundaries, suspect the serializer, not the data.** The 4 GB / 32-bit boundary is a well-known issue in many archive formats.
2. **Split large checkpoints by component.** Model and optimizer are independently useful and independently loadable. Splitting also enables resuming training from just the model weights.
3. **Always verify imports after adding `os.path` calls.** The missing `import os` in `utils.py` was caught only after a runtime crash.
