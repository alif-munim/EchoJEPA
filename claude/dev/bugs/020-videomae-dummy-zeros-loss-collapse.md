# Bug 020: VideoMAE `__getitem__` returns zero tensors on S3 failure, collapsing loss to ~0

**Severity:** CRITICAL
**Discovered:** 2026-04-21
**Status:** FIXED upstream in commit `d91b4d4` ("Fix videomae recursive calls, s3 credential expiry"). Still re-occurs when a stale `vjepa2-src.tar.gz` is deployed to HyperPod.
**Affected file:** `s3_dataset.py` (VideoMAE pretraining dataset)

## Summary

The VideoMAE pretraining dataset's `__getitem__` caught any exception and returned `torch.zeros((3, T, H, W))` as a silent fallback. When S3 reads started failing in bursts (>700 errors/min from ~30 min into training), a large fraction of each batch became all-zero frames. MAE normalizes target patches per-clip, so zero-frame targets become degenerate and the per-step loss drops from ~0.42 → ~0.001. Looks like a training instability; is actually dataloader-induced signal loss.

## Root Cause

Old `s3_dataset.py` `__getitem__` (before `d91b4d4`):

```python
def __getitem__(self, idx):
    path = self.samples[idx]
    mask = self.masked_position_generator()
    try:
        vr = self.loadvideo_decord(path)
        ...
        return frames, mask
    except Exception as e:
        print(f"[WARN] RETURNING DUMMY for {path} | Error: {e}", flush=True)
        dummy = torch.zeros((3, self.frames_per_clip, self.crop_size, self.crop_size))
        return dummy, torch.from_numpy(mask).bool()
```

Combined with `loadvideo_decord`'s weak retry loop (3 attempts, 0.1s/0.2s/0.4s sleep) and a fallthrough to `decord.VideoReader(sample)` on an `s3://` path (which fails with "Protocol not found" because decord has no S3 plugin), transient S3 hiccups are reliably converted to zero tensors rather than valid clips. The loss signal vanishes whenever S3 has a rough minute.

## How Discovered

Three failed ViT-B IN21K MAE runs (jobs 293, 295, 297) all collapsed in epoch 1 around step ~750–1050. Running avg loss stayed healthy (~0.42) because it was anchored by the earlier healthy steps; the per-step loss showed 0.41 → 0.005 → 0.001 within ~30 steps. Step time simultaneously jumped from 0.5s to ~4s.

Investigation narrowed it down:

1. **Not credential expiration per se.** The concurrent ViT-L JEPA run on a different compute node (`ip-10-0-50-35`) had no cred-refresh cron installed and was still healthy at 12+ hours. So AWS creds alone couldn't be the cause.
2. **Different dataset class.** JEPA uses `src/datasets/video_dataset.py`, which on load failure **substitutes** a random valid sample (up to 50 retries) and logs `"Retrying with new sample"`. VideoMAE uses `s3_dataset.py`, which returned zeros.
3. **Stale tarball.** The current repo root `s3_dataset.py` already has the `max_retries=50` substitution fix. The deployed `/opt/dlami/nvme/mae_b_21k_{293,295,297}/code/s3_dataset.py` on the compute node still had the old `RETURNING DUMMY` code — whoever built the tarball earlier in the session picked up a pre-`d91b4d4` version. Job 298 was launched with a freshly-rebuilt tarball (04:12:01 UTC) containing the fix, and is training healthy past the collapse point.

## Why MAE Loss Goes to Zero (Not NaN)

VideoMAE `normalize_target: True` normalizes each target patch by its mean/std within the clip. If the clip is all zeros, per-patch normalization produces `0/0 → 0` (with eps). The target is a constant, which the model can match trivially, so reconstruction MSE → 0. The optimizer happily rides the gradient down a cliff with no warning.

This is *not* a classic NaN/overflow collapse — weights remain finite, `loss_scale` stays at 65536, `grad_norm` decays smoothly toward zero. It looks like convergence.

## Impact

- Three wasted ViT-B MAE training attempts (~90 min of H100 time each, plus env setup).
- Every other VideoMAE pretraining job deployed with a stale tarball is at risk whenever S3 hiccups for more than a few minutes.
- Silent: because `print` goes to stdout and the dummy path still returns valid-shape tensors, there's no exception, no NaN, and no obvious log signature other than a burst of `[WARN] RETURNING DUMMY` lines buried in stdout.

## Fix (Already in Repo)

Commit `d91b4d4` replaced the single-attempt + dummy-zeros path with a 50-attempt substitution loop (mirrors the JEPA pipeline's behavior):

```python
def __getitem__(self, idx):
    max_retries = 50
    for attempt in range(max_retries):
        cur_idx = idx if attempt == 0 else np.random.randint(len(self.samples))
        path = self.samples[cur_idx]
        mask = self.masked_position_generator()
        try:
            vr = self.loadvideo_decord(path)
            ...
            return frames, mask
        except Exception as e:
            print(f"[WARN] Retry {attempt}/{max_retries} for {path} | Error: {e}", flush=True)
            continue
    # final fallback: raise or dummy (last resort, very unlikely)
```

## Deployment Gotcha

The HyperPod workflow deploys code via `s3://.../setup/vjepa2-src.tar.gz`. If the tarball was built before `d91b4d4` was committed, jobs pick up the old dataset and will collapse on the next S3 hiccup. **Always rebuild and re-upload the tarball after pulling latest before resubmitting VideoMAE pretraining.**

Verification that a deployed tarball has the fix:

```bash
# On controller:
srun --jobid=<RUNNING_JOB> --overlap --ntasks=1 \
  grep -c max_retries /opt/dlami/nvme/<workdir>/code/s3_dataset.py
# Expect: 1 (has the fix) or 0 (old, vulnerable)
```

## Diagnostic Signature

Symptoms that indicate this failure mode (vs. a real model instability):

1. Per-step `loss:` in the `Epoch: [N] [step/N]` line drops by ~2 orders of magnitude within ~20 steps, while the running average stays near the healthy value.
2. `time:` per step spikes (0.5s → ~4s) simultaneously with the loss drop (S3 retry latency).
3. stderr shows bursts of decord errors: `"moov atom not found"`, `"Invalid data found when processing input"`, `"Protocol not found"` (the s3:// fallthrough).
4. stdout shows repeated `[WARN] RETURNING DUMMY for s3://...` — this is the smoking gun and only appears on the buggy code path.
5. `loss_scale` stays at 65536 (no overflow); `grad_norm` decays smoothly toward zero.

## Related

- Bug 004: same general phenomenon in the JEPA pipeline (silent load-failure substitution), but JEPA substitutes a real clip, so training stays correct; only the clip↔index mapping is wrong for extraction. VideoMAE is the more dangerous variant because it substitutes *zeros*.
- `feedback_aws_creds_hyperpod.md` (user memory): STS expiration on mp.Process workers. Not the root cause here (ViT-L ran fine without cred refresh), but an orthogonal contributor to S3-read failure bursts on affected nodes.
