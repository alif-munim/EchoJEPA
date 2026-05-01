# phase_relational pretrain: launch debugging log

Running record of failures, root causes, and fixes on the first method
pretrain attempt. Covers jobs 579, 590, 591, and the aborted control
job 584. Written on 2026-05-01 while the third set of fixes is pending
resubmit.

Files referenced throughout:
- `app/vjepa_multiview/train.py` — training entrypoint
- `app/vjepa_multiview/phase_relational_head.py` — the InfoNCE-time head
- `app/main.py` — per-rank child launcher
- `src/utils/distributed.py` — `init_process_group` helper
- `src/utils/wrappers.py` — `MultiSeqWrapper` (returns nested lists)
- `classifier/phase/sampler/phase_matched_sampler.py` — sampler with hard-neg logic
- `classifier/phase/sampler/phase_matched_pair_dataset.py` — per-epoch builder
- `scripts/neurips/phase/final_phase_rel_hardneg25_paper.sbatch`
- `configs/train/vitl16/pretrain-multiview-phase-relational-hardneg-25of100-paper.yaml`
- Logs at `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/final_phase_rel25_paper_{579,590,591}/logs/`

## TL;DR

**Three sequential failures, each exposed only after the previous was fixed.**

1. **Job 579** — I cancelled prematurely on a wrong NCCL theory.
   The actual bug would have surfaced at ~27 min with the traceback
   in 590.
2. **Job 590** — `pool_tokens(z_ctx[0])` crashed because
   `MultiSeqWrapper` returns `list[list[Tensor]]` when called with
   masks. Fixed to `z_ctx[0][0]`.
3. **Job 591** — `relational_head.query(...)` crashed because DDP's
   `__getattr__` doesn't forward custom methods. Initially patched to
   route three calls through an op-dispatcher `forward(op, ...)`, then
   re-patched per user direction to a **single unified** `forward(c_a,
   view_a, view_bp, dphi, y_pos, y_neg) -> (q, y_pos, y_hard)` so the
   DDP reducer touches all head parameters in one call.

All three bugs manifest only at the first optimizer step. The 12-min
pair-build + 3-min cold S3 fetch window means it takes **~27 min per
failed attempt** to observe the crash.

## Test matrix present at the time of each failure

Before any of these runs was submitted, we had:

| Test file | Scope |
|---|---|
| `test_candidate_set_infonce.py` | 6 tests: InfoNCE column ordering, labels all 0, same-study mask, masked-all-negatives finite, `_build_predictor_inputs` arity, gradient flow through a non-DDP head |
| `test_wrong_phase_strategy.py` | 4 synthetic sampler strategy tests |
| `check_triple_sampler_yield.py` | Metadata-only yield gate on 198k-clip parquet |
| `check_triple_clip_smoke.py` | Local DataLoader plumbing on on-disk DICOMs |
| `verify_init_checkpoint.py` | sha256 allowlist preflight |

**Gap that hid bugs 590 and 591**: nothing exercised `forward_phase_relational`
end-to-end with a real `MultiSeqWrapper`-wrapped encoder or with a
DDP-wrapped head. The unit tests called `head.query(...)` / `head.target(...)`
on a bare `PhaseRelationalHead`, which is not what happens in production.

The `test_ddp_wrapped_head.py` smoke test added this turn closes that
gap for the head; there is still no end-to-end integration test for
`forward_phase_relational` + `MultiSeqWrapper`, and that is the next
testing gap to fill.

## Normal startup timeline

Per-epoch cost breakdown for a healthy run, derived from the 590 and
591 logs (identical timings to within a few seconds):

| Stage | Duration | What's happening |
|---|---|---|
| sbatch → bash trap | ~10 s | NVMe setup, AWS creds, tarball download, tar extract, path exports |
| `verify_init_checkpoint.py --strict` | ~6 s | sha256 over 5.13 GB, torch.load header, sha256 allowlist check |
| Python launch → `process_main` spawn × 8 | ~10 s | `mp.set_start_method('spawn')`, 8 child processes, each sets `CUDA_VISIBLE_DEVICES` mask |
| `init_process_group` + NCCL rendezvous | ~30-40 s | Gloo/TCPStore handshake, NCCL group creation; prints the benign "Guessing device ID" warning |
| Build encoder + predictor + target_encoder | ~30 s | ViT-L/16 + 12-layer predictor, CUDA kernel warmup |
| `PhaseMatchedStudySampler._load` pair-index cache | ~3-4 s | One-time per-study index over 6947 studies |
| Load IN21K e100 weights into encoder + target_encoder + predictor | ~3-5 s | `torch.load` + `.load_state_dict` across 3 modules |
| DDP wrap (encoder + predictor + target_encoder + head) | ~1 s | `DistributedDataParallel` constructor |
| **`builder.refresh_epoch(0)` — hard-neg triple construction** | **~12 min** | Per-rank: 20,841 anchors × up to 16 retries × same-view-then-same-family filter + circular Δφ ≥ 0.25 check; all scans linear over `study_to_rows[study_key]` |
| frame-guard (verify clip_b_pos frame counts on first 8 pair rows) | ~1 s | Sanity check |
| First-batch cold S3 fetch + decord decode | ~3 min | 3 clips × batch 32 × 8 ranks = 768 mp4 downloads over boto3, each ~300 KB to 2 MB, decoded by decord |
| **First optimizer step lands, first CSV row appears** | +0:00 | Target state |

**Total to first step (healthy)**: ≈ **16-17 min** from job start.

Observed actual delta from "FORCE-LOAD" line to "refreshed pairs rank=0":
- Job 590: 12:11 (08:11:10 → 08:23:21)
- Job 591: 12:29 (08:43:38 → 08:56:07)

These are stable within a few seconds; the cost is deterministic
Python work on the same data.

## Per-job forensic detail

### Job 579 — first submit (08:00:17 UTC)

sbatch: `final_phase_rel_hardneg25_paper.sbatch` (stage 1a).
SLURM id: 579. 1 node, 8 GPUs, 36h wall clock.

**Observed timeline** (reconstructed from the S3-synced log after cancel):

| Time (UTC) | Event |
|---|---|
| 07:44:? | sbatch start |
| 07:45-07:46 | tarball download + extract + init-checkpoint preflight |
| 07:46:09 | DDP init: rank2 prints "Guessing device ID" warning |
| 07:46:54 | Per-study pair-index cache built in 3.0 s |
| 07:46:58 | "Loaded encoder: <All keys matched successfully>" |
| 07:58:51 | [epoch 0] refreshed pairs: rank=X n=20841 (pair-build ends) |
| 08:00:?? | I polled, saw CSV header only, GPU 0%, 22:29 elapsed |
| 08:02-08:03 | I decided this was a NCCL hang (wrong) and **cancelled via scancel 579** |

**Why I cancelled**: the PyTorch warning at 07:46:09,
```
Guessing device ID based on global rank. This can cause a hang if rank
to GPU mapping is heterogeneous.
```
text-matched "hang" and I assumed causation. With 22+ min elapsed and
no CSV row + 0% GPU I decided to cut losses.

**Why that was wrong**:
- `app/main.py:39` masks `CUDA_VISIBLE_DEVICES` per child: each rank's
  `cuda:0` is correctly "the one GPU this child can see."
- This is the same path 542/548 used successfully.
- The warning is new in recent PyTorch ("Guessing device ID" was added
  in PyTorch 2.3), emitted whenever `init_process_group` runs without
  an explicit `device_id=` kwarg. It's noisy but benign under a
  per-rank CUDA mask.
- Had I waited 5 more min I would have seen the actual 590 traceback
  in 579's logs.

**Cost of the cancel**: 22 min wasted GPU time + ~5 min to restart.

**Lesson**: a "stall" with 8 CPU-active Python processes and no
traceback is not yet diagnosed. Wait for either a traceback or a truly
dead process (wchan blocked on `poll`, zero CPU, zero TCP traffic for
10+ min).

### Job 590 — second submit (08:09:17 UTC)

Same code as 579. Submitted without any changes.

**Exact stage timings**:

| Time (UTC) | Event | Δ from job start |
|---|---|---|
| 08:09:17 | sbatch submit | 0:00 |
| 08:11:10 | FORCE-LOADING pretrained model | 1:53 |
| 08:11:11 | Loaded encoder: <All keys matched> | 1:54 |
| 08:23:13 | [epoch 0] refreshed pairs: rank=7 (first rank done) | 13:56 |
| 08:23:21 | [epoch 0] refreshed pairs: rank=0 (last rank done) | 14:04 |
| 08:23:21 | [frame-guard] verifying frame counts on first 8 pair rows | 14:04 |
| 08:23:23 | [frame-guard] OK: 8 rows × 2 sides | 14:06 |
| ~08:27 | First `_step()` call → crash in `forward_phase_relational` | ~17:43 |
| 08:36:31 | NCCL cleanup + post-mortem S3 log sync done; bash trap exits | 27:14 |

**The traceback** (all 8 ranks):

```
File ".../app/vjepa_multiview/train.py", line 705, in forward_phase_relational
    c_a_pool = pool_tokens(z_ctx[0])
               ^^^^^^^^^^^^^^^^^^^^^
File ".../app/vjepa_multiview/phase_relational_head.py", line 64, in pool_tokens
    return x.mean(dim=1)
           ^^^^^^
AttributeError: 'list' object has no attribute 'mean'
```

**Root cause**: `MultiSeqWrapper.forward` in `src/utils/wrappers.py:15-27`:

```python
def forward(self, x, masks=None):
    if masks is None:
        return [self.backbone(xi) for xi in x]   # flat list[Tensor]
    outs = [[] for _ in x]
    for i, (xi, mi) in enumerate(zip(x, masks)):
        for mij in mi:
            outs[i] += [self.backbone(xi, masks=mij)]
    return outs                                   # NESTED list[list[Tensor]]
```

Called on the student side as
```python
z_ctx = encoder(pair.clip_a, pair.masks_enc)
```
with `masks_enc = list[list[Tensor]]` (outer: fpc, inner: mask-generators).
Result: `z_ctx = [[T_fpc0_mask0, T_fpc0_mask1]]` — outer length 1 (one
fpc entry), inner length 2 (context mask + target mask).

My code did `z_ctx[0]`, which is still `[T_mask0, T_mask1]` — a
**list**. Then `pool_tokens(list).mean(dim=1)` → `AttributeError`.

Teacher side was fine because `target_encoder(concat_fpc)` was called
**without masks**, hitting the `return [self.backbone(xi) for xi in x]`
branch and producing a flat `[Tensor]`. `h_b_pos[0]` was a tensor.

**Why unit tests missed it**: no unit test exercised
`forward_phase_relational` with a real `MultiSeqWrapper`-wrapped
encoder. `test_candidate_set_infonce.py` called `head.query()` /
`head.target()` with synthetic tensors.

**Fix 1 (Patch 1a)** — `app/vjepa_multiview/train.py`:
```python
# Before
c_a_pool = pool_tokens(z_ctx[0])

# After
assert isinstance(z_ctx, list) and len(z_ctx) >= 1, ...
assert isinstance(z_ctx[0], list) and len(z_ctx[0]) >= 1, ...
c_a_pool = pool_tokens(z_ctx[0][0])   # fpc=0, mask=0 = context tokens
```

The isinstance guards fail loud if `MultiSeqWrapper`'s contract
changes, avoiding another silent integration gap.

### Job 591 — third submit (08:41:44 UTC) with Fix 1

**Exact stage timings**:

| Time (UTC) | Event | Δ from job start |
|---|---|---|
| 08:41:44 | sbatch submit | 0:00 |
| 08:43:38 | FORCE-LOADING pretrained model | 1:54 |
| 08:43:43 | Loaded predictor: <All keys matched> | 1:59 |
| 08:55:39 | [epoch 0] refreshed pairs: rank=5 (first rank done) | 13:55 |
| 08:56:07 | [epoch 0] refreshed pairs: rank=0 (last rank done) | 14:23 |
| 08:56:08 | [frame-guard] OK: 8 rows × 2 sides | 14:24 |
| ~09:00 | First `_step()` → crash at `relational_head.query(...)` | ~18:16 |
| 09:08:49 | NCCL cleanup + trap exits | 27:05 |

**The traceback** (all 8 ranks):

```
AttributeError: 'DistributedDataParallel' object has no attribute 'query'
```

Occurred at the call site
```python
q_pre = relational_head.query(c_a_pool, view_a_ids, view_b_pos_ids, delta_phase_pos)
```
inside `forward_phase_relational`.

**Root cause**: when `world_size > 1`, `train.py:1302-1305` wraps
`relational_head` in `DistributedDataParallel`. DDP's `__getattr__`
(defined in `torch/nn/parallel/distributed.py`) forwards attribute
access to the wrapped module's submodules/parameters/buffers — **but
not to custom methods**. So `ddp_head.query` raises.

**Secondary concern** not visible in this crash but real: the initial
patch attempt routed three separate calls through an op-dispatcher
`forward(op: str, *args)`:
```python
q_pre = relational_head("query", c_a_pool, view_a_ids, view_b_pos_ids, delta_phase_pos)
y_pos_pre = relational_head("target", y_pos_pool)
y_hard_pre = relational_head("target", y_neg_pool)
```
This fixes the AttributeError but creates a subtler issue: each DDP
forward touches a disjoint parameter subset (`query` uses source_proj +
relation_mlp + view embeds + phase_mlp; `target` uses only target_proj).
DDP's reducer tracks parameters-marked-ready per forward, and a
3-forwards-per-step pattern with disjoint parameter subsets either
needs `find_unused_parameters=True` (slower reducer + may mask bugs)
or breaks the `static_graph` invariant.

**Fix 2 (final, per user direction)** — `phase_relational_head.py` +
`train.py`:

`PhaseRelationalHead.forward` was defined as a **single unified
forward** that returns all three projection outputs:
```python
def forward(
    self,
    c_a_pool, view_a_ids, view_b_pos_ids, delta_phase_pos,
    y_pos_pool, y_neg_pool,
):
    q_pre = self.query(c_a_pool, view_a_ids, view_b_pos_ids, delta_phase_pos)
    y_pos_pre = self.target(y_pos_pool)
    y_hard_pre = self.target(y_neg_pool)
    return q_pre, y_pos_pre, y_hard_pre
```

The `query()` / `target()` methods remain for unit tests that use an
unwrapped head. Production code calls only `relational_head(...)`, which
becomes `DDP.__call__ → DDP.forward → head.forward` and touches every
parameter of the head in one call.

DDP wrap kwargs reverted to `DistributedDataParallel(relational_head,
static_graph=False)` — no `find_unused_parameters` because every param
is touched.

**New test added: `classifier/phase/sampler/test_ddp_wrapped_head.py`**:
- Single-process DDP group over gloo (doesn't need NCCL or multi-GPU)
- Asserts `ddp_head(...)` returns `(q_pre, y_pos_pre, y_hard_pre)` with
  `[B, rel_dim]` shapes
- Asserts `loss.backward()` succeeds
- Asserts every one of the 6 submodule families (`source_proj`,
  `relation_mlp`, `view_embed_a`, `view_embed_b_pos`, `phase_mlp`,
  `target_proj`) receives gradients
- Asserts no DDP reducer warning (`unused parameter`, etc.)
- Asserts no `AttributeError` on `__call__`
- Asserts teacher-side inputs stay detached

All 2 DDP tests + 6 existing InfoNCE unit tests pass locally.

### Job 584 — control pretrain, unintended start

Submitted earlier in the chain with `Dependency=afterany:591`. When 591
FAILED at 09:08:49, SLURM immediately promoted 584 to RUNNING because
`afterany` is satisfied by *any* terminal state, including FAILED.

Cancelled at 09:13:26 (4:24 elapsed, still in the bash trap / init
phase — had not yet called `python -m app.main`).

**Control is not affected by the 590 or 591 bugs** because
`forward_intraview_only` in `train.py:467` doesn't touch
`relational_head` or the nested `z_ctx` in the same way. But running the
control on a broken method dependency would have produced a control
checkpoint with no matching method checkpoint, making the paper-clean
A/B comparison impossible.

Deliberate design fix for the future: downstream probes should use
`afterok` (strict success), not `afterany`. The control should use
`afterany` only if we're willing to run it as a standalone baseline —
which we are not.

## Data processing cost breakdown

### What makes `refresh_epoch` slow

Per `classifier/phase/sampler/phase_matched_pair_dataset.py:298-318`,
`refresh_epoch(epoch)` does:

1. `sampler.set_epoch(epoch)` → sets an epoch-seeded RNG
2. `sampler.build_records()` → **the 12-min step**
3. `_records_to_pair_dataframe(records, self.sampler._df, ...)` → DataFrame construction
4. `_records_to_anchor_table(records)` → anchor rewrite for `VideoGroupDataset`
5. `self.dataset.set_pair_dataframe(pair_df, anchors_by_index=anchors)` → atomic swap

Cost dominator is step 2. Inside `build_records`, for each of 20,841
per-rank anchors:

- Draw an anchor clip from `study_to_rows[study_key]`
- Phase-match a `clip_b_pos` from the same study with circular Δφ near
  the chosen bucket center, subject to `phase_tolerance: 0.15`
- **`_draw_hard_negative_clip`** under `same_view_then_same_family`:
  - Filter `study_to_rows[study_key]` by `view == clip_b_pos.view`
    (linear scan)
  - For each candidate, compute `circular_dist(phi_candidate,
    target_phi_pos) ≥ 0.25` (fine)
  - Exclude `clip_a.dicom_id` and `clip_b_pos.dicom_id` from candidates
  - If no same-view candidate passes, fall back to same-family filter
  - Repeat with retry up to `max_hard_neg_attempts=16`
  - If all 16 retries fail, resample the entire anchor

On 198,744 eligible clips across 6,947 studies (avg 29 clips/study),
the linear scan is small per anchor but runs `16 × 20,841 = 333,456`
times per rank, in pure Python. This is where the 12 min goes.

### Per-epoch cost over the full run

With no optimization, the cost repeats every epoch because
`build_records()` re-rolls under a new seed:

| Phase | Per-epoch cost | 25-epoch total |
|---|---|---|
| `build_records` pair-build | ~12 min | **~5 h** |
| Cold S3 fetch (first epoch only) | ~3 min | ~3 min |
| Actual training (650 steps × ~3-4 s/step after warm-up) | ~35-45 min | ~16-19 h |
| Checkpoint save at epochs {0, 5, 10, 25} | ~1 min each | ~4 min |
| **Total wall-clock** | | **~21-24 h** |

So the pair-build overhead adds ~20 % to wall-clock vs a no-hard-neg
baseline. Within the sbatch time limit (1d 12h), but not free.

### Optimization candidates (deferred)

Not applied; listed so we can revisit after the method pretrain lands:

1. **Pre-index clips per study per view on sampler `_load()`**:
   `study_to_rows_by_view: dict[study_id, dict[view, list[row_idx]]]`.
   Replaces the per-anchor linear scan with an O(1) lookup. Expected
   savings: ~10 min of the 12-min cost → epoch-0 goes from 12 min to
   ~2 min. Net: ~4 h shaved off the 25-epoch run.

2. **Cache `study_to_rows_by_family`** similarly for the fallback path.

3. **Lower `max_hard_neg_attempts`** from 16 to 8. Pair-build halves but
   some anchors that would have found a hard-neg on retry 9-16 are
   instead resampled. Net effect on training: small, but worth
   measuring once the run is stable.

4. **Parallelize `build_records()` across workers**. Currently each
   rank builds its own records on the main process. The sampler is
   deterministic per `(epoch, rank, seed)` so per-rank builds could
   proceed in a ProcessPoolExecutor. More infrastructure, bigger change.

None of these change the loss or the training behavior, only wall-clock.

### 3-clip-per-sample cost

Distinct from pair-build: the DataLoader now streams 3 mp4s per sample
instead of 2 (`clip_a`, `clip_b_pos`, `clip_b_neg_phase`). Per-step
cost during warm steady state:

- boto3 GET × 3 × 32 samples × 8 ranks = 768 mp4 fetches per step.
  S3 throughput on p5.48xlarge is tens of GB/s aggregate; per-file
  latency is more relevant than total bytes. With 4 DataLoader workers
  per rank and `prefetch_factor=2`, the prefetch should fully overlap
  the forward + backward once warm.
- decord decode: ~50 ms per 16-frame clip. 3 × 32 × 8 = 768 decodes per
  step, distributed across 32 worker processes → ~20 ms per worker per
  step.
- Teacher concat-forward is on `3B=96` clips. Roughly 1.5× the smooth_l1
  run's teacher cost.

At steady state, expected per-step wall clock is ~3-4 s (vs ~2-2.5 s
for the old 2-clip smooth_l1 run at batch 64). At 650 steps/epoch,
that's 35-45 min/epoch.

### Cold-start costs we can't avoid

- **First `ipg.refresh_epoch(0)`**: 12 min. Has to happen before the
  DataLoader can issue its first batch.
- **First-batch cold S3 GET**: 3 min for 768 files with no OS page cache
  warm. Later batches amortize down as mp4s enter the NVMe buffer cache
  (1.46 TB buff/cache available on the p5 node).
- **Per-rank CUDA kernel JIT + cuDNN benchmark on first forward**: ~1 min
  for ViT-L + predictor + relational head. Masked inside the 3-min cold
  S3 window because the dataloader is bottleneck.
- **NCCL topology discovery on first `all_reduce`**: ~5-10 s.

Total cold-start overhead, absent bugs: ~16-17 min to the first CSV
row. Observed 590/591 were crashing exactly at the end of this window,
which is why I confused "still in cold start" with "hung."

### Why first-epoch pair-build doesn't paralellize

Looking at the log times for the final rank to finish pair-build:

| Run | rank=5 | rank=4 | rank=7 | rank=2 | rank=1 | rank=3 | rank=6 | rank=0 |
|---|---|---|---|---|---|---|---|---|
| 591 | 08:55:39 | 08:55:46 | 08:55:49 | 08:55:49 | 08:55:50 | 08:55:53 | 08:55:56 | 08:56:07 |

All 8 ranks finish within a 28-second window of each other. That means
each rank's build takes almost exactly the same wall-clock time
(~12 min) — they're GIL-bound per process and the cross-rank spread is
just anchor-sampling variance. There is no DDP barrier coupling them.

That's encouraging for a parallel optimization: if we can speed up
per-rank pair-build 10×, we get 10× speedup for the whole phase
without needing more ranks.

## Queue handling during each incident

- After cancelling 579, I held all 9 downstream jobs via `scontrol
  hold`, then resubmitted. Good practice, preserved the dependency
  chain.
- When 590 crashed on its own, SLURM did **not** promote downstream
  (584) because downstream was still held from the 579 incident.
- When 591 crashed, I hadn't re-held 584 between the restart and the
  crash, so 584 was promoted by `afterany:591(FAILED)`. Recovered by
  `scancel 584; scontrol hold ...`.

Permanent improvement: consider changing all pretrain→downstream
`afterany` deps to `afterok`. The only caller that actually wanted
`afterany` was 584 (we wanted the control to run even if method
pretrain's downstream probes failed later — but we never wanted the
control to run if the method pretrain itself failed, which is the case
we actually hit). Cleanest fix: `afterok` on the method pretrain for
the control dep; separate `afterany` only between control pretrain and
its probes.

## Status as of 2026-05-01 09:20 UTC

- Both fixes (nested-list unwrap + unified head forward) committed to
  the source tree
- Local unit tests: all pass (`test_candidate_set_infonce.py` 6/6,
  `test_ddp_wrapped_head.py` 2/2)
- Tarball rebuilt (574 KB file-list mode) and uploaded to S3 at 09:20
- Controller `/tmp/vjepa2-ctrl/` re-extracted at 09:20
- Queue: 9 downstream jobs held via `scontrol hold`. Method pretrain
  **not yet resubmitted** — awaiting user go-ahead

## Fix-applied invariants preserved

- `multiview_objective: phase_relational` (method) / `intraview_only` (control)
- `rel_require_same_study_wrong_phase_negative: true` (both)
- `rel_wrong_phase_strategy: same_view_then_same_family`
- `rel_allow_missing_hard_negative: false` (fail-loud)
- `rel_hard_negative_fallback: resample_anchor`
- `rel_max_hard_neg_attempts: 16`
- `batch_size: 32` matched
- `num_clips: 3`
- `delta_phase_buckets: [0.0, 0.125, 0.25, 0.5]`
- `delta_phase_bucket_probs: [0.40, 0.30, 0.20, 0.10]`
- Init checkpoint `mimic_standard_jepa_e100` sha256
  `cdf0fabefe83e21e…` verified by preflight before each launch

## Lessons for future launches

1. **Integration test gap.** The smoke-test suite covered unit pieces
   but never exercised `forward_phase_relational` end-to-end with a
   real wrapped encoder and DDP-wrapped head. Two bugs that would have
   been trivial to catch with an integration test consumed ~90 min of
   GPU time and three restarts. Action: add a GPU-less integration
   test that runs a single `forward_phase_relational` on a tiny
   synthetic encoder and a DDP-wrapped head.

2. **Don't cancel on ambiguous signals.** "8 Python processes at 100%
   CPU with no log output for 8 min" is compatible with both a hung
   NCCL barrier *and* a legitimately busy pair-build + cold S3 fetch.
   A traceback is an unambiguous signal; CPU activity is not. Wait
   for the traceback or a truly dead process (wchan blocked on
   `poll`, zero CPU, zero TCP traffic) before destructive action.

3. **Custom module methods + DDP are a footgun.** Any
   `nn.Module` with public methods other than `forward` will
   AttributeError under DDP. The fix is a unified `forward` that
   does all the dispatching.

4. **`afterany` on a pretrain is probably wrong.** We want `afterok`
   on the pretrain→pretrain dep, and we want to use `afterany` only
   where a downstream step can absorb an upstream failure gracefully
   (which is rarely the case).

5. **The stale `/tmp/vjepa2-src.tar.gz` on the controller** — owned by
   root from a prior ssm session — silently prevents
   `sudo -u ubuntu aws s3 cp` from overwriting it. Workaround: always
   `rm -f /tmp/vjepa2-src.tar.gz` (as root, in the bash wrapper)
   before the ubuntu-user download. Ideally the deploy script should
   encapsulate this.
