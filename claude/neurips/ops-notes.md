# Operational Notes for NeurIPS Experiments

Lessons learned from running experiments on A100 (SageMaker) and HyperPod H100 clusters. Read before launching any new runs.

---

## Probe training: never chain runs

**Problem (2026-04-05):** Chaining probe training runs with `&&` in a single background task produces one shared output file. The eval logs don't include the model/checkpoint name, so when grep-ing for val MAE scores, results from different runs get mixed together. This caused JEPA IN21K e75 val MAE to be misattributed.

**Rules:**
1. **Always launch each probe as its own separate background task.** One model per task = one output file per model. Unambiguous.
2. **Never chain `evals.main` calls with `&&`.** Even with `echo` markers between them, the output interleaving is error-prone.
3. **Use absolute paths** for all output files (CSVs, logs). Relative paths resolve to the background shell's cwd, which may differ from the repo root. (This caused the first frame shuffling severity run to "lose" all output files.)
4. **Use `PYTHONUNBUFFERED=1`** for all background Python jobs. Without it, stdout buffers and progress isn't visible until the process exits.
5. **After a run completes, always verify the output file exists and has the expected content** before reporting results.

## Config file gotchas

- **`resume_checkpoint`**: Set to `true` only when you want to resume from `latest.pt` in the output folder. After a run completes, reset to `false` — otherwise the next model's run will try to load the previous model's checkpoint. The `sed` in-chain approach is fragile; prefer separate config files or manual reset.
- **`tasks_per_node`**: Must match the number of `--devices` passed. If you launch on 7 GPUs but config says `tasks_per_node: 8`, rank 7 will crash.

## Frame shuffling severity script

- The script `scripts/rebuttal/frame_shuffle_severity.py` has hardcoded model configs in `PT50_END_CONFIGS` and `ALL_CONFIGS` dicts. When adding new models, update both dicts.
- Use `--models MODEL_NAME` to run a single model per GPU. Do NOT use `--all` in a single process — it runs models sequentially on one GPU.
- For parallel runs, launch one process per model per GPU with separate output files.

## Killing GPU processes

- `pkill -f <script_name>` kills the main process but spawned multiprocessing workers may survive.
- Always follow with: `ps aux | grep multiprocessing | grep -v grep | awk '{print $2}' | xargs kill -9`
- Verify with `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` — if memory is still allocated, use `fuser -k /dev/nvidiaN`.
- Some zombie processes hold ~2GB indefinitely. These usually clear on their own after a few minutes.

## S3 checkpoint downloads

- BYOL v2 original training (e0-e50): `s3://sagemaker-echojepa-h100-march-0d224785-bucket/checkpoints/byol-vitl-imagenet-v2/`
- BYOL v2 resume (e55-e105): `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/checkpoints/byol-vitl-imagenet-v2-resume/`
- VideoMAE original (e4-e54): `.../runs/videomae_matched_2n_245/training_folder/checkpoint-N.pth`
- VideoMAE resume (e58-e116): `.../runs/videomae_resume_e54_354/training_folder/checkpoint-N.pth`
- JEPA IN21K (e0-e100): `.../runs/jepa_in21k_pretrain_376/checkpoints/eN.pt`
- SALT S2 (e4-e79): `.../runs/salt_s2_pretrain_388/checkpoints/eN.pt`

Note: BYOL original and resume runs use **different S3 buckets**. The training code's `s3_checkpoint_uri` determines which bucket checkpoints land in.

## Init confound reference

Always check the encoder init before comparing models:

| Model | Init | How to verify |
|-------|------|--------------|
| JEPA pt50 | Fully-trained 235ep JEPA | `anneal_ckpt: vitl.pt` in config |
| JEPA IN21K | ImageNet-21K | Job 376 |
| BYOL | ImageNet-21K | `anneal_ckpt: vitl_in21k.pt` |
| MAE | ImageNet | `args.finetune` in checkpoint |
| SALT S1 | ImageNet-21K | `force_load_pretrain: true` + `anneal_ckpt` |
| SALT S2 | Random (student) | `force_load_pretrain: false` — correct per SALT paper |

**Do not use JEPA pt50 in the primary comparison table.**

## SALT configuration discrepancies (discovered 2026-04-06)

The initial SALT S1+S2 runs (jobs 379, 388, 391, 392) had several hyperparameter mismatches vs the SALT paper. These are now fixed in the configs but the existing checkpoints were trained with the old values.

| Parameter | SALT Paper | Our Initial Config | Fixed Config | Impact |
|-----------|-----------|-------------------|-------------|--------|
| **S2 Loss** | L1 (SALT paper Eq 2.1: \|\|...\|\|_1) | L1 (`loss_exp: 1.0`) | L1 (`loss_exp: 1.0`) | Was incorrectly changed to 2.0, now reverted |
| **Peak LR** | 0.000625 | 1.75e-4 | 6.25e-4 | 3.6× too low → undertrained |
| **Start LR** | 0.0002 | 3.33e-5 | 2.0e-4 | 6× too low |
| **Final LR** | 1e-6 (cosine decay) | 1.75e-4 (constant) | 1e-6 | No LR decay → never fine-tunes |
| **Weight decay ramp** | 0.04→0.4 (cosine) | 0.04→0.04 (constant) | 0.04→0.4 | Missing regularization ramp |
| **ipe_scale** | 1.0 | 1.25 | 1.0 | 25% phantom steps (SALT paper explicitly disables) |
| **Resize scale** | [0.3, 1.0] | [0.5, 1.0] | [0.3, 1.0] | Less aggressive cropping |
| **Resize aspect** | [0.75, 1.35] | [0.9, 1.1] | [0.75, 1.35] | Less aspect variation |
| **Batch size** | 3072 | 512 (64×8) | 512 (GPU-limited) | 6× smaller — cannot fix without more GPUs |

**Files fixed (round 1, 2026-04-06):**
- `app/salt/train.py`: `loss_exp` default changed from 1.0 to 2.0
- `configs/train/vitl16/pretrain-salt-s1-mimic-224px-16f-hp.yaml`: LR, WD, ipe_scale, augmentation
- `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml`: LR, WD, ipe_scale, augmentation, loss_exp

**Files fixed (round 2, 2026-04-07, after full paper audit):**
- `app/salt/train.py`: `loss_exp` default reverted to 1.0 (SALT paper uses L1, not L2 — the round 1 "fix" was wrong)
- `app/salt/train.py`: Stage 2 teacher/student forward mode changed to single-level (`training_mode=False`) unless `n_output_distillation` is explicitly set. SALT paper uses standard V-JEPA embeddings, NOT V-JEPA 2.1 hierarchical output.
- `configs/train/vitl16/pretrain-salt-s2-mimic-224px-16f-hp.yaml`: `loss_exp` reverted to 1.0, `pred_num_heads` fixed from 12 to 16

**Additional discrepancies found in round 2 audit (not yet fixed in configs):**
- Warmup: 12,000 steps (40 epochs × 300 ipe) vs paper's 10,000. Should reduce to ~33 epochs.
- LR not scaled for batch size: using paper's 6.25e-4 with 512 batch (paper uses 3072). Should scale down by sqrt(512/3072) ≈ 0.41 → lr ≈ 2.6e-4.
- ImageNet-21K init for S1 teacher: deliberate for controlled comparison, but deviates from paper (trains from scratch).

**Implication:** SALT results from jobs 379/388/391/392 had MULTIPLE misconfigurations (hierarchical output, wrong loss direction, wrong num_heads, wrong LR/WD, wrong ipe_scale). A retrain with fully corrected configs is needed before drawing any conclusions about SALT's performance.

## Probe result extraction

When extracting val MAE from logs, always:
1. Identify the run boundaries (look for `Epoch 1` or model-loading messages)
2. Count exactly 20 val entries per run (20 probe epochs)
3. The `best.pt` checkpoint corresponds to the lowest val MAE across all 6 HP heads, not the last epoch
4. Cross-check: `best.pt` file timestamp should match the epoch with the lowest val MAE
