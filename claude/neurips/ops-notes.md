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

## Probe result extraction

When extracting val MAE from logs, always:
1. Identify the run boundaries (look for `Epoch 1` or model-loading messages)
2. Count exactly 20 val entries per run (20 probe epochs)
3. The `best.pt` checkpoint corresponds to the lowest val MAE across all 6 HP heads, not the last epoch
4. Cross-check: `best.pt` file timestamp should match the epoch with the lowest val MAE
