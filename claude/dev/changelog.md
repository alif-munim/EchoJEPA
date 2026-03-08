# Changelog

Chronological record of code changes, bug fixes, and extraction runs.

## 2026-03-07

### Bug Fixes

- **CRITICAL: Shuffle ordering fix** (`extract_embeddings.py`, `extract_uhn_embeddings.py`)
  Set `data_loader.sampler.shuffle = False` to prevent DistributedSampler from permuting clip order. Created `fix_shuffle_order.py` (UHN) and `fix_mimic_shuffle.py` (MIMIC) for post-hoc reordering of existing embeddings. All 7 MIMIC models reordered and verified (100% label match). See `bugs/001-shuffle-bug.md`.

- **HIGH: Encoder normalization fixes** (`panecho_encoder.py`, `echo_prime_encoder.py`, `echofm_encoder.py`)
  PanEcho: removed double ImageNet normalization. EchoPrime: added de-norm → [0,255] → model-specific norm. EchoFM: added de-norm → [0,1]. See `bugs/002-normalization-bugs.md`.

- **Moderate: EchoFM temporal padding** (`echofm_encoder.py`)
  Replaced last-frame repetition with `torch.linspace` + `index_select` for 16→32 frame adaptation. Unified upsample/downsample into single code path. See `bugs/003-echofm-padding.md`.

### Extraction Runs

- **EchoJEPA-G UHN** — 319,815 studies, 1408-dim, ~25.5h (bs=32, w=8, pf=1, pre-optimization). Post-hoc shuffle fix applied.
- **EchoJEPA-L UHN** — 319,802 studies, 1024-dim, ~12.5h (bs=128→64, w=12, pf=4). Extracted after shuffle fix.
- **EchoJEPA-L-K UHN** — in progress (bs=64, w=12, pf=4). Extracted after shuffle fix.

### Downstream Pipeline

- **UHN EchoJEPA-G**: Post-hoc shuffle fix completed. `clip_embeddings.npz` reordered to CSV order (180/18.1M clips zero-filled from incomplete extraction). `study_embeddings.npz` re-pooled (319,815 studies).
- **MIMIC all 7 models**: Downstream pipeline regenerated via `regenerate_mimic_downstream.py` — 7 models × 23 tasks, producing study-level NPZs + train/val/test splits. 4 correct models ready for probing.

### Config Changes

- Created `configs/inference/vitl/extract_uhn.yaml` (EchoJEPA-L UHN extraction)
- Created `configs/inference/vitl/extract_uhn_kinetics.yaml` (EchoJEPA-L-K UHN extraction)

### Code Review

- Full review of all 5 encoder adapters in `modelcustom/`. See `code-review.md`.
- Full review of extraction, pooling, remapping, and probe training scripts. 6 bugs identified (3 previously known, 3 new). See `code-review.md`.

- **HIGH: Video load substitution tracking** (`src/datasets/video_dataset.py`)
  Added `_substitution_count` counter and per-event WARNING logging to `__getitem__`. Each load-failure substitution now logs the original and served index/path. Removed `threading.Lock` (unnecessary — DataLoader workers are separate processes, not threads; lock also broke `mp.spawn` pickling). See `bugs/004-video-load-substitution.md`.

- **MEDIUM: `drop_last` forwarding** (`src/datasets/data_manager.py`)
  Added `drop_last=drop_last` to the `make_videodataset()` call in the `VideoDataset` branch of `init_data()`. Previously the parameter was accepted but silently ignored. See `bugs/005-drop-last-not-forwarded.md`.

### Cleanup & Re-extraction

- Deleted corrupted MIMIC embeddings for PanEcho, EchoPrime, EchoFM (~9.5GB: master NPZs, shuffled backups, study-level dirs, split dirs)
- Started MIMIC re-extraction for all 3 models (8×A100, bs=32, w=8). Script: `scripts/reextract_mimic_3models.sh`. Uses fixed adapters (normalization + EchoFM padding). Sequential: PanEcho → EchoPrime → EchoFM. ~1h each estimated.

### Additional Fixes (during re-extraction)

- **PanEcho `hubconf.py` local tasks.pkl cache** — `pd.read_pickle()` was fetching `tasks.pkl` from GitHub on every worker init. With 8 workers hitting simultaneously, GitHub returned HTTP 429. Fixed by downloading `tasks.pkl` to `PanEcho/content/tasks.pkl` and loading from local path via `os.path.join(os.path.dirname(__file__), 'content', 'tasks.pkl')`.

- **VideoDataset pickle compatibility** — Removed `threading.Lock` from `_substitution_count` tracking (bug 004 fix). Lock objects cannot be pickled, which broke `mp.spawn` multi-GPU extraction. The lock was unnecessary since DataLoader workers are separate processes (each gets its own counter copy). Per-worker counter + WARNING logging is sufficient.
