# ICML Rebuttal — Task Tracker

**Deadline:** Apr 2, 2026 (submission). Writing starts Apr 1.
**Resources:** 8× A100 80GB (GPUs 0-7). H100 node running BYOL-Video v2 pretrain (separate).

---

## P0 — CRITICAL PATH (blocks rebuttal writing)

These are the experiments the rebuttal narrative is built on. Without them, we have no mechanistic evidence section.

| # | Task | Status | Reviewer | Effort | Depends On | Notes |
|---|------|--------|----------|--------|-----------|-------|
| P0.1 | **Frame shuffling temporal ablation** | NOT STARTED | ALL (AC champion) | ~4h | None | `scripts/rebuttal/frame_shuffling.py` exists. Run JEPA-L, BYOL-L, MAE-L on LVEF + view. 3 shuffle types (tubelet, frame, matched-position) × 3 seeds. |
| P0.2 | **Generate perturbed speckle data** | NOT STARTED | — (blocks P0.3, P0.4) | ~2h | None (CPU) | Synthetic Rayleigh speckle at 3-5 intensity levels. `scripts/rebuttal/generate_perturbed_videos.py`. Can run on CPU. |
| P0.3 | **CKA speckle invariance** | NOT STARTED | ncQn (explicit ask) | ~4h | P0.2 | CKA between clean and perturbed representations. All 5 models. |
| P0.4 | **Noise-level linear probe** | NOT STARTED | ncQn (explicit ask) | ~4h | P0.2 | Predict speckle intensity from frozen features. All 5 models. |
| P0.5 | **Record MAE pt50 CAMUS results** | **DONE** | hfQ1, 6t2T | — | — | Test Dice=0.822, Val Dice=0.834. MAE best on CAMUS despite R²=0 on LVEF. |

**Why P0:** ncQn is 75-80% likely to flip 3→4/5 if P0.1-P0.4 are strong. Frame shuffling provides the AC champion sentence. These three together ARE the "mechanistic evidence" section of the rebuttal.

---

## P1 — HIGH (completes 3-way comparison across all tasks)

Without these, the controlled comparison only covers LVEF and CAMUS. RVSP adds the multi-view spatial reasoning angle.

| # | Task | Status | Reviewer | Effort | Depends On | Notes |
|---|------|--------|----------|--------|-----------|-------|
| P1.1 | **EchoMAE-L pt50 LVEF probe** | **RETRAIN IN PROGRESS** (HyperPod job 274) | hfQ1, ncQn | ~8h remaining | — | ⚠️ Job 247 probe trained without z-score normalization (Bug 017c) — invalid for inference (test MAE 719). Job 274 retraining: head 1/6 done (val MAE 7.17), head 2/6 in progress. |
| P1.2 | **Resume JEPA pt50 RVSP 41K** (ep18→20) | **RUNNING** (ep19/20, 8× A100, PID 665767) | ALL | ~20 min | — | Pearson=0.503 at ep16 (matches pt210-an25). Resumed from ep18 checkpoint. |
| P1.3 | **BYOL pt50 RVSP 41K** (20ep) | KILLED (ep0) | hfQ1 | ~10h | Config exists | Killed mid-epoch-1, no checkpoint. Full restart needed. |
| P1.4 | **MAE pt50 RVSP 41K** (20ep) | **IN PROGRESS** (HyperPod job 260, ep8/20) | ALL | ~5h remaining | — | Val MAE 9.48 (ep6), R²=0.163 (ep7), Pearson=0.406 (ep8). Running on node ip-10-0-50-184. |

**Why P1:** The rebuttal tex claims "converging evidence across LVEF, CAMUS, and RVSP." Without RVSP results for all 3 models, this is only partially supported.

---

## P2 — MEDIUM (EchoBench — addresses 3-4 reviewers at once)

External benchmark validation on public data. Differentiates from US-JEPA. Community contribution.

| # | Task | Status | Reviewer | Effort | Depends On | Notes |
|---|------|--------|----------|--------|-----------|-------|
| P2.1 | **Train pt50 EchoNet-Dynamic LVEF probes** (×3 models) | NOT STARTED | 6t2T, hfQ1, ncQn | ~6 GPU-h | Configs needed | JEPA-L-pt50, BYOL-L-pt50, MAE-L-pt50 |
| P2.2 | **Train pt50 EchoNet-Pediatric LVEF probes** (×3 models) | NOT STARTED | 6t2T, hfQ1 | ~6 GPU-h | Configs needed | Domain-shift evaluation (adult→pediatric) |
| P2.3 | **Generate perturbed EchoNet-Dynamic test videos** | NOT STARTED | ncQn | ~2h | Pipeline exists | 7 perturbation types × 3 levels |
| P2.4 | **Run perturbation matrix** (fully-trained + pt50) | NOT STARTED | ALL | ~8h | P2.1-P2.3 | Probes for 5 fully-trained models already exist |
| P2.5 | **Package EchoBench** (scripts + README) | NOT STARTED | 6t2T (novelty) | ~4h writing | P2.4 | Open-source release artifact |

**Why P2:** EchoBench reframes segmentation risk into a contribution. Existing probes for fully-trained models done. Only the pt50 probes are new. Addresses novelty (6t2T), broader tasks (hfQ1), and noise (ncQn).

---

## P3 — NICE TO HAVE (strengthens specific arguments)

| # | Task | Status | Reviewer | Effort | Depends On | Notes |
|---|------|--------|----------|--------|-----------|-------|
| P3.1 | **Single-view RVSP ablation** | NOT STARTED | L8sp | ~4h | Build CSV + 1 probe | A4C-only vs A4C+PSAX-AV. Validates multi-view contribution. Quick win. |
| P3.2 | **Few-shot label scaling** | NOT STARTED | ALL | ~8h | None | {1%, 5%, 10%, 50%, 100%} × 2 tasks × 3 models. "JEPA reaches full-data baselines with 10%." |
| P3.3 | **Linear probe view classification** (all models) | NOT STARTED | ALL | ~2h | None | Confirms ranking holds under linear probing. Already claimed in tex (70.8% vs 59.2%). |

---

## DEFER (out of scope for rebuttal)

| Task | Reason | Where Instead |
|------|--------|---------------|
| Biplane LVEF (A4C+A2C multi-view) | Data pipeline work; ICML reviewers won't appreciate clinical significance | Nature Medicine |
| EchoMAE-L 50ep retrain (corrected LR) | Current pt50 checkpoint sufficient for comparison | Camera-ready if needed |
| DINO controlled baseline | BYOL-Video already provides the contrastive comparison | Camera-ready if reviewer insists |
| Full EchoBench packaging + paper | Benchmark paper is a separate effort | Standalone release |

---

## DONE

| # | Task | Key Result | Date |
|---|------|-----------|------|
| ✓ | EchoJEPA-L pt50 LVEF (10K, 20ep) | R²=0.436, Pearson=0.667, MAE=6.329 | Mar 29 |
| ✓ | EchoBYOL-L pt50 LVEF (10K, 20ep) | R²=0.421, Pearson=0.652, MAE=6.297 | Mar 29 |
| ✓ | EchoJEPA-L pt50 LVEF test (53K) | R²=0.409, Pearson=0.650, MAE=6.508 | Mar 29 |
| ✓ | EchoBYOL-L pt50 LVEF test (53K) | R²=0.384, Pearson=0.625, MAE=6.656 | Mar 29 |
| ✓ | EchoJEPA-L pt50 CAMUS (50ep) | Test Dice=0.815 | Mar 29 |
| ✓ | EchoBYOL-L pt50 CAMUS (50ep) | Test Dice=0.821 | Mar 29 |
| ✓ | EchoMAE-L ep99 LVEF (5K) | R²~0, MAE=8.05 (no signal) | Mar 28 |
| ✓ | EchoMAE-L ep99 View (5K) | Acc=44.1%, AUROC=0.847 | Mar 28 |
| ✓ | EchoJEPA-B LVEF (10K) | R²=0.650, Pearson=0.806, MAE=5.244 | Mar 28 |
| ✓ | CAMUS seg (6 fully-trained models) | JEPA-L=0.818, MAE=0.790 | Mar 27 |
| ✓ | RVSP multi-view data audit | 96.7% genuine A4C+PSAX-AV pairs | Mar 29 |
| ✓ | Biplane LVEF feasibility check | 97% of studies have A4C+A2C | Mar 29 |
| ✓ | EchoMAE-L pt50 CAMUS (50ep) | **Test Dice=0.822** (best of 3, despite R²=0 on LVEF) | Mar 29 |
| ✓ | Update icml_rebuttal.tex with results | 3-way numbers, CAMUS, scaling, EMA finding | Mar 29 |

---

## Execution Plan (Mar 29 evening → Apr 2)

### Tonight (Mar 29) — UPDATED
- [x] P0.5: Record MAE CAMUS results — **Test Dice 0.822** ✓
- [x] P1.1: Launched MAE pt50 LVEF retrain — HyperPod job 274, node 83 (6 HP heads, ~8h)
- [x] Infrastructure: Migrated ALL 34 sbatch scripts from code.tar to deploy.sh `/opt/vjepa2` workflow
- [x] Infrastructure: Updated deploy.sh to target both compute nodes by default
- [ ] P1.4: MAE pt50 RVSP running (job 260, ep8/20) — monitor
- [ ] P0.1: Launch frame shuffling (blocked — both HyperPod nodes busy)
- [ ] P0.2: Launch speckle generation on CPU (runs ~2h, can run independently)
- [ ] P1.2: Resume JEPA RVSP 41K (blocked — need GPU)
- [ ] P1.3: Resume BYOL RVSP 41K (blocked — need GPU)

### Mar 30
- [ ] P0.3: Launch CKA (after P0.2 finishes, morning)
- [ ] P0.4: Launch noise probe (after P0.2, parallel with CKA)
- [ ] P2.1: Train EchoNet-Dynamic pt50 probes (3 models, queue overnight)
- [ ] Record all P0 results → update icml_rebuttal.tex

### Mar 31
- [ ] P2.3: Generate perturbed EchoNet-Dynamic test videos
- [ ] P2.4: Run perturbation matrix
- [ ] P3.1: Single-view RVSP ablation (if GPU free)
- [ ] Record all P1/P2 results → update docs

### Apr 1
- [ ] Write rebuttal text (all experiments done by now)
- [ ] Final numbers into icml_rebuttal.tex (replace all \tbd)
- [ ] Review narrative coherence

### Apr 2
- [ ] Final review, submit
