# ICML Rebuttal — Task Tracker

**Deadline:** Apr 2, 2026 (submission). Writing starts Apr 1.
**Resources:** 8× A100 80GB (GPUs 0-7). H100 node running BYOL-Video v2 pretrain (separate).

---

## P0 — CRITICAL PATH (blocks rebuttal writing)

These are the experiments the rebuttal narrative is built on. Without them, we have no mechanistic evidence section.

| # | Task | Status | Reviewer | Effort | Depends On | Notes |
|---|------|--------|----------|--------|-----------|-------|
| P0.1 | **Frame shuffling temporal ablation** | NOT STARTED | ALL (AC champion) | ~4h | None | `scripts/rebuttal/frame_shuffling.py` exists. Run JEPA-L, BYOL-L, MAE-L on LVEF + view. 3 shuffle types (tubelet, frame, matched-position) × 3 seeds. |
| P0.2 | **Generate perturbed test data** | NOT STARTED | — (blocks P0.3, P0.4, P0.6) | ~2h | None (CPU) | Synthetic Rayleigh speckle at 3-5 intensity levels on LVEF, RVSP, and CAMUS test sets. `scripts/rebuttal/generate_perturbed_videos.py`. Can run on CPU. Generate for ALL test sets, not just CKA subset. |
| P0.3 | **CKA speckle invariance** | NOT STARTED | ncQn (explicit ask) | ~4h | P0.2 | CKA between clean and perturbed representations. All 5 models. Representation-level stability evidence. |
| P0.4 | **Noise-level linear probe** | NOT STARTED | ncQn (explicit ask) | ~4h | P0.2 | Predict speckle intensity from frozen features. All 5 models. Representation-level information content evidence. |
| P0.5 | **Record MAE pt50 CAMUS results** | **DONE** | hfQ1, 6t2T | — | — | Test Dice=0.822, Val Dice=0.834. MAE best on CAMUS despite R²=0 on LVEF. |
| P0.6 | **Noised test inference (LVEF, CAMUS, RVSP)** | NOT STARTED | ALL | ~4-6h | P0.2 + trained probes from P1 | Run existing trained probes on perturbed test sets (3-5 speckle levels). No retraining — inference only. Probes: pt50 JEPA/BYOL/MAE on LVEF (done), CAMUS (done), RVSP (P1.2-P1.4). **Task-level complement to CKA** (P0.3 = representation stability, P0.6 = downstream task degradation). Expected: MAE degrades most, JEPA least — flips CAMUS clean ordering where MAE wins. |

**Why P0:** ncQn is 75-80% likely to flip 3→4/5 if P0.1-P0.4 are strong. Frame shuffling provides the AC champion sentence. CKA + noise probe + noised inference together provide three complementary angles on noise filtering: representation stability (CKA), noise information content (probe), and task-level degradation curves (noised inference). The clean 3-way results are close (CAMUS: 0.7pp spread, MAE wins; LVEF: 1.5pp JEPA-BYOL gap) — **perturbation testing is what separates the methods and turns the close clean results from a weakness into a strength**: *"Under clean conditions all methods converge; under realistic noise, only latent prediction maintains performance."*

---

## P1 — HIGH (completes 3-way comparison across all tasks)

Without these, the controlled comparison only covers LVEF and CAMUS. RVSP adds the multi-view spatial reasoning angle.

| # | Task | Status | Reviewer | Effort | Depends On | Notes |
|---|------|--------|----------|--------|-----------|-------|
| P1.1 | **EchoMAE-L pt50 LVEF probe** | **DONE** (HyperPod job 274) | hfQ1, ncQn | — | — | R²=0.325, Pearson=0.584, MAE=6.866 (ep18). Retrained with z-score normalization after Bug 017c. MAE trails JEPA (0.436) and BYOL (0.421) on R², confirming EMA advantage. |
| P1.2 | **Resume JEPA pt50 RVSP 41K** (ep18→20) | **DONE** (20/20) | ALL | — | — | **Pearson=0.504** (ep19), R²=0.241 (ep20), Val MAE=9.044 (ep16). Matches pt210-an25. |
| P1.3 | **BYOL pt50 RVSP 41K** (20ep) | KILLED (ep0) | hfQ1 | ~10h | Config exists | Killed mid-epoch-1, no checkpoint. Full restart needed. |
| P1.4 | **MAE pt50 RVSP 41K** (20ep) | **DONE** (HyperPod job 260) | ALL | — | — | Val MAE **9.287** (ep17), R²=0.198 (ep19), Pearson=**0.453** (ep20). Trails JEPA (0.504) by 5.1pp Pearson. |

**Why P1:** The rebuttal tex claims "converging evidence across LVEF, CAMUS, and RVSP." Without RVSP results for all 3 models, this is only partially supported.

---

## P2 — MEDIUM (EchoBench — addresses 3-4 reviewers at once)

External benchmark validation on public data. Differentiates from US-JEPA. Community contribution.

| # | Task | Status | Reviewer | Effort | Depends On | Notes |
|---|------|--------|----------|--------|-----------|-------|
| P2.1 | **Train pt50 EchoNet-Dynamic LVEF probes (224px)** (×3 models) | **DONE** (all 3) | 6t2T, hfQ1, ncQn | — | — | JEPA **R²=0.621**, Pearson=0.793, MAE=5.506. BYOL R²=0.528, Pearson=0.729, MAE=6.174. MAE R²=0.495, Pearson=0.706, MAE=6.410. **JEPA >> BYOL > MAE on cross-dataset transfer.** |
| P2.2 | **Train pt50 EchoNet-Pediatric LVEF probes (224px)** (×3 models) | **DONE** (all 3) | 6t2T, hfQ1 | — | — | MAE **5.985**, JEPA 6.093, BYOL 6.147. All converged (spread 0.16). Test inference pending. |
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
| ✓ | **EchoMAE-L pt50 LVEF (10K, 20ep)** | **R²=0.325, Pearson=0.584, MAE=6.866** (job 274, retrained) | Mar 29 |
| ✓ | EchoJEPA-L pt50 RVSP 41K (20ep) | Pearson=0.504, R²=0.241, MAE=9.044 | Mar 29 |
| ✓ | **EchoMAE-L pt50 RVSP 41K (20ep)** | **Pearson=0.453, R²=0.198, MAE=9.287** (job 260) | Mar 30 |

---

## Execution Plan (Mar 29 evening → Apr 2)

### Completed (Mar 28-29)
- [x] P0.5: Record MAE CAMUS results — **Test Dice 0.822** ✓
- [x] P1.1: Launched MAE pt50 LVEF retrain — HyperPod job 274, node 83 (6 HP heads, ~8h)
- [x] Infrastructure: Migrated ALL 34 sbatch scripts from code.tar to deploy.sh `/opt/vjepa2` workflow
- [x] Infrastructure: Updated deploy.sh to target both compute nodes by default

### Mar 30 — Mechanistic evidence day (P0 focus)
- [ ] P0.2: Generate perturbed test data on CPU (speckle at 3-5 levels on LVEF + CAMUS + RVSP test sets) — **start first, blocks P0.3/P0.4/P0.6**
- [ ] P0.1: Launch frame shuffling on A100 (JEPA/BYOL/MAE × LVEF + view × 3 shuffle types × 3 seeds)
- [ ] P0.3: Launch CKA (after P0.2 finishes, parallel with P0.4)
- [ ] P0.4: Launch noise probe (after P0.2, parallel with P0.3)
- [ ] P1.2: Check JEPA pt50 RVSP 41K final result (was ep19/20)
- [x] P1.1: MAE pt50 LVEF retrain **DONE** — R²=0.325, Pearson=0.584, MAE=6.866
- [ ] P1.4: Check MAE pt50 RVSP 41K (HyperPod job 260, ep14/20)
- [x] P2.1: JEPA EchoNet-Dynamic DONE at 224px (job 294, R²=0.621). MAE (job 296) + BYOL (A100) running
- [ ] P2.2: EchoNet-Pediatric 224px retrain in progress (all 3 on A100, ep12-14)
- [ ] P1.3: Restart BYOL pt50 RVSP 41K (if GPU available)

### Mar 31 — Task-level degradation + EchoBench
- [ ] P0.6: Noised test inference — run trained LVEF + CAMUS probes on perturbed test sets (inference only). RVSP if P1.2-P1.4 probes done.
- [ ] P0.6: Generate degradation curves (performance vs noise level, per model)
- [ ] P2.1: Train EchoNet-Dynamic pt50 probes (3 models) if GPUs free
- [ ] Record all P0 results → update icml_rebuttal.tex
- [ ] P3.1: Single-view RVSP ablation (if GPU free — quick win)

### Apr 1 — Write rebuttal
- [ ] P2.3-P2.4: EchoBench perturbation matrix (if P2.1 probes done)
- [ ] Write rebuttal text — all P0 results must be in by now
- [ ] Final numbers into icml_rebuttal.tex (replace all \tbd)
- [ ] Review narrative coherence

### Apr 2
- [ ] Final review, submit
