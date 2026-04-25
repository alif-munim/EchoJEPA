# φ-JEPA: Phase-Conditioned V-JEPA for Echo

**Status:** Gate 1 complete (PASS, 2026-04-24). Gate 2 (Run D headline pretraining) is the next step — requires dataloader/predictor/training-step code + config + sbatch.
**Last updated:** 2026-04-24 (post-Gate-1)
**Author context:** Method follow-up to the NeurIPS diagnostic paper (`experiments/frame-shuffling-results.md`). Evaluation uses the matched-frame diagnostic from that work.

---

## Pivot history

This doc replaces several earlier designs. Each iteration tightened the method:

1. **v0 — CAF-JEPA (cycle-aware cropping + stride-ranked auxiliary)**: cycle-aware frame_step at dataload time, conditioned on a mitral-brightness HR estimator on apical views. Stride-ranked RnC auxiliary loss as a secondary component. Two loosely-coupled components.
2. **v1**: upgraded Component 1 to phase-conditioned predictor; phase derived from a trained regressor.
3. **v2 — phase-conditioned predictor + stride-RnC auxiliary**: relative phase derived directly from DICOM metadata. No estimator. Component 2 retained as secondary.
4. **v3 — φ-JEPA**: single-component method. Predictor receives Δφ (phase offset from context to target) as conditioning input. Stride-ranked auxiliary dropped.
5. **v4 — φ-JEPA with attribution controls**: same core method as v3, but the ablation matrix is restructured around attribution and disagreement-case handling. Pre-registered effect-size thresholds for Gate 2. Explicit reframing from "predictor conditioning tweak" to "cardiac phase as first-class representational property."
6. **v5 (this doc, post-Gate-1) — phase-jepa**: Gate 1 metadata pilot PASSED (job 351, 99.6% HR coverage, 96.7% studies tight-rhythm). Doc renamed from `caf-jepa.md` to `phase-jepa.md` to match the method's actual scope. Next milestone is Gate 2 (Run D headline pretraining).

The key enabling finding: DICOM `(0018,1088) HeartRate` and `(0018,1063) FrameTime` are present per-clip on MIMIC-IV-Echo raw DICOMs. Initially verified on 5 sample clips from 2 studies; Gate 1 confirmed this holds at scale (see next section).

---

## Gate 1 result: DICOM metadata extraction pilot (2026-04-24, job 351)

**PASS on both criteria.** 74,314 clips across 1,000 random MIMIC studies scanned in 5m46s by `scripts/neurips/phase/extract_dicom_phase_metadata.py`. Outputs at `s3://sagemaker-hyperpod-lifecycle-495467399120-usw2/vjepa2-artifacts/runs/phi_jepa_gate1_351/`.

| Criterion | Result | Threshold | Status |
|---|---|---|---|
| Valid HR coverage (40-180 bpm) | **99.59%** (74,011/74,314) | ≥95% | **PASS** |
| Within-study HR stdev ≤ 15 bpm | **96.68%** (962/995) | ≥85% | **PASS** |
| Parse errors | 0 (0%) | — | — |

**HR distribution** (valid clips, n=74,011):
- median 70 bpm, mean 73 bpm, std 16.3 bpm
- p05 = 53 bpm, p95 = 103 bpm, tails 40-179 bpm
- Broad but clinically realistic; captures pediatric/tachycardic clips in the tail.

**Within-study HR spread** (studies with ≥2 valid clips, n=995):
- median stdev 3.37 bpm, p95 stdev 11.65 bpm, max 37.23 bpm
- 33 of 995 studies (3.3%) flagged irregular (stdev > 15) → use `<no_phase>` conditioning at train time.

**FrameTime coverage is only 55%** (40,873 / 74,314 clips). The missing 45% are single-frame DICOMs (still images, M-mode) that don't carry `FrameTime` — correctly excluded from cine pretraining. Extrapolating to the full 525K MIMIC corpus yields ~290K usable cine clips, well above the ViT-L pretraining threshold.

**Implications for the method:**

- Phase conditioning covers ~96.7% of studies (after arrhythmia filter) at near-100% within-study clip coverage. The `<no_phase>` fallback is exercised on ~3-5% of clips — enough to train the fallback robustly, small enough not to dilute the phase signal.
- HR distribution does not require special handling — Fourier embedding over [0, 1] wrapped phase handles the full 40-180 bpm range cleanly.
- **Pediatric-transfer caveat partially addressed**: the adult-pretraining HR distribution extends to p95 = 103 bpm, not a hard wall at 90 bpm. Pediatric HRs (80-140 bpm) fall within the pretraining distribution's tail rather than entirely outside it. Worth confirming if pediatric evaluation underperforms.

**Artifacts produced:**
- `s3://.../phi_jepa_gate1_351/clip_phase_metadata.csv` (14 MB, 74,314 rows) — columns: `study_id, clip_id, hr_bpm, frame_time_ms, num_frames, fps, present_hr, present_ft, present_nf, dicom_path`.
- `s3://.../phi_jepa_gate1_351/gate1_report.txt` — full human-readable report.
- Next step for this CSV: run the same script over all 525K MIMIC clips (no sampling) to produce the full `mimic_clip_phase_metadata.csv` consumed by the φ-JEPA dataloader. Expected runtime at current throughput (270 clips/s × 32 workers): ~30 min. Can be folded into the Run D sbatch's setup phase.

---

## Motivation and representational claim

### The claim

**Cardiac phase is a first-class representational property that echo video SSL should explicitly condition on.** Clinical assessment of cardiac function depends on phase-specific geometry — ventricular filling, ejection, valve opening/closure, wall thickening — and each of these is defined relative to a point in the cardiac cycle. A representation learning objective that treats all temporal positions as interchangeable is asking the encoder to learn phase implicitly, which the diagnostic paper shows is a partially-solved problem: JEPA does encode temporal dynamics (linear-diff R² = 0.376 at e100, Pearson r = 0.72 pediatric transfer), but the matched-frame gap of −0.143 at convergence indicates unused headroom, and the concentration of prediction error on reduced-EF patients (+3.7 MAE) is the clinical expression of that headroom.

φ-JEPA operationalizes the claim via architectural conditioning: the V-JEPA predictor receives cardiac phase offset Δφ between context and target tokens as explicit conditioning input, alongside its standard positional inputs. For the predictor to use Δφ productively, the encoder must produce context features from which phase-conditioned target prediction is possible. This is a hard architectural requirement on the encoder's representation, not a soft bias on the training distribution.

### Relation to Brain-JEPA

Brain-JEPA's Gradient Positioning is the closest prior art: it injects brain-specific structural coordinates (functional gradient embeddings) into JEPA's positional structure, claiming that brain SSL should condition on brain organization. Parameter-wise, Gradient Positioning is also modest. The paper frames it as a claim about what brain representations must encode, not as a positional-encoding tweak. φ-JEPA takes the same position for echo and cardiac phase.

### Two anchoring quantities from the diagnostic paper

1. **JEPA e100 matched-frame gap = −0.143.** The target metric.
2. **Reduced-EF tail MAE = +3.7 over full distribution.** The clinical failure mode the method should address.

These are what Gate 2's pre-registered thresholds (below) are calibrated against.

---

## Method

### Data pipeline

**Per-clip metadata extraction (offline, one-shot)**. A pydicom batch job over MIMIC-IV-Echo raw DICOMs emits:

```
study_id, clip_id, hr_bpm, frame_time_ms, num_frames, view, fps
s94106955, 94106955_0001, 64, 33.68, 58, A4C, 29.69
```

- Source tags: `(0018,1088)` HR, `(0018,1063)` FrameTime, `(0028,0008)` NumberOfFrames. View label joined from existing view classifier outputs.
- Cost: ~4 CPU hours over 525K clips.
- Output: `data/csv/mimic_clip_phase_metadata.csv` on S3.

**Study-level arrhythmia filter**. Compute `stdev(hr_bpm)` across clips within each study. Studies with stdev > 15 bpm are flagged `irregular`; their clips use `<no_phase>` conditioning instead of real phase.

**No runtime phase computation in the dataloader.** Δφ is computed inside the training step, using the target/context positions from `MaskCollator` and the clip's HR + FrameTime. The dataloader only attaches `hr_bpm` and `frame_time_ms` to each clip.

### Δφ: phase offset from context to target

V-JEPA's `MaskCollator` partitions tokens of a clip into a context set and one or more target sets, each with spatial and temporal positions. For each target token, Δφ is the phase offset from the context's mean frame index to the target's frame index, in cycle fractions:

```
Δφ(target_i) = (frame_target_i − frame_context_mean) × FrameTime_ms × HR_bpm / 60000
```

Δφ is per-clip and per-target. Within a typical 16-frame V-JEPA clip at 30 fps and 70 bpm, |Δφ| stays in [0, ~0.6] — well inside one cycle. Wrapping is applied inside the Fourier embedding so circular symmetry is preserved even when |Δφ| exceeds 1.

### Architecture

**Baseline V-JEPA predictor input** (unchanged):

```
predictor(context_tokens, mask_tokens)
  where mask_tokens carry target position via learned embedding + RoPE
```

**φ-JEPA predictor input**:

```
predictor(context_tokens, mask_tokens + phase_embed(Δφ))
  where phase_embed is an integer-frequency Fourier embedding of (Δφ mod 1.0), then a 2-layer MLP
```

Key design choices:

- **Integer-frequency Fourier embedding with phase wrapping**. `Δφ mod 1.0` is embedded via `{sin(2πkΔφ), cos(2πkΔφ) : k = 1..K}` with K integer frequencies. Circular: `Δφ=0.3` and `Δφ=1.3` embed identically.
- **Element-wise add** to mask tokens. Matches V-JEPA's convention for combining mask tokens with position embeddings. No tensor-shape change; no retuning of predictor projection layers.
- **Phase-blind encoder** (primary). The encoder sees context tokens without phase information; only the predictor uses Δφ. This forces the encoder to produce context features informative enough for phase-conditioned prediction.
- **`<no_phase>` embedding**. A learned token replaces the Fourier embedding when phase is unknown (irregular rhythm, invalid HR).
- **Phase dropout** at `p = 0.15`. During training, randomly replace the Fourier embedding with `<no_phase>`. Prevents the predictor from trivially ignoring context features; trains the `<no_phase>` fallback; regularizes against phase-label noise on downstream ports.

Parameter count addition is minor (a Fourier → MLP embedding and a `<no_phase>` token, ~5K params vs ViT-L's 304M). The paper leads with the representational claim and uses the parameter count as a footnote, not as the headline characterization.

### Objective

Standard V-JEPA L1 loss on predictor outputs vs EMA target encoder outputs:

```
L = L1(predictor(enc(ctx), mask + phase_embed(Δφ)), target_encoder(target))
```

No auxiliary loss. EMA schedule, masking ratios, LR schedule, weight decay all unchanged from the V-JEPA baseline.

### Pseudocode

```python
# ============================================================
# Dataset: adds hr_bpm + frame_time_ms to each clip's metadata
# ============================================================

class PhaseMetadataVideoDataset(VideoDataset):
    def __init__(self, *args, phase_metadata_csv, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta = pd.read_csv(phase_metadata_csv).set_index("clip_id")
        hr_std = self.meta.groupby("study_id")["hr_bpm"].std()
        irregular = hr_std[hr_std > 15].index
        self.meta["is_irregular"] = self.meta["study_id"].isin(irregular)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip_id = parse_clip_id_from_path(path)
        row = self.meta.loc[clip_id]
        clip = self._load_clip(path)
        hr = None if (row.is_irregular or row.hr_bpm is None
                      or not (40 <= row.hr_bpm <= 180)) else row.hr_bpm
        return clip, label, hr, row.frame_time_ms


# ============================================================
# Predictor (phase-conditioned V-JEPA)
# ============================================================

class PhaseConditionedPredictor(nn.Module):
    def __init__(self, d_model, num_blocks, num_heads,
                 phase_drop_p=0.15, n_fourier_freqs=16):
        super().__init__()
        self.blocks = nn.ModuleList([
            PredictorBlock(d_model, num_heads) for _ in range(num_blocks)
        ])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.phase_fourier = IntegerFourierEmbedding(n_freqs=n_fourier_freqs)
        self.phase_mlp = nn.Sequential(
            nn.Linear(2 * n_fourier_freqs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.no_phase_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.phase_drop_p = phase_drop_p

    def forward(self, context_tokens, target_positions, delta_phi):
        B, N_target = target_positions.shape[:2]
        mask = self.mask_token.expand(B, N_target, -1)

        if delta_phi is not None:
            wrapped = delta_phi % 1.0
            drop = torch.rand(B, N_target, device=mask.device) < self.phase_drop_p
            phase_emb = self.phase_mlp(self.phase_fourier(wrapped))
            phase_emb = torch.where(
                drop.unsqueeze(-1),
                self.no_phase_token.expand(B, N_target, -1),
                phase_emb,
            )
        else:
            phase_emb = self.no_phase_token.expand(B, N_target, -1)

        targets = mask + target_positions_emb(target_positions) + phase_emb
        x = torch.cat([context_tokens, targets], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return x[:, context_tokens.shape[1]:]


# ============================================================
# Training step
# ============================================================

def training_step(batch, encoder, predictor, target_encoder, mask_collator):
    clips, labels, hr_bpm, frame_time_ms = batch
    ctx_tokens, target_sets, target_pos = mask_collator(clips)

    ctx_mean_frame = ctx_tokens.mean_frame_index          # [B]
    tgt_frames = target_pos.frame_indices                 # [B, N_target]
    delta_phi = compute_delta_phi(
        tgt_frames, ctx_mean_frame, hr_bpm, frame_time_ms
    )

    context_enc = encoder(ctx_tokens)
    pred_targets = predictor(context_enc, target_pos, delta_phi)
    with torch.no_grad():
        true_targets = target_encoder(target_sets)
    return F.l1_loss(pred_targets, true_targets)


def compute_delta_phi(tgt_frames, ctx_mean_frame, hr_bpm, frame_time_ms):
    """Returns [B, N_target] cycle fractions, or None where HR is None."""
    frame_delta = tgt_frames - ctx_mean_frame.unsqueeze(-1)
    out = torch.full_like(frame_delta, float("nan"))
    valid = hr_bpm.isfinite()
    if valid.any():
        fd = frame_delta[valid]
        hr = hr_bpm[valid].unsqueeze(-1)
        ft = frame_time_ms[valid].unsqueeze(-1)
        out[valid] = fd * ft * hr / 60000.0
    return out if valid.any() else None
```

---

## Ablation matrix

Seven rows, organized around attribution and disagreement-case handling. Runs D, G, I are load-bearing; runs E1, E2, F, J are supporting.

| Run | Training Δφ | Inference Δφ | Encoder sees phase | Purpose |
|---|---|---|---|---|
| A | — (V-JEPA baseline) | — | no | Baseline. Exists at `runs/jepa_in21k_pretrain_376`. |
| **D** | **true Δφ** | **true Δφ** | **no** | **Full φ-JEPA, headline.** |
| G | **shuffled Δφ across batch** | shuffled | no | **Negative control #1**: phase values permuted; tests whether conditioning signal is used. |
| **I** | **Δφ from fixed HR = 72 bpm** | fixed HR | no | **Attribution test**: HR-specific phase vs population-averaged temporal conditioning. See "Attribution framing" below. |
| J | **random-vector conditioning** | random vector | no | **Capacity-matched control**: same Fourier+MLP+no_phase-token architecture, but fed a learned per-clip random vector instead of Δφ. Tests whether the parameter-count addition is what's helping. |
| E1 | always `<no_phase>` (p=1.0) | true Δφ | no | **Untrained-phase inference**: predictor never learned a phase mapping but receives phase at test. Tests whether inference-time phase alone is useful. |
| E2 | true Δφ | always `<no_phase>` | no | **Phase-free deployment**: trained with phase, deployed without. Tests whether training-time phase conditioning shapes the encoder itself (strong claim + better deployment story). |
| F | true Δφ | true Δφ | **yes** | Symmetric encoder-side phase. Tests whether phase-blind encoder is the right default. |

**Minimum viable set**: A (exists) + D + G + I. Three new runs, ~36 H100-hours. I is the strongest attribution test; paper cannot ship without it.

**Recommended set**: A + D + G + I + E2 + J. Five new runs, ~60 H100-hours. E2 strengthens the deployment story; J is the capacity-matched control that disambiguates case (c) disagreement.

**Extended set**: add E1 and F. Seven new runs, ~84 H100-hours.

### Attribution framing (why I and J are load-bearing)

The paper's claim is "cardiac phase specifically matters." The weaker version is "any reasonable temporal conditioning helps." Run I (fixed-HR = 72 bpm) distinguishes:

- **I matches D**: the contribution is "temporal-distance-aware conditioning," not "cardiac phase." Paper scope calibrates down.
- **I matches A**: population-average HR provides garbage conditioning for clips far from 72 bpm. Full D's HR-specific conditioning is doing the work. Strong claim stands.
- **I is between A and D**: both mechanisms contribute. Paper reports tiered claim — HR-specific phase > population-averaged temporal > no conditioning.

Run J further disambiguates the parameter-count concern. If J matches D, the apparent improvement is coming from predictor capacity, not phase content. If J matches A, the conditioning content is load-bearing.

### Disagreement-case handling

Gate 2 evaluates two criteria: downstream metrics (ΔR², reduced-EF MAE, pediatric r) and the direct representational test (phase-recovery linear probe, metric 4). Four cases:

- **(a) Both improve**: Clean story. Paper ships with strong representational claim.
- **(b) Neither improves**: Method fails Gate 2. Investigate or pivot.
- **(c) Downstream improves, phase recovery does not**: **Hard interpretability problem.** Something about the phase conditioning is helping but not via the representational mechanism we claimed. Run J determines whether it's a capacity effect. If yes (J matches D), paper does not ship with the representational framing; scope down to "conditioning the predictor helps." If no (J matches A), a different mechanism is at play and the paper requires additional investigation before shipping.
- **(d) Phase recovery improves, downstream does not**: The encoder encodes phase but the encoded phase is not clinically useful on the evaluated tasks. Weaker than (a) but still a real finding — reframe paper as representational study rather than clinical method. Supplementary experiments (e.g., phase-labeled clinical tasks) would be needed.

**Pre-registered commitment**: if case (c) obtains and Run J matches D, the paper does not ship with the current scope. Pre-registration prevents outcome-dependent re-framing.

---

## Evaluation

Every run is evaluated on four metrics:

1. **Matched-frame diagnostic**: clean R², matched-frame R², ΔR² on EchoNet-Dynamic LVEF. Primary metric is ΔR² — the diagnostic paper established this is the right measure of frame-order dependence.
2. **Reduced-EF tail MAE**: predictions stratified by LVEF < 40%. Secondary metric aligned with the paper's identified failure mode.
3. **Pediatric transfer Pearson r on EchoNet-Pediatric**: see caveats section below.
4. **Phase recovery linear probe**: train a linear probe on held-out EchoNet-Dynamic clips (with exact phase from ED/ES annotations) to predict phase from encoder features. **Direct representational test of the method's claim.**

### Pre-registered Gate 2 effect-size thresholds (ΔR² on EchoNet-Dynamic LVEF)

Baseline V-JEPA's matched-frame gap is −0.143. Pre-registered thresholds for Run D:

- **Clear success**: ΔR² ≤ −0.18 (improvement of ≥ 0.035 over baseline, ≥ 25% relative strengthening).
- **Marginal success**: −0.18 < ΔR² ≤ −0.155 (improvement of 0.012 to 0.035). Publishable but the paper's framing needs care around effect size.
- **Null**: −0.155 < ΔR² ≤ −0.135 (within run-to-run variance of baseline). Not a result; investigate or pivot.
- **Failure**: ΔR² > −0.135. Method is weakening temporal dependence. Debug before continuing.

**Clean R² constraint**: ±0.02 of baseline is acceptable; > +0.02 is a bonus; degradation beyond −0.03 is a fail even if ΔR² looks good.

**Phase-recovery probe**: baseline V-JEPA's encoder is expected to partially recover phase (linear-diff R² = 0.376 at e100 implies phase-probe R² in a similar order of magnitude). Run D should exceed baseline by ≥ 0.10 R² on phase recovery. Within ±0.03 of baseline puts us in case (c).

These thresholds are committed before the first run; they do not get adjusted after seeing results.

---

## Gates

**Gate 1 — DICOM metadata extraction pilot (PASSED 2026-04-24, job 351).** See "Gate 1 result" section above for full numbers. Summary: 99.59% HR coverage, 96.68% of studies within the tight-rhythm threshold, 0% parse errors. 74,314 clips scanned in 5m46s. The CSV at `s3://.../phi_jepa_gate1_351/clip_phase_metadata.csv` is the artifact consumed by the φ-JEPA dataloader.

**Gate 2 — Phase conditioning is useful (after Run D).**
- Evaluate against the pre-registered effect-size thresholds above.
- Pass criterion: clear or marginal success on downstream AND ≥0.10 R² improvement on phase-recovery probe. This is case (a).
- Case (c) handling: if downstream improves but phase recovery doesn't, run Run J immediately (before G) to determine whether it's a capacity effect.
- Fail mode: null or failure → predictor is ignoring phase. Investigate (phase-emb capacity, phase-dropout, unit/sign bugs in Δφ) before committing to further runs.

**Gate 3 — Conditioning signal isn't spurious (Run G, and Run J if Gate 2 triggered case (c) handling).**
- Pass criterion: G underperforms D on ≥1 primary metric and G matches A on phase-recovery probe.
- Fail mode: G matches D → phase conditioning carries no usable signal. Method fails.
- J criterion: if Gate 2 is in case (c), J must match A (not D) for the paper to ship with the representational framing.

**Gate 4 — Attribution (Run I, load-bearing for scope).**
- Evaluate I against D and A across all four metrics.
- Strong claim supported: I matches A (HR-specific phase is load-bearing).
- Scope-down: I matches D (temporal-distance conditioning suffices). Paper ships at lower scope with the weaker framing made explicit.
- Tiered claim: I between A and D (both mechanisms contribute). Paper reports tiered result.

---

## Compute budget

| Stage | Runs | H100-hours |
|---|---|---|
| Gate 1 (metadata pilot) | — | 0 (CPU only) |
| Run D (headline) | 1 | 12 |
| Run G (negative control) | 1 | 12 |
| Run I (attribution) | 1 | 12 |
| Run J (capacity control) | 1 | 12 |
| Run E2 (phase-free deployment) | 1 | 12 |
| Runs E1, F (optional extras) | 2 | 24 |
| Downstream evaluation (probes × multi-seed × 4 metrics) | — | ~40 |
| **Minimum viable (A + D + G + I)** | 3 new | ~40 |
| **Recommended (+ J + E2)** | 5 new | ~60 + downstream |
| **Extended (+ E1 + F)** | 7 new | ~84 + downstream |

---

## Positioning

- **Relative to V-JEPA**: architectural modification to the predictor. The ingredients are standard (conditional prediction, Fourier embeddings, DICOM metadata joins); the combination and the application are novel, and the representational claim — cardiac phase as first-class property for echo SSL — is the contribution.
- **Relative to Brain-JEPA's Gradient Positioning**: closest prior art in the "domain-specific structural conditioning of JEPA" space. Brain-JEPA injects brain-gradient positional encodings; φ-JEPA injects cardiac-phase offsets. Different organ, different structural property, shared framing.
- **Honest framing**: "cardiac phase as a first-class representational property for echo SSL, operationalized via phase-conditioned prediction in V-JEPA, evaluated via the matched-frame diagnostic from our prior work." Not "a novel SSL objective for medical video."

---

## Caveats and failure modes

### Pediatric transfer: method + pretraining-distribution, not method alone

Pediatric HRs span 80-140 bpm, largely outside the adult MIMIC-IV-Echo pretraining distribution (60-90 bpm). If Δφ values during pretraining cluster around adult cycle lengths, the Fourier embedding may not generalize to pediatric cycle lengths even though the phase math is correct. This is a pretraining-distribution coverage issue, not a method issue.

**Mitigation options (not in the primary matrix, follow-up if needed)**:
- Include EchoNet-Pediatric in pretraining data at some mix ratio.
- Synthetically upsample rare-HR clips during training.

**Paper framing**: pediatric-transfer improvement or lack thereof is a combined measure of method + distribution. If φ-JEPA fails to improve pediatric transfer, the paper should explicitly decompose "φ-JEPA trained on adult HR distribution does not transfer" from "φ-JEPA does not support cross-HR transfer." A small supplementary experiment — linear-probe φ-JEPA on a pediatric subset — would isolate method from distribution. Flag this as a known caveat rather than a method flaw.

### DICOM metadata portability

The method depends on `HeartRate` and `FrameTime` DICOM tags. UHN likely has these (Philips and GE both populate them), but some research datasets strip metadata during de-identification. Cross-institution generalization is contingent on tag preservation.

### Δφ precision vs clinical phase semantics

Δφ is derived from a per-clip scalar HR averaged over the clip's duration. For normal-rhythm clips at steady HR, this is near-exact. For subtle HR variation within a clip (~1-3 bpm over respiratory cycle), Δφ averages away the variation. This is fine for the method's representational claim (the predictor sees an approximately-correct cycle offset) but means the method is not a substitute for precise phase measurement (tissue Doppler, ECG-gated analysis). Downstream clinical tasks that depend on sub-beat phase precision would need a different approach.

---

## Open questions

1. **Per-target vs per-clip phase conditioning.** Current design injects Δφ per target token. Alternative: single per-clip `<phase>` token the predictor attends to. Per-target is more informative; per-clip is cheaper.
2. **Phase-embedding capacity.** 16 integer Fourier frequencies × 2 (sin/cos) → 32-D → d_model MLP. If Gate 2 fails with evidence of predictor ignoring phase, first knob to raise.
3. **λ_phase weighting.** Currently no separate loss weight; phase affects predictions only through the predictor's forward pass. Could add an explicit loss weight on target tokens that carry phase (vs `<no_phase>`); small pilot.
4. **Absolute-phase variant.** Alternative formulation conditions on absolute phase per target. Requires phase propagation decisions Δφ sidesteps. Potential follow-up if Run D clears gates.
5. **Stride-ranked auxiliary as phase-collapse regularizer.** Dropped from primary method. If phase-recovery probe shows encoder over-encodes phase at the expense of other temporal features, stride-RnC becomes a natural regularizer. Follow-up scope.
6. **Cross-institution generalization.** All development on MIMIC-IV-Echo. UHN validation is a natural extension once the method's MIMIC results land.
7. **Relation to ongoing SALT extended-teacher runs.** If SALT with better-trained pixel-reconstruction teachers (jobs 330, 335, 349, 350) still shows weak matched-frame gap, that strengthens the argument for latent-prediction methods like φ-JEPA over pixel-target alternatives.

---

## Job log

| Job | Experiment | State | Notes |
|---|---|---|---|
| **351** | **Gate 1 — DICOM metadata pilot on 1000 studies (74,314 clips)** | **COMPLETED 2026-04-24T11:15Z** | **PASS** — 99.59% HR coverage, 96.68% tight-rhythm studies, 0% parse errors, 5m46s runtime. CSV at `s3://.../phi_jepa_gate1_351/clip_phase_metadata.csv`. Report at `s3://.../phi_jepa_gate1_351/gate1_report.txt`. |
| pending | Gate 1 extension — run on full 525K MIMIC corpus | Not started | Reuses same script with `N_STUDIES=-1`. ~30 min. Produces the authoritative `mimic_clip_phase_metadata.csv` consumed by Run D. Can fold into Run D's sbatch setup phase. |
| pending | Run D (headline) | Not started | Needs code (dataloader, predictor, training step, config, sbatch). **Next step.** |
| pending | Run G (negative control) | Not started | Needs Run D Gate 2 pass. Submit after D completes. |
| pending | Run I (HR-attribution) | Not started | Load-bearing for paper scope. |
| pending | Run J (capacity control) | Not started | Required if Gate 2 triggers case (c). |
| pending | Run E2 (phase-free deployment) | Not started | Strengthens deployment story. |
| pending | Run E1 (untrained-phase inference) | Not started | Optional extra. |
| pending | Run F (encoder-phase variant) | Not started | Optional extra. |

### Next step

Implement Run D's code path:
1. `PhaseMetadataVideoDataset` class extending `VideoDataset` with phase-metadata join.
2. `PhaseConditionedPredictor` with Fourier phase embedding, `<no_phase>` token, phase dropout.
3. Modified training step that computes Δφ from context/target positions + per-clip HR + FrameTime.
4. Config `configs/train/vitl16/pretrain-phi-jepa-mimic-224px-16f.yaml`.
5. Sbatch `scripts/neurips/phase/pretrain_phi_jepa_vitl.sbatch`.
6. Debug pilot (10 epochs) before committing to full 100-epoch run.
