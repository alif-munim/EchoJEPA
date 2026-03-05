EchoJEPA
Start working on the innovation committee application.
Submit REB, digital ticket. We need to work with the vendor.
Use current model weights. Work with Ali on this.
Wendy has the plan for prospective study.
Final Paper Review and Master Rebuttal Template

PART I: COMPREHENSIVE PAPER AUDIT
Executive Summary
Paper Strengths:
Novel, well-motivated hypothesis (latent prediction for noisy medical imaging)
Large-scale pretraining (18M videos, largest echo corpus to date)
Comprehensive evaluation (3 tasks, 4 sites, multiple axes)
Strong results on sample efficiency and robustness
Public model release for reproducibility
Critical Vulnerabilities:
VideoMAE baseline severely underperforms literature (8.52 vs 3.78 MAE)
Low learning rate configuration (~170× below standard)
Appendix documentation inconsistencies
Missing comparisons to echo-specific MAE methods (EchoCardMAE)
Key Defensive Asset: All non-JEPA baselines (including properly-trained EchoPrime/PanEcho) cluster at 40-42% view accuracy, demonstrating the gap reflects objective limitations, not optimization failures.

Detailed Issue Inventory
TIER 1: CRITICAL (Must Address Proactively)

C1. EchoMAE-L LVEF Performance vs Literature
The Problem:
Method
LVEF MAE
Dataset
Modifications
EchoCardMAE
3.78
EchoNet-Dynamic
Sector masking, denoised targets, InfoNCE, 1600 epochs, fine-tuned
Echo-Vision-FM
~3.9
MIMIC-IV-Echo
85% mask, STF-Net head, fine-tuned
Your EchoMAE-L
8.52
EchoNet-Dynamic
Vanilla VideoMAE, ~164 epochs, frozen probe

Anticipated Attack: "Published VideoMAE methods achieve 3.78 MAE. Your baseline achieves 8.52—over 2× worse. This invalidates your comparison."
Defense Evidence:
EchoMAE-L achieves competitive RVSP (5.36 vs 5.01)—model isn't broken
EchoPrime/PanEcho (properly trained) also achieve only 5.33/5.43 on LVEF—similar to EchoJEPA-L
EchoCardMAE requires extensive modifications that explicitly work around reconstruction's limitations
Frozen probing vs fine-tuning is a fundamental protocol difference
Response:
"We clarify several important distinctions:
Protocol difference: EchoCardMAE and Echo-Vision-FM fine-tune the encoder end-to-end; we use frozen probing to isolate representation quality. Prior work shows reconstruction methods benefit disproportionately from fine-tuning to discard noise patterns learned during pretraining.
Baseline validity: EchoMAE-L achieves competitive RVSP (5.36 vs 5.01 MAE), demonstrating meaningful representation learning. A broken model would fail across all tasks.
Echo-specific modifications: EchoCardMAE achieves strong LVEF through sector-aware masking, denoised reconstruction targets, and temporal alignment losses—modifications that explicitly compensate for reconstruction's tendency to learn speckle patterns. Our baseline tests whether vanilla reconstruction transfers to ultrasound; it does not.
Supporting our hypothesis: The fact that reconstruction requires extensive domain engineering (EchoCardMAE) while latent prediction succeeds without it (EchoJEPA) directly supports our claim about objective-domain alignment."
Worst-Case Escalation: Reviewer insists comparison is unfair.
Fallback: "We acknowledge this limitation and will add EchoCardMAE as an additional baseline in the camera-ready version, pending code availability. However, we emphasize that our core claim—about the superiority of latent prediction in the frozen-probe regime—remains supported by the existing evidence."

C2. VideoMAE Learning Rate Configuration
The Problem:
Your LR: 8.79e-7 (base) → 3.5e-6 (scaled)
Standard VideoMAE LR: 1.5e-4 (base) → 6e-4 (scaled at batch 1024)
Gap: ~170× lower than standard
Anticipated Attack: "The VideoMAE baseline used an extremely low learning rate. The performance gap may reflect optimization failure."
Defense Evidence:
Model converged (loss 0.87 → 0.27)
RVSP performance is competitive (5.36 vs 5.01)
ALL non-JEPA baselines cluster at 40-42% view accuracy
Response:
"We acknowledge the learning rate was conservative. However, we emphasize:
Convergence: EchoMAE-L converged normally (final loss: 0.27, down from 0.87), indicating successful optimization.
Non-degenerate representations: EchoMAE-L achieves competitive RVSP estimation (5.36 vs 5.01), demonstrating the model learned meaningful temporal features.
Clustering pattern: ALL non-JEPA baselines—including EchoPrime (12M videos, official training) and PanEcho (1M+ videos, official training)—cluster at 40-42% view accuracy. If VideoMAE's poor view classification reflected optimization failure, properly-trained baselines should perform substantially better. They do not.
The clustering of three different objectives (supervised, contrastive, reconstruction) at identical performance, while only latent prediction achieves 85%+, indicates the gap reflects objective-domain alignment, not optimization quality."

C3. Appendix Inconsistencies
The Problems:
Location
Stated
Actual
Issue
A.1
Cooldown: 12,000 updates
24,000 (80 epochs × 300)
Typo
A.2 vs A.3
V-JEPA: 84,000 total
~87,000 actual
Mismatch
§3.2 vs A.1
24 fps
8 fps
Unclear which applies where
A.3
LR: 8.79e-7
After scaling: 3.5e-6
Confusing presentation

Response:
"We apologize for documentation errors in Appendix A. Corrections:
Cooldown updates: A.1 should state 24,000 updates (80 epochs × 300 ipe), not 12,000.
Total updates: EchoJEPA-L: ~87,000; EchoMAE-L: ~84,000. The 3.5% difference is within acceptable tolerance.
FPS: Section 3.2 describes domain adaptations for EchoJEPA-G (24 fps). The compute-matched comparison (Appendix A) uses 8 fps for both models.
Learning rate: We will clarify that the stated value (8.79e-7) is before batch-size scaling; the actual optimizer LR is 3.5e-6.
All corrections will appear in the camera-ready version. These are documentation errors; the experiments themselves were conducted correctly."

TIER 2: HIGH PRIORITY (Likely to Be Raised)

H5. Attentive Probe Architecture Mismatch
→ See `probe-fairness-rebuttal.md` for full analysis, three-pillar defense, response template, and worst-case escalation. Key finding: attentive probes underperform linear probes for CNN baselines due to token starvation (1 pre-pooled token vs 1568 spatial-temporal tokens). Gap narrows from ~45pts to ~23-28pts with linear probes but EchoJEPA remains dominant.

H1. Model Size Disparity
The Problem:
EchoJEPA-L: 307M params
EchoPrime: 35M params
PanEcho: 28M params
Anticipated Attack: "The 10× size difference explains the performance gap."
Defense Evidence:
Model
Params
View Acc
PanEcho
28M
41.9%
EchoPrime
35M
42.1%
EchoMAE-L
307M
40.4%

No performance gradient with size among non-JEPA models.
Response:
"If model size were determinative, we would expect: PanEcho (28M) < EchoPrime (35M) < EchoMAE-L (307M). Instead, all three achieve 40-42%—the 10× larger EchoMAE-L gains nothing.
The compute-matched comparison directly addresses this: EchoJEPA-L and EchoMAE-L use identical ViT-L architectures (307M params). The 45-point view classification gap is attributable only to the training objective.
We also note EchoPrime was trained on 23× more data (12M vs 525K videos). If scale compensated for objective, EchoPrime should substantially outperform EchoJEPA-L. It achieves half the accuracy."

H2. Frozen-Probe Protocol Limitations
Anticipated Attack: "EchoPrime was designed for end-to-end use. Frozen probing disadvantages it unfairly."
Response:
"We acknowledge frozen probing is one evaluation paradigm. Our goal was to isolate representation quality. We note:
Practical value: Frozen evaluation enables clinical researchers without large compute budgets to deploy foundation models.
Fair protocol: All models underwent identical evaluation with the same probe architecture and hyperparameter grid.
Claim qualification: We will revise 'state-of-the-art' to 'state-of-the-art frozen-backbone performance' where appropriate.
Precedent: EchoPrime's original paper also reports frozen/linear probe results, suggesting this is a valid evaluation paradigm."

H3. Missing Baseline Comparisons
Anticipated Attack: "You should compare to EchoFM, EchoCardMAE, DISCOVR, and other echo-specific SSL methods."
Response:
"We acknowledge this limitation. EchoFM, EchoCardMAE, and DISCOVR are concurrent or recent works. We note:
Existing diversity: Our comparison spans supervised (PanEcho), contrastive (EchoPrime), and reconstruction (EchoMAE-L)—all clustering at 40-42% view accuracy.
Cleanest test: The EchoJEPA-L vs EchoMAE-L comparison controls all variables except the objective, providing the strongest evidence for our hypothesis.
Public release: Our model and evaluation framework enable direct community comparison.
We will add EchoFM/EchoCardMAE comparisons in the camera-ready version if weights are publicly available."

H4. Task-Specific Performance Pattern
The Pattern:
Task
EchoMAE-L
Interpretation
RVSP (competitive)
5.36
Captures temporal/velocity dynamics
LVEF (terrible)
8.52
Misses precise spatial anatomy
View Class. (clusters with baselines)
40.4%
Fails at semantic discrimination

Anticipated Attack: "Why does EchoMAE do well on RVSP but terribly on LVEF? This inconsistency is confusing."
Response:
"This task-specific pattern is consistent with our hypothesis and provides important insight:
RVSP requires detecting TR jet velocity from color Doppler intensity patterns—primarily temporal/intensity dynamics that pixel reconstruction captures well.
LVEF requires precise LV boundary delineation and wall motion tracking—spatial anatomy that reconstruction conflates with speckle patterns.
View classification requires semantic category discrimination—global structure that reconstruction does not explicitly optimize for.
The pattern demonstrates reconstruction is not useless—it captures certain features—but systematically fails on tasks requiring spatial precision or semantic understanding. Latent prediction succeeds on all three."

TIER 3: MEDIUM PRIORITY (May Be Raised)

M1. Narrow Hyperparameter Grid
Response:
"Our grid follows the V-JEPA2 evaluation protocol. EchoJEPA outperforms baselines across all 6 configurations; the ranking is stable. The 45-point view classification gap is unlikely to close with hyperparameter tuning alone."

M2. Synthetic-Only Robustness
Response:
"We acknowledge this limitation (discussed in Section 5). Our perturbations are physics-grounded, not arbitrary. Multi-site evaluation provides some real distribution shift. Prospective validation remains future work."

M3. Pediatric Zero-Shot Definition
Response:
"Zero-shot means: (1) no pediatric data in pretraining corpus, (2) probes trained on adult data only, (3) frozen encoder + adult probe applied directly to EchoNet-Pediatric test split without any pediatric tuning."

TIER 4: EDITORIAL FIXES (Low Priority)
Issue
Location
Fix
"encoder processes masked frames"
Fig 1 caption
"visible (unmasked)"
"40% less than reconstruction-based"
Contributions
"86% less than next-best baseline"
"reconstruction-based models"
Table 5 caption
"alternative foundation models"
300M vs 307M params
§3.3 vs §4.1
Standardize to 307M
Missing citations (?)
Appendix B.1, B.4.1
Fix Albumentations, Smistad refs
Extra comma
Intro ¶2
Remove after EchoPrime citation


PART II: MASTER REBUTTAL TEMPLATE

Opening
We thank all reviewers for their thorough and constructive feedback. We address the key concerns below and commit to incorporating all corrections in the camera-ready version.

Core Defense: The Clustering Pattern
[Use this as your anchor argument throughout]
We emphasize our central finding: ALL non-JEPA baselines cluster at 40-42% view classification accuracy, regardless of training quality, model size, or data scale:
Model
Objective
Params
Training Data
View Acc
PanEcho
Supervised
28M
1M+ videos
41.9%
EchoPrime
Contrastive
35M
12M videos
42.1%
EchoMAE-L
Reconstruction
307M
525K videos
40.4%
EchoJEPA-L
Latent prediction
307M
525K videos
85.5%

This clustering—spanning three different objectives, 10× parameter range, and 23× data scale—while only latent prediction achieves 85%+, provides strong evidence that objective-domain alignment, not optimization or scale, determines performance on ultrasound.

Response to Compute-Matched Comparison Concerns
VideoMAE Learning Rate
We acknowledge the conservative learning rate. However:
Convergence: EchoMAE-L converged normally (loss 0.87 → 0.27).
Non-degenerate representations: Competitive RVSP (5.36 vs 5.01) demonstrates meaningful learning.
Clustering pattern: If optimization explained VideoMAE's poor view classification, properly-trained EchoPrime and PanEcho should perform substantially better. They achieve identical 40-42% accuracy.
Comparison to Literature (EchoCardMAE)
EchoCardMAE achieves 3.78 MAE through:
Sector-aware masking
Denoised reconstruction targets
Temporal alignment losses (InfoNCE)
1600 epochs (vs our 164)
End-to-end fine-tuning (vs our frozen probing)
These modifications explicitly compensate for reconstruction's tendency to learn speckle. The fact that reconstruction requires such engineering while latent prediction succeeds without it supports our hypothesis.
Documentation Errors
We apologize for inconsistencies in Appendix A:
Cooldown: 24,000 updates (not 12,000)
FPS: 24 for EchoJEPA-G, 8 for compute-matched comparison
Total updates: V-JEPA ~87K, VideoMAE ~84K (within 4%)
All corrections will appear in camera-ready.

Response to Baseline Fairness Concerns
Model Size
No performance gradient exists among non-JEPA models: the 10× larger EchoMAE-L (307M) achieves identical accuracy to PanEcho (28M). The compute-matched comparison (identical 307M architecture) isolates the objective.
Frozen Protocol
We qualify claims to "frozen-backbone performance." Frozen evaluation isolates representation quality and has practical value for resource-constrained clinical deployment. All models underwent identical evaluation.
Missing Baselines
Our comparison spans supervised, contrastive, and reconstruction objectives—all clustering at 40-42%. We will add EchoFM/EchoCardMAE if weights are available.

Response to Task-Specific Pattern
EchoMAE-L's performance pattern (competitive RVSP, poor LVEF, poor views) is consistent with our hypothesis:
RVSP: Requires temporal/velocity dynamics → reconstruction captures this
LVEF: Requires spatial anatomy → reconstruction conflates with speckle
Views: Requires semantic categories → reconstruction doesn't optimize for this
This pattern would persist regardless of learning rate; it reflects what the objective incentivizes.

Summary of Revisions
Category
Action
Documentation
Fix all Appendix A inconsistencies
Claims
Qualify "SOTA" to "frozen-backbone SOTA"
Editorial
Fix Fig 1 caption, Table 5 caption, contributions bullet, citations
Baselines
Add EchoFM/EchoCardMAE comparison if feasible


Closing
The clustering of all non-JEPA methods at 40-42%—across architectures (28M-307M), data scales (525K-12M), and objectives (supervised/contrastive/reconstruction)—while only latent prediction achieves 85%+, provides robust evidence for objective-domain alignment. We believe EchoJEPA makes a significant contribution to understanding how pretraining objectives should be matched to domain characteristics in medical imaging.

PART III: WORST-CASE SCENARIOS

Scenario 1: "Your VideoMAE baseline is broken. Reject."
Response:
Lead with RVSP competitiveness (proves model isn't broken)
Lead with clustering (proves properly-trained baselines don't do better)
Offer to rerun VideoMAE with standard LR during rebuttal if essential
Emphasize EchoJEPA-G vs EchoPrime comparison (no VideoMAE involved)

Scenario 2: "Without EchoCardMAE comparison, claims are incomplete."
Response:
Acknowledge limitation honestly
Note protocol difference (frozen vs fine-tuned)
Argue EchoCardMAE's modifications support your hypothesis
Offer to add comparison in camera-ready
Emphasize existing baseline diversity

Scenario 3: "Model size explains everything."
Response:
No-gradient argument: EchoMAE-L (307M) = PanEcho (28M)
Data scale doesn't compensate: EchoPrime (12M videos) still only 42%
Compute-matched comparison controls for size

Scenario 4: "Frozen probing is unfair to EchoPrime."
Response:
Acknowledge and qualify claims
Note EchoPrime paper also reports frozen results
Emphasize practical value of frozen evaluation
Note identical protocol for all models

Scenario 5: "The 45-point gap is too large to be believable."
Response:
Show sample efficiency curve: gap persists at 1%, 10%, 100%
Show multi-site consistency: Toronto, Chicago, Stanford
Attention visualizations support semantic vs noise interpretation
The gap is large because the domain (ultrasound) is unusual

PART IV: FINAL ASSESSMENT
Paper Verdict: Strong Accept with Minor Revisions
Strengths
Novel, well-motivated hypothesis with domain-specific reasoning
Comprehensive evaluation across tasks, sites, and data regimes
The clustering pattern is compelling evidence independent of VideoMAE issues
Sample efficiency (1% labels beating 100% baselines) is striking
Public model release enables independent validation
Weaknesses
VideoMAE configuration is a legitimate concern, partially mitigated by clustering evidence
Missing echo-specific MAE baselines (EchoCardMAE) is a gap
Documentation errors suggest rushed preparation
Why It Should Be Accepted
The clustering of all non-JEPA methods at 40-42%—including properly-trained EchoPrime and PanEcho—is the paper's strongest evidence. This pattern cannot be explained by VideoMAE optimization alone. Even if reviewers discount the compute-matched comparison entirely, the EchoJEPA-G vs EchoPrime comparison (no VideoMAE involved) shows a 45-point gap on the same evaluation protocol.
The theoretical contribution (objective-domain alignment for noisy medical imaging) is significant and the practical contribution (sample-efficient, robust foundation model) addresses real clinical needs.
Key Action Items Before Camera-Ready
Fix all Appendix A inconsistencies
Fix all editorial errors (Fig 1 caption, Table 5 caption, etc.)
Add EchoCardMAE comparison if weights available
Qualify SOTA claims to "frozen-backbone"
Add explicit discussion of LR choice in appendix

The paper is defensible. Lead with the clustering pattern. Acknowledge limitations honestly. The science is sound.

NOTE: The clustering pattern (40-42% uniformity) is partially a probe artifact — see `probe-fairness-rebuttal.md` for updated narrative, claim validity tiers, and camera-ready recommendations.



Core Differentiators
Aspect
EchoJEPA
Echo-Vision-FM / EchoCardMAE
EFNet
Objective
Latent prediction (JEPA)
Pixel reconstruction (MAE)
Supervised
Evaluation
Frozen backbone
Fine-tuned encoder
End-to-end
Scale
18M videos
~10K-100K
~10K
Domain modifications
None needed
Extensive (denoised targets, sector masking, alignment loss)
N/A


1. Training Objective: Latent vs Pixel Prediction
This is your core scientific contribution.
Method
What it predicts
Problem with ultrasound
EchoCardMAE / Echo-Vision-FM
Masked pixels
Must faithfully reconstruct speckle noise
EchoJEPA
Masked embeddings
EMA target naturally suppresses stochastic noise

EchoCardMAE achieves good results, but only by adding three modifications that explicitly work around reconstruction's limitations:
Denoised reconstruction targets (median blur to remove speckle)
Key area masking (focus on cardiac sector, ignore background)
Temporal alignment loss (InfoNCE to enforce cycle-invariance)
EchoJEPA needs none of these. The JEPA objective inherently downweights unpredictable artifacts.
Your argument: "The fact that reconstruction requires extensive domain engineering while latent prediction succeeds without it supports our hypothesis about objective-domain alignment."

2. Evaluation Protocol: Frozen vs Fine-Tuned
This is a critical methodological difference that makes raw MAE comparison misleading.
Method
Protocol
What it tests
EchoCardMAE
Fine-tune encoder + head
End-to-end system performance
Echo-Vision-FM
Fine-tune encoder + STF-Net
End-to-end system performance
EFNet
Train from scratch
Supervised learning capacity
EchoJEPA
Freeze encoder, train probe only
Representation quality

Frozen evaluation is harder—the encoder can't adapt to the task. Yet EchoJEPA-G achieves 3.97 MAE on Stanford with a frozen backbone, competitive with fine-tuned methods (3.7-3.87).
Your argument: "Our frozen-probe protocol isolates representation quality. Achieving comparable MAE to fine-tuned methods while keeping the encoder frozen demonstrates that EchoJEPA learns superior representations out of the box."

3. Scale: 18M vs ~10K Videos
Model
Pretraining Data
Labeled Fine-tuning
EchoCardMAE
7,465 videos (EchoNet-Dynamic train)
Same
Echo-Vision-FM
MIMIC-IV-Echo (~100K?)
~8,753 (EchoNet-Dynamic)
EFNet
None (supervised)
~8,000
EchoJEPA-G
18.1M videos
Frozen probe on ~8,753
EchoJEPA-L
525K (MIMIC-IV-Echo)
Frozen probe on ~8,753

Even EchoJEPA-L (525K videos, public data) outperforms these models on view classification and shows competitive LVEF.

4. Multi-View Reasoning
Novel contribution that others don't address.
Model
View handling
EchoCardMAE
Single view (A4C)
Echo-Vision-FM
Single view
EFNet
Single view
EchoJEPA
Multi-view probing framework with factorized stream embeddings

Your multi-view framework enables:
RVSP estimation (requires A4C + PSAX-AV integration)
Robustness to missing views (view dropout during training)
Standardized evaluation across models

5. Clinical Breadth
Model
Tasks
Sites
EchoCardMAE
LVEF, segmentation
EchoNet-Dynamic only
Echo-Vision-FM
LVEF, classification
EchoNet-Dynamic, CAMUS
EFNet
LVEF
EchoNet-Dynamic
EchoJEPA
LVEF, RVSP, view classification, robustness, pediatric transfer
Toronto, Chicago, Stanford, EchoNet-Pediatric


Summary: Your Unique Positioning
EchoJEPA is different because:

1. OBJECTIVE: Latent prediction instead of pixel reconstruction
   → No need for echo-specific hacks (denoised targets, sector masking)
   → Inherently ignores stochastic speckle

2. PROTOCOL: Frozen backbone evaluation
   → Tests representation quality, not fine-tuning capacity
   → Achieves 3.97 MAE frozen vs ~3.8 MAE fine-tuned for others

3. SCALE: 18M videos (35× larger than next biggest)
   → Even public EchoJEPA-L (525K) shows strong results

4. MULTI-VIEW: Novel probing framework
   → Enables RVSP and other multi-view tasks
   → Others limited to single-view A4C

5. BREADTH: 3 tasks, 4 sites, robustness + pediatric transfer
   → Most comprehensive evaluation in the literature

What to Say in the Paper/Rebuttal
"Unlike prior VideoMAE-based approaches (EchoCardMAE, Echo-Vision-FM) which require domain-specific modifications—denoised reconstruction targets, sector-aware masking, temporal alignment losses—to achieve strong performance, EchoJEPA succeeds without any such engineering. This supports our hypothesis that the JEPA objective is inherently better aligned with ultrasound's signal properties.
Furthermore, we evaluate under a frozen-probe protocol that isolates representation quality, whereas prior methods fine-tune the encoder. EchoJEPA-G achieves 3.97 MAE on Stanford with frozen backbones, competitive with fine-tuned methods reporting 3.7-3.87 MAE, demonstrating that latent prediction yields superior representations without task-specific adaptation."


