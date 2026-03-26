# Paper Review Simulation Prompt

Copy everything below this line and paste into Claude web app. Attach two files:
1. `icml_preprint.tex` (the paper)
2. `08-rebuttal-v2.md` (the rebuttal plan)

---

You are simulating the ICML 2026 review process for the attached paper. Your goal is to produce a rigorous, honest assessment that helps the authors strengthen their work before the rebuttal deadline.

## Your Role

Act as a panel of 4 ICML reviewers with complementary expertise:

- **Reviewer A**: Expert in self-supervised video learning (V-JEPA, VideoMAE, DINO, contrastive methods). Knows the V-JEPA2 paper inside-out.
- **Reviewer B**: Expert in medical imaging foundation models (echocardiography, ultrasound, clinical AI). Familiar with EchoPrime, PanEcho, EchoCardMAE, EchoFM.
- **Reviewer C**: Expert in evaluation methodology and experimental design (probing, fairness, controlled comparisons, statistical rigor).
- **Reviewer D**: Senior area chair perspective — broad ML knowledge, focused on novelty, significance, and whether the paper advances the field.

## ICML Review Criteria

Evaluate the paper on these four dimensions:

### 1. Quality (Technical Soundness)
- Are claims well-supported by theoretical analysis or experimental results?
- Are experiments properly controlled? Are baselines appropriate and fairly compared?
- Is the methodology sound? Are error bars reported?
- Is there sufficient detail for reproducibility?

### 2. Clarity (Writing & Organization)
- Is the paper clearly written and well organized?
- Can an expert reproduce the results from what is written?
- Is notation consistent? Are terms defined?
- Are figures and tables informative with self-contained captions?

### 3. Significance (Impact & Importance)
- Are the results impactful for the community?
- Will others build upon this work?
- Does it address an important problem?
- What is the potential for real-world impact?

### 4. Originality (Novelty & Contribution)
- Does this provide new insights? How does it differ from prior work?
- Is the contribution non-trivial?
- Note: "Originality does not necessarily require introducing an entirely new method. Papers that provide novel insights from evaluating existing approaches or shed light on why methods succeed can also be highly original." (NeurIPS guidelines)

## ICML Scoring Scale

| Score | Label | Description |
|-------|-------|-------------|
| 6 | Strong Accept | Groundbreaking, flawless; top 2-3% |
| 5 | Accept | Technically solid, high impact; benefits the community |
| 4 | Borderline Accept | Solid with limited evaluation; leans accept |
| 3 | Borderline Reject | Solid but weaknesses outweigh strengths; leans reject |
| 2 | Reject | Technical flaws or weak evaluation |
| 1 | Strong Reject | Known results or unaddressed ethics concerns |

**Calibration:** Typical accepted ICML paper = 4-5. Borderline = 3-4. Clear reject = 1-2.

**Confidence scale:** 1 (low) to 5 (absolute certainty, checked details carefully).

## Writing Style Guidelines for Reviews

Follow Daniel Dennett's rules for constructive criticism:
1. Re-express the paper's position fairly — show you understand it
2. List points of agreement — acknowledge what works
3. List what you learned from the paper
4. Only then critique — after establishing understanding

Each review should be specific and actionable. Avoid vague complaints ("the paper is incremental"). Instead, explain precisely what is missing and why it matters ("The paper claims X but provides no evidence of Y; specifically, a controlled experiment varying Z would be needed to support this claim").

## Review Format

For each of the 4 reviewers, provide:

```
### Reviewer [A/B/C/D] — [Expertise Area]

**Summary** (1 paragraph)
What the paper does, what it claims, and its main contribution.

**Strengths**
- S1: [Specific strength with reasoning]
- S2: ...
- S3: ...
(3-5 bullets)

**Weaknesses**
- W1: [Specific weakness with reasoning and what would fix it]
- W2: ...
- W3: ...
(3-5 bullets)

**Key Questions for Authors**
- Q1: [Question that would change your assessment if answered well]
- Q2: ...
(2-4 questions)

**Minor Issues**
- [Typos, unclear sentences, formatting — optional]

**Limitations Assessment**
Do the stated limitations adequately cover the paper's actual weaknesses?

**Soundness:** [1-4]
**Presentation:** [1-4]
**Significance:** [1-4]
**Originality:** [1-4]

**Overall Score:** [1-6]
**Confidence:** [1-5]
**Recommendation:** [1-2 sentence justification]
```

## After All 4 Reviews: Meta-Review

After producing all four reviews, write a meta-review that synthesizes the findings:

### Part 1: Consensus and Disagreement
- Where do the reviewers agree? Where do they disagree?
- What is the likely outcome at an ICML program committee meeting?

### Part 2: Cross-Reference with Actual Reviews

The paper has already been reviewed at ICML 2026. The actual scores were **2 (Reject), 3 (Weak Reject), 3 (Weak Reject), 4 (Weak Accept)**. The actual reviewer concerns were:

1. **Limited novelty over V-JEPA** — all 4 reviewers raised this. "The technical contribution appears limited... does not meet the high bar typically expected at ICML." "Core architecture is largely based on V-JEPA... changes are incremental."

2. **No segmentation evaluation** — 2 reviewers. "A vision foundation model is expected to demonstrate general-purpose visual capabilities... segmentation datasets are widely available... only evaluates classification and regression."

3. **Speckle suppression claim not validated at the representation level** — 1 reviewer. "Does not provide representation-level analyses to demonstrate that the learned embeddings are indeed more robust to speckle noise... Without feature invariance tests, representation visualization, or noise sensitivity studies, it remains unclear whether the gains truly arise from noise suppression."

4. **No contrastive pretraining comparison** — 1 reviewer. "No comparisons are provided between JEPA and contrastive pretraining approaches in this domain."

5. **Proprietary data / reproducibility** — 3 reviewers. "Difficult for other researchers to fully replicate or verify the flagship model's performance."

6. **Dataset diversity** — 1 reviewer. "The dataset appears to contain only five categories despite its scale."

7. **Double-blind violation** — 1 reviewer. "Internal dataset names suggest the affiliation of the authors."

For the meta-review, answer:
- Which of your simulated concerns overlap with the actual reviews?
- Which of your simulated concerns are NEW (not caught by actual reviewers)?
- Which actual reviewer concerns did your simulation MISS?
- What does this tell us about blind spots in both the paper and the review process?

### Part 3: Rebuttal Assessment

The authors have prepared a rebuttal plan (attached as 08-rebuttal-v2.md). Assess:
- Does the rebuttal adequately address each actual reviewer concern?
- Are the proposed experiments (speckle CKA, noise-level probing, segmentation) likely to change reviewer opinions?
- Is the EchoJEPA-G gated release announcement strategically effective?
- Is the novelty framing ("objective-domain alignment hypothesis, not architectural novelty") convincing?
- What is the single highest-impact action NOT yet in the rebuttal plan?

### Part 4: Predicted Outcome

Given the actual scores (2/3/3/4), the rebuttal plan, and your analysis:
- What is the most likely post-rebuttal score distribution?
- What is the probability of acceptance?
- What would tip the decision?

## Important Instructions

- Be honest and critical. Do not soften your assessments to be polite. The authors need to hear the hard truths before the rebuttal deadline.
- Ground every criticism in specific paper content (quote or reference specific sections, tables, equations).
- Distinguish between fatal flaws (would require a new paper) and addressable weaknesses (could be fixed in a rebuttal or camera-ready).
- Consider that this is an applications paper at a methods venue (ICML). How does that affect the novelty bar?
- The paper claims to be the first controlled evidence that pretraining objective determines representation quality for ultrasound. Assess whether this framing is sufficient for ICML-level novelty.
