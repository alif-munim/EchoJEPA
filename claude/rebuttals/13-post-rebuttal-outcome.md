# ICML Rebuttal — Post-Rebuttal Outcome (2026-04-04)

## Final Scores: Unchanged

All four reviewers maintained their original scores. Average remains 3.0.

| Reviewer | Score | Ack Category | Key Statement | Score Change |
|----------|-------|-------------|---------------|-------------|
| **hfQ1** | **2** | (c) Unresolved | "assessment remains unchanged" — dataset diversity, contrastive baselines, novelty | None |
| **6t2T** | **3** | (c) Unresolved | "did not provide evaluation with other SOTA methods, including non-JEPA" | None |
| **ncQn** | **3** | (c) Unresolved | "controlled baselines addressed" but "novelty remains my main concern" | None |
| **L8sp** | **4** | (b) Follow-up | "limited novelty and originality... will maintain my original score" | None |

**Average: 2/3/3/4 = 3.0 — Borderline reject. Decision now with AC.**

---

## What Each Reviewer Said

### hfQ1 (2 → 2)

Acknowledged detailed rebuttal but raised **new concerns** beyond the original review:
- Dataset diversity not convincing — "5 categories" is insufficient semantic space regardless of scale
- BYOL is not a "true contrastive method" — wants SimCLR/MoCo comparison
- Task-dependent behavior is "empirical observation, not principled advance"
- No IRB statement provided
- Questioned whether natural video pretraining would yield stronger representations

**Authors' follow-up addressed all points:** semantic space = view x anatomy x pathology x severity x phase; EchoJEPA-L-K (natural video init) results; EchoPrime as system-level contrastive reference (85.5% vs 42.1% despite 23x less data); optimization artifacts argument; ethics board confirmation.

**Assessment:** hfQ1 is firmly entrenched. The "moving goalposts" pattern (new concerns each round) suggests the score will not change regardless of evidence. The confidence-5 reject is effectively final.

### 6t2T (3 → 3)

Brief acknowledgement. Core remaining concern: "no evaluation with other state-of-the-art methods, including non-JEPA-based methods."

**Authors' follow-up clarified** that MAE and BYOL are non-JEPA methods, and EchoPrime (CLIP) + PanEcho (supervised) are system-level references. Asked reviewer to specify which methods they want.

**Assessment:** The concern appears to reflect a misunderstanding (thinking MAE/BYOL are JEPA variants). The follow-up clarification may help with AC, but 6t2T is unlikely to engage further.

### ncQn (3 → 3)

**Most important acknowledgement.** Explicitly stated: "The additional experiments have addressed my concern regarding the controlled baselines." However: "my main concern remains on the novelty and the level of technical contribution."

**Authors' follow-up** highlighted ranking inversion, speckle probing, and EchoBench as contributions beyond "applying V-JEPA."

**Assessment:** ncQn was the most flippable reviewer (estimated 75-80% pre-rebuttal). They acknowledged the experiments resolved their experimental concerns but pivoted to novelty — the same concern as all other reviewers. The experimental effort was successful but insufficient: the novelty bar at ICML was the real blocker.

### L8sp (4 → 4)

Maintained the only positive score. Stated: "concerned about the limited novelty and originality" but categorized as (b) follow-up rather than (c) unresolved.

**Assessment:** L8sp is the strongest supporter but won't champion the paper. Score stable at 4.

---

## Ethics Reviews

Two ethics reviewers:

| Reviewer | Recommendation | Key Concern |
|----------|---------------|-------------|
| **eftz** | Remediation needed | Anonymity (city names), IRB statement, MIMIC model release via PhysioNet |
| **tDZn** | No remediation needed | City names don't sufficiently narrow identification |

**Authors responded** (AC-confidential comment):
- Will anonymize geographic identifiers (Site A, Site B)
- Confirmed institutional ethics board approval at both sites
- EchoJEPA-L to be released exclusively via PhysioNet (MIMIC DUA compliance)
- EchoJEPA-G via institutional governance with gated access

---

## Acceptance Probability (Updated)

The pre-rebuttal estimate was **35-45%**. The actual outcome maps to the **"Worst" scenario** in the probability table (which was assigned only 15% probability).

**Revised estimate: 10-20% acceptance, dependent entirely on AC.**

Factors favoring acceptance:
- ncQn acknowledged experiments resolved their concerns
- Ethics issues are all addressable in camera-ready
- Reproducibility commitments (EchoJEPA-G, EchoBench, code) are strong
- The three-way comparison is genuinely novel for medical video SSL
- AC may value the applied contribution + resources even if reviewers focus on novelty

Factors against:
- All four reviewers cited novelty as primary blocker — universal consensus
- No score movement at all, even from the most flippable reviewer
- Average 3.0 is below typical ICML acceptance threshold
- hfQ1's confidence-5 reject anchors the panel
- 6t2T's confusion about baselines suggests limited engagement
- Application papers face an uphill battle at ICML regardless of evidence quality

---

## Key Lessons

1. **Novelty was the universal blocker.** Every reviewer, including L8sp who gave a 4, cited limited novelty. The experiments were successful (ncQn confirmed) but insufficient to move scores because the novelty concern is structural: "applying V-JEPA to a new domain" is how all reviewers framed it, regardless of the mechanistic evidence.

2. **The reframing strategy was partially successful.** The "SSL insight paper" framing worked for the *experiments* (3-way comparison, speckle probing, EchoBench were well-received as evidence) but did not change reviewers' fundamental perception of the *contribution type*.

3. **ncQn's pivot is the most telling signal.** They explicitly said the experiments addressed their concerns, then maintained their score on novelty. This confirms that no amount of additional experiments would have changed the outcome — the novelty gap was fundamental.

4. **hfQ1's moving goalposts made engagement costly.** Each response generated new concerns (dataset diversity → natural video pretraining → true contrastive → optimization artifacts). This pattern suggests the reject was decided early and the engagement was performative.

5. **The 3,500 GPU-hours of rebuttal experiments produced excellent science** (anatomy-function dissociation, speckle probing, EchoBench) that will strengthen the Nature Medicine submission regardless of ICML outcome.

---

## Path Forward

**If rejected (most likely):**
- All rebuttal experiments directly feed Nature Medicine manuscript
- Three-way comparison + EchoBench + speckle probing become core sections
- SALT implementation (added 2026-04-04) provides a fourth SSL paradigm for comparison
- The "SSL insight" framing developed for the rebuttal becomes the Nature Medicine framing
- Resubmit to a venue more receptive to applied contributions (NeurIPS, MICCAI, Nature Methods)

**If accepted (unlikely but possible):**
- Camera-ready restructuring per Section D commitments
- Anonymize geographic identifiers
- Add IRB statement
- Release EchoJEPA-G via gated access
- Release EchoJEPA-L via PhysioNet
