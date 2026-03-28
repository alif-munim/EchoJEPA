# PART III: Worst-Case Scenarios

What to do when a reviewer digs in on a specific attack vector.

---

## Scenario 1: "Your VideoMAE baseline is broken. Reject."

**Response:**
- Lead with RVSP competitiveness (proves model isn't broken)
- Lead with clustering (proves properly-trained baselines don't do better)
- Offer to rerun VideoMAE with standard LR during rebuttal if essential
- Emphasize EchoJEPA-G vs EchoPrime comparison (no VideoMAE involved)

---

## Scenario 2: "Without EchoCardMAE comparison, claims are incomplete."

**Response:**
- Acknowledge limitation honestly
- Note protocol difference (frozen vs fine-tuned)
- Argue EchoCardMAE's modifications support your hypothesis
- Offer to add comparison in camera-ready
- Emphasize existing baseline diversity

---

## Scenario 3: "Model size explains everything."

**Response:**
- No-gradient argument: EchoMAE-L (307M) = PanEcho (28M) on view classification
- Data scale doesn't compensate: EchoPrime (12M videos) still only 57.7% linear
- Compute-matched comparison controls for size

---

## Scenario 4: "Frozen probing is unfair to EchoPrime."

**Response:**
- Acknowledge and qualify claims
- Note EchoPrime paper also reports frozen results
- Emphasize practical value of frozen evaluation
- Note identical protocol for all models

---

## Scenario 5: "The 45-point gap is too large to be believable."

**Response:**
- Show sample efficiency curve: gap persists at 1%, 10%, 100%
- Show multi-site consistency: Toronto, Chicago, Stanford
- Attention visualizations support semantic vs noise interpretation
- The gap is large because the domain (ultrasound) is unusual

---

## Scenario 6: "The attentive probe is unfair to CNN baselines."

**Response:**
- Lead with LVEF: same probe achieves 4.87 MAE for EchoPrime (clinically useful)
- Show controlled comparison: EchoJEPA-L vs EchoMAE-L, identical ViT-L architectures
- Present linear probe results: ranking preserved, gap narrowed but robust (80.9% vs 57.7%)
- Offer to add both probe types in camera-ready tables
- Acknowledge the inversion openly — scientific honesty strengthens credibility

**Fallback:** "Even under the most generous interpretation — using only linear probes and discarding attentive results entirely — EchoJEPA-G achieves 80.9% vs the best baseline's 59.2%, a 21.7-point advantage. The controlled comparison (EchoJEPA-L: 70.8% vs EchoMAE-L: 59.2%) shows an 11.6-point gap with identical architectures."

---

## Scenario 7: "Your numbers are inflated by unfair comparisons."

**Response:**
- Present the comparison taxonomy:

| Comparison | Controls for | Does NOT control for | Claim it supports |
|-----------|-------------|---------------------|-------------------|
| EchoJEPA-L vs EchoMAE-L | Architecture, probe, data, compute | Scale (both small) | JEPA objective > MAE objective |
| EchoJEPA-L vs EchoJEPA-G | Objective, probe | Architecture, scale, data | Scaling benefits within JEPA |
| EchoJEPA-G vs EchoPrime | Probe | Architecture, scale, data, objective, embed dim | System-level performance |
| EchoJEPA-G vs PanEcho | Probe | Architecture, scale, data, objective, embed dim | System-level performance |

- Acknowledge each comparison's limitations explicitly
- Emphasize the controlled comparison as the anchor claim
- Note that even system-level gaps survive linear probes (80.9% vs 57.7%)
