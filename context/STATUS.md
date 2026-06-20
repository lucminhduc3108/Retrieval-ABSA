# Project Status — Retrieval-based ABSA

**Last updated:** 2026-06-20

---

## Dataset Switch: SemEval 2016 → 2014

**Reason:** SemEval 2016 had 12 fine-grained categories with severe rare category problem (8/12 Cat F1 < 0.7, DRINKS#PRICES F1 = 0.000). Stage 1 was capped at Cat F1 = 0.6962, bottlenecking Joint F1 regardless of Stage 2 quality.

**SemEval 2014 Task 4 Restaurant** has 5 balanced categories:
- food (1,166), anecdotes/miscellaneous (1,101), service (562), ambience (385), price (302)
- Train: 3,044 sentences / 3,516 opinions (after drop 196 conflict + 2 dedup)
- Test: 800 sentences / 973 opinions (after drop 52 conflict)
- No augmentation needed — data is balanced

---

## Pipeline Overview

Two-stage pipeline on SemEval 2014 Restaurant:

1. **Data Prep:** Parse XML → (category, polarity) pairs. Drop `conflict`. Deduplicate.
2. **Retrieval Engine:** DeBERTa encoder + projection head (768→256) trained with InfoNCE loss → FAISS IndexFlatIP.
3. **Stage 1 — Category Detection:** DeBERTa + optional Cat-Aware Attention → 5 sigmoid (ASL/BCE). Global threshold.
4. **Stage 2 — Sentiment Classification:** Two strategies:
   - **No-Retrieval (baseline):** DeBERTa → MLP(768→256→3).
   - **Phase 2a (Label Interpolation):** FAISS top-k + Diagonal W + polarity interpolation.
5. **Evaluation (end-to-end):** Category F1, Sentiment Acc|Correct Category, Joint F1 (cat+pol).

---

## Current Results

### Stage 1 — Category Detection (NB1)

| Config | Best Epoch | Cat F1 | Cat P | Cat R |
|--------|-----------|--------|-------|-------|
| ASL | 1 | 0.506 | 0.383 | 0.746 |
| **Cat-Aware** | **12** | **0.8368** | **0.8447** | **0.8291** |

### Stage 2 — Sentiment Classification (NB2)

**Val metrics — MAMS embedding (train-only FAISS index, val excluded):**

| Metric | Retrieval | No-Retrieval | **Aux Loss** | Delta (Aux vs Ret) |
|--------|-----------|-------------|-------------|-------------------|
| Metric | Retrieval | No-Retrieval | Aux Loss | **RD p=0.3** | RD p=0.5 |
|--------|-----------|-------------|----------|-------------|----------|
| **Best MacF1** | 0.9307 (ep4) | 0.8090 (ep16) | 0.9296 (ep3) | **0.9487 (ep8)** | 0.9297 (ep5) |
| **Best Acc** | 0.9540 | 0.8714 | 0.9558 | **0.9672** | 0.9585 |
| pos F1 | 0.981 | 0.929 | 0.985 | **0.986** | 0.985 |
| neg F1 | 0.937 | 0.834 | 0.927 | **0.953** | 0.937 |
| neu F1 | 0.874 | 0.663 | 0.877 | **0.906** | 0.867 |

RD p=0.3 has best val metrics across the board, but **does not generalize to test** (see Joint Eval below).

### End-to-End Joint Evaluation (NB3 — Test Set, per-category threshold)

| Metric | Ret (frozen, k=5) | Ret (aux, k=5) | Ret (aux, k=3) | **RD p=0.3** | No-Retrieval |
|--------|-------------------|---------------|---------------|-------------|-------------|
| Cat F1 | 0.8564 | 0.8564 | 0.8564 | 0.8564 | 0.8564 |
| **Sent Acc\|CC** | 0.8058 (668) | 0.8396 (696) | 0.8492 (704) | 0.8420 (698) | **0.8987** (745) |
| **Sent MacF1\|CC** | — | 0.7373 | 0.7456 | 0.7344 | **0.8022** |
| **Joint F1** | 0.6901 | 0.7190 | 0.7273 | 0.7211 | **0.7696** |

**RD p=0.3 worse than aux loss k=3** despite best val MacF1 (0.9487). Retrieval still loses on test.

### NB4 — Diagnostic Root Cause

Embedding polarity-blind for neg/neu → FAISS returns wrong-polarity neighbors → label_repr (64-dim) encodes wrong polarity → sentiment_head trusts label_repr (learned shortcut during training) → flips 115 correct predictions to wrong, only helps 35 → net -9.5pp.

### Retrieval Dropout — Why It Failed (2026-06-20)

RD p=0.3 achieves best val MacF1 (0.9487) but worst retrieval test Joint F1 (0.7211). Root cause: retrieval dropout is a dead-end strategy regardless of eval configuration.

| Eval strategy | Result | Problem |
|---|---|---|
| Dropout OFF (current) | Trust retrieval 100% | Wrong neighbors still corrupt predictions |
| Dropout ON (random) | Random keep/drop | Non-deterministic, drops good retrieval too |
| Zero label_repr (p=1.0) | ≈ No-retrieval | Discards all retrieval — pointless |

**Core flaw:** Dropout is random, not quality-aware. Model learns "sometimes retrieval is absent" but cannot tell WHEN retrieval is wrong at inference time. Also missing rescaling (no `/ (1-p)`), so eval-time label_repr is ~43% stronger than training expectation → amplifies damage from wrong neighbors.

### Agreement Filter Diagnostic (NB3, aux loss + MAMS embedding, k=3)

**Agreement Distribution (829 correct-category samples):**

| Agreement | Count | % | Baseline Acc | Filter Acc |
|-----------|-------|---|-------------|------------|
| Unanimous (3/3) | 726 | 87.6% | 89.4% | 89.4% (unchanged) |
| Majority (2/1) | 94 | 11.3% | 51.1% | 53.2% (+2.1pp) |
| Split (1/1/1) | 9 | 1.1% | 22.2% | 66.7% (+44.4pp) |

Filter activated on 103/829 (12.4%). Net gain: +6 samples (26 helped, 20 hurt).

**Filter metrics:** Joint F1 0.7221→0.7283 (+0.6pp), Sent Acc 0.8432→0.8504 (+0.7pp). Still loses to no-ret (0.7696).

**Per-polarity:** Filter helps pos (+0.9pp) and neg (+5.8pp) but **hurts neutral (-11.1pp)**.

### Neighbor Polarity Analysis (Test Set, MAMS embedding, k=3)

**Overall pol_match: 82.5% (2409/2919)**

| Gold | #Records | pol_match | Nb wrongly → pos | Nb wrongly → neg | Nb wrongly → neu |
|------|----------|-----------|-------------------|-------------------|-------------------|
| positive | 657 | 90.5% | — | 95 | 93 |
| negative | 222 | 70.3% | **117** (59% of wrong) | — | 81 |
| neutral | 94 | 56.0% | **73** (59% of wrong) | 51 | — |

**Key finding: positive dominance.** When neighbors are wrong, they are predominantly positive (59% of wrong neighbors for both neg and neu). Caused by positive being 59% of training data → embedding clusters by topic similarity → similar sentences tend to be positive.

**Unanimous-but-wrong breakdown:**
- Gold=positive: 35/610 wrong (5.7%) → neg=20, neu=15
- Gold=negative: 28/155 wrong (18.1%) → pos=20, neu=8
- Gold=neutral: **34/78 wrong (43.6%)** → pos=21, neg=13

43.6% of neutral samples have ALL 3 neighbors unanimously agreeing on the WRONG polarity. This is unfilterable — the agreement filter cannot help.

---

## Embedding Experiments (2026-06-18)

### Why retrieval fails: label_repr shortcut

MLP sentiment_head learns to trust label_repr (64-dim, clean polarity signal, correct 84% during training) over CLS (768-dim, noisy). On test, when neighbors have wrong polarity, label_repr corrupts predictions that CLS alone would get right.

### Approaches tried and failed

| Approach | Result | Root cause of failure |
|----------|--------|-----------------------|
| A1 v2 (polonly + cls_head) | Train pol_match 84% → test 77.6% (unchanged) | Projection head memorizes 3516 train positions |
| Exp 7 (CLS 768-dim, no projection) | Test pol_match 0.535 (worse) | Classification space ≠ retrieval space |
| Joint training v1 (lr=5e-7) | Test Sent Acc 0.799 (worse than frozen) | Gradient too weak (5-step chain + score collapse) |
| Joint training v2 (cls_head + lr=5e-6) | Best at ep2 (frozen), dropped after unfreeze | Embedding disrupted when updated |

### Current approach: Cross-polarity triplets + MAMS data

**Key insight:** Label interpolation only uses polarity — category matching is irrelevant. Current embedding wastes capacity on cat_match@5=100% while pol_match@5=77.6%.

**Fix:** Cross-polarity triplets (positive=same polarity any category, negative=different polarity) + MAMS dataset (3x more training data → prevents projection head memorization).

**A2 results (cross-polarity, SemEval only — 3516 records):**

| Polarity | A2 Test | Polonly Test | Delta |
|----------|---------|-------------|-------|
| positive | 0.924 | 0.897 | **+2.7pp** |
| negative | 0.654 | 0.580 | **+7.4pp** |
| neutral | 0.513 | 0.383 | **+13.0pp** |
| **Overall** | **0.823** | **0.775** | **+4.8pp** |

Cosine still collapsed (0.9943) but pol_match improved significantly.

**MAMS results (polonly, combined ~10K records):**

| Polarity | MAMS Test | A2 Test | Delta |
|----------|-----------|---------|-------|
| positive | 0.905 | 0.924 | -1.9pp |
| negative | **0.695** | 0.654 | **+4.1pp** |
| neutral | **0.545** | 0.513 | **+3.2pp** |
| **Overall** | **0.822** | **0.823** | -0.1pp |

Same overall as A2 but different strengths: extra MAMS data helps neg/neu, cross-polarity triplets help positive. Cosine collapsed (0.9929).

**A2+MAMS results (cross-polarity, combined ~10K records):**

| Polarity | A2+MAMS Test | A2 Test | MAMS Test |
|----------|-------------|---------|-----------|
| positive | 0.882 | 0.924 | 0.905 |
| negative | 0.672 | 0.654 | 0.695 |
| neutral | 0.519 | 0.513 | 0.545 |
| **Overall** | **0.799** | **0.823** | **0.822** |

A2+MAMS **worse than both** — cross-polarity on mixed domain introduces domain noise (SemEval paired with MAMS). Embedding experiments concluded.

**Chosen embedding: MAMS (0.822)** — most balanced polarity matching for neg/neu (the hard classes).

### New direction: Auxiliary loss on label_repr

Embedding ceiling at ~0.82. Switching to fix Stage 2: add auxiliary CE loss directly on label_repr to force polarity_embedding to encode polarity. Gradient from aux loss ~29x stronger than indirect main CE path.

---

## Code Changes

### 2026-06-18
- [x] A1 v2 (polonly + cls_head) results: train improved, **test unchanged** (memorization)
- [x] NB4 Exp 7: CLS 768-dim from no-ret DeBERTa → **failed** (pol_match=0.535)
- [x] Joint training v2: cls_head auxiliary loss + embedding_lr=5e-6 → **failed** (best at frozen ep2)
- [x] Implemented cross-polarity triplet builder (`build_cross_polarity_triplets`)
- [x] Implemented MAMS parser (`parse_mams_xml`) with category mapping
- [x] Added `--cross_polarity` and `--include_mams` flags to data prep scripts
- [x] Created 6 new embedding configs (A2/MAMS/A2+MAMS × s1/s2)
- [x] A2 (SemEval only): test pol_match@5 = **0.823** (+4.8pp over baseline)
- [x] MAMS (polonly, combined ~10K): test pol_match@5 = **0.822** (same as A2, but +4.1pp neg, +3.2pp neu)
- [x] A2+MAMS (cross-polarity, combined ~10K): test pol_match@5 = **0.799** (worse — domain noise)

### 2026-06-20
- [x] NB2 retrained all 5 runs: Retrieval, No-Ret, Aux Loss, RD p=0.3, RD p=0.5
- [x] RD p=0.3 best val MacF1=0.9487 (+1.9pp over aux loss), neu F1=0.906
- [x] RD p=0.5 val MacF1=0.9297 (dropout too aggressive, signal destroyed)
- [x] NB3 eval RD p=0.3: Joint F1=0.7211, **worse than aux loss k=3** (0.7273) and no-ret (0.7696)
- [x] Root cause analysis: retrieval dropout is dead-end — random (not quality-aware), disabled at eval, missing rescaling
- [x] Conclusion: Approach 2 (Retrieval Dropout) **failed**. Next: Approach 3 (Learnable Gate)

### 2026-06-20
- [x] Retrieval Dropout failed — best val MacF1 but worse test Joint F1 than aux loss
- [x] Implemented Learnable Gate (`RetrievalGate` in `sentiment_model.py`)
- [x] Gate input: [cls (768), label_repr (64), mean_neighbor_score (1)] = 833 dims
- [x] New config: `stage2_2014_gate.yaml` (gate + aux loss, no retrieval_dropout)
- [x] Updated NB2 with gate training cells, NB3 with gate eval cells
- [ ] Awaiting Kaggle training results

### 2026-06-19
- [x] Embedding experiments concluded — MAMS chosen (0.822, most balanced)
- [x] Implemented auxiliary CE loss on label_repr (`aux_label_repr_weight` in model/trainer/script)
- [x] New config: `stage2_2014_auxloss.yaml` (lambda_aux=0.1)
- [x] Migrated to new Kaggle account `duclm318`
- [x] Uploaded datasets: `p5-embed-v4` (MAMS embedding), `p5-nb1-stage1` (Cat-Aware)
- [x] NB2 updated: retrieval + no-retrieval + **aux loss** (replaced joint v2)
- [x] NB2 complete: aux loss **best val MacF1=0.9235** (+2.4pp over retrieval, +10.0pp over no-ret)
- [x] NB3 aux loss test: Sent Acc 0.8396 (+3.4pp over frozen), Joint F1 0.7190 (+2.9pp) — still loses to no-ret
- [x] Reduced retrieval top_k from 5 → 3: Sent Acc 0.8492 (+1.0pp), Joint F1 0.7273 (+0.8pp)
- [x] NB3 simplified: `--pred_strategy per_category` only (removed global/topk)
- [x] Implemented agreement filter (`--agreement_filter` in 05_evaluate_joint.py)
- [x] Agreement filter diagnostic: only 12.4% disagree, 87.6% unanimous → filter +0.6pp Joint F1, insufficient
- [x] Neighbor polarity analysis: pol_match pos=90.5%, neg=70.3%, **neu=56.0%**
- [x] Root cause confirmed: positive dominance — 59% of wrong neighbors are positive (bias from class imbalance)
- [x] 43.6% neutral samples have unanimous-wrong neighbors → unfilterable

### 2026-06-17
- [x] A1 (polarity-only, no cls_head) on Kaggle — **failed** (score collapse)
- [x] Added `num_polarities: 3` + `cls_polarity_weight: 1.0` to polonly configs
- [x] A1 v2 (polonly + cls_head) — test pol_match = 0.776 (no improvement)

### 2026-06-15
- [x] NB4 diagnostic — 6 experiments identifying root cause
- [x] Embedding polarity-blind for neg/neu; Stage 1 NOT the problem

### 2026-06-14
- [x] CLS head + split loss + 2-pass training implemented
- [x] NB0-NB3 full pipeline run on SemEval 2014
- [x] Retrieval loses to no-retrieval on test (-9.5pp Sent Acc)

### 2026-06-13
- [x] SemEval 2016 → 2014 migration complete (all code, configs, notebooks)

---

## Kaggle Notebooks (SemEval 2014)

| # | Notebook | Purpose | Key Input |
|---|----------|---------|-----------|
| NB0 | `p5_nb_embed_v4.ipynb` | Embedding training (A2/MAMS/A2+MAMS) | `semeval-absa-restaurant` |
| NB1 | `p5_nb1_stage1.ipynb` | Category Detection (Cat-Aware) | `semeval-absa-restaurant` |
| NB2 | `p5_nb2_stage2.ipynb` | Sentiment Classification (ret + no-ret + aux loss + gate) | `p5-embed-v4` |
| NB3 | `p5_nb3_eval.ipynb` | End-to-end evaluation | `p5-nb1-stage1`, `p5-nb2-stage2`, `p5-embed-v4` |
| NB4 | `p5_nb4_diagnostic.ipynb` | Diagnostic (7 experiments) | `p5-nb1-stage1`, `p5-nb2-stage2`, `p5-embed-v4` |

## Evaluation Metrics Note

Sent Macro F1|CC and Sent Macro R|CC added (commit `2def055`). NB3 now uses `per_category` threshold strategy only.

## Next Actions

### Adjusting how model uses retrieval (embedding ceiling at ~82% pol_match)

Embedding improvements exhausted (A2, MAMS, cross-polarity, joint training all tried). Core problem: pol_match@5 = 82% overall (pos 90%, neg 70%, neu 55%) — model blindly interpolates all neighbors including wrong-polarity ones.

**Approach 1 (Agreement Filter) — DONE, insufficient:**
- [x] Tested: only 12.4% of samples have disagreeing neighbors
- [x] Result: +0.6pp Joint F1 (0.7221→0.7283), still -4.1pp behind no-ret
- [x] Root cause: 87.6% unanimous neighbors, but **unanimous-wrong** is the real problem (43.6% for neutral)
- [x] Conclusion: disagreement is NOT the bottleneck; positive-biased embedding is

**Approach 2 — Retrieval Dropout: FAILED**
- [x] Implemented and tested p=0.3, p=0.5
- [x] Best val (p=0.3 MacF1=0.9487) but test Joint F1=0.7211 (worse than aux loss)
- [x] Dead-end: random dropout disabled at eval, missing rescaling, not quality-aware

**Approach 3 — Learnable Gate (implemented, awaiting Kaggle training):**
- Gate α = σ(W·[cls (768), label_repr (64), mean_neighbor_score (1)]) → gated_repr = α·label_repr
- Model learns per-instance when to trust retrieval vs ignore it
- Key advantage: gate is ON at eval (learned parameter, not dropout)
- Combined with aux loss (0.1) — aux loss improves label_repr, gate handles remaining errors
- Init: bias=0.847 → α≈0.70 at start (avoids gate collapse)
- Gate and retrieval_dropout are mutually exclusive

- [x] Implement Approach 3 (learnable gate) in `sentiment_model.py` — `RetrievalGate` class
- [x] New config: `stage2_2014_gate.yaml`
- [x] Updated NB2 (training cells) and NB3 (eval cells)
- [ ] Train on Kaggle, compare
- [ ] Target: Joint F1 > 0.7696 (beat no-retrieval baseline)
- [ ] Kaggle account: `duclm318` (migrated from `lcminhc`)

---

## Training Environment

- **GPU:** Kaggle T4.
- **Local:** Windows 11, no GPU training — code and tests only.
