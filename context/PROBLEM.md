# Problem Analysis — Retrieval-based ABSA Pipeline

**Date:** 2026-06-12

---

## All NB3 Test Results (Global threshold 0.80, Cat F1 = 0.6962)

| Run | Augmented | Ret Joint F1 | Ret Sent Acc | No-Ret Joint F1 | No-Ret Sent Acc | Gap (Ret - NoRet) |
|-----|-----------|-------------|-------------|----------------|----------------|-------------------|
| STATUS.md best (v1) | No | 0.6180 | 89.26% | **0.6304** | **91.05%** | -1.24pp |
| NB3 v4 (reference, v2) | No | 0.6139 | 88.67% | 0.6235 | 90.06% | -0.96pp |
| NB3 + 150 aug | 150 neutral | 0.6043 | 87.28% | 0.6249 | 90.26% | -2.06pp |
| NB3 + 650 aug | 650 neutral | 0.6029 | 87.08% | 0.6277 | 90.66% | -2.48pp |

**Conclusion:** Retrieval loses to No-Retrieval in every single configuration tested.

---

## 1. Data Issues

### 1.1 SemEval 2016 Duplicate & Conflict

SemEval 2016 SB1 annotates at target level. When converting to category-level ACSA, same sentence + same category can produce:
- **Exact duplicates:** 2 targets in same sentence, same category, same polarity -> 2 identical records
- **Polarity conflicts:** 2 targets in same sentence, same category, DIFFERENT polarity -> contradictory training signal

Impact: Model receives identical input with opposite labels -> gradient conflict -> limits accuracy ceiling. Not yet cleaned.

### 1.2 Class Imbalance

| Polarity | Train (aug) | Train % | Test | Test % |
|----------|------------|---------|------|--------|
| positive | 1,657 | 59.0% | 611 | 71.1% |
| negative | 749 | 26.7% | 204 | 23.7% |
| neutral | 401 | 14.3% | 44 | 5.1% |

Neutral severely underrepresented in test (5.1%). Model learns to never predict neutral because ignoring 5% of test costs less than risking wrong predictions on the other 95%.

### 1.3 Augmentation Side Effects

- No-Ret benefits from augmentation: 0.6235 -> 0.6249 (150) -> 0.6277 (650)
- Retrieval is HURT by augmentation: 0.6139 -> 0.6043 (150) -> 0.6029 (650)
- More augmented neutral in FAISS index -> FAISS returns augmented neighbors -> model overfits to augmented patterns on val but fails on real test data

---

## 2. Stage 1 — Category Detection (Cat F1 = 0.6962)

### Per-category breakdown (global threshold)

| Group | Categories | Test Support | F1 Range |
|-------|-----------|-------------|----------|
| Good (F1 >= 0.7) | FOOD#QUALITY (0.861), SERVICE#GENERAL (0.816), AMBIENCE#GENERAL (0.708), RESTAURANT#GENERAL (0.707) | 570 (77%) | 0.71 - 0.86 |
| Mid (0.3 - 0.7) | DRINKS#STYLE_OPTIONS (0.364), RESTAURANT#PRICES (0.353), FOOD#STYLE_OPTIONS (0.352), FOOD#PRICES (0.343), DRINKS#QUALITY (0.333) | 124 (17%) | 0.33 - 0.36 |
| Bad (F1 < 0.3) | LOCATION#GENERAL (0.250), RESTAURANT#MISCELLANEOUS (0.250), DRINKS#PRICES (0.000) | 49 (6%) | 0.00 - 0.25 |

**Root cause:** Rare categories lack training data. LOCATION#GENERAL has 126 train records (mostly from MAMS augmentation) but recall only 15.4% — augmented data distribution doesn't match SemEval test.

**Impact:** 8/12 categories have F1 < 0.7, covering 23% of test support. Each additional correct category detection directly improves Joint F1.

---

## 3. Stage 2 — Sentiment Classification

### 3.1 No-Retrieval (Best: Joint F1 = 0.6304)

- Sent Acc|CC ~ 90-91%, already very strong
- **Neutral F1 = 0.000** on val in 150 aug run (model never predicts neutral)
- With 650 aug + class_weights: neutral F1 improves to 0.240, but test Sent Acc slightly lower
- DeBERTa 768-dim encodes sentiment well enough without retrieval

### 3.2 Retrieval — Phase 2a (Consistently loses)

**Attempts and results:**

| Version | Changes | Val MacF1 | Test Joint F1 |
|---------|---------|-----------|---------------|
| Phase 2a v1 | Full W (65K params) | 0.7916 | 0.6180 |
| Phase 2a v2 (no aug) | Diagonal W (256), -inf padding, wired margin | -- | 0.6139 |
| Phase 2a v2 (150 aug) | + augmented index | 0.7245 | 0.6043 |
| Phase 2a v2 (650 aug) | + more augmented | 0.7907 | 0.6029 |

All variants lose to No-Retrieval. Higher val MacF1 correlates with WORSE test performance (overfit signal).

### 3.3 Why Retrieval Fails

1. **DeBERTa already captures sentiment:** 768-dim representation achieves 91% Sent Acc. Adding 64-dim polarity interpolation is noise, not signal.
2. **Frozen embedding is polarity-blind:** Training dynamics show margin_neg1 (polarity) ~ 0.001 across all epochs while margin_neg2 (category) reaches 0.703. Model optimizes category separation (easy) and ignores polarity (hard).
3. **Triplet design has conflicting signals:** neg1 (same cat, diff pol) provides polarity signal, but neg2 (diff cat, same pol) provides anti-polarity signal. Both get equal weight in InfoNCE loss. Model takes the easy path (neg2/category).
4. **Augmented data amplifies the problem:** More neutral records in FAISS index -> model overfits to augmented patterns on val -> fails on real test.

---

## 4. Embedding — Score Collapse Fixed, Polarity Still Blind

### Old Embedding (p3s2-embedding-flat)

- Mean cosine: 0.9961 (score collapse)
- Polarity match @3: 61.6% overall, neutral 8.6%
- Category match @3: 99.8%

### Embedding V4 (tau=0.07, 2-stage, augmented triplets)

- Score collapse fixed: inter cosine 0.94 -> 0.30
- But polarity match UNCHANGED: 58.9% -> 58.3%
- Training margin analysis (9 epochs):

| Epoch | m1 (polarity) | m2 (category) | Ratio |
|-------|--------------|--------------|-------|
| 1 | -0.003 | 0.006 | 2x |
| 4 (best) | 0.001 | 0.532 | 532x |
| 9 (final) | 0.001 | 0.703 | 703x |

Model reduces loss entirely through category separation (neg2). Polarity margin stays at ~0.001 for all 9 epochs.

### Root Cause: neg2 Provides Easy Optimization Path

Current triplet design:
- positive: same category + same polarity
- neg1: same category + different polarity (polarity signal)
- neg2: different category + same polarity (category signal)

InfoNCE loss can be reduced by increasing margin with EITHER neg1 or neg2. Category separation (neg2) is much easier because topic words are strong features. Model takes the easy path and never learns polarity.

### Evaluated Options

| Option | Description | Assessment |
|--------|------------|------------|
| A: Drop neg2 | Only use neg1, force polarity learning | Fixes root cause, but weaker contrastive signal (1 neg) |
| B: Cross-category polarity triplets | pos=any same-pol, neg=any diff-pol | Maximum polarity signal, large pools |
| C: Weighted loss | Keep both neg, reduce neg2 weight | Complex, uncertain effectiveness |
| D: Skip embedding fix | Use old embedding + augmented index | Zero effort, but embedding is polarity-blind |

**Decision:** Embedding improvement deferred. Even with perfect polarity matching, retrieval may still not beat No-Retrieval because DeBERTa already captures sentiment well.

---

## 5. RLI Paper Insights

Reference: "Making Better Use of Training Corpus: Retrieval-based ASTE via Label Interpolation" (ACL 2023 Findings)

Key differences from our approach:

| Aspect | RLI (paper) | Our approach |
|--------|------------|-------------|
| Retriever | **Joint trained** with task | Frozen embedding |
| Retrieve unit | Triplet (aspect+opinion pair) | Sentence |
| Relevance | Learned W matrix, backprop through | Cosine on fixed vectors |
| Pre-training | Contrastive on pseudo-labeled data | Separate contrastive training |
| Ablation | w/o joint training: -1-2% F1 | N/A (never joint trained) |

Paper explicitly states: "semantic similar neighbors with different polarities will be counterproductive" — exactly our problem. Their solution: joint training so retriever learns to fetch sentiment-consistent neighbors.

**Implication:** Joint training of retriever with Stage 2 model is the paper-backed path to make retrieval work. This is a larger architecture change deferred for later.

---

## 6. Priority Assessment

| Direction | Potential Impact | Effort | Risk |
|-----------|-----------------|--------|------|
| **Clean data (dedup + conflict)** | Medium — removes noise floor | Low | Low |
| **Improve Stage 1 rare categories** | High — each correct cat = +Joint F1 | Medium | Medium |
| **Stage 2 No-Ret neutral** | Low — only 44/859 test (5.1%) | Low | Low |
| **Joint training retriever (RLI-style)** | Unknown — DeBERTa may make retrieval redundant | High | High |
| **Fix embedding (drop neg2 / polarity triplets)** | Low — polarity match doesn't translate to better retrieval | Medium | Medium |
