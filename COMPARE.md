# Controlled Experiment: SemEval 2014 vs MAMS

**Date:** 2026-06-21

## Motivation

Embedding pol_match ceiling at ~82% on SemEval 2014. All embedding improvements exhausted (A2, MAMS augmentation, deep projection, tau tuning). All Stage 2 fixes exhausted (aux loss, gate, dropout, perturbation, two-head).

Key question: **is the ceiling dataset-driven or method-limited?**

Prior analysis found MAMS (used as augmentation) has fundamental conflicts with SemEval:
- 95.4% of MAMS sentences have mixed polarity across aspects (by design)
- MAMS polarity distribution is inverted vs SemEval (neu=44% vs pos=62%)
- Per-category polarity mismatch (MAMS food=63% neutral, SemEval food=74% positive)

This experiment isolates the variable: train and test each dataset independently with identical architecture. No cross-dataset augmentation.

---

## Dataset Comparison

### Scale

| | SemEval 2014 | MAMS |
|---|---|---|
| Train sentences | 3,044 | 3,549 (train+val merged) |
| Train opinions | 3,516 | 6,924 |
| Test sentences | 800 | 400 |
| Test opinions | 973 | 760 |
| Avg opinions/sentence (train) | 1.15 | 1.95 |
| Avg words/sentence | 14 | 24 |

### Polarity Distribution

| Polarity | SemEval Train | SemEval Test | MAMS Train | MAMS Test |
|----------|-------------|-------------|-----------|----------|
| positive | 2,176 (61.9%) | 657 (67.5%) | 1,813 (26.2%) | 195 (25.7%) |
| negative | 839 (23.9%) | 222 (22.8%) | 2,104 (30.4%) | 233 (30.7%) |
| neutral | 501 (14.2%) | 94 (9.7%) | 3,007 (43.4%) | 332 (43.7%) |

SemEval is positive-dominant. MAMS is neutral-dominant with balanced pos/neg.

### Category Distribution

| Category | SemEval Train | SemEval Test | MAMS Train+Val |
|----------|-------------|-------------|---------------|
| food | 1,166 (33.2%) | 402 (41.3%) | 2,539 (36.7%) |
| service | 562 (16.0%) | 167 (17.2%) | 1,983 (28.6%) |
| anecdotes/misc | 1,101 (31.3%) | 219 (22.5%) | 1,083 (15.6%) |
| ambience | 385 (10.9%) | 105 (10.8%) | 952 (13.7%) |
| price | 302 (8.6%) | 80 (8.2%) | 367 (5.3%) |

MAMS uses 8 native categories mapped to 5 SemEval categories: food+menu→food, service+staff→service, ambience+place→ambience, miscellaneous→anecdotes/miscellaneous, price→price.

### Per-Category Polarity (Key Difference)

| Category | SemEval Train top polarity | MAMS Train top polarity |
|----------|--------------------------|------------------------|
| food | **positive (74%)** | **neutral (63%)** |
| service | **positive (58%)** | **negative (65%)** |
| ambience | positive (68%) | neutral (48%) |
| price | positive (59%) | neutral (42%) |
| anecdotes/misc | positive (50%) | neutral (56%) |

Almost every category has a different dominant polarity between datasets.

### Structural Difference

| Property | SemEval 2014 | MAMS |
|----------|-------------|------|
| Mixed-polarity sentences | 4.5% | **95.4%** |
| Multi-aspect sentences | ~15% | **95.9%** |
| Design goal | Natural review data | Forced multi-aspect multi-sentiment |

MAMS was designed so every sentence has at least 2 aspects with different sentiments. This makes category detection harder (more multi-label) but polarity distribution more balanced.

---

## Architecture (Identical for Both Datasets)

### Embedding
- DeBERTa-v3-base → Linear(768→256) → GELU → LayerNorm → L2-norm
- InfoNCE loss, polonly triplets (no neg2), tau=0.07
- CLS polarity head: `num_polarities: 3, cls_polarity_weight: 1.0`
- Two-stage: random triplets → hard negative mining → fine-tune

### Stage 1 — Category Detection
- DeBERTa-v3-base + **Cat-Aware Attention** (category-specific queries + MultiheadAttention)
- BCE loss (not ASL), pos_weight_cap=3.0
- Per-category sigmoid, global threshold tuned on val

### Stage 2 — Sentiment Classification

**Variant A — No-Retrieval (primary):**
- DeBERTa → CLS (768) → MLP(768→256→3)
- CrossEntropy with class weights

**Variant B — Retrieval + Aux Loss (secondary):**
- DeBERTa → CLS (768) + FAISS top-k=3 → LearnableRetriever (Diagonal W) → label_repr (64)
- Concat [CLS; label_repr] = 832 → MLP(832→256→3)
- Aux polarity head on label_repr (weight=0.1)
- Ranking loss (lambda=0.01, margin=0.5)

---

## Data Flow

### SemEval Track (existing)
```
SemEval-2014/Restaurants_Train.xml → data/processed/ → checkpoints/*_2014/ → indexes/
SemEval-2014/Restaurants_Test_Gold.xml ─────────────────────────────────────→ eval on test
```

### MAMS Track (new)
```
data/mams/.../train.xml + val.xml → data/processed_mams/ → checkpoints/*_mams/ → indexes/mams/
data/mams/.../test.xml ────────────────────────────────────────────────────────→ eval on test
```

MAMS val (400 sentences) merged into train. MAMS test (400 sentences) is held-out.

---

## Expected Results Table

| Metric | SemEval No-Ret | SemEval Ret+Aux | MAMS No-Ret | MAMS Ret+Aux |
|--------|---------------|----------------|-------------|-------------|
| **pol_match@5** | 0.822 | 0.822 | ? | ? |
| **Cat F1** | 0.8564 | 0.8564 | ? | ? |
| **Sent Acc\|CC** | 0.8987 | 0.8492 | ? | ? |
| **Sent MacF1\|CC** | 0.8022 | 0.7456 | ? | ? |
| **Joint F1** | 0.7696 | 0.7273 | ? | ? |
| pos F1 | ? | ? | ? | ? |
| neg F1 | ? | ? | ? | ? |
| neu F1 | ? | ? | ? | ? |
| Retrieval gap | — | **-4.2pp** | — | **?** |

---

## What We'll Learn

### Scenario 1: MAMS pol_match >> 82%
Balanced polarity distribution improves embedding quality. The ceiling is **dataset-driven** — SemEval's positive dominance (62%) biases the embedding. Implication: retrieval needs polarity-balanced training data.

### Scenario 2: MAMS pol_match ~ 82%
Ceiling is a **method limitation** — InfoNCE + DeBERTa projection can't separate polarity regardless of distribution. Implication: need fundamentally different embedding approach (SupCon, cross-encoder, etc.)

### Scenario 3: MAMS retrieval gap < SemEval's -4.2pp
Better pol_match reduces retrieval harm. Confirms positive dominance as the key blocker for retrieval on SemEval. May indicate retrieval can work on naturally balanced datasets.

### Scenario 4: MAMS retrieval gap >= SemEval's
Retrieval is fundamentally harmful for ABSA even with balanced polarity. DeBERTa CLS (768-dim) is sufficient; 64-dim label_repr adds noise regardless of neighbor quality. Implication: abandon retrieval approach.

### Per-polarity comparison
- MAMS should have much better **neutral F1** (44% test vs 10% SemEval)
- MAMS may have lower **positive F1** (26% vs 68% SemEval)
- The polarity breakdown reveals whether "retrieval hurts neutral" (seen on SemEval) is dataset-specific

---

## Implementation

### Code changes
1. `src/data/mams_mapping.py` — add `MAMS_TEST_XML` path
2. `scripts/01_prepare_data_mams.py` — new script for MAMS standalone data prep

### New configs (6)
- `configs/embedding_mams_s1.yaml`, `configs/embedding_mams_s2.yaml`
- `configs/stage1_mams.yaml`
- `configs/stage2_mams_noret.yaml`, `configs/stage2_mams_auxloss.yaml`
- `configs/retrieval_mams.yaml`

### Execution (10 steps)
1. Data prep → `data/processed_mams/`
2. Embedding stage 1 (random triplets)
3. Hard negative mining
4. Embedding stage 2 (hard triplets)
5. Build FAISS index → `indexes/mams/`
6. Stage 1 Cat-Aware training
7. Stage 2 No-Retrieval training
8. Stage 2 Retrieval+AuxLoss training
9. Joint eval (no-retrieval)
10. Joint eval (retrieval)
