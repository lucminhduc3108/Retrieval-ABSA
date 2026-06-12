# Project Status — Retrieval-based ABSA

**Last updated:** 2026-06-12 (Embedding v4 + data augmentation plan ready)

---

## Pipeline Overview (Final)

Two-stage pipeline on SemEval 2016 Restaurant SB1:

1. **Data Prep & Augmentation:** Parse XML → (category, polarity) pairs. Drop `conflict`. Augment neutral from MAMS-ACSA (5 safe 1-to-1 category mappings).
2. **Retrieval Engine:** DeBERTa ContrastiveEmbedder (InfoNCE) → 256-dim → FAISS IndexFlatIP. Only used by Phase 2a retrieval strategy.
3. **Stage 1 — Category Detection:** DeBERTa + Cat-Aware Attention → 12 sigmoid (BCE). Global threshold. **FINAL: Cat-Aware R5.**
4. **Stage 2 — Sentiment Classification:** Predict sentiment per detected category. Two strategies:
   - **No-Retrieval (baseline):** DeBERTa → MLP(768→256→3).
   - **Phase 2a (Label Interpolation):** FAISS top-k + Diagonal W + polarity interpolation → MLP(832→256→3) + ranking loss.
5. **Evaluation (end-to-end):** Category F1, Sentiment Acc|Correct Category, Joint F1 (cat+pol).

---

## Current Best Results (Test Set — NB3 v9, 2026-06-11)

Stage 1: Cat-Aware R5 | Stage 2: Phase 2a v1 vs No-Ret

| Strategy | Cat F1 | Phase 2a Joint F1 | Phase 2a Sent Acc\|CC | No-Ret Joint F1 | No-Ret Sent Acc\|CC |
|----------|--------|-------------------|----------------------|-----------------|---------------------|
| per_category | 0.6858 | 0.6096 | 0.8935 | 0.6213 | 0.9106 |
| **global (0.80)** | **0.6962** | 0.6180 | 0.8926 | **0.6304** | **0.9105** |
| topk (k=1) | 0.6797 | 0.6054 | 0.8960 | 0.6218 | 0.9204 |

**Best: No-Ret global → Joint F1 = 0.6304, Sent Acc|CC = 0.9105**

---

## Stage 1: Category Detection (FINAL)

| Run | Val Cat F1 | Test Cat F1 | Note |
|-----|-----------|------------|------|
| R4 ContextPooler | 0.7482 | 0.6844 | Baseline |
| **R5 Cat-Aware** | **0.7243** | **0.6962** | 12 learnable queries + MHA, 30ep |

---

## Stage 2: Sentiment Classification

| Run | Val Acc | Val Macro F1 | Test Joint F1 (global) | Note |
|-----|---------|-------------|----------------------|------|
| Run 2 (frozen cosine) | 0.8980 | 0.5973 | 0.6139 | tau=0.5, no augmentation |
| Phase 2a v1 (full W) | 0.8215 | 0.7916 | 0.6180 | 65K params, overfit |
| **No-ret** | **0.8943** | 0.6737 | **0.6304** | **Best test Joint F1** |
| Phase 2a v2 (diagonal W) | -- | -- | -- | NB2 v17 on Kaggle |

---

## Phase 2a v1: FAILED (2026-06-11)

**Bugs found:** padding bias (zero-vec got weight), rank_margin dead key.
**Why retrieval lost:** 65K params overfit, padding distortion, train-test neutral mismatch.

## Phase 2a v2: Diagonal W + Bug Fixes (2026-06-11)

| Change | v1 | v2 |
|--------|----|----|
| W mode | full (65K params) | **diagonal (256 params)** |
| Padding | zero-vec gets weight | **masked to -inf** |
| rank_margin | dead key (0.1) | **wired (0.5)** |
| tau | 0.1 | **0.3** |
| lambda_rank | 0.1 | **0.01** |

---

## Embedding v4 + Data Improvement (2026-06-12) — IN PROGRESS

### Root cause analysis

Retrieval thua no-ret vì embedding quality kém:
- **Score collapse:** mean cosine = 0.9961 → retrieval gần random
- **Polarity match:** 61.6% overall, neutral chỉ 8.6%
- **Contrastive triplets:** chỉ 100 neutral (4%) → embedding mù neutral
- **Hard negative mining chưa chạy:** `hard_contrastive_triplets.jsonl` chưa tồn tại

### Data augmentation (DONE — local)

MAMS-ACSA neutral augmentation với 5 safe 1-to-1 mappings:
- place → LOCATION#GENERAL, miscellaneous → RESTAURANT#MISCELLANEOUS
- ambience → AMBIENCE#GENERAL, service/staff → SERVICE#GENERAL
- 3 nhãn mơ hồ (food, menu, price) **không dùng** — 1-to-nhiều mapping với SemEval

| File | Records | Neutral | Neutral % |
|------|---------|---------|-----------|
| `classification_aug.jsonl` | 2,807 train | 401 | 14.3% |
| `contrastive_triplets_aug.jsonl` | 2,805 | 400 anchors | 14.3% |
| `sentiment_records_aug_300.jsonl` | 2,807 train | 401 | 14.3% |

Neutral triplets: 100 → **400** (4x tăng).

### Embedding v4 configs (DONE — local)

| Config | Triplets | tau | Note |
|--------|----------|-----|------|
| `embedding_v4_s1.yaml` | contrastive_triplets_aug.jsonl | **0.07** | Stage 1: augmented random triplets |
| `embedding_v4_s2.yaml` | hard_contrastive_triplets_aug.jsonl | **0.07** | Stage 2: hard mined triplets |

Key change: tau 0.12 → 0.07 (sharpen distribution, force larger margin).

### Kaggle workflow (PENDING)

```
1. Train embedding Stage 1 (augmented triplets, tau=0.07)
2. Hard negative mining from Stage 1 model
3. Train embedding Stage 2 (hard triplets)
4. Build FAISS index from Stage 2 model
5. Train Phase 2a v2 with new embedding + sentiment_records_aug_300
6. Eval in NB3
```

---

## Next Actions

- [x] Data augmentation: MAMS neutral → contrastive triplets + sentiment records
- [x] Embedding v4 configs (tau=0.07, separate ckpt dirs)
- [ ] Upload augmented data + configs to Kaggle dataset
- [ ] Kaggle: Embedding v4 two-stage training
- [ ] Kaggle: Build FAISS index + train Phase 2a v2
- [ ] Kaggle: NB3 eval — target Joint F1 > 0.6304

---

## Training Environment

- **GPU:** Kaggle T4x2 (1 GPU used). Stage 1 batch=16, Stage 2 batch=32 grad_accum=2.
- **Local:** Windows 11, no GPU training — code and tests only.
- **Kaggle datasets:** `semeval-absa-restaurant`, `p5-nb1-stage1`, `p5-nb2-stage2`, `p3s2-embedding-flat`.
