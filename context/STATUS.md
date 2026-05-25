# Project Status — Retrieval-ABSA

**Last updated:** 2026-05-25 (Sprint 2 implementation)

---

## Pipeline Overview

| Phase | Script | Status | Output |
|-------|--------|--------|--------|
| 1. Data Prep | `scripts/01_prepare_data.py` | Done | 5,865 records (bio + cls + triplets) |
| 2. Embedding | `scripts/02_train_embedding.py` | Done | `checkpoints/embedding/best.pt` (736MB) |
| 3. FAISS Index | `scripts/03_build_index.py` | Done | `indexes/train.faiss` (4,161 vectors) |
| 4. ABSA Train | `scripts/04_train_absa.py` | Done | `checkpoints/absa/best.pt` (701MB) |
| 5. Evaluate | `scripts/05_evaluate.py` | Done | See results below |

---

## Phase 2 Results (Test Set, 859 opinions — SemEval 2016 SB1 only)

| Metric | Retrieval | No-retrieval | Delta |
|--------|-----------|--------------|-------|
| BIO Token F1 | 0.5418 | **0.6198** | +7.8pp |
| Span F1 | 0.6074 | **0.6489** | +4.2pp |
| Sentiment Acc | 0.8976 | **0.9243** | +2.7pp |
| Sentiment MacF1 | 0.7589 | **0.8234** | +6.5pp |
| **Joint F1** | 0.5460 | **0.6104** | **+6.4pp** |

**Key findings:**
- **Retrieval is hurting** — no-retrieval wins every metric by 2.7–7.8pp
- **Span detection remains bottleneck** — Span F1 vs Sent Acc gap = 27.5pp
- **Sqrt class weights working** — Sent MacF1 0.8234 (up from MVP 0.7619)
- **Implicit opinions**: BIO/Span/Joint = 0 (expected, no target span)

**Full report:** `report_4_5.md`

---

## Phase 3 CRF Experiments (Test Set, 859 opinions — No Retrieval)

| Metric | P2 no-ret | S1 CRF unnorm | S1A CRF norm |
|--------|-----------|---------------|-------------|
| BIO Token F1 | **0.6198** | 0.6373 | 0.5835 |
| Span F1 | 0.6489 | **0.6667** | 0.6237 |
| Sentiment Acc | **0.9243** | 0.8615 | 0.9034 |
| Sentiment MacF1 | **0.8234** | 0.5664 | 0.7612 |
| **Joint F1** | **0.6104** | 0.5648 | 0.5694 |

**Verdict: CRF dropped.** Tried unnormalized, normalized, both worse than plain CE. Phase 2 no-retrieval CE (Joint F1=0.6104) remains best baseline.

**Reports:** `report_5_7_gd3_s1.md` (S1), `gd3-session1a-crf-normalized.ipynb` (S1A)

---

## Phase 3 Session 2 Results — Improved Retrieval (Test Set, 859 opinions)

| Metric | P2 no-ret | P2 retrieval | **S2 retrieval** | vs P2 ret | vs P2 no-ret |
|--------|-----------|--------------|-----------------|-----------|-------------|
| BIO Token F1 | 0.6198 | 0.5418 | **0.4946** | -4.7pp | -12.5pp |
| Span F1 | 0.6489 | 0.6074 | **0.6212** | +1.4pp | -2.8pp |
| Sentiment Acc | 0.9243 | 0.8976 | **0.9034** | +0.6pp | -2.1pp |
| Sentiment MacF1 | 0.8234 | 0.7589 | **0.7951** | +3.6pp | -2.8pp |
| **Joint F1** | **0.6104** | 0.5460 | **0.5725** | +2.7pp | -3.8pp |

Config: `embedding_v2.yaml` (tau=0.12, accum=4) + `retrieval_v2.yaml` (top_k=2, threshold=0.3)

---

## Phase 3 Session 3 Results — Hard Negatives + Split BIO (Test Set, 859 opinions)

| Metric | P2 no-ret | S2 ret | **S3 ret** | vs P2 no-ret | vs S2 ret |
|--------|-----------|--------|-----------|-------------|-----------|
| BIO Token F1 | 0.6198 | 0.4946 | **0.5505** | -6.9pp | **+5.6pp** |
| Span F1 | 0.6489 | 0.6212 | **0.6823** | **+3.3pp** | **+6.1pp** |
| Sentiment Acc | 0.9243 | 0.9034 | **0.9197** | -0.5pp | **+1.6pp** |
| Sentiment MacF1 | 0.8234 | 0.7951 | **0.7991** | -2.4pp | +0.4pp |
| **Joint F1** | 0.6104 | 0.5725 | **0.6237** | **+1.3pp** | **+5.1pp** |

Config: `embedding_v3.yaml` (hard negatives, tau=0.12, batch=128) + `retrieval_v2.yaml` (top_k=2, threshold=0.3) + `absa.yaml` (split_bio=true, bio_max_length=128)

**Key findings:**
- **Retrieval lần đầu beat no-retrieval baseline** — Joint F1 0.6237 > 0.6104 (+1.3pp)
- **Span F1 = 0.6823 — best ever** (+3.3pp vs P2 no-ret). Split BIO giải phóng BIO head khỏi attention dilution
- Hard negatives + batch=128 cải thiện embedding quality → neighbors relevant hơn
- BIO Token F1 recover +5.6pp vs S2, nhưng vẫn -6.9pp vs no-ret do shared backbone bị kéo 2 hướng
- Sentiment MacF1 gap -2.4pp do neutral class imbalance (4%) + recall@3 vẫn thấp

**Training log (NB3 ABSA):** Best epoch 6, val joint_f1=0.6010
**Kaggle notebook:** [lcminhc/p3s2-nb4-eval](https://www.kaggle.com/code/lcminhc/p3s2-nb4-eval)
**Full report:** `report_gd3_s3.md`

---

## Phase Progress

**Phase 0 (code logic fixes) — DONE ✅**
**Phase 1 (data quality fixes) — DONE ✅**
**Phase 2 (retrain on clean data) — DONE ✅**
**Phase 3 (retrieval improvements) — DONE ✅**
- [x] 3A: CRF — DROPPED (S1 + S1A)
- [x] 3B-1: Improved retrieval hyperparams — DONE (S2, Joint F1=0.5725)
- [x] 3B-2: Hard negative mining — DONE (S3, embedding_v3, batch=128)
- [x] 3C: Split BIO head — DONE (S3, two-pass forward, Joint F1=0.6237)

**Phase 4 (measure & minor fix) — IN PROGRESS 🔄**
- [x] Sprint 1: Retrieval quality analysis — DONE (polarity @3=61.6%, category @3=99.8%, score collapsed)
- [x] Sprint 2: ABSA hardening — code done, notebook ready to push
  - E1: S3 + cls_class_weights [1.00, 1.70, 4.33] (isolate weight effect)
  - E2: E1 + retrieval_dropout=0.15 + lambda_cls=0.75 (full hardening)
- [ ] Sprint 3: Label interpolation (deferred — wait for Sprint 2 results)
- [ ] Sprint 4: Fix retrieval pipeline (deferred — wait for Sprint 2 results)

---

## Phase 3: Code Changes

**3A: CRF Layer** — ❌ DROPPED

**3B-1: Retrieval Hyperparams** — ✅ DONE
- `configs/embedding_v2.yaml` (tau=0.12, accum=4), `configs/retrieval_v2.yaml` (top_k=2, threshold=0.3)

**3B-2: Hard Negative Mining** — ✅ DONE (S3)
- [x] `build_hard_negative_triplets()` in contrastive_builder
- [x] `scripts/build_hard_triplets.py` created
- [x] `configs/embedding_v3.yaml` (batch=128, tau=0.12, gradient_checkpointing)
- [x] Chạy Kaggle S3 — embedding v3 trained

**3C: Tách BIO head (Split BIO)** — ✅ DONE (S3)
- [x] `src/absa/model.py` — two forward passes (bio_input_ids → BIO head, full input_ids → sentiment head)
- [x] `src/absa/dataset.py` — build bio_input_ids/bio_attention_mask khi split_bio=True
- [x] `src/absa/trainer.py` — transparent pass-through
- [x] `configs/absa.yaml` — split_bio=true, bio_max_length=128

**Tests:** 70/70 pass

---

## Kaggle Sessions

| Session | Config | Status | Result |
|---------|--------|--------|--------|
| ~~1: CRF no-retrieval~~ | CRF unnorm | ✅ Done | CRF dropped |
| ~~1A: CRF normalized~~ | CRF norm | ✅ Done | CRF dropped |
| ~~2: Improved retrieval~~ | embedding_v2 + retrieval_v2 | ✅ Done | Joint F1=0.5725 |
| ~~3: Hard neg + Split BIO~~ | embedding_v3 + retrieval_v2 + split_bio | ✅ Done | **Joint F1=0.6237** |

Sessions use plain CE (no CRF). Full pipeline split into nb1–nb4 to avoid T4 OOM.

---

## Kaggle Notebook Status

| Notebook | Kaggle URL | Status |
|----------|-----------|--------|
| P3-S1: CRF no-retrieval | [lcminhc/gd3-session1-crf-no-retrieval](https://www.kaggle.com/code/lcminhc/gd3-session1-crf-no-retrieval) | Done — CRF unnorm, sentiment collapsed |
| P3-S1A: CRF normalized | local `gd3-session1a-crf-normalized.ipynb` | Done — CRF norm, span regressed |
| P3-S2-NB1: Embedding | [lcminhc/p3s2-embedding dataset](https://www.kaggle.com/datasets/lcminhc/p3s2-embedding) | Done — tau=0.12, recall@3=0.136 |
| P3-S2-NB2: Index | [lcminhc/p3s2-index dataset](https://www.kaggle.com/datasets/lcminhc/p3s2-index) | Done — 4,161 vectors |
| P3-S2-NB3: ABSA Train | [lcminhc/p3s2-nb3-absa-train](https://www.kaggle.com/code/lcminhc/p3s2-nb3-absa-train) | Done — best epoch 8, val joint_f1=0.646 |
| P3-S2-NB4: Evaluate | [lcminhc/p3s2-nb4-eval](https://www.kaggle.com/code/lcminhc/p3s2-nb4-eval) | Done — S2: Joint F1=0.5725, **S3: Joint F1=0.6237** |
| **P4-Sprint01: Retrieval Quality** | [lcminhc/sprint01-retrieval-quality](https://www.kaggle.com/code/lcminhc/sprint01-retrieval-quality) | Done — results in `retrieval_quality.md` |
| **P4-Sprint02: ABSA Hardening** | `kaggle_upload/push_sprint02/` | **Ready to push** |

---

## Training Environment

- **GPU:** Kaggle T4 (free tier)
- **Local:** Windows 11, no GPU training
- **Checkpoints stored:** Kaggle datasets (`lcminhc/p3s2-embedding`, `lcminhc/p3s2-index`, `lcminhc/p3s2-absa-ckpt`, `lcminhc/p3s2-embedding-flat`)

---

## Vấn đề tồn đọng (sau S3)

1. **BIO Token F1 gap**: 0.5505 vs 0.6198 (P2 no-ret) = -6.9pp. Shared backbone bị kéo 2 hướng (BIO short input vs sentiment long input). Span F1 >> Token F1 gap = 13.2pp cho thấy boundary confusion (B/I).
2. **Sentiment MacF1 gap**: 0.7991 vs 0.8234 = -2.4pp. Neutral class imbalance (4%) + recall@3 vẫn thấp.
3. **Overfitting**: Val loss diverges từ epoch 5, 10 epochs chạy hết mà early stop không trigger.

**Full report:** `report_gd3_s3.md`

---

## Phase 4: Improve Part 2 — Measure First, Minor Fix

**Plan:** `IMPROVE_PART2.md`

**Strategy:** Brainstorm + code audit cho thay embedding da sentiment-aware (contrastive pairs dung category+polarity), index da per-opinion, encoder da encode (sentence, aspect_category). Thay vi rewrite pipeline theo RLI paper, chon "measure first, minor fix".

**Sprints:**

| Sprint | Goal | Status |
|--------|------|--------|
| Sprint 1 | Do retrieval quality (polarity/category match rate) | **IN PROGRESS** |
| Sprint 2 | Focal loss, majority-vote, retrieval dropout | Pending (depends on Sprint 1 results) |
| Sprint 3 | Label interpolation (conditional) | Pending |
| Sprint 4 | Fix retrieval pipeline (conditional) | Pending |

**Decision criteria (Sprint 1 -> next sprint):**

| Polarity match @3 (train) | Action |
|---|---|
| > 60% | Retriever OK -> Sprint 2 (fix ABSA model) |
| 40-60% | Trung binh -> Sprint 2 + Sprint 3 |
| < 40% | Retriever te -> Sprint 4 (fix embedding/retrieval) |

### Sprint 1: Retrieval Quality Analysis

- **Script:** `scripts/analyze_retrieval_quality.py` (reusable CLI)
- **Notebook:** `sprint01/s01_retrieval_quality.ipynb` (self-contained, no repo clone)
- **Dataset:** `lcminhc/p3s2-embedding-flat` (gop embedding + index + data vao 1 dataset flat)
- **Kaggle:** [lcminhc/sprint01-retrieval-quality](https://www.kaggle.com/code/lcminhc/sprint01-retrieval-quality) — pushed, pending run
- **Metrics:** polarity match @1/2/3, category match, per-class breakdown, neutral analysis, score distribution
- **Config:** top_k=3, threshold=0.0 (analysis mode), self-exclusion for train queries
- **Design doc:** `SPRINT01.md`

### Kaggle Dataset: p3s2-embedding-flat

Gop `p3s2-embedding` + `p3s2-index` vao 1 dataset flat, them hard negatives + v3 log:
- `embedding_best.pt` (736MB), `classification.jsonl`, `bio_tagging.jsonl`
- `train.faiss`, `train_metadata.jsonl`, `train_vectors.npy`
- `contrastive_triplets.jsonl`, `hard_contrastive_triplets.jsonl`
- `embedding_v2.jsonl`, `embedding_v3.jsonl`
- URL: https://www.kaggle.com/datasets/lcminhc/p3s2-embedding-flat

---

## Next Actions

- [ ] Run Sprint 1 notebook tren Kaggle, doc ket qua retrieval quality
- [ ] Dua vao ket qua, quyet dinh Sprint 2 (fix ABSA) hay Sprint 4 (fix retrieval)
