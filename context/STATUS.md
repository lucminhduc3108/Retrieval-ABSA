# Project Status — Retrieval-ABSA

**Last updated:** 2026-05-02

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

## MVP Results (Test Set, 1,704 samples)

| Metric | Value |
|--------|-------|
| BIO Token F1 | 0.6988 |
| Span F1 | 0.7088 |
| Sentiment Acc | 0.9079 |
| Sentiment Macro F1 | 0.7619 |
| **Joint F1** | **0.6379** |

**Bottleneck:** Span detection (not sentiment)
**Key insight:** When span is correct, polarity accuracy ~90%

---

## Known Issues

1. **Data leakage:** 49.8% test sentences overlap with train (SemEval 2015+2016 share data)
2. **Class imbalance:** positive 67%, negative 29%, neutral 4.1%
3. **Low embedding recall:** Recall@3 = 9.88% (but 14x random baseline)
4. **Retrieval impact unknown:** No ablation study yet

---

## Current Phase: GD 1 — Ablation Experiments

**Plan:** IMPROVE.md (Huong D: Measure Before Fix)

**GD 0 (code changes) — DONE:**
- [x] Early stopping: span_f1 -> joint_f1
- [x] Split evaluation: implicit vs explicit
- [x] Class weights configurable
- [x] --no_retrieval flag
- [x] Duplicate analysis script
- [x] Gradient accumulation
- [x] Stratified validation split
- [x] Experiment configs (exp_b, exp_c)

**GD 1 (ablation, next) — TODO:**
- [ ] Exp A: No-retrieval baseline
- [ ] Exp B: Retrieval + inverse-freq class weights
- [ ] Exp C: Retrieval + sqrt class weights

**Pending decisions (after GD 1):**
- Q1: Does retrieval help? (compare Exp A vs B/C)
- Q2: Which class weights? (compare B vs C)
- Q3: Improve embedding? (only if retrieval helps)
- Q4: Data augmentation? (only if neutral still bad)

---

## Training Environment

- **GPU:** Kaggle T4 (free tier)
- **Local:** Windows 11, no GPU training
- **Checkpoints stored:** Kaggle datasets (`lcminhc/retrieval-absa-embedding-ckpt`, `lcminhc/retrieval-absa-ckpt`)

---

## Uncommitted Changes (as of 2026-05-02)

- `configs/absa.yaml` — class weights, grad_accum additions
- `src/absa/model.py` — class weights parameter
- `src/absa/trainer.py` — joint_f1 early stop, grad accumulation
- `scripts/04_train_absa.py` — --no_retrieval, stratified split, grad_accum
- `scripts/05_evaluate.py` — --no_retrieval, implicit/explicit split
- New: `configs/absa_exp_b.yaml`, `configs/absa_exp_c.yaml`
- New: `scripts/analyze_duplicates.py`

---

## Next Actions

1. Commit GD 0 changes
2. Prepare Kaggle notebook for 3 ablation experiments
3. Run GD 1 on Kaggle (~2-3 hours T4)
4. Analyze results, answer Q1-Q4
5. Execute GD 3 based on answers
