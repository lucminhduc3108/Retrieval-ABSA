# Project Status — Retrieval-ABSA

**Last updated:** 2026-06-05 (NB3 run 2 done — Stage 1 model fail, Cat F1=0.23 on test, needs retrain)

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

## Phase 4 E3 Results — Quick Fixes A1+A2+A3 (Test Set, 859 opinions)

| Metric | P2 no-ret | S3 ret | **E3 ret** | E3 vs S3 | E3 vs P2 |
|--------|-----------|--------|-----------|----------|----------|
| BIO Token F1 | 0.6198 | 0.5505 | 0.5464 | -0.4pp | -7.3pp |
| Span F1 | 0.6489 | 0.6823 | **0.6813** | -0.1pp | **+3.2pp** |
| Sentiment Acc | 0.9243 | 0.9197 | **0.9220** | **+0.2pp** | -0.2pp |
| Sentiment MacF1 | 0.8234 | 0.7991 | **0.8157** | **+1.7pp** | -0.8pp |
| **Joint F1** | 0.6104 | 0.6237 | **0.6374** | **+1.4pp** | **+2.7pp** |

Config: S3 config + val_ratio=0.2 + patience=5 + differential LR (encoder_lr=2e-6, head_lr=2e-4)

**Key findings:**
- **Joint F1 = 0.6374 — best ever** (+1.4pp vs S3, +2.7pp vs P2 no-ret)
- **A3 (differential LR) là hero:** Sentiment MacF1 +1.7pp, gap vs P2 no-ret thu hẹp từ 2.4pp → 0.8pp
- **A1 (val_ratio=0.2):** Val reliable hơn (val=0.602, test=0.637 — model generalize tốt)
- **A2 (patience=5):** Không trigger, max_epochs=10 vẫn là limiter
- **BIO Token F1 thấp (-7.3pp vs no-ret) không ảnh hưởng Joint F1** — boundary confusion (B/I), nhưng Span F1 vẫn best ever (0.6813). BIO Token F1 là diagnostic metric, không feed vào Joint F1
- **Span detection ổn định:** 0.6813 ≈ S3's 0.6823, split BIO vẫn hoạt động tốt

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
- **Sentiment MacF1 gap -2.4pp — phát hiện config mismatch:** S3 dùng `absa.yaml` (cls_class_weights: null), P2 no-ret dùng `absa_exp_c.yaml` (weights: [1.00, 1.70, 4.33]). Gap có thể do thiếu weights, không phải retrieval noise

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
- [x] Sprint 1: Retrieval quality analysis — DONE (results in `retrieval_quality.md`)
- [x] Sprint 2: ABSA hardening — **DONE — FAIL**
  - E1: S3 + cls_class_weights [1.00, 1.70, 4.33] — val joint=0.6349 nhưng **test joint=0.5684** (thua S3 -5.5pp, overfit)
  - E2: E1 + retrieval_dropout=0.15 + lambda_cls=0.75 — val joint=0.5714 (tệ hơn E1)
  - **Root cause:** patience=999 + val set quá nhỏ (~250 samples) + class weights khuếch đại minority overfitting
  - **Kết luận:** ~~S3 vẫn là best model~~ → E3 là best model (Joint F1=0.6374)
- [x] Improvement roadmap cập nhật trong `IMPROVE_PART2.md` (6 nhóm phương án mới)
- [x] E3: Quick fixes (A1+A2+A3) — **DONE — SUCCESS**
  - A1: val_ratio 0.1→0.2 (~501 val samples vs ~251)
  - A2: removed --patience 999, config patience=5 active
  - A3: differential LR — encoder_lr=2e-6, head_lr=2e-4
  - Commit: `b32813a`
  - **Test Joint F1 = 0.6374 (+1.4pp vs S3, best ever)**

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

**Tests:** 70/70 pass (Phase 3), 85/85 pass (sau Sprint 2), 86/86 pass (sau E3), 90/90 pass (sau B1)

---

## Kaggle Sessions

| Session | Config | Status | Result |
|---------|--------|--------|--------|
| ~~1: CRF no-retrieval~~ | CRF unnorm | ✅ Done | CRF dropped |
| ~~1A: CRF normalized~~ | CRF norm | ✅ Done | CRF dropped |
| ~~2: Improved retrieval~~ | embedding_v2 + retrieval_v2 | ✅ Done | Joint F1=0.5725 |
| ~~3: Hard neg + Split BIO~~ | embedding_v3 + retrieval_v2 + split_bio | ✅ Done | **Joint F1=0.6237** |
| **E3: Quick fixes** | S3 + val_ratio=0.2 + patience=5 + differential LR | ✅ Done | **Joint F1=0.6374 (best)** |
| **B1: Focal Loss** | E3 + focal gamma=2.0 / 0.5 | ✅ Done | **FAIL** (val Joint F1=0.314/0.265) |

Sessions use plain CE (no CRF). Full pipeline split into nb1–nb4 to avoid T4 OOM.

---

## Phase 5 Kaggle Notebooks

| Notebook | Kaggle URL | Status | Output dataset |
|----------|-----------|--------|----------------|
| **P5-NB1: Stage 1 Train** | [lcminhc/p5-nb1-stage1-train](https://www.kaggle.com/code/lcminhc/p5-nb1-stage1-train) | ✅ Done | `p5-nb1-stage1` |
| **P5-NB2: Stage 2 Train** | [lcminhc/p5-nb2-stage2-train](https://www.kaggle.com/code/lcminhc/p5-nb2-stage2-train) | ✅ Run 2 done (ret), pending no-ret | `p5-nb2-stage2` |
| **P5-NB3: Joint Eval** | [lcminhc/p5-nb3-joint-eval](https://www.kaggle.com/code/lcminhc/p5-nb3-joint-eval) | ✅ Run 2 done — Stage 1 FAIL | metrics report |

**Stage 1 results (2026-06-04):** Cat F1=0.8991 (val only, P=1.0, R=0.82), best epoch 1, early stop epoch 6. Fix: encoder_lr 2e-5→2e-6 (commit `8b26e92`).
**Batch sizes:** Stage 1 batch=64, Stage 2 batch=32, grad_accum=2

### NB3 Run 1 — FAILED (2026-06-05, old code)

Bugs: (1) eval val split missing `stratify` → threshold tuned on training data, (2) Sent Acc|CC metric was global not per-sentence. Fixed in commits `9a5d112`, `f140e40`, `0afdcd1`, `e997645`.

### NB3 Run 2 — Stage 1 FAIL (2026-06-05)

**Fixes applied:** stratify match, 3 prediction strategies (per_category/global/topk), diagnostic logging.

**Strategy Comparison (test set, 859 opinions — identical for ret/no-ret since Stage 1 shared):**

| Strategy | Cat P | Cat R | Cat F1 | Avg Preds | Config |
|----------|-------|-------|--------|-----------|--------|
| per_category | 0.1375 | 0.9152 | 0.2391 | 8.42 | per-cat thresholds |
| global | 0.1478 | 0.4670 | 0.2246 | 4.00 | threshold=0.45 |
| topk | 0.1516 | 0.4791 | 0.2303 | 4.00 | k=4 |

**Joint F1 (retrieval vs no-retrieval):**

| Strategy | Ret Joint F1 | No-Ret Joint F1 | Ret Sent Acc\|CC | No-Ret Sent Acc\|CC |
|----------|-------------|-----------------|----------------|------------------|
| per_category | 0.2079 | **0.2121** | 0.8706 | **0.8882** |
| global | 0.2008 | **0.2014** | 0.8963 | **0.8991** |
| topk | 0.2078 | **0.2085** | 0.9045 | **0.9073** |

**Root cause — Stage 1 model cannot generalize:**
1. **pos_weight quá cao cho rare categories** (cap=5.0) — BCE loss phạt false negative nặng gấp 5x → model học bias cao cho mọi category → fire mọi câu
2. **encoder_lr=2e-6 quá thấp** — DeBERTa gần frozen, [CLS] features không adapt cho 12-way classification
3. **Chỉ 1 epoch hiệu quả** (best epoch 1, early stop epoch 6) — Linear(768,12) chỉ kịp học bias, chưa học discriminative weights
4. **Val Cat F1=0.8991 misleading** — val nhỏ (~342 samples), threshold tuned trên chính val → optimistic, gap val→test = 66pp

**Stage 2 OK:** Sent Acc|CC = 0.87-0.91 khi category đúng. No-retrieval slightly better.

**Kết luận:** Cần retrain Stage 1 trước khi so sánh retrieval vs no-retrieval có ý nghĩa.

### NB3 Run 3 — Stage 1 R4 + Stage 2 Run 2 (2026-06-08)

**Test Set Results (859 opinions):**

| Strategy | Ret Cat F1 | Ret Joint F1 | Ret Sent Acc\|CC | No-Ret Cat F1 | No-Ret Joint F1 | No-Ret Sent Acc\|CC |
|----------|------------|--------------|----------------|---------------|-----------------|-------------------|
| per_category | 0.6765 | 0.5977 | 0.8880 | 0.6765 | 0.6028 | 0.8956 |
| global (0.6) | **0.6844** | 0.6067 | 0.8913 | **0.6844** | **0.6120** | 0.8990 |
| topk (k=1) | 0.6602 | 0.5889 | 0.8975 | 0.6602 | 0.5979 | **0.9112** |

**Key findings:**
1. **Stage 1 R4 is a huge success:** Category F1 reached **0.6844** on the hidden test set (up from the disastrous 0.23 in Run 2). The model successfully learned to predict categories without overfitting.
2. **No-Retrieval strictly beats Retrieval** across all metrics and all strategies (Joint F1 +0.53pp, Sent Acc +0.77pp on global strategy).
3. **Gap narrowed significantly:** In Phase 2, No-Ret beat Ret by 6.4pp. Now the gap is only ~0.5pp. This indicates label interpolation with retrieval embeddings is much less noisy but still slightly lags behind direct sequence classification.
4. **Best configuration:** Global Strategy (threshold=0.60) + No-Retrieval yields Joint F1 = **0.6120**. This establishes the new robust baseline for Phase 5.


### NB2 Run 1 Results (Val metrics, 10 epochs each)

| Metric | Retrieval (ep10) | No-Ret (ep7 best) | Gap |
|--------|------------------|--------------------|-----|
| Sent Acc | 0.8589 | **0.8795** | -2.1pp |
| Macro F1 | 0.5727 | **0.5940** | -2.1pp |

**Problems identified:**
- Retrieval **chưa converge** ở epoch 10 (Acc/F1 vẫn đang tăng, patience 0/5)
- Macro F1 thấp (~0.57-0.59) dù Acc ~0.88 — **neutral class (4%) bị ignore**
- Early stopping theo Acc (misleading với 66% positive majority)
- tau=0.05 quá sharp → label interpolation gần one-hot, khuếch đại wrong-polarity noise

### NB2 Run 2 Fixes (commit `5397f32`)

| Fix | Change | Config |
|-----|--------|--------|
| P1: Early stop Macro F1 | `sentiment_trainer.py`: best model selected by `sentiment_macro_f1` | — |
| P2: Sqrt class weights | `sentiment_model.py` + `04b_train_stage2.py`: weights [1.00, 1.485, 3.910] | — |
| P3: Softer tau | `stage2.yaml`: tau 0.05 → **0.5** | retrieval only |
| P4: More epochs | `stage2.yaml`: epochs 10 → **20**, patience 5 → **7** | retrieval only |

### NB2 Run 2 Results — Retrieval (Val metrics, 20 epochs)

| Metric | Run 1 Ret (ep10) | **Run 2 Ret (ep20)** | Run 1 No-Ret (ep7) | Run 2 vs Run 1 Ret | Run 2 vs No-Ret |
|--------|-------------------|----------------------|---------------------|--------------------|-----------------|
| Sent Acc | 0.8589 | **0.8980** | 0.8795 | **+3.9pp** | **+1.9pp** |
| Macro F1 | 0.5727 | **0.5973** | 0.5940 | **+2.5pp** | **+0.3pp** |

**Key findings:**
- **Retrieval lần đầu beat no-retrieval trên val** — Macro F1 0.5973 > 0.5940 (+0.3pp), Acc 0.8980 > 0.8795 (+1.9pp)
- **4 fixes đều có tác dụng:** +2.5pp Macro F1, +3.9pp Acc vs run 1
- **Chưa converge hoàn toàn:** best ở epoch 20 (cuối), patience chưa trigger. Gains nhỏ dần (ep13→ep20: +0.5pp Macro F1)
- **Vẫn thiếu no-ret run 2** (với P1+P2 fixes) để so sánh fair

---

## Phase 1–4 Kaggle Notebooks (archived)

| Notebook | Kaggle URL | Status |
|----------|-----------|--------|
| P3-S1: CRF no-retrieval | [lcminhc/gd3-session1-crf-no-retrieval](https://www.kaggle.com/code/lcminhc/gd3-session1-crf-no-retrieval) | Done — CRF dropped |
| P3-S2-NB3: ABSA Train | [lcminhc/p3s2-nb3-absa-train](https://www.kaggle.com/code/lcminhc/p3s2-nb3-absa-train) | Done — Joint F1=0.6237 |
| P3-S2-NB4: Evaluate | [lcminhc/p3s2-nb4-eval](https://www.kaggle.com/code/lcminhc/p3s2-nb4-eval) | Done — S3 Joint F1=0.6237 |
| P4-Sprint01 | [lcminhc/sprint01-retrieval-quality](https://www.kaggle.com/code/lcminhc/sprint01-retrieval-quality) | Done |
| P4-Sprint02 | [lcminhc/sprint-2-absa-train-e1-e2](https://www.kaggle.com/code/lcminhc/sprint-2-absa-train-e1-e2) | Done — FAIL |

Old notebooks + push dirs moved to `archive/kaggle_old/`.

---

## Training Environment

- **GPU:** Kaggle T4x2 (2×16GB VRAM, pipeline uses 1 GPU)
- **Local:** Windows 11, no GPU training
- **Checkpoints stored:** Kaggle datasets (`lcminhc/p3s2-embedding-flat` for Phase 3 assets)

---

## Phase 4: Improve Part 2

**Plan:** `IMPROVE_PART2.md` (cập nhật 2026-05-27)

**Strategy cũ:** "measure first, minor fix" — Sprint 1 (đo) + Sprint 2 (fix nhẹ). Sprint 2 fail → chuyển sang roadmap mới.

**Strategy mới:** 6 nhóm phương án ưu tiên từ thấp risk đến cao:
- A: Quick fixes (val_ratio, early stopping, differential LR)
- B: Focal Loss, Majority-vote (chưa thử từ plan cũ)
- C: Fix retrieval (re-ranking, ensemble)
- D: Data augmentation (balance pos/neu/neg)
- E: Label interpolation
- F: Thay pipeline theo RLI paper (incremental)

| Sprint | Goal | Status |
|--------|------|--------|
| Sprint 1 | Retrieval quality analysis | **DONE** |
| Sprint 2 | ABSA hardening | **DONE — FAIL** |
| E3 | Quick fixes A1+A2+A3 | **DONE — SUCCESS** |
| Next | Focal Loss (B1) hoặc combine | **TODO** |

### Sprint 1: Retrieval Quality Analysis — DONE

**Results** (`retrieval_quality.md`):

| Metric | @1 | @3 | Random |
|--------|----|----|--------|
| Category match (train) | 99.9% | 99.8% | ~8.3% |
| Polarity match (train) | 64.9% | 61.6% | ~53% |

- **Category:** xuất sắc (99.8%), embedding cluster theo category rất tốt
- **Polarity:** bề mặt OK (61.6% > 60% threshold) nhưng bị inflate bởi positive dominance
  - positive: 73.4% (@3) vs 66% random = **+7pp lift** (gần random)
  - negative: 42.7% vs 30% random = **+13pp lift** (sai 57%)
  - neutral: 8.6% vs 4% random = **thảm họa** (91% neighbors sai polarity)
- **Score collapse:** mean=0.9961, match/mismatch cùng score → threshold filtering vô dụng
- **Kết luận:** Embedding chỉ biết category, gần mù polarity. Quyết định: Sprint 2 (harden ABSA model)

### Sprint 2: ABSA Hardening — DONE (FAIL)

**Giả thuyết ban đầu:** S3 train với `absa.yaml` (cls_class_weights: null) — config mismatch so với P2 no-ret (dùng weights [1.00, 1.70, 4.33]). **Giả thuyết bị bác bỏ.**

**Code changes** (commit `42db1a5`):
- `configs/absa_e1.yaml`: S3 + cls_class_weights [1.00, 1.70, 4.33]
- `configs/absa_e2.yaml`: E1 + retrieval_dropout=0.15 + lambda_cls=0.75
- `src/absa/dataset.py`: retrieval dropout (training-only, tự tắt ở eval)
- `scripts/04_train_absa.py`: `--lambda_cls`, `--cls_class_weights` CLI overrides
- `tests/test_absa_dataset.py`: +3 tests (85/85 pass)

**Kết quả test set (859 opinions):**

| Metric | S3 ret | E1 ret | E2 (val only) | E1 vs S3 |
|--------|--------|--------|---------------|----------|
| BIO Token F1 | 0.5505 | 0.5171 | — | -3.3pp |
| Span F1 | 0.6823 | 0.6226 | — | -6.0pp |
| Sentiment Acc | 0.9197 | 0.9034 | — | -1.6pp |
| Sentiment MacF1 | 0.7991 | 0.7719 | — | -2.7pp |
| Joint F1 | 0.6237 | 0.5684 | 0.5714 (val) | -5.5pp |

**Root cause analysis:**
- Val set quá nhỏ (~250 samples, neutral ~10 samples) → val metrics misleading
- Class weights [1.00, 1.70, 4.33] khuếch đại overfitting trên minority val samples
- E1 best epoch=10 (overfit), S3 best epoch=6 (ít overfit hơn)
- Patience=999 trong cả S3 lẫn E1 — không phải differentiator
- Val loss diverge: 0.271 (epoch 3) → 0.376 (epoch 10) nhưng val joint_f1 vẫn tăng (memorize, không generalize)

**Bài học:**
1. Val set nhỏ + class weights = combo overfit
2. Val tốt không có nghĩa test tốt — cần tăng val_ratio
3. Patience=999 là sai lầm — cần early stopping đúng

**Kaggle notebooks:**
- Train: [lcminhc/sprint-2-absa-train-e1-e2](https://www.kaggle.com/code/lcminhc/sprint-2-absa-train-e1-e2)
- Eval E1: [lcminhc/sprint-02-eval-e1](https://www.kaggle.com/code/lcminhc/sprint-02-eval-e1)

**Skipped (chưa thử):** Focal Loss, Majority-vote — vẫn là phương án khả thi trong roadmap mới

### Kaggle Dataset: p3s2-embedding-flat

Gop `p3s2-embedding` + `p3s2-index` vao 1 dataset flat:
- `embedding_best.pt` (736MB), `classification.jsonl`, `bio_tagging.jsonl`
- `train.faiss`, `train_metadata.jsonl`, `train_vectors.npy`
- `contrastive_triplets.jsonl`, `hard_contrastive_triplets.jsonl`
- URL: https://www.kaggle.com/datasets/lcminhc/p3s2-embedding-flat

---

## Next Actions

- [x] ~~Đọc kết quả Sprint 2 từ Kaggle (E1, E2 val logs)~~
- [x] ~~So sánh E1 vs S3 (class weights effect), E2 vs E1 (dropout+lambda effect)~~
- [x] ~~Eval E1 trên test set~~ — Joint F1=0.5684, thua S3
- [x] ~~Cập nhật IMPROVE_PART2.md~~ — roadmap mới với 6 nhóm phương án
- [x] ~~Chờ NB3 E3 xong → chạy NB4 eval → so sánh với S3~~ — **Done, E3 Joint F1=0.6374 (best ever)**
- [x] B1: Focal Loss — **DONE — FAIL**
  - gamma=2.0: Sent Acc stuck 0.6614 (majority class) epochs 1-6, val Joint F1=0.314
  - gamma=0.5: Sent Acc stuck 0.6614 epochs 1-4, val Joint F1=0.265
  - Root cause: Focal Loss giảm gradient quá mạnh cho dataset nhỏ (~1600 train) + lambda_cls=0.5 + encoder_lr=2e-6
  - **Kết luận:** B1 dropped. E3 vẫn là best model (Joint F1=0.6374)
- [x] ~~Next: C1, C2, E~~ → **Cancelled — pipeline redesign (Phase 5)**

---

## Phase 5: Pipeline Redesign — BIO → Category + Sentiment (IN PROGRESS 🔄)

**Started:** 2026-06-01
**Decision:** 2026-06-02 — **Combo 6→5** (Two-stage, phased retriever upgrade)
**Design refined:** 2026-06-02 — architecture + loss details chốt sau review

**Motivation:** Chuyển từ BIO tagging (aspect term extraction) sang Category Detection (aspect category classification). BIO head → Category head. Lý do: (1) BIO + given category quá đơn giản cho graduation thesis, (2) Category detection thực tế hơn (real-world không cho trước category).

**Quyết định đã chốt:**
- Model tự dự đoán category (KHÔNG cho trước category trong input)
- 12 categories: AMBIENCE#GENERAL, DRINKS#PRICES, DRINKS#QUALITY, DRINKS#STYLE_OPTIONS, FOOD#PRICES, FOOD#QUALITY, FOOD#STYLE_OPTIONS, LOCATION#GENERAL, RESTAURANT#GENERAL, RESTAURANT#MISCELLANEOUS, RESTAURANT#PRICES, SERVICE#GENERAL
- Sentiment head giữ nguyên 3 classes (positive, negative, neutral)
- **Separate backbone** — Stage 1 DeBERTa và Stage 2 DeBERTa train riêng biệt

**Chosen approach: Combo 6→5**

| Phase | Description | Retriever | Kaggle sessions |
|-------|------------|-----------|-----------------|
| **Phase 1 (Combo 6)** | Two-stage + frozen retriever + label interpolation | Frozen cosine | 1-2 (+1 no-ret baseline) |
| **Phase 2a** | Learnable W (256×256) + ranking loss, ContrastiveEmbedder frozen | Learnable W | 1-2 additional |
| **Phase 2b** (conditional) | Unfreeze ContrastiveEmbedder, re-encode store ~26s/epoch | Joint training | 1-2 additional |

**Architecture details (Phase 1):**
- Stage 1: DeBERTa → BCE với sqrt pos_weight (range 1.23–9.19) + **per-category** threshold tuning
- Stage 2: label interpolation dùng `softmax(score/tau)` với tau ∈ {0.02, 0.05, 0.1}
- polarity_embedding dim = 64d → final_repr = 768+64 = 832d → Linear(832,256)→ReLU→Dropout→Linear(256,3)
- nb_text = **chỉ sentence** (RLI-style, không có [ASP]/[POL])
- Stage 2 training dùng **gold categories**, predicted khi inference
- Conflict pairs (42 train, 8 test): **giữ cả 2 records**
- Self-exclusion: loại trừ theo **sentence text** (không chỉ op_id)

**Metrics changed:** Old (Span F1, Joint F1 span+pol) → New (Category F1, Joint F1 cat+pol). E3 results NOT comparable.
**New metric:** Sentiment Acc | Correct Category (tách lỗi Stage 1 vs Stage 2)

**Full design:** `REDESIGN_DISCUSSION.md`

**Status:** NB1 done, NB2 run 2 ret done, NB3 run 2 done — **Stage 1 model fail (Cat F1=0.23), needs retrain**

### Code Changes (2026-06-03)

**New files (17):**
- `src/data/category_builder.py` — CATEGORY_LIST, build_category_records(), build_sentiment_records()
- `src/absa/category_model.py` — CategoryDetector (DeBERTa + 12 sigmoid + BCE)
- `src/absa/category_dataset.py` — CategoryDataset (sentence → 12-dim multi-label)
- `src/absa/category_trainer.py` — CategoryTrainer (per-category threshold tuning)
- `src/absa/label_interpolation.py` — LabelInterpolation (polarity_embedding + softmax/tau)
- `src/absa/sentiment_model.py` — SentimentPredictor (DeBERTa + LabelInterp + MLP 832→3)
- `src/absa/sentiment_dataset.py` — SentimentDataset (RLI-style nb_text, sentence-level self-exclusion)
- `src/absa/sentiment_trainer.py` — SentimentTrainer
- `src/evaluation/category_metrics.py` — category_f1, joint_cat_sent_f1, sent_acc|correct_cat
- `scripts/04a_train_stage1.py`, `scripts/04b_train_stage2.py`, `scripts/05_evaluate_joint.py`
- `configs/stage1.yaml`, `configs/stage2.yaml`, `configs/stage2_noret.yaml`

**Modified files (3):**
- `scripts/01_prepare_data.py` — output category_detection.jsonl + sentiment_records.jsonl
- `src/retrieval/retriever.py` — thêm exclude_sentence parameter (C1 fix)
- `tests/test_retriever.py` — thêm test exclude_sentence

**Tests:** 146/146 pass (143 Phase 5 + 2 new: class weights + early stop metric + 1 per-sentence metric)
**Smoke tests:** Stage 1 CPU ✅, Stage 2 no-retrieval CPU ✅

### Kaggle Datasets (Phase 5)

| Dataset | URL | Contents |
|---------|-----|----------|
| `p5-nb1-stage1` | [lcminhc/p5-nb1-stage1](https://www.kaggle.com/datasets/lcminhc/p5-nb1-stage1) | `stage1_best.pt`, `category_detection.jsonl`, `sentiment_records.jsonl`, logs |

### Next Actions

- [x] Lên implementation plan chi tiết cho Phase 1
- [x] Rewrite dataset (gom opinions theo sentence, multi-label format)
- [x] Rewrite model (category head 12 sigmoid + conditioned sentiment head + label interpolation)
- [x] Rewrite metrics (category P/R/F1, new Joint F1, Sentiment Acc|Correct Category)
- [x] Rewrite trainer (Stage 1 train → Stage 2 train separately)
- [x] Train Stage 1 trên Kaggle — **Done** (Cat F1=0.8991, encoder_lr fix 2e-5→2e-6)
- [x] Train Stage 2 run 1 — **Done** (retrieval not converged, Macro F1 low)
- [x] Phân tích + implement 4 fixes — **Done** (commit `5397f32`)
- [x] Train Stage 2 run 2 trên Kaggle (retrieval) — **Done** (Macro F1=0.5973, best ep20, beat no-ret run 1)
- [x] NB3 run 1 — **FAILED** (threshold overfitting + metric bug)
- [x] Fix: stratify match, 3 strategies, diagnostic logging (commits `9a5d112`→`e997645`)
- [x] NB3 run 2 — **Stage 1 FAIL** (Cat F1=0.23, model can't distinguish categories on test)
- [ ] **Retrain Stage 1** — fix pos_weight cap, tăng encoder_lr, train đủ epochs
- [ ] Neutral augmentation từ MAMS dataset — **Plan ready, chờ nghiên cứu thêm**

---

## Neutral Augmentation Plan (Pending)

**Vấn đề:** Neutral chỉ 101/2,507 samples (4%) → model ignore neutral (Acc=0.88, Macro F1=0.59)

**Approach:** Lấy neutral sentences từ MAMS-ACSA dataset, LLM map sang 12 SemEval categories.

**MAMS dataset:** https://github.com/siat-nlp/MAMS-for-ABSA
- Paper: EMNLP-IJCNLP 2019, restaurant domain, mỗi câu ≥2 aspects với polarity khác nhau
- Neutral available: ~190 (train) + ~468 (val) = **~658 neutral opinions**
- 8 coarse categories (food, service, ambience...) → cần map sang 12 SemEval fine-grained

**Category mapping:**
- 5/8 map trực tiếp (High/Medium confidence): service→SERVICE#GENERAL, ambience→AMBIENCE#GENERAL, miscellaneous→RESTAURANT#MISCELLANEOUS, staff→SERVICE#GENERAL, place→RESTAURANT#GENERAL
- 3/8 cần LLM annotate (food→FOOD#QUALITY/PRICES/STYLE_OPTIONS, price→*#PRICES, menu→?)

**Target:** 15-20% neutral (thêm ~240-425 samples)

**Quy trình:** Extract MAMS → LLM annotate categories → Human spot-check → Merge vào sentiment_records.jsonl → Retrain Stage 2

**Nguồn bổ sung đã khảo sát:** QUAD dataset (~96 neutral, cùng 12 categories nhưng overlap SemEval), back-translation, LLM-generated, oversampling

**Full plan:** `.claude/plans/context-after-nb2-training-curried-volcano.md`
