# Project Status — Retrieval-ABSA

**Last updated:** 2026-06-08 (Stage 1 improvement in progress — ASL dropped (2 attempts), Cat-Aware v2 running on Kaggle NB1 v6 (30 epochs, ep3/30))

---

## Phase 1–4 Summary (Old BIO Pipeline — ARCHIVED)

Phase 1–4 dùng pipeline cũ: BIO tagging + given category → sentiment. **Không còn active.** Tất cả code/notebooks đã archive vào `archive/kaggle_old/`.

| Phase | Kết quả tốt nhất | Ghi chú |
|-------|-----------------|---------|
| P2 (retrain clean data) | Joint F1=0.6104 (no-ret) | Retrieval thua no-ret 6.4pp |
| P3 (retrieval improvements) | Joint F1=0.6237 (S3 ret) | Hard negatives + Split BIO. Retrieval lần đầu beat no-ret |
| P4 E3 (quick fixes) | Joint F1=**0.6374** (ret) | diff LR + val_ratio=0.2 + patience=5. Best ever của pipeline cũ |
| P4 B1 (Focal Loss) | FAIL | Gradient collapse trên dataset nhỏ |

**Lý do chuyển sang Phase 5:** BIO + given category quá đơn giản cho thesis; chuyển sang Category Detection (model tự predict category, không cho trước).

---

## Phase 5: Pipeline Redesign — Category Detection + Sentiment

### Architecture

**Two-stage pipeline:**
- **Stage 1:** DeBERTa → ContextPooler → Linear(768, 12) với 12 sigmoid (BCE + sqrt pos_weight). Per-category threshold tuning. Inputs: raw sentence only.
- **Stage 2:** DeBERTa + label interpolation (cosine retrieval, `softmax(score/tau=0.5)`) → polarity_embedding(64d) → MLP(832→256→3). Inputs: sentence + retrieved neighbors (RLI-style, no [ASP]/[POL]).
- **Training:** Stage 1 và Stage 2 train riêng biệt. Stage 2 dùng gold categories khi train, predicted categories khi inference.
- **Metrics:** Category F1, Joint F1 (cat+pol), Sentiment Acc|Correct Category.
- 12 categories: AMBIENCE#GENERAL, DRINKS#PRICES, DRINKS#QUALITY, DRINKS#STYLE_OPTIONS, FOOD#PRICES, FOOD#QUALITY, FOOD#STYLE_OPTIONS, LOCATION#GENERAL, RESTAURANT#GENERAL, RESTAURANT#MISCELLANEOUS, RESTAURANT#PRICES, SERVICE#GENERAL.

**Full design:** `REDESIGN_DISCUSSION.md`

### Current Best Results (Phase 5, Test Set — NB3 Run 3, 2026-06-08)

Stage 1: R4 | Stage 2: Run 2 (retrieval)

| Strategy | Cat F1 | No-Ret Joint F1 | No-Ret Sent Acc\|CC | Ret Joint F1 | Ret Sent Acc\|CC |
|----------|--------|-----------------|---------------------|-------------|-----------------|
| per_category | 0.6765 | 0.6028 | 0.8956 | 0.5977 | 0.8880 |
| **global (0.60)** | **0.6844** | **0.6120** | 0.8990 | 0.6067 | 0.8913 |
| topk (k=1) | 0.6602 | 0.5979 | **0.9112** | 0.5889 | 0.8975 |

**Best config:** Global strategy (threshold=0.60) + No-Retrieval → **Joint F1 = 0.6120**
No-retrieval beats retrieval by ~0.5pp (down from 6.4pp in Phase 2 old pipeline).

### Kaggle Notebooks (Phase 5)

| Notebook | Kaggle URL | Status | Output dataset |
|----------|-----------|--------|----------------|
| **P5-NB1: Stage 1 Train** | [lcminhc/p5-nb1-stage1-train](https://www.kaggle.com/code/lcminhc/p5-nb1-stage1-train) | 🔄 v6 running (Cat-Aware 30ep) | `p5-nb1-stage1` |
| **P5-NB2: Stage 2 Train** | [lcminhc/p5-nb2-stage2-train](https://www.kaggle.com/code/lcminhc/p5-nb2-stage2-train) | ✅ Run 2 done (ret) | `p5-nb2-stage2` |
| **P5-NB3: Joint Eval** | [lcminhc/p5-nb3-joint-eval](https://www.kaggle.com/code/lcminhc/p5-nb3-joint-eval) | ⏳ Pending NB1 v6 output | — |

### Stage 1 Training History

| Run | Val Cat F1 | Test Cat F1 | Config | Note |
|-----|-----------|------------|--------|------|
| R1/R2/R3 | — | 0.23 | old configs | encoder_lr too low, pos_weight too high |
| **R4** | 0.7482 (ep16) | **0.6844** | `stage1_r4.yaml` | ContextPooler, encoder_lr=2e-5, pos_weight_cap=3.0, fp16=false |

### Stage 2 Training History

| Run | Val Sent Acc | Val Macro F1 | Note |
|-----|-------------|-------------|------|
| Run 1 (ret) | 0.8589 | 0.5727 | tau=0.05 too sharp, not converged |
| **Run 2 (ret)** | **0.8980** | **0.5973** | tau=0.5, sqrt class weights, 20 epochs — beat no-ret on val |

Run 2 no-retrieval baseline: pending (P1+P2 fixes not yet applied to no-ret config).

### Phase 5 Key Code (2026-06-03)

New: `src/data/category_builder.py`, `src/absa/category_{model,dataset,trainer}.py`, `src/absa/{label_interpolation,sentiment_model,sentiment_dataset,sentiment_trainer}.py`, `src/evaluation/category_metrics.py`, `scripts/04a_train_stage1.py`, `scripts/04b_train_stage2.py`, `scripts/05_evaluate_joint.py`, `configs/stage1*.yaml`, `configs/stage2*.yaml`

Modified: `scripts/01_prepare_data.py` (output category_detection.jsonl + sentiment_records.jsonl), `src/retrieval/retriever.py` (exclude_sentence param)

---

## Stage 1 Improvement (Phase A — 2026-06-08)

**Goal:** Cat F1 0.6844 → ≥0.74. Root cause: discrimination failure (correlated labels, shared CLS pooler).

**Code changes (commits `7c158b3`, `81e8038`, `c7a66e2`):**
- `src/absa/category_model.py`: `AsymmetricLoss` + `CategoryDetector` with `use_asl`/`use_cat_attention` flags
- `scripts/04a_train_stage1.py`: multi-label stratified split, ASL/cat-attention support
- `scripts/05_evaluate_joint.py`: **bug fix** — pass `use_cat_attention` from config when loading Stage 1 model
- `configs/stage1_r5.yaml` / `configs/stage1_r5_cataware.yaml`
- `stage1_improvement.md`: full diagnostic + Phase A/B plan
- **Tests:** 158/158 pass

### Experiment Results

| Experiment | Best Val Cat F1 | Note |
|------------|----------------|------|
| ASL v1 (gamma_neg=4, no pos_weight) | 0.6718 (ep13) | Thua R4. R>>P, rare categories get near-zero gradient |
| ASL v2 (gamma_neg=2, pos_weight=3.0) | 0.3391 (ep7) | Worse. pos_weight + ASL interaction broken: amplified positive loss overwhelms focusing |
| **Cat-Aware v1** (epochs=20) | **0.7222 (ep19)** | P≈R balanced (0.724/0.721). Not converged — session lost before upload |
| **Cat-Aware v2** (epochs=30) | 🔄 Running (ep3/30) | Slightly behind v1 in warmup phase, expected to recover ep4+ |

**ASL: DROPPED.** Không tương thích với dataset nhỏ — pos_weight/gamma overcorrect theo những hướng khác nhau.

**Cat-Aware Attention:** Learnable query per category attends vào token sequence thay vì shared CLS. Đây là kiến trúc promising (first model đạt P≈R balance).

**After NB1 v6 finishes:** Upload outputs to `p5-nb1-stage1` dataset → run NB3 → compare test Cat F1 vs R4 (0.6844).

**Decision criteria:**
- Test Cat F1 ≥ 0.74 → dừng Phase A, đủ cho thesis
- Test Cat F1 0.68–0.73 → cân nhắc thêm epochs hoặc Phase B
- Test Cat F1 < 0.68 → Phase B (Hierarchical Entity→Attribute)

---

## Stage 1 Per-Category Diagnosis (R4 Baseline)

| Tier | Category | P | R | F1 | Train |
|------|----------|-----|-----|-----|-------|
| Cao | SERVICE#GENERAL | 0.917 | 0.759 | 0.830 | 419 |
| Cao | FOOD#QUALITY | 0.750 | 0.903 | 0.819 | 681 |
| Cao | RESTAURANT#GENERAL | 0.809 | 0.655 | 0.724 | 421 |
| Trung bình | FOOD#PRICES | 0.567 | 0.773 | 0.654 | 82 |
| Trung bình | AMBIENCE#GENERAL | 0.485 | 0.877 | 0.625 | 226 |
| Trung bình | RESTAURANT#PRICES | 0.444 | 0.571 | 0.500 | 80 |
| Thấp | DRINKS#QUALITY | 0.462 | 0.286 | 0.353 | 46 |
| Thấp | DRINKS#STYLE_OPTIONS | 0.500 | 0.250 | 0.333 | 30 |
| Thấp | FOOD#STYLE_OPTIONS | 0.333 | 0.229 | 0.272 | 128 |
| Thấp | RESTAURANT#MISC | 0.222 | 0.242 | 0.232 | 97 |
| Zero | LOCATION#GENERAL | 0.500 | 0.077 | 0.133 | 28 |
| Zero | DRINKS#PRICES | 0.000 | 0.000 | 0.000 | 20 |

---

## Next Actions

- [ ] **Chờ NB1 v6 xong** → Upload outputs to `p5-nb1-stage1` dataset
- [ ] **Chạy NB3** với Cat-Aware checkpoint (`stage1_r5_cataware_best.pt`, config `stage1_r5_cataware.yaml`)
- [ ] **Quyết định Phase B** dựa trên test Cat F1
- [ ] Neutral augmentation từ MAMS dataset — pending Stage 1 improvement

---

## Neutral Augmentation Plan (Pending)

**Vấn đề:** Neutral chỉ 101/2,507 samples (4%) → model ignore neutral (Stage 2 Macro F1 ~0.59)

**Approach:** Lấy neutral sentences từ MAMS-ACSA dataset, LLM map sang 12 SemEval categories.

**MAMS dataset:** https://github.com/siat-nlp/MAMS-for-ABSA — ~658 neutral opinions available.

**Category mapping:** 5/8 map trực tiếp (service, ambience, miscellaneous, staff, place); 3/8 cần LLM annotate (food, price, menu).

**Target:** 15–20% neutral (+240–425 samples). **Full plan:** `.claude/plans/context-after-nb2-training-curried-volcano.md`

---

## Training Environment

- **GPU:** Kaggle T4x2 (pipeline uses 1 GPU). Stage 1 batch=16, Stage 2 batch=32 grad_accum=2.
- **Local:** Windows 11, no GPU training.
- **Key datasets on Kaggle:** `lcminhc/semeval-absa-restaurant` (raw XML), `lcminhc/p5-nb1-stage1` (Stage 1 ckpt + data), `lcminhc/p5-nb2-stage2` (Stage 2 ckpts), `lcminhc/p3s2-embedding-flat` (embedding + FAISS index).
