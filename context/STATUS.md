# Project Status — Retrieval-ABSA

**Last updated:** 2026-06-10 (NB2 v4 đang chạy trên Kaggle — Phase 2a (W matrix) + no-ret baseline training)

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

### Current Best Results (Phase 5, Test Set — NB3 v4, 2026-06-09)

Stage 1: Cat-Aware R5 | Stage 2: Run 2 (retrieval)

| Strategy | Cat F1 | No-Ret Joint F1 | No-Ret Sent Acc\|CC | Ret Joint F1 | Ret Sent Acc\|CC |
|----------|--------|-----------------|---------------------|-------------|-----------------|
| per_category | 0.6858 | 0.6135 | 0.8992 | 0.6057 | 0.8878 |
| **global (0.80)** | **0.6962** | **0.6235** | 0.9006 | 0.6139 | 0.8867 |
| topk (k=1) | 0.6797 | 0.6158 | **0.9115** | 0.6024 | 0.8916 |

**Best config:** Global strategy (threshold=0.80) + No-Retrieval → **Joint F1 = 0.6235**
No-retrieval beats retrieval by ~1pp. Cat-Aware improved Cat F1 +1.18pp and Joint F1 +1.15pp over R4.

### Previous Best (R4 baseline, NB3 Run 3)

Global (0.60): Cat F1=0.6844 | No-Ret Joint F1=0.6120 | Ret Joint F1=0.6067

### Kaggle Notebooks (Phase 5)

| Notebook | Kaggle URL | Status | Output dataset |
|----------|-----------|--------|----------------|
| **P5-NB1: Stage 1 Train** | [lcminhc/p5-nb1-stage1-train](https://www.kaggle.com/code/lcminhc/p5-nb1-stage1-train) | ✅ v6 done (Cat-Aware 30ep) | `p5-nb1-stage1` |
| **P5-NB2: Stage 2 Train** | [lcminhc/p5-nb2-stage2-train](https://www.kaggle.com/code/lcminhc/p5-nb2-stage2-train) | ✅ Run 2 done (ret) | `p5-nb2-stage2` |
| **P5-NB3: Joint Eval** | [lcminhc/p5-nb3-joint-eval](https://www.kaggle.com/code/lcminhc/p5-nb3-joint-eval) | ✅ v4 done (Cat-Aware eval) | — |

### Stage 1 Training History

| Run | Val Cat F1 | Test Cat F1 | Config | Note |
|-----|-----------|------------|--------|------|
| R1/R2/R3 | — | 0.23 | old configs | encoder_lr too low, pos_weight too high |
| **R4** | 0.7482 (ep16) | 0.6844 | `stage1_r4.yaml` | ContextPooler, encoder_lr=2e-5, pos_weight_cap=3.0, fp16=false |
| **R5 Cat-Aware** | 0.7243 (ep26) | **0.6962** | `stage1_r5_cataware.yaml` | Cat-Aware Attention (12 learnable queries + MHA), encoder_lr=1e-5, head_lr=5e-4, 30ep |

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

| Experiment | Best Val Cat F1 | Test Cat F1 | Note |
|------------|----------------|------------|------|
| ASL v1 (gamma_neg=4, no pos_weight) | 0.6718 (ep13) | — | Thua R4. R>>P, rare categories get near-zero gradient |
| ASL v2 (gamma_neg=2, pos_weight=3.0) | 0.3391 (ep7) | — | Worse. pos_weight + ASL interaction broken |
| Cat-Aware v1 (epochs=20) | 0.7222 (ep19) | — | Session lost before upload |
| **Cat-Aware v2** (epochs=30) | **0.7243 (ep26)** | **0.6962** | Val→test gap 2.8pp (better than R4's 6.4pp). Plateau ep12-15, overfit after |

**ASL: DROPPED.** Không tương thích với dataset nhỏ.

**Cat-Aware Attention: ACCEPTED as new baseline.** Test Cat F1=0.6962 (+1.18pp over R4), Joint F1=0.6235 (+1.15pp). Improves AMBIENCE (+8.3pp), FOOD#STYLE_OPTIONS (+8pp), LOCATION (+11.7pp) but hurts FOOD#PRICES (-31pp), RESTAURANT#PRICES (-15pp) due to high global threshold.

**Decision:** Cat F1=0.6962 falls in 0.68-0.73 range → Cat-Aware tuning unlikely to reach 0.74 (val plateau at 0.7243, architecture ceiling). **Proceed to Phase B: Hierarchical Entity→Attribute.**

**Side fix for next training:** `CategoryTrainer.evaluate()` only uses per_category threshold for model selection. Add global threshold F1 for checkpoint selection (~10 lines, zero GPU cost).

---

## Phase B: Hierarchical Entity→Attribute (2026-06-10)

**Goal:** Cat F1 0.6962 → ≥0.74 by decomposing 12-flat into 6-entity + 3-attribute.

### Architecture

```
DeBERTa → CLS → ContextPooler(768→768, Tanh, Dropout)
  ├── Entity Head: Linear(768, 6) — 6 sigmoid, multi-label
  ├── FOOD Attr Head: Linear(768, 3) — PRICES / QUALITY / STYLE_OPTIONS
  ├── DRINKS Attr Head: Linear(768, 3) — PRICES / QUALITY / STYLE_OPTIONS
  └── RESTAURANT Attr Head: Linear(768, 3) — GENERAL / MISCELLANEOUS / PRICES

AMBIENCE/LOCATION/SERVICE → entity detected = emit ENTITY#GENERAL (no attr head)
Loss = L_entity + L_food_attr(masked) + L_drinks_attr(masked) + L_restaurant_attr(masked)
Attribute loss only computed for samples where gold entity is active.
```

### Code Changes (complete, 2026-06-10)

| File | Change |
|------|--------|
| `src/data/category_builder.py` | `ENTITY_LIST`, `ENT2IDX`, `ENTITY2ATTRS`, `MULTI_ATTR_ENTITIES`, `ATTR2IDX`; `build_category_records` adds `entity_vector` + 3 `*_attr_vector` |
| `src/absa/category_model.py` | `HierarchicalCategoryDetector` — entity head + 3 attr heads, masked attr loss |
| `src/absa/category_dataset.py` | `HierarchicalCategoryDataset` — returns entity_labels + 3 attr_labels |
| `src/absa/category_trainer.py` | `tune_entity_thresholds`, `tune_attr_thresholds`, `hierarchical_decode`, `HierarchicalCategoryTrainer` |
| `scripts/04a_train_stage1.py` | `use_hierarchical` branch + `compute_entity_pos_weight` / `compute_attr_pos_weight` |
| `scripts/05_evaluate_joint.py` | Hierarchical model loading, `collect_hierarchical_logits`, hierarchical decode with val threshold tuning |
| `configs/stage1_hierarchical.yaml` | `use_hierarchical: true`, encoder_lr=1e-5, head_lr=5e-4, 30 epochs |
| Tests | 14 new tests (model/trainer/builder/dataset), 175/175 pass |

### NB1 v7 Training Results (2026-06-10)

| Epoch | Loss | Cat F1 | Cat P | Cat R |
|-------|------|--------|-------|-------|
| 1 | 2.3715 | 0.2768 | 0.1629 | 0.9202 |
| 14 | 0.8049 | 0.6261 | 0.5688 | 0.6962 |
| **22** | **0.4762** | **0.7002** | **0.6640** | **0.7406** |
| 27 | 0.3860 | 0.7002 | 0.6640 | 0.7406 |

Early stopped epoch 27 (patience=5). Best val Cat F1 = **0.7002** at epoch 22.

### Verdict: DROPPED

- Val Cat F1=0.7002 — **4.8pp below R4** (0.7482), **2.4pp below Cat-Aware R5** (0.7243)
- Root causes: two-stage thresholding compounds errors, attr heads data-starved (DRINKS ~77 samples), silent recall loss when entity predicted but no attr clears threshold
- Estimated test Cat F1 ~0.67 (worse than both baselines)

**Decision:** Hierarchical dropped. **Cat-Aware R5 accepted as Stage 1 FINAL** (test Cat F1=0.6962, Joint F1=0.6235).

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

- [x] **NB1 v6 done** → Cat-Aware v2, outputs uploaded (2026-06-08)
- [x] **NB3 v4 done** → Cat-Aware test Cat F1=0.6962, Joint F1=0.6235 (2026-06-09)
- [x] **Decision:** Cat-Aware accepted as baseline. Proceed to Phase B (Hierarchical)
- [x] **Phase B code complete** (2026-06-10) — 175/175 tests, smoke test OK
- [x] **NB1 v7 done** → Hierarchical val Cat F1=0.7002, thua Cat-Aware (0.7243). **DROPPED.**
- [x] **Decision:** Cat-Aware R5 = Stage 1 FINAL. Move to Phase 2a (learnable retriever).
- [x] **Phase 2a code complete** (2026-06-10) — 197/197 tests, smoke test OK
  - `src/absa/learnable_retriever.py` (NEW): W matrix + ranking loss
  - `src/retrieval/retriever.py`: adds `faiss_idx` to results
  - `src/absa/sentiment_dataset.py`: `query_vec`, `neighbor_vecs`, `query_polarity`; -inf padding fix
  - `src/absa/label_interpolation.py`: handles -inf padding correctly
  - `src/absa/sentiment_model.py`: `use_learnable_retriever` flag
  - `src/absa/sentiment_trainer.py`: combined loss, per-class F1 logging
  - `scripts/04b_train_stage2.py`, `scripts/05_evaluate_joint.py`: full support
  - `configs/stage2_phase2a.yaml` (NEW), `configs/stage2_noret.yaml` fixed
- [ ] **NB2 v4:** 🔄 Đang chạy trên Kaggle — Phase 2a (W matrix) + no-ret (20ep). Sau khi xong: upload outputs → dataset `p5-nb2-stage2`
- [ ] **NB3 v5:** Joint eval Cat-Aware R5 + Phase 2a sentiment
- [ ] Neutral augmentation từ MAMS dataset — pending (deferred)

---

## Neutral Augmentation Plan (Pending)

**Vấn đề:** Neutral chỉ 101/2,507 samples (4%) → model ignore neutral (Stage 2 Macro F1 ~0.59)

**Approach:** Lấy neutral sentences từ MAMS-ACSA dataset, LLM map sang 12 SemEval categories.

**MAMS dataset:** https://github.com/siat-nlp/MAMS-for-ABSA — ~658 neutral opinions available.

**Category mapping:** 5/8 map trực tiếp (service, ambience, miscellaneous, staff, place); 3/8 cần LLM annotate (food, price, menu).

**Target:** 15–20% neutral (+650 samples). 
**Update 2026-06-10:** Đã thực hiện trích xuất 650 mẫu neutral từ 5 danh mục an toàn (1-1 map: place, miscellaneous, ambience, service, staff) của MAMS-ACSA. Dữ liệu mới đã được ghi vào `data/processed/sentiment_records_aug.jsonl`, nâng tỷ lệ Neutral trong tập Train từ 4% lên ~20%.

---

## Phase 2a: Learnable Retrieval Alignment

**Problem:** No-ret beats ret by ~1pp (Joint F1 0.6235 vs 0.6139). Frozen cosine retriever is polarity-blind.

**Approach:** Replace `score(q,k) = q^T k` with `score(q,k) = q^T W k`. W là learnable 256×256 matrix (65K params). FAISS vẫn select candidates, W re-score cho label interpolation. Ranking loss: same-polarity neighbors score cao hơn.

**Code:** `src/absa/learnable_retriever.py` — commit `68edc6d`. Sửa lỗi toán học dùng hard positive `.min()` thay vì `.max()` để ranking loss có tác dụng.

**Config:** `configs/stage2_phase2a.yaml` — `retriever_lr=5e-4`, `lambda_rank=0.1`, `tau=0.1` (giảm để tăng độ sắc nét nội suy), `rank_margin=0.5` (tăng để kéo dãn không gian vector), 20 epochs, data dùng `sentiment_records_aug.jsonl`.

**Status:** NB2 v4 đang train trên Kaggle (2026-06-10)

**Backup:** Nếu W re-score không đủ → full PyTorch scoring (W score toàn bộ corpus ~2500 vectors, bypass FAISS).

---

## Training Environment

- **GPU:** Kaggle T4x2 (pipeline uses 1 GPU). Stage 1 batch=16, Stage 2 batch=32 grad_accum=2.
- **Local:** Windows 11, no GPU training.
- **Key datasets on Kaggle:** `lcminhc/semeval-absa-restaurant` (raw XML), `lcminhc/p5-nb1-stage1` (Stage 1 ckpt + data), `lcminhc/p5-nb2-stage2` (Stage 2 ckpts), `lcminhc/p3s2-embedding-flat` (embedding + FAISS index).
