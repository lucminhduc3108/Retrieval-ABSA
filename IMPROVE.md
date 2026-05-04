# IMPROVE.md — Fix First, Then Train

## Context

Retrieval-ABSA pipeline đã hoàn thành MVP với kết quả test: Joint F1 = 0.6379, Span F1 = 0.7088, Sentiment Acc = 0.9079, Macro F1 = 0.7619. Bottleneck chính là span detection.

**Quyết định cố định:**
- Giữ retrieval architecture. Không bỏ retrieval bất kể kết quả.
- **Chỉ dùng SemEval 2016 SB1** (bỏ SemEval 2015). Lý do: 2016 train chứa toàn bộ 2015, gộp lại chỉ tạo thêm leakage và giảm train size sau dedup.
- Sqrt class weights `[1.00, 1.70, 4.33]`.

**Hướng đi:** Fix code logic + data quality → retrain pipeline trên SemEval 2016 sạch → fine-tune model.

---

## GĐ 0: Code Logic Fixes — DONE ✅

- [x] **0a. Early stopping: span_f1 → joint_f1** — `src/absa/trainer.py`
- [x] **0b. Split evaluation: implicit vs explicit** — `scripts/05_evaluate.py`
- [x] **0c. Class weights cho sentiment loss** — `src/absa/model.py`, `configs/absa.yaml`
- [x] **0d. Flag --no_retrieval** — `scripts/04_train_absa.py`, `scripts/05_evaluate.py`
- [x] **0e. Script phân tích data duplication** — `scripts/analyze_duplicates.py`
- [x] **0f. Gradient accumulation** — `src/absa/trainer.py`, `scripts/04_train_absa.py`
- [x] **0g. Stratified validation split** — `scripts/04_train_absa.py`

---

## GĐ 1: Data Quality Fixes — DONE ✅

**Mục tiêu:** Tạo dataset sạch, reliable trước khi invest GPU.

### Tasks — Hoàn thành

- [x] **1a. Deduplicate annotations** — `scripts/01_prepare_data.py`
- [x] **1b. Handle test leakage** — output `clean_test_ids.json`
- [x] **1c. Re-run data preparation** — verified
- [x] **1d. Recalculate class weights** — `configs/absa_exp_c.yaml`
- [x] **1e. Chuyển sang SemEval 2016 only** — bỏ SemEval 2015

### Dataset SemEval 2016 SB1 (dùng cho GĐ 2+)

```
Source files:
  Train: SemEval 2016 Task 5/Restaurant Training/ABSA16_Restaurants_Train_SB1_v2.xml
  Test:  SemEval 2016 Task 5/Phase B/Gold Annotation/Restaurant/EN_REST_SB1_TEST.xml.gold

Stats:
  Train: 2,000 sentences (1,995 unique), 2,507 opinions
  Test:  676 sentences (675 unique), 859 opinions
  Train/Test ratio: 2.92:1
  Train-test overlap: 1 sentence (0.1%) — gần như 0 leakage
  Conflict polarity: 0 (không cần drop)
  Polarity: positive 66.1%, negative 29.9%, neutral 4.0%
  Categories: 12 (không có FOOD#GENERAL)
  Implicit opinions: 25%
  Class weights (sqrt): [1.00, 1.70, 4.33]
```

### Lý do bỏ SemEval 2015

SemEval 2016 train chứa toàn bộ SemEval 2015 (train + test). Gộp 2 dataset:
- Không thêm data mới (2015 đã nằm trong 2016)
- Phình test set (thêm 2015 test) → dedup phải loại train trùng test
- Kết quả: train giảm từ 2,507 → 1,707 sau dedup, train/test ratio = 1:1 (bất thường)
- Dùng chỉ 2016: giữ nguyên 2,507 train, ratio 3:1, 0.1% leakage

---

## GĐ 2: Retrain Pipeline on SemEval 2016 (1 Kaggle session)

**Mục tiêu:** Train lại toàn bộ pipeline trên SemEval 2016 sạch + code fixes.

**Cần sửa trước:** `scripts/01_prepare_data.py` — chỉ đọc 2 file SemEval 2016 SB1, bỏ dedup phức tạp (gần 0 leakage).

### Steps

- [ ] **2a. Sửa 01_prepare_data.py** — SemEval 2016 only
- [ ] **2b. Retrain embedding** trên clean triplets (batch=32, epochs=15)
- [ ] **2c. Rebuild FAISS index**
- [ ] **2d. Train ABSA** — 2 experiments:
  - **Retrieval + sqrt weights** `[1.00, 1.70, 4.33]` → `exp_retrieval.pt`
  - **No-retrieval ablation** (baseline) → `exp_no_retrieval.pt`
- [ ] **2e. Evaluate** — cả 2 experiments, full test + implicit/explicit split

### Output mong đợi

| Metric | MVP (dirty 2015+2016) | Retrieval | No-retrieval |
|---|---|---|---|
| Joint F1 | 0.6379 | ? | ? |
| Span F1 | 0.7088 | ? | ? |
| Sent Acc | 0.9079 | ? | ? |
| Sent Macro F1 | 0.7619 | ? | ? |

Lưu ý: so sánh với MVP KHÔNG fair (khác dataset, khác test set). MVP dùng inflated test (leakage 49.8%). Retrieval vs no-retrieval là so sánh fair (cùng data, cùng config).

---

## GĐ 3: Architecture Improvements (conditional, 1-2 Kaggle sessions)

Dựa trên kết quả GĐ 2, chọn improvements có ROI cao nhất:

### 3A. ABSA model (nếu span F1 vẫn là bottleneck)
1. **CRF layer** cho BIO head (enforce B-before-I sequential constraint)
2. **Differential LR**: encoder_lr=1e-5, head_lr=5e-4
3. **Tune lambda_cls** (0.5 → 1.0) nếu sentiment macro F1 vẫn thấp

### 3B. Embedding + Retrieval (nếu retrieval contribution thấp <2%)
1. **Larger batch** cho InfoNCE (32→64)
2. **Tune tau** (0.07 → 0.10-0.15)
3. **Data augmentation** cho neutral triplets
4. **Tune retrieval params**: top_k (3→5?), threshold (0.0→0.3?)

### 3C. Data augmentation (nếu neutral F1 vẫn thấp sau class weights)
1. **Gộp QUAD dataset** — repo `NilsHellwig/ABSA-QUAD-updated-2024` (đã clone tại `temp_quad/`)
   - 335 sentences mới, 975 annotations mới (same 13 categories)
   - Cần string-match target để tạo BIO tags (không có char offsets)
   - Category mapping: `food quality` → `FOOD#QUALITY`
2. **LLM-generated data** — target neutral class specifically
3. Oversampling / synonym replacement nếu cần thêm

---

## Thứ tự thực hiện

```
GĐ 0 ✅ DONE — Code logic fixes (local)
       │
       ▼
GĐ 1 (local, ~2-3 giờ code work)
  ├── 1a: Deduplicate annotations
  ├── 1b: Handle test leakage (report both splits)
  ├── 1c: Re-run 01_prepare_data.py
  └── 1d: Recalculate class weights
       │
       ▼
GĐ 2 (1-2 Kaggle sessions)
  ├── 2a: Sửa 01_prepare_data.py (SemEval 2016 only)
  ├── 2b: Retrain embedding
  ├── 2c: Rebuild FAISS index
  ├── 2d: Train ABSA (retrieval + sqrt weights)
  └── 2e: Evaluate
       │
       ▼
GĐ 3 (conditional, 1-2 Kaggle sessions)
  ├── 3A: CRF + diff LR (nếu span bottleneck)
  ├── 3B: Better embedding (nếu retrieval delta thấp)
  └── 3C: Data augmentation (nếu neutral vẫn kém)
```

---

## Files cần sửa/tạo

| File | Thay đổi | GĐ |
|---|---|---|
| `scripts/01_prepare_data.py` | SemEval 2016 only, simplify dedup | 2a |
| `configs/absa_exp_c.yaml` | Sqrt class weights (recalc if needed) | 2d |
| `configs/embedding_v2.yaml` | batch=32, epochs=15, patience=5 | 2b |
| Kaggle notebook | Full pipeline: embed → index → train → eval | 2b-e |

---

## Verification

Sau sửa 01_prepare_data.py (local):
```bash
python scripts/01_prepare_data.py
# Expected: ~2,500 BIO records, ~2,200 CLS records, ~2,000 triplets (SemEval 2016 only)
python scripts/analyze_duplicates.py
# Expected: ~0% leakage
```

Sau GĐ 2 (Kaggle):
```bash
python scripts/05_evaluate.py --config configs/absa_exp_c.yaml \
  --checkpoint checkpoints/absa/best.pt \
  --embedding_ckpt checkpoints/embedding/best.pt --index_dir indexes/
```
