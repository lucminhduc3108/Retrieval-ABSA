# IMPROVE.md — Fix First, Then Train

## Context

Retrieval-ABSA pipeline đã hoàn thành MVP với kết quả test: Joint F1 = 0.6379, Span F1 = 0.7088, Sentiment Acc = 0.9079, Macro F1 = 0.7619. Bottleneck chính là span detection.

**Quyết định cố định:** Giữ retrieval architecture. Không bỏ retrieval bất kể kết quả.

**Hướng đi mới:** Fix code logic + data quality trước → retrain toàn bộ pipeline trên data sạch → fine-tune model. Tránh chạy experiments trên data bẩn rồi phải chạy lại.

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

## GĐ 1: Data Quality Fixes (local, không cần GPU) — DONE ✅

**Mục tiêu:** Tạo dataset sạch, reliable trước khi invest GPU.

### Tasks — Hoàn thành

- [x] **1a. Deduplicate annotations** — `scripts/01_prepare_data.py`
  - Loại internal duplicates trong train (SemEval 2015 ⊂ 2016)
  - Loại train records trùng exact với test (chống leakage)
- [x] **1b. Handle test leakage** — output `clean_test_ids.json`
  - Test giữ nguyên, report cả "full test" và "clean test" metrics
- [x] **1c. Re-run data preparation** — verified
- [x] **1d. Recalculate class weights** — `configs/absa_exp_b.yaml`, `configs/absa_exp_c.yaml`

### Kết quả

```
Data sau dedup:
  Train: 1,707 BIO / 1,542 CLS (từ 4,161 — SemEval 2016 train chứa toàn bộ 2015)
  Test: 1,704 (unchanged)
  Test leakage: 49.8% → 0.3% (chỉ 3 sentences)
  Clean test: 1,698/1,704 (99.6%)
  Contrastive triplets: 1,537
  Class weights (inv-freq): [1.00, 2.88, 18.75]
  Class weights (sqrt): [1.00, 1.70, 4.33]
```

---

## GĐ 2: Retrain Pipeline (1-2 Kaggle sessions)

**Mục tiêu:** Train lại toàn bộ pipeline trên data sạch + code fixes.

### Session 1: Embedding + Index + Experiments

- [ ] **2a. Retrain embedding** trên clean triplets
  - Config changes: batch_size 16→32 (nếu GPU cho phép), tune tau
  - Output: `checkpoints/embedding/best_v2.pt`

- [ ] **2b. Rebuild FAISS index** từ embedding mới
  - Output: `indexes/train_v2.faiss` + metadata

- [ ] **2c. Train ABSA experiments** (3 experiments)
  - **Exp A:** No-retrieval baseline (quantify retrieval contribution)
  - **Exp B:** Retrieval + inverse-freq class weights
  - **Exp C:** Retrieval + sqrt class weights
  - Tất cả dùng: joint_f1 early stop, grad_accum=8, stratified split

### Session 2 (nếu cần): Fine-tune

- [ ] **2d. Analyze results** → chọn best class weights
- [ ] **2e. Fine-tune winner** nếu cần thêm tuning (lambda_cls, top_k, threshold)

### Output mong đợi: Bảng so sánh

| Metric | MVP (dirty data) | Exp A (no-ret) | Exp B (inv-freq) | Exp C (sqrt) |
|---|---|---|---|---|
| Joint F1 | 0.6379 | ? | ? | ? |
| Span F1 | 0.7088 | ? | ? | ? |
| Sent Acc | 0.9079 | ? | ? | ? |
| Sent Macro F1 | 0.7619 | ? | ? | ? |
| Joint F1 (explicit) | — | ? | ? | ? |
| Joint F1 (clean test) | — | ? | ? | ? |

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
  ├── 2a: Retrain embedding (clean data, larger batch)
  ├── 2b: Rebuild FAISS index
  ├── 2c: Train Exp A/B/C on clean data
  └── 2d-e: Analyze + fine-tune
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
| `scripts/01_prepare_data.py` | Dedup logic, clean test split output | 1a, 1b |
| `configs/absa_exp_b.yaml` | Recalculate class weights | 1d |
| `configs/absa_exp_c.yaml` | Recalculate class weights | 1d |
| `configs/embedding.yaml` | batch_size, tau tuning | 2a |
| Kaggle notebook | Full pipeline: embed → index → 3 experiments | 2a-c |

---

## Verification

Sau GĐ 1 (local):
```bash
# Re-run data prep with dedup
python scripts/01_prepare_data.py --raw_dir data/raw --out_dir data/processed

# Verify no annotation duplicates
python scripts/analyze_duplicates.py

# Verify clean test split exists
wc -l data/processed/bio_tagging.jsonl
```

Sau GĐ 2 (Kaggle):
```bash
# Evaluate on both full and clean test
python scripts/05_evaluate.py --config configs/absa.yaml \
  --checkpoint checkpoints/absa/best.pt \
  --embedding_ckpt checkpoints/embedding/best.pt --index_dir indexes/

python scripts/05_evaluate.py --config configs/absa.yaml \
  --checkpoint checkpoints/absa/best.pt --no_retrieval
```
