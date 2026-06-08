# Stage 1 Improvement Plan — Category Detection

## 1. Chẩn đoán

### Hiện trạng
- Cat F1 = 0.6844 (global strategy, threshold=0.60) trên test set (587 sentences, 743 category instances)
- Train: 1,708 sentences → 80/20 split → 1,366 train / 342 val
- Architecture: DeBERTa-v3-base → ContextPooler → Linear(768, 12) với 12 sigmoid
- Config: `stage1_r4.yaml` — encoder_lr=2e-5, head_lr=1e-4, batch=16, seq_len=256, pos_weight_cap=3.0

### Per-Category F1 (Global Strategy, Test Set)

| Tier | Category | P | R | F1 | Train | Test |
|------|----------|-----|-----|-----|-------|------|
| Cao | SERVICE#GENERAL | 0.917 | 0.759 | 0.830 | 419 | 145 |
| Cao | FOOD#QUALITY | 0.750 | 0.903 | 0.819 | 681 | 226 |
| Cao | RESTAURANT#GENERAL | 0.809 | 0.655 | 0.724 | 421 | 142 |
| Trung bình | FOOD#PRICES | 0.567 | 0.773 | 0.654 | 82 | 22 |
| Trung bình | AMBIENCE#GENERAL | 0.485 | 0.877 | 0.625 | 226 | 57 |
| Trung bình | RESTAURANT#PRICES | 0.444 | 0.571 | 0.500 | 80 | 21 |
| Thấp | DRINKS#QUALITY | 0.462 | 0.286 | 0.353 | 46 | 21 |
| Thấp | DRINKS#STYLE_OPTIONS | 0.500 | 0.250 | 0.333 | 30 | 12 |
| **Thấp** | **FOOD#STYLE_OPTIONS** | 0.333 | 0.229 | **0.272** | **128** | 48 |
| **Thấp** | **RESTAURANT#MISC** | 0.222 | 0.242 | **0.232** | **97** | 33 |
| Zero | LOCATION#GENERAL | 0.500 | 0.077 | 0.133 | 28 | 13 |
| Zero | DRINKS#PRICES | 0.000 | 0.000 | 0.000 | 20 | 3 |

### Nguyên nhân gốc

**#1 — Correlated label interference (chính):**
FOOD#STYLE_OPTIONS (128 train, F1=0.272) co-occurs 58.6% với FOOD#QUALITY. BCE penalizes FOOD#STYLE_OPTIONS head mỗi khi gặp FOOD#QUALITY-only sentence (vì cũng food-related) → model học "nếu là food, đừng fire STYLE_OPTIONS trừ khi rất chắc" → recall collapse.

**#2 — Semantic boundary diffusion:**
RESTAURANT#MISCELLANEOUS (97 train, F1=0.232) chỉ 2.1% co-occurs với RESTAURANT#GENERAL, nhưng semantic boundary không rõ (misc covers atmosphere, crowd, wait time — dễ nhầm với GENERAL hoặc AMBIENCE).

**#3 — AMBIENCE over-firing:**
P=0.485, R=0.877. Shared CLS pooler không phân biệt được "nice decor" (pure AMBIENCE) vs "great ambiance and food" (AMBIENCE + FOOD#QUALITY).

**#4 — Training dynamics plateau:**
Val loss plateau sau epoch 7. Improvement epoch 7→16 hoàn toàn do threshold re-tuning, không phải model tốt hơn.

**Kết luận: Discrimination failure > Data scarcity.** 4 rare categories (< 50 train) chỉ chiếm 6.6% test. Biggest impact: fix FOOD#STYLE_OPTIONS (48 test) + RESTAURANT#MISC (33 test) + AMBIENCE precision (57 test) = 138/743 = 18.6% test.

---

## 2. Lộ trình cải thiện

### Phase A — ASL + Category-Aware Attention (1 Kaggle session)

Quick wins, low-medium risk. Nếu Cat F1 ≥ 0.74 → dừng. Nếu < 0.71 → chuyển Phase B.

### Phase B — Hierarchical Architecture (1-2 Kaggle sessions, conditional)

Tách 12 categories thành Entity (6-class) → Attribute (3-class per entity). Trực tiếp giải quyết discrimination. Reuse ASL loss từ Phase A.

---

## 3. Phase A: ASL + Category-Aware Attention

### Experiment 1: Asymmetric Loss (ASL)

**Mục đích:** Giảm gradient interference giữa correlated labels. ASL (Ben-Baruch et al., ICCV 2021) dùng khác focusing parameter cho positive vs negative, + probability-shifting margin zero out easy negatives.

**Risk: Low | Effort: Low | Expected: +2-4pp Cat F1**

**Tại sao ASL giải quyết được:**
Khi train FOOD#STYLE_OPTIONS-positive sentence, FOOD#QUALITY thường fire với p≈0.7 (food-related). BCE loss cho negative này = 1.20 → kéo gradient ngược. ASL (gamma_neg=4, margin=0.05) giảm loss này xuống ~0.19 → model không còn bị suppress bởi dominant siblings.

**Code changes:**

`src/absa/category_model.py`:
- Thêm `AsymmetricLoss(nn.Module)` class: gamma_neg, gamma_pos, margin params
- `CategoryDetector.__init__`: thêm `use_asl=False, asl_gamma_neg=4, asl_gamma_pos=0, asl_margin=0.05`
- Khi `use_asl=True`: `self.loss_fn = AsymmetricLoss(...)` thay vì `BCEWithLogitsLoss`
- Forward interface giữ nguyên: `{logits, loss}`

`scripts/04a_train_stage1.py`:
- Đọc ASL params từ config, pass vào `CategoryDetector`
- Khi `use_asl=True`: skip `compute_pos_weight()`, pass `pos_weight=None`

`configs/stage1_r5.yaml` (copy từ r4):
```yaml
use_asl: true
asl_gamma_neg: 4
asl_gamma_pos: 0
asl_margin: 0.05
pos_weight_cap: null
ckpt_dir: checkpoints/stage1_r5
log_path: logs/stage1_r5_training.jsonl
```

Tests: Thêm `test_asl_loss_forward()` trong `tests/test_category_model.py`

### Experiment 2: Category-Aware Attention Pooling

**Mục đích:** Thay shared CLS pooler (1 vector cho 12 categories) bằng 12 category-specific attention views. Mỗi category có query vector riêng, attend vào tokens liên quan (FOOD#STYLE_OPTIONS attend "menu/variety/option", FOOD#QUALITY attend "taste/fresh").

**Risk: Medium | Effort: Medium | Expected: +3-7pp Cat F1**

**Code changes:**

`src/absa/category_model.py`:
- Thêm `use_cat_attention=False` param
- Khi `True`, thay `self.pooler` bằng:
  - `self.cat_queries = nn.Embedding(12, 768)` — learnable query per category
  - `self.cat_attention = nn.MultiheadAttention(768, num_heads=8, batch_first=True, dropout=0.1)`
  - `self.cat_norm = nn.LayerNorm(768)`
- Forward:
  ```python
  q = self.cat_queries.weight.unsqueeze(0).expand(batch, -1, -1)  # (B, 12, 768)
  k = v = last_hidden_state  # (B, seq_len, 768)
  attn_out, _ = self.cat_attention(q, k, v, key_padding_mask=~attention_mask.bool())
  attn_out = self.cat_norm(attn_out)  # (B, 12, 768)
  logits = (attn_out * self.category_head.weight.unsqueeze(0)).sum(-1) + self.category_head.bias
  ```
- Output shape vẫn `(batch, 12)` — downstream compatible

`scripts/04a_train_stage1.py`:
- Đọc `use_cat_attention`, thêm attention params (cat_queries, cat_attention, cat_norm) vào `head_lr` group

`configs/stage1_r5_cataware.yaml` (copy từ r4):
```yaml
use_cat_attention: true
use_asl: false
head_lr: 5e-4    # cao hơn cho random-init attention weights
encoder_lr: 1e-5  # thấp hơn để bảo vệ pretrained
ckpt_dir: checkpoints/stage1_r5_cataware
log_path: logs/stage1_r5_cataware_training.jsonl
```

Tests: Thêm `test_cat_attention_forward()` — check shape (B, 12) và loss non-None

### Bundled: Multi-Label Stratified Split (kèm Exp 1, zero cost)

Thay `stratify_key = min(num_cats, 2)` bằng `MultilabelStratifiedShuffleSplit` (package `iterative-stratification`). Đảm bảo mỗi category có val samples tỉ lệ — đặc biệt quan trọng cho RESTAURANT#MISC (hiện chỉ 12 val samples).

`scripts/04a_train_stage1.py`:
```python
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    label_matrix = np.array([r["category_vector"] for r in train_records])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    for train_idx, val_idx in msss.split(label_matrix, label_matrix):
        train_split = [train_records[i] for i in train_idx]
        val_split = [train_records[i] for i in val_idx]
except ImportError:
    # fallback to current stratify method
```

### Kaggle Session 1 Structure (~100 min)

1. Setup + pip install iterative-stratification
2. **Exp 1 (ASL):** Train ~40 min → eval per-category F1
3. **Exp 2 (Cat-Attention):** Train ~50 min → eval per-category F1
4. So sánh Exp1 vs Exp2 vs R4 baseline (per-category breakdown)
5. Save checkpoints + logs

### Quyết định sau Session 1

| Kết quả | Hành động |
|---------|-----------|
| Cat F1 ≥ 0.74 | **Dừng** — đủ tốt cho thesis |
| Cat F1 = 0.71-0.73 | Cân nhắc — tuỳ per-category improvement |
| Cat F1 < 0.71 | **Phase B** — Hierarchical |

Nếu cả 2 experiments improve: chạy thêm **Exp 3 (ASL + Cat-Attention combined)** trong cùng session hoặc session 2.

---

## 4. Phase B: Hierarchical Architecture (Conditional)

### Entity-level distribution (train)

| Entity | Sentences | Attributes | Ghi chú |
|--------|----------|------------|---------|
| FOOD | 757 | QUALITY 90%, PRICES 10.8%, STYLE_OPTIONS 16.9% | 3-way attribute |
| RESTAURANT | 564 | GENERAL 74.6%, MISC 17.2%, PRICES 14.2% | 3-way attribute |
| SERVICE | 419 | GENERAL only | No attribute prediction |
| AMBIENCE | 226 | GENERAL only | No attribute prediction |
| DRINKS | 79 | QUALITY 58.2%, PRICES 25.3%, STYLE_OPTIONS 38.0% | 3-way attribute |
| LOCATION | 28 | GENERAL only | No attribute prediction |

### Architecture

```
DeBERTa encoder → last_hidden_state

Step 1: Entity Head
  [CLS] → ContextPooler → Linear(768, 6) với 6 sigmoid
  Output: predicted entities (multi-label)

Step 2: Attribute Heads (chỉ cho FOOD, RESTAURANT, DRINKS)
  [CLS] → entity-specific Linear(768, 3) với 3 sigmoid
  - food_attr_head: {QUALITY, PRICES, STYLE_OPTIONS}
  - restaurant_attr_head: {GENERAL, MISCELLANEOUS, PRICES}
  - drinks_attr_head: {QUALITY, PRICES, STYLE_OPTIONS}

Training: dùng gold entities để activate attribute heads
Inference: dùng predicted entities

Loss: L_entity (ASL) + L_attribute (ASL, chỉ cho active entities)
```

### Tại sao Hierarchical giải quyết discrimination

- FOOD#STYLE_OPTIONS chỉ cạnh tranh với 2 siblings (QUALITY, PRICES) thay vì 11 categories khác
- Entity prediction dễ hơn: 6 entities rất khác biệt semantically, data per entity nhiều hơn (FOOD: 757 vs FOOD#STYLE_OPTIONS: 128)
- 3-way attribute prediction đơn giản hơn 12-way flat prediction

### Code changes (estimated)

- `src/absa/category_model.py`: Thêm `HierarchicalCategoryDetector` class (hoặc modify `CategoryDetector` với `use_hierarchical=True`)
- `src/data/category_builder.py`: Thêm entity/attribute mapping, build entity labels
- `src/absa/category_trainer.py`: Thêm hierarchical evaluation (entity F1 + attribute F1 + combined)
- `scripts/04a_train_stage1.py`: Support hierarchical config
- `scripts/05_evaluate_joint.py`: Decode hierarchical predictions → 12 categories
- `configs/stage1_hierarchical.yaml`: New config
- Tests: update/add tests cho hierarchical mode

### Reuse từ Phase A

| Component | Reusable? |
|-----------|-----------|
| `AsymmetricLoss` class | **Có** — dùng cho entity head + attribute heads |
| Multi-label stratification | **Có** — áp dụng cho entity-level |
| Cat-Attention module | Không — kiến trúc khác |
| Per-category F1 baseline | **Có** — so sánh |

---

## 5. Verification

### Local (trước mỗi Kaggle session)

```bash
# Tất cả existing tests phải pass
pytest tests/ -v

# Smoke test Stage 1 training (CPU)
python scripts/04a_train_stage1.py --config configs/stage1_r5.yaml --limit 16 --epochs 1
```

### Kaggle

- Per-category F1 breakdown cho cả 3 strategies
- So sánh trực tiếp với R4 baseline trên cùng test set
- NB3 joint eval nếu Cat F1 cải thiện đáng kể
