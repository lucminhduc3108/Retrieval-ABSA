# Báo cáo training — Retrieval-ABSA

**Ngày:** 01/05/2026

---

## Bước 1: Chuẩn bị dữ liệu (`scripts/01_prepare_data.py`)

**Mục tiêu:** Parse XML SemEval 2015 + 2016 (Restaurant domain) → 3 file JSONL phục vụ các bước training sau.

**Cách làm:**
- Parse 4 file XML (2015 train/test + 2016 train/test) bằng `lxml`
- Loại bỏ opinion có `polarity="conflict"` → giữ 3 lớp: positive / negative / neutral
- Với mỗi opinion, tạo 3 loại record:
  - **BIO record** (explicit): gán nhãn `B-ASP`, `I-ASP`, `O` cho từng token dựa trên char offset
  - **Implicit record**: aspect không có target text → tất cả BIO tag = `O`, đánh dấu `implicit: true`
  - **Classification record**: chỉ gồm sentence + aspect_category + polarity (không có BIO)
- Xây dựng **contrastive triplets** từ train split: mỗi triplet gồm anchor / positive (cùng category+polarity) / negative (khác polarity hoặc khác category)

**Kết quả:**

| File output | Số record | Train | Test |
|---|---|---|---|
| `bio_tagging.jsonl` | 5,865 | 4,161 | 1,704 |
| `classification.jsonl` | 5,865 | 4,161 | 1,704 |
| `contrastive_triplets.jsonl` | 4,158 | 4,158 | — |

**Phân bố polarity (cả 2 file):**

| Polarity | Số lượng | Tỉ lệ |
|---|---|---|
| positive | 3,920 | 66.8% |
| negative | 1,702 | 29.0% |
| neutral | 243 | 4.1% |

**Phân bố explicit vs implicit:**

| Loại | Train | Test | Tổng |
|---|---|---|---|
| Explicit (có span) | 3,159 | 1,247 | 4,406 (75%) |
| Implicit (không có span) | 1,002 | 457 | 1,459 (25%) |

**Nhận xét:**
- **Mất cân bằng lớp nghiêm trọng**: positive chiếm 67%, neutral chỉ 4%. Điều này sẽ ảnh hưởng trực tiếp đến sentiment macro F1.
- **Dữ liệu trùng lặp**: SemEval 2015 và 2016 dùng chung nhiều câu → ~2,491 ID trùng (74% record có bản sao). Pipeline hiện tại không deduplicate.
- **13 aspect categories**: FOOD#QUALITY, SERVICE#GENERAL, RESTAURANT#GENERAL, v.v.

---

## Bước 2: Train Contrastive Embedding (`scripts/02_train_embedding.py`)

**Mục tiêu:** Fine-tune DeBERTa-v3-base thành sentence embedder, sao cho các câu cùng (category, polarity) nằm gần nhau trong không gian vector 256 chiều. Mục đích: phục vụ retrieval ở bước 4 — tìm ví dụ tương tự để bổ sung context cho ABSA model.

**Cách làm:**
- **Kiến trúc**: `DeBERTa-v3-base → [CLS] → Linear(768, 256) → GELU → LayerNorm → L2-normalize`
- **Loss**: Symmetric InfoNCE (in-batch negatives + explicit hard negatives), `tau = 0.07`
  - Công thức: `loss = 0.5 * (CE(anchor→positive) + CE(positive→anchor))`
  - Negative gồm: in-batch negatives + 2 explicit negatives mỗi triplet (hard-neg cùng category khác polarity, semi-hard neg khác category)
- **Training config**:
  - Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
  - Scheduler: linear warmup (10%) + linear decay
  - batch_size=16, max_seq_length=128, epochs=10, patience=3
  - fp16 mixed precision (trên Kaggle T4 GPU)
  - Val split: 10% random từ triplets (seed=42)
- **Early stopping**: dựa trên `recall@3` trên validation set

**Metric giải thích — Recall@k:**
- Recall@k đo: "Trong top-k vector gần nhất (theo cosine similarity), có chứa true positive không?"
- Recall@1: top-1 chính xác là positive → rất strict
- Recall@3: true positive nằm trong top-3 → metric chính để early stop
- Recall@5: true positive nằm trong top-5 → dễ hơn
- Đánh giá trên toàn bộ validation set bằng matrix multiplication: `anchors @ positives.T`

**Kết quả training (trên Kaggle T4 GPU, 7 epochs, early stopping patience=3):**

| Epoch | Train Loss | Val Recall@3 | Ghi chú |
|---|---|---|---|
| 1 | 3.1491 | 0.0723 | Best saved |
| 2 | 1.8369 | 0.0795 | Best saved |
| 3 | 1.7049 | 0.0771 | patience 1/3 |
| 4 | 1.6856 | **0.0988** | **Best saved (final best)** |
| 5 | 1.6717 | 0.0916 | patience 1/3 |
| 6 | 1.6735 | 0.0988 | patience 2/3 (tied, not strictly better) |
| 7 | 1.6758 | 0.0892 | patience 3/3 → Early stopping |

- Checkpoint: `checkpoints/embedding/best.pt` (736 MB)
- Đã lưu thành Kaggle dataset: `lcminhc/retrieval-absa-embedding-ckpt`

**Đánh giá kết quả embedding:**
- **Recall@3 = 9.88%** — rất thấp. Nghĩa là chỉ ~10% trường hợp true positive nằm trong top-3 nearest neighbors
- **Loss giảm mạnh epoch 1→2** (3.15 → 1.84) rồi gần như bão hòa (1.70 → 1.68) → model hội tụ nhanh nhưng ở mức thấp
- **Loss không tiếp tục giảm đáng kể** sau epoch 3 → model có thể đã đạt capacity limit hoặc data quality không đủ

**Nguyên nhân recall@3 thấp (~10%):**

| Nguyên nhân có thể | Cách xác thực | Cách cải tiến |
|---|---|---|
| **Evaluation metric quá strict** — recall@3 tính trên toàn bộ val set (416 samples), true positive chỉ có 1 trong 416 candidates | Kiểm tra kích thước val set, tính random baseline (3/416 ≈ 0.7%) | Nếu random baseline ~0.7% thì 9.88% là tốt hơn random ~14 lần |
| **Neutral chỉ có 152 triplet (~3.6%)** → model học kém lớp này | Tính recall@k riêng cho từng polarity trên val set | Data augmentation cho lớp thiểu số (synonym replacement, paraphrase) |
| **Duplicate records tạo "shortcut"** — model chỉ cần nhớ câu thay vì học semantic | Deduplicate trước khi tạo triplet, so sánh recall trước/sau | Deduplicate ở bước 01 |
| **tau=0.07 quá sharp** → gradient bị saturation với hard negatives | Thử tau=0.1 hoặc 0.15, so sánh convergence | Tune temperature |
| **max_seq_length=128 có thể cắt câu dài** | Thống kê % câu bị truncate | Tăng lên 192 nếu GPU cho phép |
| **batch_size=16 quá nhỏ cho InfoNCE** — in-batch negatives ít → loss signal yếu | Thử batch_size=32 hoặc 64 (cần kiểm tra GPU memory) | Tăng batch size hoặc dùng memory bank |

---

## Bước 3: Build FAISS Index (`scripts/03_build_index.py`)

**Mục tiêu:** Encode toàn bộ train-split classification records thành vector 256 chiều, lưu vào FAISS index để retrieval real-time khi train ABSA.

**Cách làm:**
- Load embedding model từ checkpoint
- Encode tất cả 4,161 train records → L2-normalized 256-dim vectors
- Build `faiss.IndexFlatIP` (inner product = cosine similarity vì vector đã L2-normalized)
- Lưu metadata kèm theo: `{id, sentence, aspect_category, polarity, tokens, bio_tags}`

**Kết quả:**
- `indexes/train.faiss`: FAISS index với 4,161 vectors
- `indexes/train_vectors.npy`: raw numpy vectors
- `indexes/train_metadata.jsonl`: metadata cho từng vector

**Nhận xét:**
- IndexFlatIP = brute-force exact search (phù hợp với ~4K vectors, không cần approximate)
- Thời gian encoding: 34.8 giây trên Kaggle T4 GPU
- Kích thước index: 4.3 MB (train.faiss) + 4.3 MB (train_vectors.npy)

---

## Bước 4: Train Multi-task ABSA (`scripts/04_train_absa.py`)

**Mục tiêu:** Train model DeBERTa-v3-base để đồng thời:
1. **BIO tagging**: xác định vị trí aspect term trong câu (B-ASP, I-ASP, O)
2. **Sentiment classification**: phân loại polarity (positive/negative/neutral)

Model được augment bằng retrieved examples từ FAISS index.

**Cách làm:**

*Kiến trúc model (`RetrievalABSA`):*
```
DeBERTa-v3-base
    ├── sequence_output → Linear(768, 3) → BIO logits [B, T, 3]
    └── [CLS] output → Dropout(0.1) → Linear(768, 3) → Sentiment logits [B, 3]
```

*Input format (retrieval-augmented):*
```
[CLS] query_sentence [SEP] aspect_category [SEP] 
  neighbor1_sent [ASP] neighbor1_asp [POL] neighbor1_pol [SEP]
  neighbor2_sent [ASP] neighbor2_asp [POL] neighbor2_pol [SEP]
  neighbor3_sent [ASP] neighbor3_asp [POL] neighbor3_pol [SEP]
```
- max_length = 512 tokens
- query_budget = 100 tokens (query không bao giờ bị cắt trước)
- top_k = 3 neighbors

*Loss function:*
```
L_total = L_bio + 0.5 * L_sentiment
```
- `L_bio`: CrossEntropy(ignore_index=-100) — chỉ tính trên query tokens
- `L_sentiment`: CrossEntropy trên [CLS] token
- `lambda_cls = 0.5` → BIO task được ưu tiên gấp đôi sentiment

*Training config:*
- batch_size=4 (giảm từ 16 do OOM trên T4), fp16 mixed precision
- lr=2e-5, AdamW, linear warmup (10%) + decay
- epochs=10, patience=5, grad_clip=1.0
- Embedding model chuyển sang CPU sau khi tạo dataset → giải phóng ~1GB GPU cho ABSA model
- Val split: 10% random từ train (seed=42)

*Điểm đặc biệt:*
- BIO labels = -100 cho TẤT CẢ token retrieved (không supervise phần retrieved)
- Implicit records: BIO labels = -100 cho toàn bộ → chỉ train sentiment head
- Retrieval xảy ra online trong `__getitem__()` → mỗi epoch đều query FAISS
- Self-exclusion: truyền `query_id` để retriever không trả về chính câu query

**Kết quả training (trên Kaggle T4 GPU):**

Data split: Train 3,745 / Val 416 / Test 1,704

| Epoch | Train Loss | Val Span F1 | Val Sent Acc | Val Joint F1 | Ghi chú |
|---|---|---|---|---|---|
| 1 | 0.5592 | 0.5621 | 0.8654 | 0.4837 | Best saved (span_f1=0.5621) |
| 2 | 0.2410 | 0.7748 | 0.9038 | 0.6907 | Best saved (span_f1=0.7748) |
| 3 | 0.1486 | **0.7765** | **0.9135** | 0.6882 | **Best saved (span_f1=0.7765)** |
| 4 | 0.0954 | 0.7500 | 0.9303 | 0.6970 | patience 1/5 |
| 5 | 0.0694 | 0.7421 | 0.9423 | 0.6855 | patience 2/5 |
| 6 | 0.0440 | 0.7297 | 0.9447 | 0.6689 | patience 3/5 |
| 7 | 0.0322 | 0.7697 | 0.9615 | 0.7003 | patience 4/5 |
| 8 | 0.0264 | 0.7738 | 0.9543 | 0.7016 | patience 5/5 → Early stopping |

- Checkpoint: `checkpoints/absa/best.pt` (701 MB)
- Đã lưu thành Kaggle dataset: `lcminhc/retrieval-absa-ckpt`

**Phân tích xu hướng training:**
- **Train loss giảm liên tục** (0.56 → 0.03) → model đang fit training data tốt
- **Val span F1 đạt peak sớm (epoch 3)**, sau đó dao động nhưng không vượt → dấu hiệu overfit trên span detection
- **Val sentiment acc tăng đều** (0.87 → 0.96) → sentiment head tiếp tục học tốt dù span head đã bão hòa
- **Val joint F1 tăng dần ở epoch 7-8** (0.70-0.70) nhờ sentiment cải thiện, nhưng early stopping dựa trên span_f1 nên không chọn epoch 7-8
- **Lưu ý**: Nếu early stopping dựa trên joint_f1 thay vì span_f1, model tốt nhất có thể là epoch 8 (joint_f1=0.7016 > epoch 3 joint_f1=0.6882)

---

## Bước 5: Evaluation (`scripts/05_evaluate.py`)

**Mục tiêu:** Đánh giá model ABSA trên toàn bộ test set (1,704 records) với 5 metrics.

**Kết quả trên Test Set:**

| Metric | Giá trị | Ý nghĩa |
|---|---|---|
| **BIO Token F1** | 0.6988 | Token-level F1 cho nhãn B-ASP và I-ASP (bỏ qua O). Micro-average. |
| **Span F1** | 0.7088 | Exact-match span F1 — span phải khớp chính xác (start, end) mới tính TP |
| **Sentiment Acc** | 0.9079 | Accuracy phân loại polarity 3 lớp |
| **Sentiment Macro F1** | 0.7619 | Macro-average F1 cho 3 lớp polarity (trung bình F1 mỗi lớp, không cân theo tần suất) |
| **Joint F1** | 0.6379 | **Metric chính** — span + polarity phải đúng cả hai mới tính TP |

---

### Giải thích chi tiết từng metric

**1. BIO Token F1 = 0.6988**
- Đo khả năng gán đúng nhãn BIO ở mức từng token
- Chỉ tính trên B-ASP (1) và I-ASP (2), bỏ qua O (0) và -100
- Token bị gán sai B vs I vẫn tính lỗi
- **Nhận xét**: 70% là khá, nhưng có gap đáng kể so với state-of-the-art (~80-85% trên SemEval datasets)

**2. Span F1 = 0.7088**
- Strict hơn token F1: toàn bộ span (start, end) phải khớp chính xác
- Ví dụ: gold = "grilled salmon" (tokens 3-4), nếu model dự đoán chỉ "salmon" (token 4) → sai
- Span F1 > Token F1 ở đây (0.7088 > 0.6988) — có thể do model predict đúng span boundary nhưng đôi khi gán sai B/I label bên trong span

**3. Sentiment Accuracy = 0.9079**
- 91% accuracy nghe cao, nhưng cần xem xét baseline
- **Baseline naive**: nếu model luôn đoán "positive" → accuracy = 66.8% (do class imbalance)
- Model đạt 91% → tốt hơn baseline 24%

**4. Sentiment Macro F1 = 0.7619**
- Macro F1 trung bình F1 của 3 lớp **không cân theo tần suất**
- Gap lớn giữa accuracy (0.9079) và macro F1 (0.7619) → **model kém ở lớp thiểu số (neutral)**
- Neutral chỉ chiếm 4.1% → model có thể gần như bỏ qua lớp này mà vẫn giữ accuracy cao
- Macro F1 = (F1_pos + F1_neg + F1_neu) / 3 ≈ 0.76 → ước tính F1_neutral có thể chỉ ~0.4-0.5

**5. Joint F1 = 0.6379 (metric quan trọng nhất)**
- Yêu cầu: đúng span (start, end) VÀ đúng polarity → mới tính true positive
- `joint_f1 ≈ span_f1 × P(polarity đúng | span đúng)`
- 0.6379 / 0.7088 ≈ 0.90 → khi model tìm đúng span, ~90% lần nó cũng đoán đúng polarity
- **Bottleneck chính là span detection, không phải sentiment classification**

---

### So sánh Validation vs Test

| Metric | Val (epoch 3) | Val (epoch 8) | Test | Gap (e3→test) |
|---|---|---|---|---|
| Span F1 | 0.7765 | 0.7738 | 0.7088 | -6.8% |
| Sentiment Acc | 0.9135 | 0.9543 | 0.9079 | -0.6% |
| Joint F1 | 0.6882 | 0.7016 | 0.6379 | -5.0% |

- Sentiment gần như không thay đổi → generalize tốt
- Span F1 giảm ~7% → có dấu hiệu overfit nhẹ hoặc distribution shift giữa val/test
- Joint F1 giảm tương ứng (do bị kéo bởi span F1)
- **Lưu ý**: Epoch 8 có val joint_f1 cao nhất (0.7016) nhưng không được chọn vì early stopping theo span_f1. Nếu dùng epoch 8, test joint F1 có thể khác

---

### Phân tích nguyên nhân và hướng cải tiến

#### Vấn đề 1: Span F1 thấp (0.7088) — Bottleneck chính

| Nguyên nhân có thể | Cách xác thực | Cách cải tiến |
|---|---|---|
| **Không có CRF layer** — BIO tagging phụ thuộc vào sequential dependency (B phải đứng trước I), Linear head predict từng token độc lập | So sánh confusion matrix B/I/O, đếm số lần xuất hiện I mà không có B trước đó | Thêm CRF layer lên trên BIO head (đã defer trong MVP) |
| **25% records là implicit** (BIO = all O) — model thấy 25% data không có span nào, có thể bias toward predicting O | Tách evaluation implicit vs explicit, xem span F1 chỉ trên explicit records | Train riêng 2 model hoặc thêm indicator flag cho implicit |
| **Data duplication** (~74% trùng) — model có thể memorize câu thay vì học pattern | Deduplicate, retrain, so sánh kết quả | Deduplicate ở bước 01 |
| **Val/Test distribution shift** (gap 7%) — Val là random 10% từ train, test từ file riêng | Kiểm tra phân bố aspect category và polarity giữa val vs test | Stratified split hoặc cross-validation |
| **Whitespace tokenization** không khớp subword tokenization — BIO alignment có thể lệch | Kiểm tra số record có alignment mismatch trong dataset | Verify alignment logic trong `_align_bio_labels()` |

#### Vấn đề 2: Sentiment Macro F1 thấp hơn Accuracy nhiều (0.76 vs 0.91)

| Nguyên nhân có thể | Cách xác thực | Cách cải tiến |
|---|---|---|
| **Class imbalance nghiêm trọng**: positive 67%, negative 29%, neutral 4% | Tính confusion matrix 3x3, xem precision/recall/F1 từng lớp | Class weights trong loss, oversampling neutral, focal loss |
| **Neutral quá ít** (243/5,865 = 4.1%) — model gần như không thấy neutral examples | Xem F1 riêng cho lớp neutral | Data augmentation cho neutral, hoặc 2-class (pos/neg) rồi post-hoc neutral |
| **Lambda_cls = 0.5** → sentiment loss bị downweight | Thử lambda_cls = 1.0 hoặc 1.5, so sánh | Tune lambda trên val set |

#### Vấn đề 3: Joint F1 (0.6379) — Hệ quả của Span F1

- Joint F1 bị giới hạn bởi span detection (bottleneck)
- Khi span đúng, polarity accuracy ~90% → sentiment classification đã tốt
- **Ưu tiên cải tiến: tập trung vào span detection** (CRF, deduplicate, tách implicit)

---

### Vai trò của Retrieval trong kết quả

**Retrieval augmentation hoạt động như thế nào:**
- Mỗi query lấy 3 neighbors từ FAISS (top_k=3, threshold=0.0)
- Neighbors mang theo labeled polarity (`[POL] positive/negative/neutral`) → "in-context learning" within fine-tuned model
- Model thấy ví dụ tương tự + nhãn polarity → giúp sentiment classification
- BIO head chỉ supervise trên query → retrieved context là context bổ sung, không phải target

**Đánh giá impact của retrieval:**
- Chưa có ablation study (so sánh có retrieval vs không retrieval) → đây là feature defer trong MVP
- Không thể xác định chính xác retrieval đóng góp bao nhiêu % vào kết quả
- **Cách xác thực**: Train model ABSA không có retrieval (chỉ `[CLS] query [SEP] aspect [SEP]`), so sánh joint F1

---

## Tổng kết Pipeline

```
SemEval XML (2015+2016)
    |
    v
[Buoc 1] Parse + BIO tag + Triplets
    |
    |-- bio_tagging.jsonl (5,865)
    |-- classification.jsonl (5,865)
    +-- contrastive_triplets.jsonl (4,158)
            |
            v
[Buoc 2] Train Embedding (InfoNCE, DeBERTa -> 256d)
            |
            v
        best.pt (736 MB)
            |
            v
[Buoc 3] Build FAISS Index (4,161 vectors)
            |
            v
        train.faiss + metadata
            |
            v
[Buoc 4] Train ABSA (BIO + Sentiment, retrieval-augmented)
    |   Early stop epoch 8, best epoch 3
    |   Val: span_f1=0.7765, joint_f1=0.6882
            |
            v
[Buoc 5] Evaluate tren Test (1,704 samples)
    |
    v
+-----------------------------+
| BIO Token F1:     0.6988    |
| Span F1:          0.7088    |
| Sentiment Acc:    0.9079    |
| Sentiment MacF1:  0.7619    |
| Joint F1:         0.6379    |
+-----------------------------+

Bottleneck: Span detection (CRF, deduplicate, implicit handling)
Strength:   Sentiment classification (90% acc khi span dung)
Unknown:    Retrieval contribution (can ablation study)
```
