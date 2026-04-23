# Kế hoạch MVP: Aspect-aware Retrieval + Contrastive Embedding + Multi-task ABSA

> **Dành cho engineer thực thi:** SUB-SKILL BẮT BUỘC — dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi từng task. Các step dùng checkbox (`- [ ]`) để track.

## Context

Plan này bám sát sườn 6 bước trong `FIRST_PLAN.md` và chủ ý **cắt toàn bộ tính năng nâng cao** để ra được khung sườn chạy end-to-end trước. Sau khi MVP chạy xong, các cải tiến (CRF, AMP, differential LR, stratified split, ablation 7 điều kiện, E2E fine-tune) sẽ được thêm ở một plan riêng.

**Quyết định đã chốt:**
- Dataset: SemEval 2015 + 2016 Restaurant (không dùng 2014).
- Multi-task: BIO tagging + Sentiment classification (3 class: positive/negative/neutral, drop `conflict`).
- Viết lại từ đầu, không tái sử dụng code KGAN.
- Ngôn ngữ: prose tiếng Việt, code/identifier/commit tiếng Anh.
- Phạm vi MVP = Step 1 → Step 4 + Step 6 của FIRST_PLAN. Step 5 (E2E fine-tune) và ablation đa điều kiện **không** nằm trong MVP.

**Goal:** Pipeline ABSA end-to-end chạy được trên SemEval 15+16 Restaurant, xuất ra span F1 + sentiment accuracy/F1 + joint F1 trên test split.

**Architecture:** 3 pha tuần tự —
1. Train DeBERTa encoder bằng InfoNCE với hard negatives (cùng aspect, khác polarity).
2. Encode toàn bộ train split, build `faiss.IndexFlatIP`.
3. Train multi-task ABSA (BIO head + sentiment head) với input = query + top-k retrieved context.

**Tech Stack:** PyTorch, HuggingFace Transformers (`microsoft/deberta-v3-base`), FAISS (`faiss-cpu`), scikit-learn, PyYAML, pytest, lxml, tqdm.

**Không dùng trong MVP:** torchcrf, `torch.cuda.amp`, gradient accumulation, differential LR, stratified split, ablation runner.

---

## Cấu trúc project

```
C:\Users\admin\Desktop\Retrieval ABSA\
├── FIRST_PLAN.md
├── PLAN.md
├── README.md
├── .gitignore
├── requirements.txt
│
├── configs/
│   ├── embedding.yaml
│   ├── absa.yaml
│   └── retrieval.yaml
│
├── data/
│   ├── raw/semeval15/{ABSA15_RestaurantsTrain.xml, ABSA15_Restaurants_Test.xml}
│   ├── raw/semeval16/{ABSA16_Restaurants_Train_SB1_v2.xml, EN_REST_SB1_TEST.xml.gold}
│   └── processed/{bio_tagging.jsonl, classification.jsonl, contrastive_triplets.jsonl}
│
├── src/
│   ├── __init__.py
│   ├── data/         (xml_parser, bio_builder, cls_builder, contrastive_builder, datasets)
│   ├── embedding/    (model, loss, trainer)
│   ├── retrieval/    (encoder, index, retriever)
│   ├── absa/         (model, dataset, trainer)
│   ├── evaluation/   (metrics)
│   └── utils/        (io, seed)
│
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_embedding.py
│   ├── 03_build_index.py
│   ├── 04_train_absa.py
│   └── 05_evaluate.py
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/toy_restaurant.xml
│   └── test_*.py
│
├── checkpoints/{embedding,absa}/
├── indexes/  (train.faiss, train_metadata.jsonl, train_vectors.npy)
└── logs/
```

---

## Quyết định kỹ thuật (MVP)

### Step 1 — Data preparation
- SemEval 15/16 XML cùng schema: `<sentence><text>...</text><Opinions><Opinion target=".." category="FOOD#QUALITY" polarity=".." from=".." to=".."/></Opinions></sentence>`.
- `target="NULL"` → record đóng góp cho classification + contrastive, **không** vào BIO.
- Polarity: giữ `positive/negative/neutral`. Drop `conflict`.
- BIO mapping: whitespace-tokenize giữ char offset → token đầu tiên overlap `[from, to)` gán `B-ASP`, các token tiếp theo `I-ASP`.
- Aspect category: giữ nguyên full `Category#Attribute` (ví dụ `FOOD#QUALITY`, `SERVICE#GENERAL`). Không rút gọn.
- Split: train = SemEval15 train + SemEval16 train; test = SemEval15 test + SemEval16 test. Val = 10% random (seed cố định) lấy từ train (không stratified — phần này để giai đoạn cải tiến).
- Contrastive triplets: mỗi anchor ghép 1 positive (cùng `aspect_category` + cùng `polarity`, khác `id`) + 2 negatives: (1) hard negative — cùng `aspect_category` + khác `polarity`; (2) semi-hard negative — khác `aspect_category` + cùng `polarity`. Skip anchor nếu không đủ candidate cho positive hoặc bất kỳ negative nào (log warning). **⚠️ CHƯA QUYẾT:** Data augmentation (Synonym Replacement, Back-Translation, LLM Prompting) cho bucket hiếm thay vì skip — quyết định trước khi implement T4.

### Step 2 — Contrastive embedding
- Backbone `microsoft/deberta-v3-base`. Input `[CLS] sentence [SEP] aspect_category [SEP]`. Pooling = CLS.
- Projection head: `Linear(768, 256) → GELU → LayerNorm(256)`, output L2-normalized.
- Loss InfoNCE, `tau=0.07`. **⚠️ CHƯA QUYẾT:** (A) triplet-only InfoNCE — denominator chỉ gồm 1 positive + 1 hard negative per anchor, hoặc (C) hybrid — denominator gồm in-batch negatives + hard negatives (sim matrix `(B, 2B)`). Quyết định trước khi implement T7.
- Hyperparams: `batch_size=32`, `epochs=10`, `lr=2e-5` (single LR toàn model, không differential), `weight_decay=0.01`, `warmup_ratio=0.1`, `max_seq_length=128`, `grad_clip=1.0`.
- Validation: Recall@{1,3,5} trên val pairs. Save best theo Recall@3. Early stopping patience 3.

### Step 3 — Retrieval index
- ~4000-5000 vector dim 256 → `faiss.IndexFlatIP` đủ. Không cần HNSW.
- Lưu 3 artifact trong `indexes/`: `train.faiss`, `train_metadata.jsonl` (chứa `{id, sentence, tokens, bio_tags, aspect_category, polarity, split}`), `train_vectors.npy`.
- Default `top_k=3`, `threshold=0.0` (không filter theo similarity ở MVP — để giai đoạn cải tiến tune). **Self-exclusion bằng `query_id` vẫn bắt buộc** — đây là correctness, không phải optimization.

### Step 4 — ABSA multi-task model
- Backbone DeBERTa **riêng** (không share với embedding model ở MVP).
- Input: `[CLS] query_sent [SEP] query_aspect [SEP] ret1_sent [ASP] ret1_asp [POL] ret1_pol [SEP] ret2_sent [ASP] ... [SEP]`.
- Budget token (max_length 512): query tối đa 100 token, mỗi retrieved item `(512 - 100) // top_k` token; truncate phần sentence của retrieved khi vượt. Nếu query > 100, giảm budget retrieved (ưu tiên giữ query nguyên vẹn, log warning).
- BIO head: `Linear(768, 3)` (O=0, B-ASP=1, I-ASP=2). **Không CRF** ở MVP.
- Sentiment head: `Dropout(0.1) → Linear(768, 3)` trên `[CLS]`.
- Loss: `L_total = L_bio + 0.5 * L_cls`. BIO dùng `ignore_index=-100` cho `[CLS]`, `[SEP]`, padding, và **toàn bộ token thuộc retrieved portion** (chỉ supervise token của query). Đây là correctness, không phải optimization.
- Training: AdamW single LR `2e-5`, warmup 10%, `grad_clip=1.0`, `batch_size=16`, `epochs=10`. Không grad accumulation, không AMP. Early stopping patience 5 theo span F1 trên val.

### Step 6 — Evaluation (basic, không ablation)
- Metrics: BIO token-level P/R/F1, span exact-match F1, sentiment accuracy + macro-F1, joint F1 (span đúng AND polarity đúng).
- Script `05_evaluate.py`: load best checkpoint, chạy trên test split của BIO dataset (retrieval bật), in bảng metrics ra stdout và ghi `logs/eval_results.md`.
- **Không** chạy 7-condition ablation ở MVP.

---

## Task decomposition (17 tasks)

### T1 — Project scaffold

**Files tạo:**
- `requirements.txt`, `.gitignore`, `README.md`
- `configs/embedding.yaml`, `configs/absa.yaml`, `configs/retrieval.yaml`
- `src/__init__.py` và các `__init__.py` trong `src/{data,embedding,retrieval,absa,evaluation,utils}/`
- `src/utils/io.py`, `src/utils/seed.py`

- [ ] **Step 1:** Tạo `requirements.txt`:
  ```
  torch>=2.1
  transformers>=4.40
  faiss-cpu>=1.8
  scikit-learn>=1.3
  numpy
  pandas
  pyyaml
  pytest
  tqdm
  lxml
  ```
- [ ] **Step 2:** Tạo `.gitignore`:
  ```
  __pycache__/
  *.pyc
  data/raw/
  data/processed/
  checkpoints/
  indexes/
  logs/
  .venv/
  *.pt
  *.faiss
  *.npy
  ```
- [ ] **Step 3:** Tạo các `__init__.py` rỗng cho `src/` và tất cả sub-package.
- [ ] **Step 4:** Tạo `configs/embedding.yaml`:
  ```yaml
  model_name: microsoft/deberta-v3-base
  proj_dim: 256
  tau: 0.07
  batch_size: 32
  epochs: 10
  lr: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_seq_length: 128
  grad_clip: 1.0
  patience: 3
  seed: 42
  triplets_path: data/processed/contrastive_triplets.jsonl
  val_ratio: 0.1
  ckpt_dir: checkpoints/embedding
  log_path: logs/embedding_training.jsonl
  ```
- [ ] **Step 5:** Tạo `configs/absa.yaml`:
  ```yaml
  model_name: microsoft/deberta-v3-base
  num_bio_labels: 3
  num_sent_labels: 3
  lambda_cls: 0.5
  dropout: 0.1
  batch_size: 16
  epochs: 10
  lr: 2.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  max_seq_length: 512
  query_budget: 100
  grad_clip: 1.0
  patience: 5
  seed: 42
  bio_path: data/processed/bio_tagging.jsonl
  val_ratio: 0.1
  ckpt_dir: checkpoints/absa
  log_path: logs/absa_training.jsonl
  ```
- [ ] **Step 6:** Tạo `configs/retrieval.yaml`:
  ```yaml
  top_k: 3
  threshold: 0.0
  index_dir: indexes
  ```
- [ ] **Step 7:** Viết `src/utils/io.py` với `read_jsonl(path)`, `write_jsonl(records, path)`, `load_yaml(path)`.
- [ ] **Step 8:** Viết `src/utils/seed.py` với `set_seed(seed)` cover `random`, `numpy`, `torch`, `torch.cuda`.
- [ ] **Step 9:** Commit: `chore: initialize project structure, configs, and utilities`.

---

### T2 — SemEval XML parser

**Files:**
- Create: `src/data/xml_parser.py`, `tests/fixtures/toy_restaurant.xml`, `tests/conftest.py`, `tests/test_xml_parser.py`

**Interface:**
```python
def parse_semeval_xml(path: str) -> list[dict]:
    """Returns [{sentence_id, text, opinions: [{target, category, polarity, from_char, to_char}]}].
    target is None for NULL. Opinions with polarity='conflict' are dropped."""
```

- [ ] **Step 1: Viết fixture `tests/fixtures/toy_restaurant.xml`** — 3 sentence: (a) 1 opinion explicit với `target`, `from`, `to` hợp lệ; (b) 1 opinion `target="NULL"`; (c) 1 opinion `polarity="conflict"`.
- [ ] **Step 2: Viết test `tests/test_xml_parser.py`:**
  ```python
  from src.data.xml_parser import parse_semeval_xml

  def test_parse_returns_three_sentences():
      out = parse_semeval_xml("tests/fixtures/toy_restaurant.xml")
      assert len(out) == 3

  def test_null_target_is_none():
      out = parse_semeval_xml("tests/fixtures/toy_restaurant.xml")
      null_sent = next(s for s in out if s["sentence_id"] == "null_case")
      assert null_sent["opinions"][0]["target"] is None

  def test_conflict_is_dropped():
      out = parse_semeval_xml("tests/fixtures/toy_restaurant.xml")
      conflict_sent = next(s for s in out if s["sentence_id"] == "conflict_case")
      assert conflict_sent["opinions"] == []
  ```
- [ ] **Step 3:** Chạy `pytest tests/test_xml_parser.py -v` → Expected: FAIL (`ModuleNotFoundError`).
- [ ] **Step 4:** Implement `src/data/xml_parser.py` dùng `lxml.etree`. Giữ nguyên category full (ví dụ `FOOD#QUALITY`). `target="NULL"` → `None`. Skip opinion có polarity `conflict`.
- [ ] **Step 5:** Chạy lại test → Expected: PASS.
- [ ] **Step 6:** Commit: `feat(data): add SemEval 15/16 XML parser with NULL and conflict handling`.

---

### T3 — BIO tag builder

**Files:**
- Create: `src/data/bio_builder.py`, `tests/test_bio_builder.py`

**Interface:**
```python
def build_bio_records(parsed: list[dict], split: str) -> list[dict]:
    """Converts char-offset opinions to word-level BIO. Skips NULL targets.
    Returns one record per non-NULL opinion."""
```

- [ ] **Step 1: Viết test:**
  ```python
  from src.data.bio_builder import build_bio_records

  def test_single_word_span():
      parsed = [{"sentence_id": "s1", "text": "The food is great",
                 "opinions": [{"target": "food", "category": "FOOD",
                              "polarity": "positive", "from_char": 4, "to_char": 8}]}]
      recs = build_bio_records(parsed, split="train")
      assert len(recs) == 1
      assert recs[0]["tokens"] == ["The", "food", "is", "great"]
      assert recs[0]["bio_tags"] == ["O", "B-ASP", "O", "O"]
      assert recs[0]["split"] == "train"

  def test_multi_word_span():
      parsed = [{"sentence_id": "s2", "text": "The pad thai is nice",
                 "opinions": [{"target": "pad thai", "category": "FOOD",
                              "polarity": "positive", "from_char": 4, "to_char": 12}]}]
      recs = build_bio_records(parsed, split="train")
      assert recs[0]["bio_tags"] == ["O", "B-ASP", "I-ASP", "O", "O"]

  def test_null_target_skipped():
      parsed = [{"sentence_id": "s3", "text": "The restaurant is fine",
                 "opinions": [{"target": None, "category": "RESTAURANT",
                              "polarity": "positive", "from_char": 0, "to_char": 0}]}]
      assert build_bio_records(parsed, split="train") == []
  ```
- [ ] **Step 2:** Chạy test → FAIL.
- [ ] **Step 3:** Implement: whitespace tokenize giữ char offset, gán tag bằng linear scan. Mỗi opinion không-NULL sinh 1 record `{id, sentence, tokens, bio_tags, aspect_category, polarity, split}`. `id` = `f"{sentence_id}_op{opinion_idx}"`.
- [ ] **Step 4:** Chạy lại → PASS.
- [ ] **Step 5:** Commit: `feat(data): character-offset to word-level BIO tag converter`.

---

### T4 — Classification records + contrastive triplets

**Files:**
- Create: `src/data/cls_builder.py`, `src/data/contrastive_builder.py`, `tests/test_cls_builder.py`, `tests/test_contrastive_builder.py`

**Interfaces:**
```python
def build_cls_records(parsed: list[dict], split: str) -> list[dict]:
    """One record per opinion (including NULL target).
    Returns [{id, sentence, aspect_category, polarity, split}]."""

def build_contrastive_triplets(cls_records: list[dict], seed: int = 42) -> list[dict]:
    """For each anchor, sample 1 positive (same aspect + same polarity, different id)
    and 2 negatives:
      neg1 (hard): same aspect + different polarity
      neg2 (semi-hard): different aspect + same polarity
    Returns [{anchor_id, anchor_sentence, anchor_aspect, anchor_polarity,
              positive_id, positive_sentence, positive_aspect, positive_polarity,
              neg1_id, neg1_sentence, neg1_aspect, neg1_polarity,
              neg2_id, neg2_sentence, neg2_aspect, neg2_polarity}].
    Skip anchor if no candidate for positive OR any negative (log warning)."""
```

- [ ] **Step 1: Viết `tests/test_cls_builder.py`:**
  ```python
  from src.data.cls_builder import build_cls_records

  def test_one_record_per_opinion_including_null():
      parsed = [{"sentence_id": "s1", "text": "Great food and fast service",
                 "opinions": [
                     {"target": "food", "category": "FOOD#QUALITY", "polarity": "positive",
                      "from_char": 6, "to_char": 10},
                     {"target": None, "category": "SERVICE#GENERAL", "polarity": "positive",
                      "from_char": 0, "to_char": 0},
                 ]}]
      recs = build_cls_records(parsed, split="train")
      assert len(recs) == 2
      assert recs[1]["aspect_category"] == "SERVICE#GENERAL"
      assert recs[1]["split"] == "train"
  ```
- [ ] **Step 2: Viết `tests/test_contrastive_builder.py`:**
  ```python
  from src.data.contrastive_builder import build_contrastive_triplets

  def _mk(i, asp, pol):
      return {"id": f"r{i}", "sentence": f"s{i}", "aspect_category": asp,
              "polarity": pol, "split": "train"}

  def test_positive_shares_aspect_and_polarity():
      recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
              _mk(2, "FOOD#QUALITY", "negative"), _mk(3, "SERVICE#GENERAL", "positive")]
      triplets = build_contrastive_triplets(recs, seed=0)
      for t in triplets:
          assert t["anchor_aspect"] == t["positive_aspect"]
          assert t["anchor_polarity"] == t["positive_polarity"]
          assert t["anchor_id"] != t["positive_id"]

  def test_hard_neg_same_aspect_different_polarity():
      recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
              _mk(2, "FOOD#QUALITY", "negative"), _mk(3, "SERVICE#GENERAL", "positive")]
      triplets = build_contrastive_triplets(recs, seed=0)
      for t in triplets:
          assert t["anchor_aspect"] == t["neg1_aspect"]
          assert t["anchor_polarity"] != t["neg1_polarity"]
          assert t["anchor_id"] != t["neg1_id"]

  def test_semi_hard_neg_different_aspect_same_polarity():
      recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
              _mk(2, "FOOD#QUALITY", "negative"), _mk(3, "SERVICE#GENERAL", "positive")]
      triplets = build_contrastive_triplets(recs, seed=0)
      for t in triplets:
          assert t["anchor_aspect"] != t["neg2_aspect"]
          assert t["anchor_polarity"] == t["neg2_polarity"]
          assert t["anchor_id"] != t["neg2_id"]

  def test_skip_when_no_positive_candidate():
      recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "negative")]
      triplets = build_contrastive_triplets(recs, seed=0)
      assert triplets == []

  def test_skip_when_no_hard_neg_candidate():
      recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
              _mk(2, "SERVICE#GENERAL", "positive")]
      triplets = build_contrastive_triplets(recs, seed=0)
      assert triplets == []

  def test_skip_when_no_semi_hard_neg_candidate():
      recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
              _mk(2, "FOOD#QUALITY", "negative")]
      triplets = build_contrastive_triplets(recs, seed=0)
      assert triplets == []
  ```
- [ ] **Step 3:** Chạy 2 test file → FAIL.
- [ ] **Step 4:** Implement `build_cls_records` (trivial loop). Implement `build_contrastive_triplets`: index records theo `(aspect, polarity)`; với mỗi anchor, positive bucket = cùng `(aspect, polarity)` loại chính nó; neg1 bucket (hard) = cùng `aspect` + khác `polarity`; neg2 bucket (semi-hard) = khác `aspect` + cùng `polarity`; skip nếu bất kỳ bucket nào rỗng (log warning).
- [ ] **Step 5:** Chạy lại → PASS.
- [ ] **Step 6:** Commit: `feat(data): classification records and contrastive triplets with hard negatives`.

---

### T5 — Data preparation script

**Files:**
- Create: `scripts/01_prepare_data.py`

- [ ] **Step 1:** Viết CLI: `python scripts/01_prepare_data.py --raw_dir data/raw --out_dir data/processed`.
- [ ] **Step 2:** Script logic:
  - Parse 4 XML (15 train/test, 16 train/test), gộp thành list parsed với `split` label đúng (train cho train files, test cho test files).
  - Gọi `build_bio_records`, `build_cls_records`, `build_contrastive_triplets` (chỉ tạo triplet từ record `split=="train"`).
  - Ghi 3 jsonl vào `out_dir`: `bio_tagging.jsonl`, `classification.jsonl`, `contrastive_triplets.jsonl`.
  - Print counts: `#sentence`, `#opinion`, `#bio_records`, `#cls_records`, `#contrastive_triplets`.
- [ ] **Step 3: Manual verify** — chạy script trên toy XML fixture (copy fixture thành giả 4 file) và assert 3 file jsonl non-empty. Không cần unit test riêng (script là orchestration).
- [ ] **Step 4:** Commit: `feat(scripts): end-to-end data preparation pipeline`.

---

### T6 — ContrastiveEmbedder

**Files:**
- Create: `src/embedding/model.py`, `tests/test_embedding_model.py`

**Interface:**
```python
class ContrastiveEmbedder(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 proj_dim: int = 256, dropout: float = 0.1): ...
    def encode(self, input_ids, attention_mask) -> torch.Tensor:
        """Returns L2-normalized (B, proj_dim)."""
    def forward(self, anchor_ids, anchor_mask, pos_ids, pos_mask,
                 neg1_ids=None, neg1_mask=None,
                 neg2_ids=None, neg2_mask=None) -> dict:
        """Returns {'anchor_vecs': (B, D), 'pos_vecs': (B, D),
                    'neg1_vecs': (B, D) | None, 'neg2_vecs': (B, D) | None}."""
```

- [ ] **Step 1: Viết test:**
  ```python
  import torch
  from src.embedding.model import ContrastiveEmbedder

  def test_encode_returns_normalized_vectors():
      m = ContrastiveEmbedder(proj_dim=256)
      ids = torch.randint(0, 1000, (2, 16))
      mask = torch.ones_like(ids)
      out = m.encode(ids, mask)
      assert out.shape == (2, 256)
      norms = out.norm(dim=1)
      assert torch.allclose(norms, torch.ones(2), atol=1e-4)
  ```
- [ ] **Step 2:** Chạy → FAIL.
- [ ] **Step 3:** Implement: `AutoModel.from_pretrained(model_name)`, CLS pooling (`last_hidden_state[:, 0]`), projection head `Linear(hidden, proj_dim) → GELU → LayerNorm(proj_dim)`, `F.normalize(x, p=2, dim=-1)` cuối cùng.
- [ ] **Step 4:** Chạy lại → PASS (có thể cần download weight; nếu CI offline, mock bằng `AutoConfig` + `AutoModel.from_config`).
- [ ] **Step 5:** Commit: `feat(embedding): ContrastiveEmbedder with DeBERTa and L2-normed projection`.

---

### T7 — InfoNCE loss with hard negatives

**⚠️ CHƯA QUYẾT:** Implementation phụ thuộc vào lựa chọn phương án loss:
- **(A) Triplet-only:** `sim = cat([sim(a,p), sim(a,n1), sim(a,n2)], dim=1)` shape `(B, 3)` — mỗi anchor chỉ thấy 1 positive + 2 negatives.
- **(C) Hybrid:** `sim = cat([a @ p.T, a @ n1.T, a @ n2.T], dim=1)` shape `(B, 3B)` — mỗi anchor thấy 1 positive + (B-1) in-batch negatives + 2B hard/semi-hard negatives.
Quyết định trước khi implement. Code dưới đây viết theo phương án C (sẽ cập nhật nếu chọn A).

**Files:**
- Create: `src/embedding/loss.py`, `tests/test_infonce_loss.py`

**Interface:**
```python
def infonce_loss(anchor: torch.Tensor, positive: torch.Tensor,
                 negatives: list[torch.Tensor] | None = None, tau: float = 0.07) -> torch.Tensor:
    """InfoNCE with optional hard negatives. Inputs must be L2-normalized (B, D).
    negatives is a list of negative tensors (e.g. [neg1_vecs, neg2_vecs]),
    each appended as additional columns in the similarity matrix."""
```

- [ ] **Step 1: Viết test:**
  ```python
  import torch
  import torch.nn.functional as F
  from src.embedding.loss import infonce_loss

  def test_loss_is_positive_scalar():
      torch.manual_seed(0)
      a = F.normalize(torch.randn(4, 8), dim=-1)
      p = F.normalize(torch.randn(4, 8), dim=-1)
      loss = infonce_loss(a, p, tau=0.07)
      assert loss.dim() == 0
      assert loss.item() > 0

  def test_identical_inputs_give_low_loss():
      a = F.normalize(torch.randn(4, 8), dim=-1)
      loss_same = infonce_loss(a, a, tau=0.07)
      loss_rand = infonce_loss(a, F.normalize(torch.randn(4, 8), dim=-1), tau=0.07)
      assert loss_same < loss_rand

  def test_hard_negatives_increase_loss():
      torch.manual_seed(0)
      a = F.normalize(torch.randn(4, 8), dim=-1)
      p = a.clone()
      n1 = F.normalize(torch.randn(4, 8), dim=-1)
      n2 = F.normalize(torch.randn(4, 8), dim=-1)
      loss_no_neg = infonce_loss(a, p, tau=0.07)
      loss_one_neg = infonce_loss(a, p, negatives=[n1], tau=0.07)
      loss_two_neg = infonce_loss(a, p, negatives=[n1, n2], tau=0.07)
      assert loss_one_neg >= loss_no_neg
      assert loss_two_neg >= loss_one_neg
  ```
- [ ] **Step 2:** Chạy → FAIL.
- [ ] **Step 3:** Implement:
  ```python
  import torch
  import torch.nn.functional as F

  def infonce_loss(anchor, positive, negatives=None, tau=0.07):
      sim_ap = (anchor @ positive.T) / tau
      if negatives:
          neg_sims = [anchor @ n.T / tau for n in negatives]
          sim = torch.cat([sim_ap] + neg_sims, dim=1)
      else:
          sim = sim_ap
      labels = torch.arange(anchor.size(0), device=sim.device)
      loss_a = F.cross_entropy(sim, labels)
      loss_p = F.cross_entropy(sim_ap.T, labels)
      return 0.5 * (loss_a + loss_p)
  ```
- [ ] **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit: `feat(embedding): InfoNCE loss with hard negative support`.

---

### T8 — Contrastive dataset + trainer

**Files:**
- Create: `src/data/datasets.py` (class `ContrastiveTripletDataset`), `src/embedding/trainer.py`, `tests/test_embedding_trainer.py`

**Interfaces:**
```python
class ContrastiveTripletDataset(Dataset):
    def __init__(self, triplets_path: str, tokenizer, max_length: int = 128): ...
    # __getitem__ returns {anchor_input_ids, anchor_attention_mask,
    #                      pos_input_ids, pos_attention_mask,
    #                      neg1_input_ids, neg1_attention_mask,
    #                      neg2_input_ids, neg2_attention_mask}

class ContrastiveTrainer:
    def __init__(self, model, optimizer, scheduler, tau, device, log_path, grad_clip=1.0): ...
    def train(self, train_loader, val_loader, epochs, patience=3) -> list[dict]: ...
    def evaluate_recall(self, val_loader, k_list=(1, 3, 5)) -> dict: ...
```

- [ ] **Step 1: Viết test dataset:**
  ```python
  from transformers import AutoTokenizer
  from src.data.datasets import ContrastiveTripletDataset
  from src.utils.io import write_jsonl

  def test_dataset_item_has_expected_keys(tmp_path):
      p = tmp_path / "triplets.jsonl"
      write_jsonl([{"anchor_id": "a", "anchor_sentence": "food is good",
                    "anchor_aspect": "FOOD#QUALITY", "anchor_polarity": "positive",
                    "positive_id": "b", "positive_sentence": "great food",
                    "positive_aspect": "FOOD#QUALITY", "positive_polarity": "positive",
                    "neg1_id": "c", "neg1_sentence": "food was bad",
                    "neg1_aspect": "FOOD#QUALITY", "neg1_polarity": "negative",
                    "neg2_id": "d", "neg2_sentence": "great service",
                    "neg2_aspect": "SERVICE#GENERAL", "neg2_polarity": "positive"}], str(p))
      tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
      ds = ContrastiveTripletDataset(str(p), tok, max_length=32)
      item = ds[0]
      assert set(item.keys()) == {"anchor_input_ids", "anchor_attention_mask",
                                  "pos_input_ids", "pos_attention_mask",
                                  "neg1_input_ids", "neg1_attention_mask",
                                  "neg2_input_ids", "neg2_attention_mask"}
  ```
- [ ] **Step 2: Viết test trainer (smoke):**
  ```python
  def test_one_step_decreases_loss(tmp_path):
      # Setup 4 triplet với anchor == positive, negative random → loss giảm sau 1 step
      ...
  ```
- [ ] **Step 3:** Chạy → FAIL.
- [ ] **Step 4:** Implement dataset: tokenize anchor với `[CLS] sentence [SEP] aspect [SEP]`, tokenize positive, neg1, neg2 tương tự, return tensor.
- [ ] **Step 5:** Implement trainer: forward `model(...)` → `infonce_loss(anchor_vecs, pos_vecs, negatives=[neg1_vecs, neg2_vecs], tau)`, backward, `clip_grad_norm_`, `optimizer.step`, `scheduler.step`. Log per epoch vào jsonl. `evaluate_recall`: encode toàn bộ val anchor và positive, build similarity matrix `(N, N)`, cho mỗi anchor check top-k có chứa index của positive không, return dict `{'recall@1': ..., 'recall@3': ..., 'recall@5': ...}`. Early stopping theo Recall@3.
- [ ] **Step 6:** Chạy → PASS.
- [ ] **Step 7:** Commit: `feat(embedding): contrastive triplet dataset and trainer with Recall@k validation`.

---

### T9 — Embedding training script

**Files:**
- Create: `scripts/02_train_embedding.py`

- [ ] **Step 1:** CLI: `python scripts/02_train_embedding.py --config configs/embedding.yaml`.
- [ ] **Step 2:** Script logic:
  - `load_yaml(config)`, `set_seed(cfg.seed)`.
  - Load tokenizer + model (`ContrastiveEmbedder`).
  - Load `contrastive_triplets.jsonl`, random split train/val với tỉ lệ `val_ratio` (seed cố định).
  - Create `ContrastivePairDataset` cho train/val, `DataLoader(batch_size=cfg.batch_size, shuffle=True)`.
  - AdamW, linear warmup scheduler (`get_linear_schedule_with_warmup`).
  - `ContrastiveTrainer.train(...)`, save best `model.state_dict()` vào `{ckpt_dir}/best.pt` theo Recall@3.
- [ ] **Step 3: Manual smoke test** — thêm flag `--limit N` giới hạn số pair và `--epochs 1` để chạy nhanh xác nhận không crash.
- [ ] **Step 4:** Commit: `feat(scripts): contrastive embedding training entry point`.

---

### T10 — FAISS index builder

**Files:**
- Create: `src/retrieval/encoder.py`, `src/retrieval/index.py`, `tests/test_index.py`

**Interfaces:**
```python
def encode_records(records: list[dict], model: ContrastiveEmbedder, tokenizer,
                   batch_size: int = 64, device: str = "cuda") -> np.ndarray:
    """Returns float32 (N, proj_dim), L2-normalized."""

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP: ...
def save_index(index, metadata: list[dict], out_dir: str): ...
def load_index(out_dir: str) -> tuple[faiss.IndexFlatIP, list[dict]]: ...
```

- [ ] **Step 1: Viết test:**
  ```python
  import numpy as np, faiss
  from src.retrieval.index import build_index, save_index, load_index

  def test_self_retrieval(tmp_path):
      rng = np.random.default_rng(0)
      v = rng.standard_normal((10, 8)).astype("float32")
      v = v / np.linalg.norm(v, axis=1, keepdims=True)
      idx = build_index(v)
      D, I = idx.search(v[:1], 1)
      assert I[0, 0] == 0

  def test_roundtrip(tmp_path):
      v = np.eye(4, dtype="float32")
      idx = build_index(v)
      meta = [{"id": f"r{i}"} for i in range(4)]
      save_index(idx, meta, str(tmp_path))
      idx2, meta2 = load_index(str(tmp_path))
      assert meta2 == meta
      assert idx2.ntotal == 4
  ```
- [ ] **Step 2:** Chạy → FAIL.
- [ ] **Step 3:** Implement `build_index`: `faiss.normalize_L2(v)` (safety, idempotent cho vector đã normed), `idx = faiss.IndexFlatIP(dim); idx.add(v); return idx`. Implement `save_index`/`load_index` dùng `faiss.write_index` + ghi `train_metadata.jsonl` + `train_vectors.npy`. Implement `encode_records`: tokenize từng batch, `model.encode(...)`, concat, cast `float32`.
- [ ] **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit: `feat(retrieval): FAISS IndexFlatIP builder and persistence`.

---

### T11 — Retriever với self-exclusion

**Files:**
- Create: `src/retrieval/retriever.py`, `tests/test_retriever.py`

**Interface:**
```python
class Retriever:
    def __init__(self, index, metadata: list[dict], top_k: int = 3,
                 threshold: float = 0.0): ...
    def retrieve(self, query_vec: np.ndarray, query_id: str | None = None) -> list[dict]:
        """Returns up to top_k metadata dicts, excluding query_id, filtered by cos_sim >= threshold."""
```

- [ ] **Step 1: Viết test:**
  ```python
  import numpy as np, faiss
  from src.retrieval.retriever import Retriever
  from src.retrieval.index import build_index

  def _setup():
      v = np.eye(4, dtype="float32")
      meta = [{"id": f"r{i}", "sentence": f"s{i}"} for i in range(4)]
      return Retriever(build_index(v), meta, top_k=2)

  def test_self_excluded():
      r = _setup()
      out = r.retrieve(np.eye(4, dtype="float32")[0:1], query_id="r0")
      ids = [m["id"] for m in out]
      assert "r0" not in ids

  def test_no_exclusion_when_id_none():
      r = _setup()
      out = r.retrieve(np.eye(4, dtype="float32")[0:1], query_id=None)
      assert any(m["id"] == "r0" for m in out)

  def test_threshold_filters_all():
      r = Retriever(build_index(np.eye(4, dtype="float32")),
                    [{"id": f"r{i}"} for i in range(4)],
                    top_k=2, threshold=1.01)
      assert r.retrieve(np.eye(4, dtype="float32")[0:1], query_id=None) == []
  ```
- [ ] **Step 2:** Chạy → FAIL.
- [ ] **Step 3:** Implement: search `top_k + 1`, loop kết quả, skip item có `metadata[i]["id"] == query_id`, filter theo `score >= threshold`, trả về tối đa `top_k`.
- [ ] **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit: `feat(retrieval): Retriever with self-exclusion and similarity threshold`.

---

### T12 — Index build script

**Files:**
- Create: `scripts/03_build_index.py`

- [ ] **Step 1:** CLI: `python scripts/03_build_index.py --embedding_ckpt checkpoints/embedding/best.pt --input data/processed/classification.jsonl --out_dir indexes/`.
- [ ] **Step 2:** Script logic:
  - Load tokenizer + `ContrastiveEmbedder`, `load_state_dict` từ checkpoint, `eval()`.
  - Load `classification.jsonl`, filter `split == "train"`.
  - **Join BIO info:** load `bio_tagging.jsonl` và build map `id → (tokens, bio_tags)` để embedding metadata có sẵn BIO cho dataset retrieval (các record NULL sẽ không có BIO → field để `None`).
  - `encode_records(...)` → vectors.
  - `build_index(vectors)` → `save_index(...)` ghi `train.faiss`, `train_metadata.jsonl`, `train_vectors.npy` trong `out_dir`.
  - Log counts + thời gian + kích thước file.
- [ ] **Step 3:** Commit: `feat(scripts): FAISS index build script`.

---

### T13 — Evaluation metrics

**Files:**
- Create: `src/evaluation/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
```python
def bio_token_metrics(pred_seqs: list[list[int]], gold_seqs: list[list[int]]) -> dict:
    """Token-level P/R/F1 over labels {1:B-ASP, 2:I-ASP}. Ignore -100 positions."""

def extract_spans(bio_seq: list[int]) -> list[tuple[int, int]]:
    """Converts BIO ids (0=O, 1=B, 2=I) to list of (start, end) spans (end exclusive)."""

def span_f1(pred_spans: list[list[tuple]], gold_spans: list[list[tuple]]) -> dict: ...

def sentiment_metrics(preds: list[int], golds: list[int]) -> dict: ...

def joint_f1(pred_with_pol: list[list[tuple]], gold_with_pol: list[list[tuple]]) -> float:
    """Each element: list of (start, end, polarity_id). Micro-F1 over exact (span+pol) match."""
```

- [ ] **Step 1: Viết test:**
  ```python
  from src.evaluation.metrics import (
      bio_token_metrics, extract_spans, span_f1, sentiment_metrics, joint_f1)

  def test_extract_spans_basic():
      assert extract_spans([0, 1, 2, 0, 1, 0]) == [(1, 3), (4, 5)]

  def test_bio_token_perfect():
      m = bio_token_metrics([[0, 1, 2, 0]], [[0, 1, 2, 0]])
      assert m["f1"] == 1.0

  def test_bio_token_ignore_index():
      m = bio_token_metrics([[0, 1, 2, 0]], [[-100, 1, 2, 0]])
      assert m["f1"] == 1.0  # -100 bị bỏ qua

  def test_span_f1_partial():
      m = span_f1([[(1, 3), (4, 5)]], [[(1, 3), (4, 6)]])
      assert 0 < m["f1"] < 1

  def test_sentiment_metrics_keys():
      m = sentiment_metrics([0, 1, 2], [0, 1, 2])
      assert m["accuracy"] == 1.0
      assert m["macro_f1"] == 1.0

  def test_joint_f1_requires_both():
      assert joint_f1([[(1, 3, 0)]], [[(1, 3, 1)]]) == 0.0
      assert joint_f1([[(1, 3, 1)]], [[(1, 3, 1)]]) == 1.0
  ```
- [ ] **Step 2:** Chạy → FAIL.
- [ ] **Step 3:** Implement:
  - `bio_token_metrics`: flatten, filter `-100`, treat label 0 = O, 1 = B, 2 = I; P/R/F1 trên positive class (`labels=[1,2]`, `average='micro'`) bằng `sklearn.metrics.precision_recall_fscore_support`.
  - `extract_spans`: linear scan, `B` mở span, `I` extend, `O` đóng.
  - `span_f1`: `pred_set`, `gold_set` gom từ tất cả sentence, micro F1 = `2·TP / (|pred| + |gold|)`.
  - `sentiment_metrics`: `accuracy_score`, `f1_score(..., average='macro')`.
  - `joint_f1`: tương tự `span_f1` nhưng key `(sent_idx, start, end, polarity)`.
- [ ] **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit: `feat(evaluation): token, span, sentiment, and joint F1 metrics`.

---

### T14 — RetrievalABSA multi-task model

**Files:**
- Create: `src/absa/model.py`, `tests/test_absa_model.py`

**Interface:**
```python
class RetrievalABSA(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_bio_labels: int = 3, num_sent_labels: int = 3,
                 lambda_cls: float = 0.5, dropout: float = 0.1): ...
    def forward(self, input_ids, attention_mask,
                bio_labels=None, sentiment_label=None) -> dict:
        """Returns {'bio_logits': (B, L, 3), 'sentiment_logits': (B, 3),
                    'loss': scalar | None, 'loss_bio': ..., 'loss_cls': ...}."""
```

- [ ] **Step 1: Viết test:**
  ```python
  import torch
  from src.absa.model import RetrievalABSA

  def test_forward_shapes():
      m = RetrievalABSA()
      ids = torch.randint(0, 1000, (2, 32))
      mask = torch.ones_like(ids)
      out = m(ids, mask)
      assert out["bio_logits"].shape == (2, 32, 3)
      assert out["sentiment_logits"].shape == (2, 3)
      assert out["loss"] is None

  def test_forward_with_labels_returns_scalar_loss():
      m = RetrievalABSA()
      ids = torch.randint(0, 1000, (2, 32))
      mask = torch.ones_like(ids)
      bio = torch.zeros(2, 32, dtype=torch.long)
      sent = torch.zeros(2, dtype=torch.long)
      out = m(ids, mask, bio_labels=bio, sentiment_label=sent)
      assert out["loss"].dim() == 0
      assert out["loss"].item() > 0
  ```
- [ ] **Step 2:** Chạy → FAIL.
- [ ] **Step 3:** Implement: `AutoModel`, BIO head `Linear(hidden, 3)`, sentiment head `Dropout → Linear(hidden, 3)`. CE cho BIO dùng `ignore_index=-100`, CE cho sentiment mặc định. `loss = loss_bio + lambda_cls * loss_cls`.
- [ ] **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit: `feat(absa): multi-task RetrievalABSA model with BIO and sentiment heads`.

---

### T15 — RetrievalABSADataset

**Files:**
- Create: `src/absa/dataset.py`, `tests/test_absa_dataset.py`

**Interface:**
```python
class RetrievalABSADataset(Dataset):
    def __init__(self, bio_records: list[dict], retriever: "Retriever | None",
                 tokenizer, embedding_model: "ContrastiveEmbedder | None",
                 max_length: int = 512, query_budget: int = 100,
                 top_k: int = 3, device: str = "cpu"): ...
    def __getitem__(self, idx) -> dict:
        """Keys: input_ids, attention_mask, bio_labels (-100 for non-query/special tokens),
                 sentiment_label, query_id."""
```

Polarity → id map: `{"positive": 0, "negative": 1, "neutral": 2}`. BIO → id map: `{"O": 0, "B-ASP": 1, "I-ASP": 2}`.

- [ ] **Step 1: Viết test:**
  ```python
  def test_no_retriever_behaves_like_plain_tokenization(...):
      # retriever=None, top_k=0 → no retrieved tokens
      ...

  def test_retrieved_tokens_have_ignore_label(...):
      # mock retriever trả 2 record cố định, assert bio_labels cho retrieved portion == -100
      ...

  def test_query_id_passed_to_retriever(...):
      # spy mock retriever, assert nhận đúng query_id
      ...
  ```
- [ ] **Step 2:** Chạy → FAIL.
- [ ] **Step 3:** Implement:
  1. Với mỗi record, encode query qua `embedding_model.encode(...)` (nếu retriever bật) → gọi `retriever.retrieve(vec, query_id=record["id"])`.
  2. Tokenize query `[CLS] sentence [SEP] aspect [SEP]`, đồng thời align BIO label token→subword (sử dụng `return_offsets_mapping=True` hoặc manual align). Sau tokenize, `bio_labels` cho token của `sentence` bằng id BIO tương ứng, còn lại (`[CLS]`, `[SEP]`, aspect tokens, padding) = `-100`.
  3. Nếu query_token > `query_budget` → cảnh báo, shrink budget retrieved thay vì cắt query.
  4. Với mỗi retrieved item, tokenize `ret_sent [ASP] ret_asp [POL] ret_pol [SEP]`, truncate sao cho tổng ≤ `(max_length - len(query_tokens)) // top_k`.
  5. Concat query tokens + retrieved tokens, pad/truncate đến `max_length`. `bio_labels` của retrieved portion đều `-100`.
  6. `sentiment_label` = id của `record["polarity"]`.
- [ ] **Step 4:** Chạy → PASS.
- [ ] **Step 5:** Commit: `feat(absa): retrieval-augmented dataset with truncation-safe context injection`.

---

### T16 — ABSA trainer + training script

**Files:**
- Create: `src/absa/trainer.py`, `tests/test_absa_trainer.py`, `scripts/04_train_absa.py`

**Interface:**
```python
class ABSATrainer:
    def __init__(self, model, optimizer, scheduler, device,
                 patience: int = 5, grad_clip: float = 1.0,
                 log_path: str = "", polarity_id_map: dict | None = None): ...
    def train(self, train_loader, val_loader, epochs: int) -> list[dict]: ...
    def evaluate(self, loader) -> dict:
        """Returns {'loss', 'bio_token_f1', 'span_f1', 'sentiment_acc',
                    'sentiment_macro_f1', 'joint_f1'}."""
```

- [ ] **Step 1: Viết test smoke:**
  ```python
  def test_one_step_decreases_loss(...):
      # 4 sample giả, AdamW lr=1e-3, 1 step → loss giảm
      ...

  def test_evaluate_returns_expected_keys(...):
      ...
  ```
- [ ] **Step 2:** Chạy → FAIL.
- [ ] **Step 3:** Implement `ABSATrainer`:
  - Loop epoch: forward, loss, `clip_grad_norm_`, `optimizer.step`, `scheduler.step`. Log per-epoch jsonl.
  - `evaluate`: argmax BIO logits, argmax sentiment logits → `extract_spans` → tính các metric qua module T13. Joint F1 = spans ghép với sentiment pred.
  - Early stopping theo `span_f1`, save best `model.state_dict()` vào `{ckpt_dir}/best.pt`.
- [ ] **Step 4:** Chạy test → PASS.
- [ ] **Step 5: Viết `scripts/04_train_absa.py`:**
  - CLI: `python scripts/04_train_absa.py --config configs/absa.yaml --embedding_ckpt checkpoints/embedding/best.pt --index_dir indexes/ --retrieval_config configs/retrieval.yaml`.
  - Logic:
    - `set_seed`, load tokenizer, load `ContrastiveEmbedder` (eval mode, for encoding query at train-time), load FAISS index + metadata, khởi tạo `Retriever`.
    - Load `bio_tagging.jsonl`, split theo field `split`: records `split=="train"` → train+val (random 90/10 bằng `val_ratio`); `split=="test"` → test.
    - Tạo `RetrievalABSADataset` cho 3 split, `DataLoader`.
    - Init `RetrievalABSA`, AdamW single LR, linear warmup scheduler.
    - `ABSATrainer.train(...)`, lưu best checkpoint vào `checkpoints/absa/best.pt`.
- [ ] **Step 6:** Commit: `feat(absa): multi-task trainer and training entry point`.

---

### T17 — Evaluation script (basic)

**Files:**
- Create: `scripts/05_evaluate.py`

- [ ] **Step 1:** CLI: `python scripts/05_evaluate.py --config configs/absa.yaml --checkpoint checkpoints/absa/best.pt --embedding_ckpt checkpoints/embedding/best.pt --index_dir indexes/`.
- [ ] **Step 2:** Script logic:
  - Load mọi thứ như T16 bước train (retriever, embedding, tokenizer, model).
  - Load `RetrievalABSA` state từ checkpoint.
  - Dùng `ABSATrainer.evaluate(test_loader)` trên test split.
  - In bảng markdown ra stdout và ghi `logs/eval_results.md`:
    ```
    | Metric           | Value |
    |------------------|-------|
    | BIO token F1     | ...   |
    | Span F1          | ...   |
    | Sentiment Acc    | ...   |
    | Sentiment MacF1  | ...   |
    | Joint F1         | ...   |
    ```
- [ ] **Step 3:** Commit: `feat(scripts): basic evaluation on test split`.

---

## Dependency graph

```
T1 ─┬─ T2 ─ T3 ─┬─ T5 ─ T6 ─ T7 ─ T8 ─ T9 ─ T10 ─ T11 ─ T12 ─ T14 ─ T15 ─ T16 ─ T17
    │           │                                                   ↑
    │           └─ T4 ─┘                                             │
    │                                                                │
    └─────────────────────────────────── T13 ─────────────────────────┘
```

T13 (metrics) độc lập — làm càng sớm càng tốt sau T1.

---

## Rủi ro & mitigation (MVP)

1. **Download SemEval 15/16** — XML phải đăng ký tại metashare/ELDA. Là blocker cho T5 trở đi chạy trên data thật. *Mitigation:* dùng `tests/fixtures/toy_restaurant.xml` cho unit test; user download song song với T1-T8.

2. **Retrieval leakage** — query retrieve chính nó → model "cheat". *Mitigation:* `Retriever.retrieve(query_id=...)` loại trừ; `RetrievalABSADataset` luôn pass `query_id`. Kiểm tra qua test T11 và T15.

3. **BIO alignment với retrieved context** — nếu loss BIO tính trên retrieved portion sẽ sai nhãn. *Mitigation:* set `-100` cho toàn bộ token retrieved (test trong T15).

4. **Truncation cắt span trong query** — query dài > budget 100 sẽ cắt mất aspect span. *Mitigation:* ưu tiên giữ query nguyên vẹn, giảm budget retrieved thay vì cắt query. Log warning.

5. **Overfit do dataset nhỏ** — *Mitigation MVP:* `dropout=0.1`, `weight_decay=0.01`, early stopping patience 5. (Stratified split để giai đoạn cải tiến.)

6. **Không có VRAM fp16** — DeBERTa base fp32 có thể chật ở `batch_size=16, seq_len=512`. *Mitigation MVP:* nếu OOM, giảm `batch_size` xuống 8 (vẫn đủ cho khung sườn). AMP để giai đoạn cải tiến.

---

## Phạm vi bị cắt khỏi MVP (làm sau)

Các mục dưới đây ghi lại rõ để giai đoạn cải tiến biết cần thêm gì:

1. **CRF head** cho BIO (`use_crf` flag + `torchcrf`).
2. **AMP / fp16** (`torch.cuda.amp.GradScaler`).
3. **Gradient accumulation** (`grad_accum_steps`).
4. **Differential LR** (encoder `2e-5` vs heads `1e-4`).
5. **Stratified val split** theo `aspect_category`.
6. **Similarity threshold tuning** cho retriever (MVP dùng `threshold=0.0`).
7. **End-to-end fine-tune** (Step 5 của FIRST_PLAN): share encoder + periodic FAISS refresh.
8. **Ablation 7 conditions** (no-retrieval, random retrieval, k ∈ {1,3,5}, + threshold, + E2E).
9. **Cross-domain / ngôn ngữ khác SemEval Restaurant.**

---

## Verification end-to-end (MVP)

Sau khi hoàn thành T1–T17, verify theo thứ tự:

```bash
# 0. Cần download SemEval 15/16 XML trước vào data/raw/
pytest tests/ -v
# Kỳ vọng: all green

python scripts/01_prepare_data.py --raw_dir data/raw --out_dir data/processed
# Kỳ vọng: 3 file jsonl non-empty

python scripts/02_train_embedding.py --config configs/embedding.yaml
# Kỳ vọng: checkpoints/embedding/best.pt, Recall@3 tăng theo epoch

python scripts/03_build_index.py --embedding_ckpt checkpoints/embedding/best.pt \
       --input data/processed/classification.jsonl --out_dir indexes/
# Kỳ vọng: indexes/{train.faiss, train_metadata.jsonl, train_vectors.npy}

python scripts/04_train_absa.py --config configs/absa.yaml \
       --embedding_ckpt checkpoints/embedding/best.pt --index_dir indexes/ \
       --retrieval_config configs/retrieval.yaml
# Kỳ vọng: checkpoints/absa/best.pt, val span_f1 non-trivial (>0.3 là đã qua baseline)

python scripts/05_evaluate.py --config configs/absa.yaml \
       --checkpoint checkpoints/absa/best.pt \
       --embedding_ckpt checkpoints/embedding/best.pt --index_dir indexes/
# Kỳ vọng: logs/eval_results.md có đủ 5 metric
```

Chỉ tuyên bố MVP hoàn thành khi toàn bộ 6 bước trên chạy không lỗi và `logs/eval_results.md` có đủ metric. Các cải tiến (xem "Phạm vi bị cắt") sẽ được plan riêng sau đó.
