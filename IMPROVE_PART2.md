# IMPROVE_PART2.md — Phase 4: Improvement Roadmap

## Context

**Best model:** S3 retrieval — Joint F1 = 0.6237 (beat no-retrieval baseline 0.6104 by +1.3pp)

**Sprint 1 (DONE):** Retrieval quality analysis cho thấy embedding cluster tốt theo category (99.8%) nhung gan mu polarity (61.6%, neutral chi 8.6%).

**Sprint 2 (DONE — FAIL):**
- E1 (class weights [1.00, 1.70, 4.33]): val joint=0.6349 nhung test joint=0.5684 — **overfit nang** do val set qua nho (~250 samples) + class weights khuech dai minority overfitting
- E2 (E1 + retrieval_dropout=0.15 + lambda_cls=0.75): val joint=0.5714 — te hon E1
- **Root cause:** patience=999 + val set nho + class weights = combo overfit. Khong phai loi notebook.

**Van de ton dong:**

| Van de | Chi so | Gap |
|--------|--------|-----|
| BIO Token F1 gap | 0.5505 vs 0.6198 (no-ret) | -6.9pp |
| Sentiment MacF1 gap | 0.7991 vs 0.8234 (no-ret) | -2.4pp |
| Overfitting | val loss diverge tu epoch 5 | |
| Retrieval polarity noise | 61.6% match, neutral 8.6% | |

---

## Phuong an cai thien

### A. Quick Fixes (chua thu — uu tien cao nhat)

#### A1. Tang val_ratio (10% -> 20%)

**Van de:** Val set chi ~250 samples, neutral chi ~10 samples. Val metrics misleading — E1 val tot nhung test te.

**Thay doi:** `val_ratio: 0.2` trong config YAML.

**Effort:** Rat thap (doi 1 so). **Risk:** Khong. **Files:** `configs/absa*.yaml`

#### A2. Early stopping dung (patience=3-5)

**Van de:** Patience=999 trong training notebook — model chay het 10 epochs, checkpoint epoch 10 overfit nang.

**Thay doi:** Bo `--patience 999` trong notebook, de config patience=5 hoat dong.

**Effort:** Rat thap. **Risk:** Khong. **Files:** Kaggle notebook

#### A3. Differential LR

**Van de:** Backbone DeBERTa bi catastrophic forgetting khi train 10 epochs voi LR=2e-5 cho tat ca layers.

**Thay doi:** Backbone LR = 2e-6, heads LR = 2e-4 (10x ratio).

**Effort:** Thap — sua `04_train_absa.py` tach param groups. **Risk:** Thap.

**Files:**

| File | Thay doi |
|------|---------|
| `scripts/04_train_absa.py` | Tach optimizer param groups (backbone vs heads) |
| `configs/absa*.yaml` | Them `backbone_lr`, `head_lr` |

---

### B. Sprint 2 Plan Goc (chua thu)

#### B1. Focal Loss (gamma=2.0)

**Khac voi Weighted CE (E1 da fail):** Focal Loss khong tang weight co dinh ma **focus vao hard examples** — samples ma model dang predict sai. It overfit hon class weights vi khong amplify minority class co dinh.

**Thay doi:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        ...
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
```

**Effort:** Thap. **Risk:** Thap. **Files:** `src/absa/model.py`, config moi

#### B2. Majority-vote Sentiment Baseline

**Muc tieu:** Test nhanh xem polarity cua retrieved neighbors co signal thuc su khong.

**Thay doi:** Concat `neighbor_pol_dist` (3-dim) vao sentiment head input (768 -> 771).

**Effort:** Thap. **Risk:** Thap.

**Files:** `src/absa/dataset.py`, `src/absa/model.py`, `src/absa/trainer.py`

---

### C. Fix Retrieval

#### C1. Polarity-aware Re-ranking

**Van de:** Embedding gan mu polarity (61.6% match). Score collapse (mean=0.9961) -> threshold filtering vo dung.

**Thay doi:** Sau FAISS search, boost score cho same-polarity neighbors:
```python
def retrieve(self, query_vec, query_id=None, query_polarity=None):
    # FAISS search nhu cu
    if query_polarity is not None:
        for r in results:
            if r["polarity"] == query_polarity:
                r["score"] *= 1.2
        results.sort(key=lambda x: x["score"], reverse=True)
    return results[:self.top_k]
```

**Luu y:** Chi dung khi train (da biet polarity). Khi test: dung predicted polarity tu first-pass hoac khong boost.

**Effort:** Thap. **Risk:** Thap. **Files:** `src/retrieval/retriever.py`

#### C2. Ensemble S3 + No-retrieval

**Y tuong:** S3 co Span F1 tot (0.6823), no-retrieval co Sentiment tot (0.8234). Combine strengths.

**Cach lam:** Average logits hoac voting tu 2 models.

**Effort:** Thap. **Risk:** Thap. **Files:** Script eval moi

---

### D. Data Augmentation (balance pos/neu/neg)

**Van de:** Positive 66%, Negative 30%, Neutral 4% (~100 samples). Neutral qua it de model hoc.

**Nguon data kha thi:**
- **Back-translation** minority classes (negative, neutral) — dich sang ngon ngu khac roi dich lai
- **Synonym replacement** cho aspect/opinion terms trong minority samples
- **LLM-generated** neutral restaurant reviews voi annotation
- **External reviews** (Yelp/Amazon) — nhung khong co SemEval-style annotation

**Rang buoc quan trong:**
- **Chi augment cho classification head** (sentiment) — khong dung BIO data vi co the pha Span F1
- Phai retrain embedding + index + ABSA neu thay doi training data
- Synthetic data co the khong representative -> can validate ky

**Effort:** Trung binh. **Risk:** Cao (annotation quality, distribution shift). **Files:** Script augmentation moi, retrain full pipeline

---

### E. Label Interpolation (Sprint 3 tu plan cu)

**Y tuong:** Thay vi hard-code polarity signal (majority-vote), dung learnable module.

**Thay doi:**
```python
class LabelInterpolation(nn.Module):
    def __init__(self, num_labels=3, hidden_dim=64):
        self.label_embed = nn.Embedding(num_labels, hidden_dim)
        self.score_proj = nn.Linear(1, 1, bias=False)

    def forward(self, neighbor_labels, neighbor_scores):
        embeds = self.label_embed(neighbor_labels)     # (batch, k, hidden_dim)
        weights = F.softmax(self.score_proj(neighbor_scores.unsqueeze(-1)).squeeze(-1), dim=1)
        return (weights.unsqueeze(-1) * embeds).sum(dim=1)  # (batch, hidden_dim)
```

Sentiment head: `[CLS](768) + label_vector(64) = 832-dim`

**Effort:** Trung binh. **Risk:** Trung binh.

**Files:** `src/absa/model.py`, `src/absa/dataset.py`, `src/absa/trainer.py`

---

### F. Thay Pipeline theo RLI Paper

**Reference:** Yu et al., 2023. "Making Better Use of Training Corpus: Retrieval-based ASTE via Label Interpolation" (ACL Findings 2023)

**3 thay doi chinh so voi pipeline hien tai:**

| | Hien tai | RLI |
|--|----------|-----|
| Retrieval unit | Sentence + aspect category | Aspect-opinion pair triplet |
| Retriever | Frozen FAISS (cosine) | Jointly trained (bilinear K^T W K_i) |
| Su dung neighbor | Concatenate text tokens | Label interpolation (learnable sentiment embedding) |
| Pre-training | Contrastive (sentence-level) | Contrastive on pseudo-labeled external data |

**Cach implement incremental (khong rewrite cung luc):**

1. **Step 1:** Label interpolation (muc E tren) — test isolated
2. **Step 2:** Trainable relevance function — thay cosine bang bilinear
3. **Step 3:** Joint training retriever + ABSA — neu step 1+2 cho signal tot
4. **Step 4:** Triplet-level store — thay sentence-level bang triplet-level

**Rui ro:**
- Gan nhu **rewrite toan bo** `src/absa/`, `src/retrieval/`, `src/embedding/`
- Joint training can **nhieu VRAM hon** — T4 16GB co the khong du
- RLI dung span-level ASTE, ta dung BIO tagging — **paradigm khac**
- Debug cycle dai tren Kaggle

**Effort:** Rat cao. **Risk:** Rat cao. **Potential:** Rat cao.

---

## Thu tu uu tien de xuat

```
Uu tien 1: A1 + A2 + A3 (quick fixes)
  |-- Mien phi, fix root cause cua E1 fail
  |-- 1 Kaggle session
  |
Uu tien 2: B1 Focal Loss
  |-- Low effort, chua thu, co che khac weighted CE
  |-- Co the gop chung session voi uu tien 1
  |
Uu tien 3: C1 Polarity re-ranking
  |-- Truc tiep fix retrieval weakness
  |-- Code change nho
  |
Uu tien 4: E Label interpolation
  |-- Buoc dau huong RLI, rui ro vua phai
  |-- 1 Kaggle session
  |
Uu tien 5: D Data augmentation
  |-- Chi augment sentiment, khong dung BIO
  |-- Can validate ky truoc khi train
  |
Uu tien 6: F RLI pipeline (incremental)
  |-- Implement tung phan, khong rewrite cung luc
  |-- Nhieu Kaggle sessions
```

---

## Metrics muc tieu

| Metric | S3 (hien tai) | Target |
|--------|---------------|--------|
| BIO Token F1 | 0.5505 | >= 0.5505 (giu) |
| Span F1 | 0.6823 | **>= 0.6823 (khong giam)** |
| Sentiment Acc | 0.9197 | >= 0.9200 |
| Sentiment MacF1 | 0.7991 | **>= 0.8234** |
| Joint F1 | 0.6237 | **>= 0.6400** |

**Constraint:** Span F1 khong duoc giam — day la thanh qua quan trong nhat tu Phase 3.

---

## Bai hoc tu Sprint 2

1. **Val set qua nho** (10% = ~250 samples) -> val metrics khong dang tin. Tang val_ratio truoc khi chay bat ky experiment nao.
2. **Class weights + small val = overfit.** Weighted CE voi neutral weight 4.33x khuech dai overfitting tren ~10 neutral val samples.
3. **Patience=999 la sai lam.** De patience=5 (hoac thap hon) de early stop dung luc.
4. **Val tot khong co nghia test tot.** Luon skeptical voi val improvements, dac biet khi val loss diverge.
