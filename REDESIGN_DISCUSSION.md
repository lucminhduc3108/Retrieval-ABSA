# Pipeline Redesign: BIO → Category + Sentiment (Combo 6→5)

**Date:** 2026-06-02
**Decision:** Combo 6→5 (Two-stage, phased retriever upgrade)
**Reference paper:** `2023.findings-acl.303.pdf` — RLI: Retrieval-based ASTE via Label Interpolation (ACL 2023)

---

## 1. Background

### Current Pipeline (E3 — BIO + Sentiment)
- Input per record: `[CLS] query_sentence [SEP] aspect_category [SEP] retrieved_context...`
- **Category is GIVEN as input** in both train and test
- **BIO head**: Linear(768, 3) → predicts O/B-ASP/I-ASP for aspect term extraction
- **Sentiment head**: Linear(768, 3) on [CLS] → predicts positive/negative/neutral
- Metrics: BIO Token F1, Span F1, Sentiment Acc/MacroF1, Joint F1 (span + polarity match)
- Best result: E3 Joint F1 = 0.6374

### Why Redesign
- Category given as input → model never learns to detect category → too simple for graduation thesis
- Real-world applications don't have category labels at inference time
- BIO tagging + known category = insufficient contribution

### New Task
- **Input:** sentence only (no category given)
- **Output:** list of (category, sentiment) pairs
- 12 categories: AMBIENCE#GENERAL, DRINKS#PRICES, DRINKS#QUALITY, DRINKS#STYLE_OPTIONS, FOOD#PRICES, FOOD#QUALITY, FOOD#STYLE_OPTIONS, LOCATION#GENERAL, RESTAURANT#GENERAL, RESTAURANT#MISCELLANEOUS, RESTAURANT#PRICES, SERVICE#GENERAL
- 3 sentiments: positive, negative, neutral
- Example: "The ramen is great but service is slow" → [(FOOD#QUALITY, positive), (SERVICE#GENERAL, negative)]

### Metrics Change
- **Old:** BIO Token F1, Span F1, Joint F1 (span + polarity)
- **New:** Category P/R/F1, Sentiment Acc/MacroF1, Joint F1 (category + polarity)
- E3 results (old metrics) **cannot be compared** with new pipeline results — different task entirely

---

## 2. Chosen Approach: Combo 6→5

Two-phase plan. Phase 1 builds the new pipeline with frozen retriever. Phase 2 upgrades retriever to joint training.

### Phase 1 (Combo 6): Two-stage + Frozen Retriever + Label Interpolation

```
Input: "The ramen is great but service is slow"

Stage 1 — Category Detection (NO retrieval)
  DeBERTa encode sentence → [CLS] → category_head (12 sigmoid outputs)
  → FOOD#QUALITY = 1, SERVICE#GENERAL = 1, 10 others = 0
  Loss: BCE with sqrt pos_weight per category
        pos_weight[i] = sqrt((N - n_i) / n_i)  — range 1.23 (FOOD#QUALITY) → 9.19 (DRINKS#PRICES)
  Threshold: single global threshold tuned on val set (grid search [0.2, 0.3, 0.4, 0.5])

Stage 2 — Sentiment Prediction per Category (WITH frozen retrieval)
  For each detected category k:
    1. query = ContrastiveEmbedder(sentence, category_k) → 256d    ← EXISTING embedding
    2. FAISS search → top-k neighbors                               ← EXISTING index
    3. Label interpolation:
       alpha_l = softmax(faiss_score_l / tau)                       ← tau ∈ {0.02, 0.05, 0.1}, ablation point
       label_repr = sum(alpha_l * polarity_embedding(neighbor_l.polarity))  ← learnable, dim=64d
    4. Input: [CLS] sentence [SEP] category_k [SEP] nb_text... [SEP]
       DeBERTa → [CLS] (768d)
       final_repr = concat([CLS], label_repr) → (768 + 64 = 832d)
       sentiment = Linear(832, 256) → ReLU → Dropout(0.1) → Linear(256, 3)
  Loss: CE

Total Loss: L_category + lambda * L_sentiment
```

> **[REVIEW NEEDED]** Với separate backbone (đã chốt), Stage 1 và Stage 2 được train trong 2 run riêng biệt — không phải joint loss.
> - Stage 1 training: `L = L_BCE` (category detection, DeBERTa_1 only)
> - Stage 2 training: `L = L_CE` (sentiment, DeBERTa_2 only)
> - "Total Loss" trên chỉ mang tính conceptual. Cần quyết định: có fine-tune Stage 1 sau khi Stage 2 converge không? Hay luôn frozen Stage 1 khi train Stage 2?

**Reused from current project:** Embedding (736MB), FAISS index (4,161 vectors), ContrastiveEmbedder code, Retriever code
**Built new:** Dataset (group by sentence), Category head (12 sigmoid), Label interpolation module, Metrics (category-based), Trainer logic
**Kaggle sessions:** 1-2 (+ 1 for no-retrieval baseline)

### Phase 2 (Upgrade → Combo 5): Joint Retriever Training — Two Sub-phases

#### Phase 2a: Learnable Alignment (W matrix, frozen embedding)

```
ONLY change vs Phase 1: replace cosine with learnable W

Phase 1:  relevance = cosine(query, neighbor)             ← fixed
Phase 2a: relevance = query^T * W * neighbor              ← W learnable (256×256 = 65K params)

+ Add ranking loss to training
- No re-encoding needed (ContrastiveEmbedder frozen → store vectors stable)
```

**Kept from Phase 1:** All of Stage 1, dataset format, metrics, label interpolation, ContrastiveEmbedder
**Changed:** Relevance function (cosine → learned W), add ranking loss
**Kaggle sessions:** 1-2 additional
**Feasibility:** W trivial, VRAM unchanged vs Phase 1, zero re-encoding overhead

#### Phase 2b: Full Joint Training (unfreeze ContrastiveEmbedder)

```
ONLY change vs Phase 2a: unfreeze ContrastiveEmbedder

Phase 2a: W learnable, ContrastiveEmbedder frozen
Phase 2b: W learnable + ContrastiveEmbedder fine-tuned   ← end-to-end gradient

+ Periodic re-encode store every N epochs (embedding changes → store stale)
```

**Kept from Phase 2a:** W matrix, ranking loss, Stage 1 and Stage 2 unchanged
**Changed:** ContrastiveEmbedder unfrozen, add re-encoding loop every epoch
**Kaggle sessions:** 1-2 additional
**Feasibility:** VRAM ~5.3GB (both DeBERTa trainable, T4 16GB → 10GB headroom).
Re-encoding 4,161 vectors ≈ 26s/epoch — negligible overhead (4.4 min per 10-epoch run).
**Note:** Phase 2b is conditional — skip if Phase 2a results already strong or timeline tight.

---

## 3. Summary Table

| | Current (E3) | Phase 1 | Phase 2a | Phase 2b |
|---|---|---|---|---|
| **Task** | BIO tagging + sentiment | **Category detection + sentiment** | Category detection + sentiment | Category detection + sentiment |
| **Input** | sentence + category (given) | sentence only (predict) | sentence only (predict) | sentence only (predict) |
| **Head 1** | BIO head (O/B/I) | Category head (12 sigmoid) | Category head (12 sigmoid) | Category head (12 sigmoid) |
| **Head 2** | Sentiment head (3 class) | Sentiment head (3 class) | Sentiment head (3 class) | Sentiment head (3 class) |
| **Retriever** | Frozen contrastive | Frozen cosine | **Learnable W** (frozen emb) | **Joint training** (unfrozen emb) |
| **Embedding** | Existing | Existing | Existing (frozen) | **Fine-tuned jointly** |
| **Re-encoding** | — | — | Not needed | ~26s/epoch |
| **Metrics** | Span F1, Joint F1 (span+pol) | **Cat F1, Joint F1 (cat+pol)** | Cat F1, Joint F1 (cat+pol) | Cat F1, Joint F1 (cat+pol) |
| **Baseline** | P2 no-retrieval | **New no-retrieval baseline** | Phase 1 | Phase 2a |

---

## 4. Risk Assessment

| Risk | Description | Mitigation |
|---|---|---|
| **No fallback to E3** | Different task, different metrics — E3 results not comparable | Accept: this is a task change, not incremental improvement |
| **Phase 1 is non-trivial** | Rewrite dataset, model, trainer, metrics from scratch | Phase 1 simpler than Combo 2 (frozen retriever, no joint training) |
| **Stage 1 imbalance** | 12 categories, ratio 42:1 (FOOD#QUALITY 681 vs DRINKS#PRICES 20 sentences) | BCE with sqrt pos_weight (range 1.23–9.19) + global threshold tuning on val |
| **Error propagation** | Stage 1 miss category → Stage 2 can't recover | Global threshold tuned on val set for recall-precision tradeoff; report per-category recall |
| **Frozen retriever noise** | Embedding is polarity-blind (61.6% match, neutral 8.6%) | Label interpolation with temperature scaling mitigates; Phase 2b fixes root cause |
| **Score collapse** | FAISS scores mean=0.9961 → alpha_l near uniform without scaling | Temperature scaling: softmax(score/tau), tau ablated {0.02, 0.05, 0.1} |

---

## 5. Evaluation Strategy

**Cannot compare with E3** — different task, different metrics.

Must create **internal baselines** within the new pipeline:

| Experiment | Purpose |
|---|---|
| Phase 1 + retrieval | Main model |
| Phase 1 + **no retrieval** | New baseline — isolates retrieval contribution |
| Phase 2a + learnable W | Intermediate — frozen embedding, learnable alignment |
| Phase 2b + joint retrieval | Full upgrade — compare vs Phase 1 and Phase 2a |

Ablation opportunities:
- With/without retrieval (Phase 1)
- Frozen vs learnable W vs joint retriever (Phase 1 vs Phase 2a vs Phase 2b)
- With/without label interpolation
- Temperature tau sensitivity {0.02, 0.05, 0.1}
- Stage 1 threshold sensitivity [0.2, 0.3, 0.4, 0.5]
- Top-k neighbors (1, 2, 3)

Additional metric:
- **Sentiment Acc | Correct Category** — in các lần Stage 1 predict đúng category, Stage 2 đúng sentiment bao nhiêu %. Tách lỗi Stage 1 khỏi Stage 2 cho thesis analysis.

---

## 6. Eliminated Options

| Option | Reason eliminated |
|---|---|
| **Combo 1 (Multi-label + retrain embedding)** | Low thesis novelty, must retrain everything, no phased approach |
| **Combo 2 (Multi-label + joint training)** | No fallback — must complete entire pipeline before any results. Same ceiling as Phase 2 of 6→5 but higher risk |
| **Combo 5 (Two-stage + joint training)** | Absorbed as Phase 2 of Combo 6→5 |
| Multi-aspect B (1 record/opinion) | Same input, different labels → training signal conflict |
| Multi-aspect C (category hint) | Hint too specific = given; hint too vague = still conflict |
| Retrieval B (keep mismatch) | Query format mismatch, quality degraded |
| Retrieval C (drop retrieval) | Removes core contribution |
