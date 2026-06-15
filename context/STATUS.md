# Project Status — Retrieval-based ABSA

**Last updated:** 2026-06-15

---

## Dataset Switch: SemEval 2016 → 2014

**Reason:** SemEval 2016 had 12 fine-grained categories with severe rare category problem (8/12 Cat F1 < 0.7, DRINKS#PRICES F1 = 0.000). Stage 1 was capped at Cat F1 = 0.6962, bottlenecking Joint F1 regardless of Stage 2 quality.

**SemEval 2014 Task 4 Restaurant** has 5 balanced categories:
- food (1,166), anecdotes/miscellaneous (1,101), service (562), ambience (385), price (302)
- Train: 3,044 sentences / 3,516 opinions (after drop 196 conflict + 2 dedup)
- Test: 800 sentences / 973 opinions (after drop 52 conflict)
- No augmentation needed — data is balanced

---

## Pipeline Overview

Two-stage pipeline on SemEval 2014 Restaurant:

1. **Data Prep:** Parse XML → (category, polarity) pairs. Drop `conflict`. Deduplicate.
2. **Retrieval Engine:** DeBERTa ContrastiveEmbedder (InfoNCE) → 256-dim → FAISS IndexFlatIP.
3. **Stage 1 — Category Detection:** DeBERTa + optional Cat-Aware Attention → 5 sigmoid (ASL/BCE). Global threshold.
4. **Stage 2 — Sentiment Classification:** Two strategies:
   - **No-Retrieval (baseline):** DeBERTa → MLP(768→256→3).
   - **Phase 2a (Label Interpolation):** FAISS top-k + Diagonal W + polarity interpolation.
5. **Evaluation (end-to-end):** Category F1, Sentiment Acc|Correct Category, Joint F1 (cat+pol).

---

## Current Results

### Embedding (NB0) — Polarity Fix with CLS Head (Mức 1)

**Problem solved:** Polarity blindness — DeBERTa CLS vectors for same-category different-polarity had cosine ≈ 0.999. InfoNCE gradient ≈ 0.

**Fix:** Added `cls_head` (nn.Linear(768→3) + CE loss) to force backbone polarity capacity. Split loss (separate L_pol + L_cat). Two-pass training: random triplets → hard negative mining → hard triplets with `--resume_from`.

| Metric | Before fix | Pass 1 (ep15) | Pass 2 (ep11) |
|--------|-----------|---------------|---------------|
| m1 (margin) | 0.007 | 0.077 | **0.159** |
| cls_acc | — | 0.779 | **0.901** |
| pol_match@5 | ~0.50 | — | **0.832** |
| cat_match@5 | — | — | **1.000** |

**Conclusion:** Mức 1 improved aggregate pol_match@5 to 0.832 (mixed pool). However, NB4 diagnostic revealed this is **insufficient per-polarity** — see Diagnostic section below.

Checkpoint: `kaggle_upload/outputs_p5_embed_v4/embedding_v4_s2_best.pt` (703 MB)
FAISS index: `kaggle_upload/outputs_p5_embed_v4/train.faiss` + `train_vectors.npy` + `train_metadata.jsonl`

### Stage 1 — Category Detection (NB1)

| Config | Best Epoch | Cat F1 | Cat P | Cat R |
|--------|-----------|--------|-------|-------|
| ASL | 1 | 0.506 | 0.383 | 0.746 |
| **Cat-Aware** | **12** | **0.8368** | **0.8447** | **0.8291** |

ASL collapsed (predict all 5 categories → P=0.245, R=1.0 from epoch 2). Cat-Aware is the winner.

### Stage 2 — Sentiment Classification (NB2)

**Val metrics (train-only FAISS index, val excluded):**

| Metric | Retrieval (Phase 2a) | No-Retrieval | Delta |
|--------|---------------------|-------------|-------|
| **Best MacF1** | **0.8480** (ep8) | 0.8176 (ep16) | **+3.0pp** |
| **Best Acc** | **0.8937** (ep8) | 0.8686 (ep16) | **+2.5pp** |
| F1 pos | 0.956 | 0.918 | +3.8pp |
| F1 neg | 0.856 | 0.841 | +1.5pp |
| F1 neu | 0.732 | 0.694 | +3.8pp |

Retrieval wins on val (fair eval, val not in index). Converges faster (ep8 vs ep16).

### End-to-End Joint Evaluation (NB3 — Test Set)

| Metric | Retrieval | No-Retrieval | Delta |
|--------|-----------|-------------|-------|
| Cat F1 | 0.8564 | 0.8564 | 0 (same Stage 1) |
| **Sent Acc\|CC** | 0.8058 (668/829) | **0.9011** (747/829) | **-9.5pp** |
| **Joint F1** | 0.6901 | **0.7717** | **-8.2pp** |

**Retrieval loses on test.** Val performance doesn't transfer: retrieval neighbors are less relevant for unseen test sentences. Model over-relies on 64-dim retrieval signal — when neighbors are noisy, it corrupts predictions that text-only would get right.

**Root cause:** Diagnosed in NB4 — see Diagnostic section below.

### NB4 — Diagnostic Results (6 experiments)

**Exp 2 — Retrieval Quality:** Embedding is polarity-blind for negative/neutral on test.

| Condition | pol_match@5 |
|-----------|-------------|
| Train → train-only index | 0.851 |
| Test → full index (aggregate) | 0.775 |
| Test positive (n=657) | **0.897** |
| Test negative (n=222) | **0.580** |
| Test neutral (n=94) | **0.383** |

NB0's reported 0.832 was on mixed train+test pool — overestimates real test-time quality. Avg cosine = 0.9937 (near score collapse).

**Exp 1 — Oracle Category:** Stage 1 errors are NOT the problem.

| Condition | Sent Acc |
|-----------|----------|
| Retrieval (predicted cat) | 0.8002 |
| Retrieval (oracle/gold cat) | 0.7986 |
| No-retrieval (predicted cat) | 0.8995 |

Stage 1 contribution: **-0.2pp** only. Intrinsic architecture gap: **+10.1pp**. Retrieval loses even with perfect categories.

**Exp 3 — Signal Ablation (oracle categories):**

| Mode | Sent Acc | Interpretation |
|------|----------|----------------|
| text+vector (standard) | 0.7986 | Baseline |
| text-only (zero label_repr) | 0.7646 | -3.4pp → vector signal is the main retrieval signal |
| vector-only (no neighbor text) | 0.7975 | -0.1pp → text signal nearly useless |
| neither | 0.6393 | Model collapses without retrieval (trained expecting 832-dim) |

**Exp 5 — Error Analysis:** 115 hurt cases (no-ret correct, ret wrong) vs 35 help cases. Net -80 records.
- Mean pol_match@5 in hurt cases: **0.14** (nearly all neighbors wrong polarity)
- 111/115 hurt cases have pol_match@5 < 0.5
- Error directions scattered: neg→neu (26), pos→neg (23), neg→pos (22)

**Exp 6 — Confidence:** Retrieval model less confident (0.9008 vs 0.9510 mean max_prob).

**Root cause chain:**
```
Embedding polarity-blind for neg/neu → FAISS returns wrong-polarity neighbors
→ label_repr (64-dim) encodes wrong polarity → sentiment_head trusts label_repr
(learned dependency during training when neighbors were good)
→ flips 115 correct predictions to wrong, only helps 35 → net -9.5pp
```

DeBERTa 768-dim CLS already captures sentiment well (no-ret 90%). The 64-dim label_repr from bad neighbors actively corrupts predictions.

### Previous Best (SemEval 2016 — archived)

| Metric | No-Retrieval | Phase 2a v1 |
|--------|-------------|-------------|
| Cat F1 (global 0.80) | 0.6962 | 0.6962 |
| Joint F1 | **0.6304** | 0.6180 |
| Sent Acc\|CC | **91.05%** | 89.26% |

---

## Code Changes

### 2026-06-15
- [x] NB4 diagnostic notebook created (`p5_nb4_diagnostic.ipynb`) — 6 experiments, inference-only
- [x] NB4 pushed to Kaggle, ran successfully on T4
- [x] Diagnosed root cause: embedding polarity-blind for neg/neu → label_repr corrupts predictions
- [x] Confirmed: Stage 1 NOT the problem (-0.2pp), intrinsic architecture gap = 10.1pp
- [x] Confirmed: vector signal (label_repr) is the main retrieval signal; text signal nearly useless

### 2026-06-14
- [x] Staged polarity fix: CLS head + proj_head + attention pool (mức 1/2/3) implemented
- [x] Split loss (separate L_pol + L_cat) with configurable weights
- [x] `--resume_from` for 2-pass training (random → hard triplets)
- [x] `strict=False` on all `load_state_dict` calls (handles cls_head keys)
- [x] NB0 trained successfully on Kaggle — pol_match@5=0.832, FAISS index built
- [x] NB2 updated: independent from NB1 (self data prep), uses NB0 embedding_v4_s2_best.pt
- [x] NB1 trained: Cat-Aware Cat F1=0.8368 (ASL collapsed)
- [x] NB2 v1 trained: Retrieval val Acc=0.8830 beats No-Retrieval val Acc=0.8626
- [x] NB3 v1 end-to-end: Retrieval Sent Acc|CC=0.8034 LOSES to No-Retrieval=0.8963
- [x] Diagnosed val leakage: val records were in FAISS index → inflated retrieval val metrics
- [x] Fixed `04b_train_stage2.py`: build FAISS index from train-only subset (exclude val)
- [x] NB2 v2 trained: Retrieval val MacF1=0.848 (ep8), No-Ret val MacF1=0.818 (ep16)
- [x] NB3 v2 end-to-end: Retrieval Sent Acc|CC=0.8058, No-Ret=0.9011 — fix didn't help
- [x] Root cause: retrieval doesn't generalize to test (architecture issue, not data leak)
- [x] Uploaded `p5-embed-v4` dataset to Kaggle

### 2026-06-13
- [x] XML parser: added `parse_semeval2014_xml()` for 2014 format
- [x] Category builder: 12 → 5 categories, removed entity/attribute hierarchy
- [x] Data preparation: updated for 2014 paths, removed BIO builder
- [x] Category model/dataset/trainer: removed HierarchicalCategoryDetector
- [x] Evaluation script: removed hierarchical branches
- [x] Created configs: `stage1_2014.yaml`, `stage1_2014_cataware.yaml`, `embedding_2014.yaml`, `stage2_2014.yaml`, `stage2_2014_noret.yaml`
- [x] All 197 tests passing
- [x] Data pipeline generates correct output (3,516 train / 973 test opinions)
- [x] Kaggle notebooks NB1/NB2/NB3 updated for SemEval 2014 (configs, paths, dataset refs)
- [x] Kernel metadata updated: `p3s2-embedding-flat` → `p5-embed-v4`

---

## Kaggle Notebooks (SemEval 2014)

All 5 notebooks for SemEval 2014. Run order:

| # | Notebook | Config | Kaggle Dataset Input | Output Dataset |
|---|----------|--------|---------------------|----------------|
| NB0 | `p5_nb_embed_v4.ipynb` | `embedding_2014_cls_s1/s2.yaml` | `semeval-absa-restaurant` | `p5-embed-v4` |
| NB1 | `p5_nb1_stage1.ipynb` | `stage1_2014_cataware.yaml` | `semeval-absa-restaurant` | `p5-nb1-stage1` |
| NB2 | `p5_nb2_stage2.ipynb` | `stage2_2014.yaml` + `stage2_2014_noret.yaml` | `semeval-absa-restaurant`, `p5-embed-v4` | `p5-nb2-stage2` |
| NB3 | `p5_nb3_eval.ipynb` | All above | `p5-nb1-stage1`, `p5-nb2-stage2`, `p5-embed-v4` | Metrics only |
| NB4 | `p5_nb4_diagnostic.ipynb` | All above | `p5-nb1-stage1`, `p5-nb2-stage2`, `p5-embed-v4` | Diagnostic report |

**Pre-requisite:** Upload 2014 XMLs (`Restaurants_Train.xml`, `Restaurants_Test_Gold.xml`) to `semeval-absa-restaurant` Kaggle dataset.

NB1 trains Cat-Aware only (ASL collapsed). NB1 and NB2 run in parallel (NB2 no longer depends on NB1).

## Next Actions

- [x] Upload SemEval 2014 XMLs to Kaggle dataset `semeval-absa-restaurant`
- [x] Kaggle: Run NB0 — Train embedding (CLS head mức 1, split loss, 2-pass)
- [x] Kaggle: Run NB1 — Cat-Aware Cat F1=0.8368
- [x] Kaggle: Run NB2 v1 — Retrieval val Acc=0.8830 > No-Ret val Acc=0.8626
- [x] Kaggle: Run NB3 v1 — End-to-end: Retrieval LOSES (0.8058 vs 0.9011 Sent Acc|CC)
- [x] Fix val leak → retrain NB2 v2 → re-eval NB3 v2 — no improvement
- [x] NB4 diagnostic — identified root cause (embedding polarity-blind for neg/neu)
- [ ] **Implement joint training (RLI-style)** — see plan below

## Joint Training Plan

**Goal:** End-to-end backprop from sentiment CE loss through label_repr → LearnableRetriever → embedding model, so embedding learns to retrieve polarity-consistent neighbors.

**Why:** Current frozen embedding is trained with contrastive loss (category-focused), not sentiment-aligned. RLI paper ablation shows -1 to -2% F1 without joint training. NB4 confirmed the vector signal (label_repr) is the main retrieval channel — fixing embedding quality directly fixes the root cause.

**Key challenge: Stale index.** Joint training updates embedding weights each step, but FAISS index holds vectors from the old embedding. Query and index vectors end up in different spaces → cosine similarity meaningless.

**Solution: Re-index every epoch.** After each epoch, re-encode all train records with updated embedding → rebuild FAISS index + store_vectors. Cost: ~2800 records × 1 forward pass ≈ 10-15s on T4. Acceptable.

**Implementation steps:**
1. Modify `04b_train_stage2.py` training loop: add `rebuild_index()` call at start of each epoch
2. Memory management: 2× DeBERTa on T4 (sentiment encoder + embedding encoder). Options: gradient checkpointing on embedding, or freeze embedding encoder and only train projection head
3. Gradient flow: CE loss → sentiment_head → cat([CLS, label_repr]) → label_repr → polarity_embedding × alpha → softmax(scores/tau) → scores = neighbor_vecs @ (W_diag * query_vec) → query_vec = embedding.encode(). The `embedding.encode()` must be called in forward pass (not pre-computed) for gradients to flow
4. Config: `stage2_2014_joint.yaml` — `joint_training: true`, `embedding_lr: 1e-5`, `gradient_checkpointing: true`
5. New Kaggle notebook: NB2-joint (combines NB0 embedding + NB2 Stage 2 training)
6. Re-eval with NB3

**Existing infrastructure:** `SentimentPredictor` already accepts `embedding_model`, `forward()` supports `embed_input_ids`, `04b_train_stage2.py` has `--joint_training` flag, optimizer has embedding param group. Need to add: epoch-level re-indexing loop, memory optimization

---

## Training Environment

- **GPU:** Kaggle T4.
- **Local:** Windows 11, no GPU training — code and tests only.
