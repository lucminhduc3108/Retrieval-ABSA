# Project Status — Retrieval-based ABSA

**Last updated:** 2026-06-13

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

No results yet on SemEval 2014 — pipeline adapted, training pending on Colab.

### Previous Best (SemEval 2016 — archived)

| Metric | No-Retrieval | Phase 2a v1 |
|--------|-------------|-------------|
| Cat F1 (global 0.80) | 0.6962 | 0.6962 |
| Joint F1 | **0.6304** | 0.6180 |
| Sent Acc\|CC | **91.05%** | 89.26% |

---

## Code Changes (2026-06-13)

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

All 4 notebooks updated for SemEval 2014. Run order:

| # | Notebook | Config | Kaggle Dataset Input | Output Dataset |
|---|----------|--------|---------------------|----------------|
| NB0 | `p5_nb_embed_v4.ipynb` | `embedding_v4_s1/s2.yaml` | `semeval-absa-restaurant` | `p5-embed-v4` |
| NB1 | `p5_nb1_stage1.ipynb` | `stage1_2014.yaml` + `stage1_2014_cataware.yaml` | `semeval-absa-restaurant` | `p5-nb1-stage1` |
| NB2 | `p5_nb2_stage2.ipynb` | `stage2_2014.yaml` + `stage2_2014_noret.yaml` | `p5-nb1-stage1`, `p5-embed-v4` | `p5-nb2-stage2` |
| NB3 | `p5_nb3_eval.ipynb` | All above | `p5-nb1-stage1`, `p5-nb2-stage2`, `p5-embed-v4` | Metrics only |

**Pre-requisite:** Upload 2014 XMLs (`Restaurants_Train.xml`, `Restaurants_Test_Gold.xml`) to `semeval-absa-restaurant` Kaggle dataset.

NB1 trains both ASL and Cat-Aware configs → compare val Cat F1 → set `STAGE1_VARIANT` in NB3.

## Next Actions

- [ ] Upload SemEval 2014 XMLs to Kaggle dataset `semeval-absa-restaurant`
- [ ] Kaggle: Run NB0 — Train embedding v4 (2-stage)
- [ ] Kaggle: Run NB1 — Train Stage 1 (ASL + Cat-Aware)
- [ ] Kaggle: Run NB2 — Train Stage 2 (retrieval + no-retrieval)
- [ ] Kaggle: Run NB3 — End-to-end evaluation — target Cat F1 > 0.85

---

## Training Environment

- **GPU:** Google Colab T4.
- **Local:** Windows 11, no GPU training — code and tests only.
