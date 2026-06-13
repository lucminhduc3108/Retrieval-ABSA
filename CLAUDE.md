# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project snapshot

**Retrieval-based ABSA** on SemEval 2014 Task 4 Restaurant.  
**Pipeline:** Two-stage — Category Detection (Stage 1) then Sentiment Classification (Stage 2).  
**Goal:** Prove retrieval-augmented sentiment (Phase 2a) can beat the no-retrieval baseline.  
Live status tracked in `context/STATUS.md`.

## Pipeline (5 steps)

1. **Data Preparation:** Parse SemEval 2014 XML → sentence-level (category, polarity) pairs. Drop `conflict`. Deduplicate opinions per (sentence, category).
2. **Retrieval Engine:** ContrastiveEmbedder (DeBERTa, InfoNCE) → 256-dim vectors → FAISS IndexFlatIP. Not needed for no-retrieval strategy.
3. **Stage 1 — Category Detection:** DeBERTa (+ optional Cat-Aware Attention) → 5 sigmoid (ASL/BCE loss). Global threshold tuned on val.
4. **Stage 2 — Sentiment Classification:** Per detected category, predict positive/negative/neutral. Two strategies:
   - **No-Retrieval (baseline):** DeBERTa → MLP(768→256→3).
   - **Phase 2a (Label Interpolation):** FAISS top-k → Diagonal W alignment → polarity embedding interpolation → MLP(832→256→3) + ranking loss.
5. **Evaluation (end-to-end):** Stage 1 predicts categories → Stage 2 predicts sentiment per category. Metrics: Category F1, Sentiment Acc|Correct Category, Joint F1 (cat+pol).

## Commands

```bash
# Setup
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

# Tests
pytest tests/ -v

# Pipeline (run in order)
python scripts/01_prepare_data.py
python scripts/02_train_embedding.py --config configs/embedding_2014.yaml
python scripts/03_build_index.py --embedding_ckpt checkpoints/embedding_2014/best.pt \
       --input data/processed/classification.jsonl --out_dir indexes/
python scripts/04a_train_stage1.py --config configs/stage1_2014.yaml
python scripts/04b_train_stage2.py --config configs/stage2_2014.yaml \
       --embedding_ckpt checkpoints/embedding_2014/best.pt --index_dir indexes/
python scripts/04b_train_stage2.py --config configs/stage2_2014_noret.yaml --no_retrieval
python scripts/05_evaluate_joint.py --stage1_ckpt checkpoints/stage1_2014/best.pt \
       --stage2_ckpt checkpoints/stage2_2014/best.pt \
       --embedding_ckpt checkpoints/embedding_2014/best.pt --index_dir indexes/ \
       --stage1_config configs/stage1_2014.yaml --stage2_config configs/stage2_2014.yaml
```

## Non-obvious correctness invariants

These are **not optimizations** — violating any of them silently breaks the model:

- **Retriever self-exclusion is mandatory.** At train time the query is already in the index; without `query_id` exclusion the top-1 result is always the query itself. `Retriever.retrieve(query_vec, query_id=record["id"])` — never pass `None` at train time.
- **Drop `conflict` polarity at parse time.** 3-class polarity: `positive / negative / neutral`.
- **Label Interpolation Padding:** Padded neighbors MUST be masked with `-inf` to avoid padding bias towards positive polarity.
- **Dedup at parse level.** `deduplicate_opinions()` runs after XML parsing, before builders. Majority vote for polarity conflicts, drop tied groups.

## SemEval XML paths

**Using SemEval 2014 Task 4 Restaurant** (sentence-level ACSA). 5 aspect categories: ambience, anecdotes/miscellaneous, food, price, service.

| Role | Actual path on disk |
|---|---|
| **Train** | `SemEval-2014/Restaurants_Train.xml` |
| **Test (gold)** | `SemEval-2014/Restaurants_Test_Gold.xml` |

Stats: Train 3,044 sentences / 3,516 opinions (after drop conflict+dedup), Test 800 sentences / 973 opinions. Train-test overlap: 0.1% (1 sentence). 5 categories, balanced distribution.

## Other gotchas

- **Windows bash shell.** Use `.venv/Scripts/activate` (not `bin/`). Quote paths containing spaces.
- **Language convention:** prose in Vietnamese, code / identifiers / commit messages in English.
- **Training on Google Colab.** All GPU training runs on Colab T4. Local machine is Windows 11 with no GPU — code/test only.
- **Switched from SemEval 2016 to 2014.** 2016 had 12 fine-grained categories with severe rare category problem (8/12 F1<0.7). 2014 has 5 balanced categories, removing Stage 1 as bottleneck.
