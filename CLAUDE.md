# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project snapshot

**Retrieval-based ABSA** on SemEval 2016 Restaurant (SB1).  
**Pipeline:** Two-stage — Category Detection (Stage 1) then Sentiment Classification (Stage 2).  
**Goal:** Improve retrieval-augmented sentiment (Phase 2a) to beat the no-retrieval baseline.  
Live status tracked in `context/STATUS.md`.

## Pipeline (5 steps)

1. **Data Preparation & Augmentation:** Parse SemEval 2016 XML → sentence-level (category, polarity) pairs. Drop `conflict`. Augment neutral from MAMS-ACSA (~20% neutral in train).
2. **Retrieval Engine (reused, supports Stage 2 retrieval only):** ContrastiveEmbedder (DeBERTa, InfoNCE) → 256-dim vectors → FAISS IndexFlatIP. Not needed for no-retrieval strategy.
3. **Stage 1 — Category Detection:** DeBERTa + Cat-Aware Attention (12 learnable queries + MHA) → 12 sigmoid (BCE loss with sqrt pos_weight). Global threshold tuned on val. **FINAL: Cat-Aware R5.**
4. **Stage 2 — Sentiment Classification:** Per detected category, predict positive/negative/neutral. Two strategies compared:
   - **No-Retrieval (baseline):** DeBERTa → MLP(768→256→3). Currently leading.
   - **Phase 2a (Label Interpolation):** FAISS top-k → Diagonal W alignment → polarity embedding interpolation → MLP(832→256→3) + ranking loss.
5. **Evaluation (end-to-end):** Stage 1 predicts categories → Stage 2 predicts sentiment per category. Metrics: Category F1, Sentiment Acc|Correct Category, Joint F1 (cat+pol).

## Commands

```bash
# Setup
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

# Tests
pytest tests/ -v
pytest tests/test_xml_parser.py::test_parse_returns_three_sentences -v  # single test

# Pipeline (run in order)
python scripts/01_prepare_data.py --raw_dir data/raw --out_dir data/processed
python scripts/02_train_embedding.py --config configs/embedding.yaml
python scripts/02_train_embedding.py --config configs/embedding.yaml --limit 64 --epochs 1  # smoke test
python scripts/03_build_index.py --embedding_ckpt checkpoints/embedding/best.pt \
       --input data/processed/classification.jsonl --out_dir indexes/
python scripts/04_train_absa.py --config configs/absa.yaml \
       --embedding_ckpt checkpoints/embedding/best.pt --index_dir indexes/ \
       --grad_accum_steps 8
python scripts/04_train_absa.py --config configs/absa.yaml --no_retrieval \
       --grad_accum_steps 8  # no-retrieval ablation (no embedding/index needed)
python scripts/05_evaluate.py --config configs/absa.yaml \
       --checkpoint checkpoints/absa/best.pt \
       --embedding_ckpt checkpoints/embedding/best.pt --index_dir indexes/
python scripts/05_evaluate.py --config configs/absa.yaml \
       --checkpoint checkpoints/absa/best.pt --no_retrieval  # evaluate no-retrieval model

# Utilities
python scripts/analyze_duplicates.py  # data duplication analysis
```

## Non-obvious correctness invariants

These are **not optimizations** — violating any of them silently breaks the model:

- **Retriever self-exclusion is mandatory.** At train time the query is already in the index; without `query_id` exclusion the top-1 result is always the query itself. `Retriever.retrieve(query_vec, query_id=record["id"])` — never pass `None` at train time.
- **Aspect category = full Category#Attribute.** Keep `FOOD#QUALITY`, `SERVICE#GENERAL`, etc. Do not reduce to coarse prefix.
- **Drop `conflict` polarity at parse time.** 3-class polarity: `positive / negative / neutral`.
- **Label Interpolation Padding:** Padded neighbors MUST be masked with `-inf` to avoid padding bias towards positive polarity.

## SemEval XML paths

**Using SemEval 2016 SB1 only** (sentence-level ABSA). SemEval 2015 is excluded — 2016 train is a superset of 2015.

| Role | Actual path on disk |
|---|---|
| **Train** | `SemEval-Dataset/SemEval 2016 Task 5/Restaurant Training/ABSA16_Restaurants_Train_SB1_v2.xml` |
| **Test (gold)** | `SemEval-Dataset/SemEval 2016 Task 5/Phase B/Gold Annotation/Restaurant/EN_REST_SB1_TEST.xml.gold` |

Stats: Train 2,000 sentences / 2,507 opinions, Test 676 sentences / 859 opinions. Train-test overlap: 0.1% (1 sentence). No conflict polarity. 12 categories.

## Other gotchas

- **SemEval-Dataset/** has its own nested `.git/` — leave it alone; gitignore it or reference its XMLs read-only.
- **Windows bash shell.** Use `.venv/Scripts/activate` (not `bin/`). Quote paths containing spaces.
- **Language convention:** prose in Vietnamese, code / identifiers / commit messages in English.
- **SemEval 2016 only.** SemEval 2015 dropped — its train+test is a subset of 2016 train, causing massive dedup loss and 1:1 train/test ratio. Using 2016 alone gives 2,507 train opinions, 0.1% leakage, 3:1 ratio.
- **SB2 is unusable.** SB2 files have review-level opinions (no target, no char offset) — completely different task from sentence-level ABSA.
- **Training on Kaggle.** All GPU training runs on Kaggle T4. Local machine is Windows 11 with no GPU — code/test only.
