# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project snapshot

Pipeline: contrastive DeBERTa embedding → FAISS index → multi-task retrieval-ABSA (BIO tagging + sentiment classification) on **SemEval 2016 Restaurant only** (SB1). **Status: preparing GĐ 2 retrain** — MVP complete on old 2015+2016 data, now switching to clean SemEval 2016 only. Live status tracked in `context/STATUS.md`.

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

## Architecture

Three phases that are intentionally separate (don't merge their backbones in MVP):

1. **Contrastive embedding** (`src/embedding/`): `ContrastiveEmbedder` = DeBERTa-v3-base + `Linear(768,256)→GELU→LayerNorm` projection, trained with symmetric InfoNCE in-batch loss (`tau=0.07`). Output: L2-normed 256-dim vectors. Validated by Recall@{1,3,5}.

2. **Retrieval index** (`src/retrieval/`): `encode_records` builds `faiss.IndexFlatIP` from train-split. Persists `train.faiss` + `train_metadata.jsonl` (containing `tokens`, `bio_tags`, `aspect_category`, `polarity` per record) + `train_vectors.npy`. `Retriever.retrieve(query_vec, query_id)` does self-exclusion at query time so the ABSA dataset gets clean neighbors.

3. **Multi-task ABSA** (`src/absa/`): separate DeBERTa backbone. Input = `[CLS] query [SEP] query_aspect [SEP] ret1_sent [ASP] ret1_asp [POL] ret1_pol [SEP] ...`. BIO head `Linear(768,3)` + sentiment head `Dropout→Linear(768,3)` on `[CLS]`. Loss = `L_bio + 0.5 * L_cls`. Metrics in `src/evaluation/metrics.py` (token F1, span F1, sentiment acc/macro-F1, joint F1).

## Non-obvious correctness invariants

These are **not optimizations** — violating any of them silently breaks the model:

- **Retriever self-exclusion is mandatory.** At train time the query is already in the index; without `query_id` exclusion the top-1 result is always the query itself. `Retriever.retrieve(query_vec, query_id=record["id"])` — never pass `None` at train time.
- **BIO labels must be `-100` for all non-query tokens.** `RetrievalABSADataset` sets `bio_labels = -100` for every token in the retrieved portion, plus `[CLS]`, `[SEP]`, aspect tokens, and padding. BIO CE loss uses `ignore_index=-100`. Only query sentence tokens carry real BIO labels.
- **Query is never truncated first.** Token budget: query ≤ `query_budget=100`, remaining split evenly across `top_k` retrieved items. If query exceeds 100 tokens, shrink retrieved budget and log a warning — do NOT shorten the query.
- **Aspect category = full Category#Attribute.** Giữ nguyên `FOOD#QUALITY`, `SERVICE#GENERAL`, v.v. Không rút gọn thành coarse prefix.
- **Drop `conflict` polarity at parse time; drop `target="NULL"` from BIO only.** NULL opinions still go into classification + contrastive builders. 3-class polarity: `positive / negative / neutral`.

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
- **Improvement scope.** Follow `IMPROVE.md` for current improvement plan. Features still deferred: CRF layer, differential LR, hard negatives, E2E fine-tune. Only implement when ablation results (GĐ 2) justify them.
- **SemEval 2016 only.** SemEval 2015 dropped — its train+test is a subset of 2016 train, causing massive dedup loss and 1:1 train/test ratio. Using 2016 alone gives 2,507 train opinions, 0.1% leakage, 3:1 ratio.
- **SB2 is unusable.** SB2 files have review-level opinions (no target, no char offset) — completely different task from sentence-level ABSA.
