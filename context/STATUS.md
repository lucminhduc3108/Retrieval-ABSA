# Project Status — Retrieval-ABSA

**Last updated:** 2026-05-18

---

## Pipeline Overview

| Phase | Script | Status | Output |
|-------|--------|--------|--------|
| 1. Data Prep | `scripts/01_prepare_data.py` | Done | 5,865 records (bio + cls + triplets) |
| 2. Embedding | `scripts/02_train_embedding.py` | Done | `checkpoints/embedding/best.pt` (736MB) |
| 3. FAISS Index | `scripts/03_build_index.py` | Done | `indexes/train.faiss` (4,161 vectors) |
| 4. ABSA Train | `scripts/04_train_absa.py` | Done | `checkpoints/absa/best.pt` (701MB) |
| 5. Evaluate | `scripts/05_evaluate.py` | Done | See results below |

---

## Phase 2 Results (Test Set, 859 opinions — SemEval 2016 SB1 only)

| Metric | Retrieval | No-retrieval | Delta |
|--------|-----------|--------------|-------|
| BIO Token F1 | 0.5418 | **0.6198** | +7.8pp |
| Span F1 | 0.6074 | **0.6489** | +4.2pp |
| Sentiment Acc | 0.8976 | **0.9243** | +2.7pp |
| Sentiment MacF1 | 0.7589 | **0.8234** | +6.5pp |
| **Joint F1** | 0.5460 | **0.6104** | **+6.4pp** |

**Key findings:**
- **Retrieval is hurting** — no-retrieval wins every metric by 2.7–7.8pp
- **Span detection remains bottleneck** — Span F1 vs Sent Acc gap = 27.5pp
- **Sqrt class weights working** — Sent MacF1 0.8234 (up from MVP 0.7619)
- **Implicit opinions**: BIO/Span/Joint = 0 (expected, no target span)

**Full report:** `report_4_5.md`

---

## Phase 3 CRF Experiments (Test Set, 859 opinions — No Retrieval)

| Metric | P2 no-ret | S1 CRF unnorm | S1A CRF norm |
|--------|-----------|---------------|-------------|
| BIO Token F1 | **0.6198** | 0.6373 | 0.5835 |
| Span F1 | 0.6489 | **0.6667** | 0.6237 |
| Sentiment Acc | **0.9243** | 0.8615 | 0.9034 |
| Sentiment MacF1 | **0.8234** | 0.5664 | 0.7612 |
| **Joint F1** | **0.6104** | 0.5648 | 0.5694 |

**S1 (CRF unnormalized):** CRF NLL per-sequence >> CE per-token → sentiment collapsed (MacF1 -25.7pp). Span F1 +1.8pp but Joint F1 -4.6pp.

**S1A (CRF normalized per-token):** Loss scale fixed (epoch 1 loss 1.05 vs 6.44). Sentiment recovered (MacF1 0.76) but span detection regressed below Phase 2 baseline (Span F1 0.6237 vs 0.6489). Joint F1 still -4.1pp vs Phase 2.

**Verdict: CRF dropped.** Tried unnormalized, normalized, both worse than plain CE. BIO label space (3 classes) too simple for CRF to add value over DeBERTa contextual representations. Phase 2 no-retrieval CE (Joint F1=0.6104) remains best baseline.

**Reports:** `report_5_7_gd3_s1.md` (S1), `gd3-session1a-crf-normalized.ipynb` (S1A)

---

## Phase Progress

**Phase 0 (code logic fixes) — DONE ✅**
**Phase 1 (data quality fixes) — DONE ✅**
**Phase 2 (retrain on clean data) — DONE ✅**
- [x] 2a: `01_prepare_data.py` updated for SemEval 2016 only
- [x] 2b: Retrain embedding on clean triplets
- [x] 2c: Rebuild FAISS index
- [x] 2d: Train ABSA (retrieval + no-retrieval)
- [x] 2e: Evaluate both experiments

---

## Phase 3: Code Changes

**3A: CRF Layer** — ❌ DROPPED (tested S1 + S1A, both worse than CE baseline)

**3B-1: Retrieval Hyperparams** — code complete, ready for Kaggle
- [x] `ContrastiveTrainer`: `grad_accum_steps` added
- [x] Script wired, `configs/embedding_v2.yaml` (tau=0.12, accum=4), `configs/retrieval_v2.yaml` (top_k=2, threshold=0.3)

**3B-2: Hard Negative Mining** — code complete, ready for Kaggle
- [x] `build_hard_negative_triplets()` in contrastive_builder
- [x] `scripts/build_hard_triplets.py` created
- [x] `configs/embedding_v3.yaml` created

**Tests:** 70/70 pass

---

## Kaggle Sessions Needed

| Session | What to run | Config |
|---------|------------|--------|
| ~~1: CRF no-retrieval~~ | ~~Done~~ | ~~CRF dropped~~ |
| ~~1A: CRF normalized~~ | ~~Done~~ | ~~CRF dropped~~ |
| **2: Improved retrieval** | retrain embedding → rebuild index → train ABSA CE with retrieval | `embedding_v2.yaml` + `retrieval_v2.yaml` + `absa.yaml` |
| **3: Hard negatives** | `build_hard_triplets.py` → retrain embedding → rebuild index → train ABSA CE | `embedding_v3.yaml` + `retrieval_v2.yaml` + `absa.yaml` |

Sessions 2–3 use plain CE (no CRF). Full pipeline split into nb1–nb4 to avoid T4 OOM.

---

## Kaggle Notebook Status

| Notebook | Kaggle URL | Status |
|----------|-----------|--------|
| P3-S1: CRF no-retrieval | [lcminhc/gd3-session1-crf-no-retrieval](https://www.kaggle.com/code/lcminhc/gd3-session1-crf-no-retrieval) | Done — CRF unnorm, sentiment collapsed |
| P3-S1A: CRF normalized | local `gd3-session1a-crf-normalized.ipynb` | Done — CRF norm, span regressed, CRF dropped |

---

## Training Environment

- **GPU:** Kaggle T4 (free tier)
- **Local:** Windows 11, no GPU training
- **Checkpoints stored:** Kaggle datasets (`lcminhc/retrieval-absa-embedding-ckpt`, `lcminhc/retrieval-absa-ckpt`)

---

## Ý tưởng đang cân nhắc (chưa quyết định)

### Tách retrieval khỏi BIO head

**Vấn đề:** Retrieval gây nhiễu cho BIO/span detection vì:
- Neighbors không mang nhãn BIO → không cung cấp tín hiệu span
- Input phình từ ~128 → ~512 tokens → attention bị pha loãng, BIO head khó tập trung vào query
- Retrieval chỉ thực sự giúp sentiment (nhờ nhãn `[POL]` tường minh)

**Ý tưởng:** Cho BIO head chỉ nhận input ngắn (query + aspect), sentiment head vẫn nhận input dài (query + aspect + neighbors). Retrieval pipeline không thay đổi, chỉ thay đổi cách ABSA model sử dụng nó.

**Các cách implement (đang cân nhắc):**
1. **Hai forward pass** — đơn giản nhất, forward 1 (input ngắn) → BIO, forward 2 (input dài) → sentiment. Tốn gấp đôi compute nhưng dễ implement.
2. **Custom attention mask** — một forward pass, BIO attend chỉ query tokens, sentiment attend toàn bộ. Phức tạp hơn.
3. **Hai backbone riêng** — tách biệt hoàn toàn, nhưng tốn gấp đôi GPU memory (T4 có thể không đủ).

**Trạng thái:** Đang thảo luận, chưa quyết định đưa vào Phase nào (3 hay 4).

---

## Next Actions

1. **Session 2: Improved retrieval (no CRF)** — retrain embedding (tau=0.12, accum=4) → rebuild index (top_k=2, threshold=0.3) → train ABSA CE with retrieval → evaluate. Goal: retrieval delta > 0 vs Phase 2 no-ret baseline (Joint F1=0.6104).
2. **Session 3: Hard negatives (no CRF)** — build hard negative triplets → retrain embedding → rebuild index → train ABSA CE → evaluate. Goal: further improve retrieval contribution.
3. Dựa trên kết quả Sessions 2–3, quyết định có tách retrieval khỏi BIO head hay không
4. Analyze results, plan Phase 4 if needed
