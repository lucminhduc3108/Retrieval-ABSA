# GĐ 3: CRF Layer + Retrieval Improvements

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CRF layer to BIO head (improve span detection), then improve retrieval quality via larger effective batch, tuned hyperparams, and hard negative mining.

**Architecture:** Phase 3A adds a CRF layer on top of the existing Linear(768,3) emission layer in `RetrievalABSA`, using `pytorch-crf`. The model accepts a `use_crf` flag for backward compatibility. Phase 3B adds gradient accumulation to the embedding trainer (effective batch 64), tunes retrieval hyperparams (tau, top_k, threshold), and introduces embedding-based hard negative mining for contrastive triplets.

**Tech Stack:** PyTorch, pytorch-crf (torchcrf), HuggingFace Transformers, FAISS, numpy.

---

## File Map

### Phase 3A: CRF Layer

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `requirements.txt` | Add `pytorch-crf` |
| Modify | `src/absa/model.py` | Add CRF layer, `use_crf` flag, CRF loss + decode |
| Modify | `src/absa/dataset.py` | Add `crf_mask` tensor to output |
| Modify | `src/absa/trainer.py` | Pass `crf_mask`, CRF decode in evaluate |
| Modify | `scripts/04_train_absa.py` | Wire `use_crf` from config |
| Modify | `scripts/05_evaluate.py` | Wire `use_crf` from config |
| Create | `configs/absa_crf.yaml` | CRF-enabled ABSA config |
| Modify | `tests/test_absa_model.py` | CRF model tests |
| Modify | `tests/test_absa_dataset.py` | crf_mask test |
| Modify | `tests/test_absa_trainer.py` | CRF trainer tests |

### Phase 3B-1: Retrieval Hyperparams

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/embedding/trainer.py` | Add `grad_accum_steps` |
| Modify | `scripts/02_train_embedding.py` | Wire `grad_accum_steps` from config |
| Create | `configs/embedding_v2.yaml` | Embedding config with grad_accum + tau=0.12 |
| Create | `configs/retrieval_v2.yaml` | top_k=2, threshold=0.3 |
| Modify | `tests/test_embedding_trainer.py` | Grad accum test |

### Phase 3B-2: Hard Negative Mining

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/data/contrastive_builder.py` | Add `build_hard_negative_triplets()` |
| Create | `scripts/build_hard_triplets.py` | Script to encode records + mine hard negatives |
| Create | `configs/embedding_v3.yaml` | Config pointing to hard triplets |
| Modify | `tests/test_contrastive_builder.py` | Hard negative mining tests |

---

## Phase 3A: CRF Layer

### Task 1: Add pytorch-crf and CRF to ABSA model

**Files:**
- Modify: `requirements.txt`
- Modify: `src/absa/model.py`
- Modify: `tests/test_absa_model.py`

- [ ] **Step 1: Write failing tests for CRF model**

Append to `tests/test_absa_model.py`:

```python
def test_crf_forward_shapes():
    m = RetrievalABSA(use_crf=True)
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones_like(ids)
    out = m(ids, mask)
    assert out["bio_logits"].shape == (2, 32, 3)
    assert out["sentiment_logits"].shape == (2, 3)
    assert out["loss"] is None


def test_crf_forward_with_labels_returns_loss():
    m = RetrievalABSA(use_crf=True)
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones_like(ids)
    bio = torch.zeros(2, 32, dtype=torch.long)
    bio[:, 0] = -100  # [CLS]
    sent = torch.zeros(2, dtype=torch.long)
    crf_mask = bio != -100
    out = m(ids, mask, bio_labels=bio, sentiment_label=sent, crf_mask=crf_mask)
    assert out["loss"].dim() == 0
    assert out["loss"].item() > 0


def test_crf_all_ignored_returns_zero_bio_loss():
    m = RetrievalABSA(use_crf=True)
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones_like(ids)
    bio = torch.full((2, 32), -100, dtype=torch.long)
    sent = torch.zeros(2, dtype=torch.long)
    crf_mask = bio != -100  # all False
    out = m(ids, mask, bio_labels=bio, sentiment_label=sent, crf_mask=crf_mask)
    assert out["loss_bio"].item() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_absa_model.py -v -k "crf"`
Expected: FAIL — `RetrievalABSA() got an unexpected keyword argument 'use_crf'`

- [ ] **Step 3: Add pytorch-crf to requirements.txt**

Add line to `requirements.txt`:
```
pytorch-crf>=0.7.2
```

Run: `pip install pytorch-crf`

- [ ] **Step 4: Implement CRF in model.py**

Replace full content of `src/absa/model.py`:

```python
import torch
import torch.nn as nn
from transformers import AutoModel


class RetrievalABSA(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_bio_labels: int = 3, num_sent_labels: int = 3,
                 lambda_cls: float = 0.5, dropout: float = 0.1,
                 cls_class_weights: list[float] | None = None,
                 use_crf: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.bio_head = nn.Linear(hidden, num_bio_labels)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_sent_labels),
        )
        self.lambda_cls = lambda_cls
        self.use_crf = use_crf
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_bio_labels, batch_first=True)
        self.bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        cls_weight = torch.tensor(cls_class_weights, dtype=torch.float32) if cls_class_weights else None
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=cls_weight)

    def forward(self, input_ids, attention_mask,
                bio_labels=None, sentiment_label=None,
                crf_mask=None) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]

        bio_logits = self.bio_head(sequence_output)
        sentiment_logits = self.sentiment_head(cls_output)

        loss = None
        loss_bio = None
        loss_cls = None

        if bio_labels is not None and sentiment_label is not None:
            if self.use_crf and crf_mask is not None and crf_mask.any():
                safe_labels = bio_labels.clone()
                safe_labels[safe_labels == -100] = 0
                loss_bio = -self.crf(bio_logits.float(), safe_labels,
                                     mask=crf_mask, reduction='mean')
            elif self.use_crf and (crf_mask is None or not crf_mask.any()):
                loss_bio = torch.tensor(0.0, device=input_ids.device)
            else:
                loss_bio = self.bio_loss_fn(
                    bio_logits.view(-1, bio_logits.size(-1)), bio_labels.view(-1))
                if torch.isnan(loss_bio):
                    loss_bio = torch.tensor(0.0, device=input_ids.device)
            loss_cls = self.cls_loss_fn(sentiment_logits, sentiment_label)
            loss = loss_bio + self.lambda_cls * loss_cls

        return {
            "bio_logits": bio_logits,
            "sentiment_logits": sentiment_logits,
            "loss": loss,
            "loss_bio": loss_bio,
            "loss_cls": loss_cls,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_absa_model.py -v`
Expected: ALL PASS (both old and new tests)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/absa/model.py tests/test_absa_model.py
git commit -m "feat(absa): add CRF layer option to RetrievalABSA model"
```

---

### Task 2: Add crf_mask to dataset

**Files:**
- Modify: `src/absa/dataset.py`
- Modify: `tests/test_absa_dataset.py`

- [ ] **Step 1: Write failing test for crf_mask**

Append to `tests/test_absa_dataset.py`:

```python
def test_crf_mask_present_and_matches_bio_labels():
    rec = _make_bio_record(bio_tags=["O", "B-ASP", "O", "O"])
    ds = RetrievalABSADataset([rec], retriever=None, tokenizer_name="microsoft/deberta-v3-base",
                               embedding_model=None, max_length=64, top_k=0)
    item = ds[0]
    assert "crf_mask" in item
    assert item["crf_mask"].dtype == torch.bool
    assert item["crf_mask"].shape == (64,)
    expected_mask = item["bio_labels"] != -100
    assert (item["crf_mask"] == expected_mask).all()


def test_crf_mask_implicit_all_false():
    rec = _make_bio_record(implicit=True, bio_tags=["O", "O", "O", "O"])
    ds = RetrievalABSADataset([rec], retriever=None, tokenizer_name="microsoft/deberta-v3-base",
                               embedding_model=None, max_length=64, top_k=0)
    item = ds[0]
    assert not item["crf_mask"].any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_absa_dataset.py -v -k "crf_mask"`
Expected: FAIL — `KeyError: 'crf_mask'`

- [ ] **Step 3: Add crf_mask to dataset __getitem__**

In `src/absa/dataset.py`, replace the return statement in `__getitem__` (line 143-149):

```python
        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bio_labels": torch.tensor(all_labels, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "query_id": record["id"],
            "crf_mask": torch.tensor([l != IGNORE_INDEX for l in all_labels], dtype=torch.bool),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_absa_dataset.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/absa/dataset.py tests/test_absa_dataset.py
git commit -m "feat(absa): add crf_mask tensor to dataset output"
```

---

### Task 3: Update trainer for CRF decode

**Files:**
- Modify: `src/absa/trainer.py`
- Modify: `tests/test_absa_trainer.py`

- [ ] **Step 1: Write failing test for CRF trainer**

In `tests/test_absa_trainer.py`, modify `_make_loader` to include `crf_mask` and add a CRF test:

Replace the `_make_loader` function:

```python
def _make_loader(n=4, seq_len=32):
    input_ids = torch.randint(0, 1000, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    bio_labels = torch.zeros(n, seq_len, dtype=torch.long)
    bio_labels[:, 1] = 1  # B-ASP
    bio_labels[:, 2] = 2  # I-ASP
    bio_labels[:, 0] = -100  # [CLS]
    sentiment_label = torch.tensor([0, 1, 2, 0][:n], dtype=torch.long)
    crf_mask = bio_labels != -100

    ds = TensorDataset(input_ids, attention_mask, bio_labels, sentiment_label, crf_mask)

    class _Wrapper:
        def __init__(self, ds, batch_size):
            self._ds = ds
            self._bs = batch_size

        def __len__(self):
            return (len(self._ds) + self._bs - 1) // self._bs

        def __iter__(self):
            loader = DataLoader(self._ds, batch_size=self._bs)
            for ids, mask, bio, sent, crf_m in loader:
                yield {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "bio_labels": bio,
                    "sentiment_label": sent,
                    "crf_mask": crf_m,
                }

    return _Wrapper(ds, batch_size=n)
```

Add new test:

```python
def test_crf_evaluate_returns_expected_keys():
    torch.manual_seed(0)
    model = RetrievalABSA(use_crf=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ABSATrainer(model, optimizer, scheduler=None,
                          device="cpu", log_path="")
    loader = _make_loader()
    result = trainer.evaluate(loader)
    expected_keys = {"loss", "bio_token_f1", "span_f1",
                     "sentiment_acc", "sentiment_macro_f1", "joint_f1"}
    assert expected_keys.issubset(set(result.keys()))
    assert 0 <= result["sentiment_acc"] <= 1


def test_crf_train_returns_history():
    torch.manual_seed(0)
    model = RetrievalABSA(use_crf=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ABSATrainer(model, optimizer, scheduler=None,
                          device="cpu", log_path="")
    loader = _make_loader()
    history = trainer.train(loader, loader, epochs=2)
    assert len(history) == 2
    assert "train_loss" in history[0]
    assert "joint_f1" in history[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_absa_trainer.py -v -k "crf"`
Expected: FAIL — trainer doesn't handle CRF decode

- [ ] **Step 3: Update trainer.py**

In `src/absa/trainer.py`, modify `_run_batch` to pass `crf_mask`:

```python
    def _run_batch(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast("cuda", enabled=self.use_fp16):
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bio_labels=batch["bio_labels"],
                sentiment_label=batch["sentiment_label"],
                crf_mask=batch.get("crf_mask"),
            )
        return out
```

Replace the `evaluate` method (lines 113-175) with:

```python
    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        total_loss = 0
        all_bio_preds = []
        all_bio_golds = []
        all_sent_preds = []
        all_sent_golds = []
        all_pred_spans_with_pol = []
        all_gold_spans_with_pol = []
        num_batches = 0

        use_crf = getattr(self.model, 'use_crf', False)

        for batch in loader:
            out = self._run_batch(batch)
            if out["loss"] is not None:
                total_loss += out["loss"].item()
            num_batches += 1

            bio_logits = out["bio_logits"]
            bio_golds = batch["bio_labels"].to(self.device)

            if use_crf:
                crf_mask = batch["crf_mask"].to(self.device)
                decoded = self.model.crf.decode(bio_logits.float(), mask=crf_mask)
            else:
                decoded = None

            sent_logits = out["sentiment_logits"]
            sent_preds = sent_logits.argmax(dim=-1)
            sent_golds = batch["sentiment_label"].to(self.device)

            for i in range(bio_golds.size(0)):
                mask = bio_golds[i] != -100
                gold_seq = bio_golds[i][mask].cpu().tolist()

                if decoded is not None:
                    pred_seq = decoded[i]
                else:
                    pred_seq = bio_logits[i].argmax(dim=-1)[mask].cpu().tolist()

                all_bio_preds.append(pred_seq)
                all_bio_golds.append(gold_seq)

                pred_sp = extract_spans(pred_seq)
                gold_sp = extract_spans(gold_seq)

                sp = sent_preds[i].item()
                sg = sent_golds[i].item()
                all_sent_preds.append(sp)
                all_sent_golds.append(sg)

                all_pred_spans_with_pol.append(
                    [(s, e, sp) for s, e in pred_sp])
                all_gold_spans_with_pol.append(
                    [(s, e, sg) for s, e in gold_sp])

        avg_loss = total_loss / max(num_batches, 1)
        bio_m = bio_token_metrics(all_bio_preds, all_bio_golds)
        span_m = span_f1(
            [extract_spans(s) for s in all_bio_preds],
            [extract_spans(s) for s in all_bio_golds],
        )
        sent_m = sentiment_metrics(all_sent_preds, all_sent_golds)
        joint = joint_f1(all_pred_spans_with_pol, all_gold_spans_with_pol)

        return {
            "loss": avg_loss,
            "bio_token_f1": bio_m["f1"],
            "span_f1": span_m["f1"],
            "sentiment_acc": sent_m["accuracy"],
            "sentiment_macro_f1": sent_m["macro_f1"],
            "joint_f1": joint,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_absa_trainer.py -v`
Expected: ALL PASS (both old and new tests)

- [ ] **Step 5: Commit**

```bash
git add src/absa/trainer.py tests/test_absa_trainer.py
git commit -m "feat(absa): CRF decode support in trainer evaluate loop"
```

---

### Task 4: Wire use_crf in scripts + create config

**Files:**
- Modify: `scripts/04_train_absa.py:109-116`
- Modify: `scripts/05_evaluate.py:95-102`
- Create: `configs/absa_crf.yaml`

- [ ] **Step 1: Create CRF config**

Create `configs/absa_crf.yaml`:

```yaml
model_name: microsoft/deberta-v3-base
num_bio_labels: 3
num_sent_labels: 3
lambda_cls: 0.5
dropout: 0.1
batch_size: 4
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
log_path: logs/absa_crf.jsonl
cls_class_weights: [1.00, 1.70, 4.33]
grad_accum_steps: 8
use_crf: true
```

- [ ] **Step 2: Wire use_crf in 04_train_absa.py**

In `scripts/04_train_absa.py`, modify the model creation (lines 109-116) to add `use_crf`:

```python
    model = RetrievalABSA(
        model_name=cfg["model_name"],
        num_bio_labels=cfg["num_bio_labels"],
        num_sent_labels=cfg["num_sent_labels"],
        lambda_cls=cfg["lambda_cls"],
        dropout=cfg["dropout"],
        cls_class_weights=cfg.get("cls_class_weights"),
        use_crf=cfg.get("use_crf", False),
    ).to(device)
```

- [ ] **Step 3: Wire use_crf in 05_evaluate.py**

In `scripts/05_evaluate.py`, modify the model creation (lines 95-102) to add `use_crf`:

```python
    model = RetrievalABSA(
        model_name=cfg["model_name"],
        num_bio_labels=cfg["num_bio_labels"],
        num_sent_labels=cfg["num_sent_labels"],
        lambda_cls=cfg["lambda_cls"],
        dropout=cfg["dropout"],
        cls_class_weights=cfg.get("cls_class_weights"),
        use_crf=cfg.get("use_crf", False),
    ).to(device)
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Smoke test (local, no GPU)**

Run: `python scripts/04_train_absa.py --config configs/absa_crf.yaml --no_retrieval --limit 8 --epochs 1 --grad_accum_steps 1`
Expected: completes without error, prints epoch log with joint_f1

- [ ] **Step 6: Commit**

```bash
git add configs/absa_crf.yaml scripts/04_train_absa.py scripts/05_evaluate.py
git commit -m "feat(absa): wire use_crf flag in training and evaluation scripts"
```

---

### Kaggle Session 1: CRF No-Retrieval

After Task 4, create Kaggle notebook `nb3_absa_crf.ipynb`:
- **Input:** Kaggle dataset with embedding checkpoint (reuse from GĐ 2), processed data
- **Run:** `python scripts/04_train_absa.py --config configs/absa_crf.yaml --no_retrieval --grad_accum_steps 8`
- **Output:** `checkpoints/absa/best.pt` → save to Kaggle dataset

Then `nb4_eval_crf.ipynb`:
- **Input:** ABSA checkpoint from nb3
- **Run:** `python scripts/05_evaluate.py --config configs/absa_crf.yaml --checkpoint checkpoints/absa/best.pt --no_retrieval`
- **Compare:** CRF no-retrieval vs GĐ 2 no-retrieval (Joint F1 0.6104, Span F1 0.6489)

---

## Phase 3B-1: Retrieval Hyperparams

### Task 5: Add gradient accumulation to embedding trainer

**Files:**
- Modify: `src/embedding/trainer.py`
- Modify: `tests/test_embedding_trainer.py`

- [ ] **Step 1: Write failing test for grad_accum**

Append to `tests/test_embedding_trainer.py`:

```python
def test_trainer_with_grad_accum(tmp_path):
    torch.manual_seed(0)
    p = tmp_path / "triplets.jsonl"
    write_jsonl([_make_triplet(i) for i in range(8)], str(p))
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    ds = ContrastiveTripletDataset(str(p), tok, max_length=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)

    model = ContrastiveEmbedder(proj_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ContrastiveTrainer(model, optimizer, scheduler=None,
                                 tau=0.07, device="cpu", log_path="",
                                 grad_accum_steps=2)
    result = trainer.evaluate_recall(loader, k_list=(1, 3))
    assert "recall@1" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedding_trainer.py -v -k "grad_accum"`
Expected: FAIL — `ContrastiveTrainer() got an unexpected keyword argument 'grad_accum_steps'`

- [ ] **Step 3: Add grad_accum_steps to embedding trainer**

Replace `src/embedding/trainer.py` with:

```python
import json
import logging
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from src.embedding.loss import infonce_loss

logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    def __init__(self, model, optimizer, scheduler, tau, device,
                 log_path, grad_clip=1.0, use_fp16=False,
                 grad_accum_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tau = tau
        self.device = device
        self.log_path = log_path
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.use_fp16 = use_fp16 and device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_fp16 else None
        if self.use_fp16:
            logger.info("fp16 mixed precision enabled")
        if self.grad_accum_steps > 1:
            logger.info("Gradient accumulation: %d steps", self.grad_accum_steps)

    def _run_batch(self, batch):
        keys = ["anchor_input_ids", "anchor_attention_mask",
                "pos_input_ids", "pos_attention_mask",
                "neg1_input_ids", "neg1_attention_mask",
                "neg2_input_ids", "neg2_attention_mask"]
        batch = {k: batch[k].to(self.device) for k in keys}
        with autocast("cuda", enabled=self.use_fp16):
            out = self.model(
                batch["anchor_input_ids"], batch["anchor_attention_mask"],
                batch["pos_input_ids"], batch["pos_attention_mask"],
                batch["neg1_input_ids"], batch["neg1_attention_mask"],
                batch["neg2_input_ids"], batch["neg2_attention_mask"],
            )
            negatives = []
            if out["neg1_vecs"] is not None:
                negatives.append(out["neg1_vecs"])
            if out["neg2_vecs"] is not None:
                negatives.append(out["neg2_vecs"])
            loss = infonce_loss(out["anchor_vecs"], out["pos_vecs"],
                                negatives=negatives if negatives else None,
                                tau=self.tau)
        return loss, out

    def train(self, train_loader, val_loader, epochs, patience=3,
              ckpt_path=None) -> list[dict]:
        history = []
        best_recall = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            self.optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                loss, _ = self._run_batch(batch)
                scaled_loss = loss / self.grad_accum_steps
                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                total_loss += loss.item()

                is_accum_step = (step + 1) % self.grad_accum_steps == 0
                is_last_step = (step + 1) == len(train_loader)
                if is_accum_step or is_last_step:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        if self.grad_clip > 0:
                            clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.grad_clip > 0:
                            clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            recall = self.evaluate_recall(val_loader)
            record = {"epoch": epoch, "train_loss": avg_loss, **recall}
            history.append(record)
            logger.info("Epoch %d: loss=%.4f recall@3=%.4f", epoch, avg_loss, recall["recall@3"])

            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            if recall["recall@3"] > best_recall:
                best_recall = recall["recall@3"]
                patience_counter = 0
                if ckpt_path:
                    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info("Saved best model (recall@3=%.4f)", best_recall)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return history

    @torch.no_grad()
    def evaluate_recall(self, val_loader, k_list=(1, 3, 5)) -> dict:
        self.model.eval()
        all_anchor = []
        all_pos = []
        for batch in val_loader:
            batch_dev = {k: batch[k].to(self.device) for k in batch}
            a_vec = self.model.encode(batch_dev["anchor_input_ids"],
                                      batch_dev["anchor_attention_mask"])
            p_vec = self.model.encode(batch_dev["pos_input_ids"],
                                      batch_dev["pos_attention_mask"])
            all_anchor.append(a_vec.cpu())
            all_pos.append(p_vec.cpu())

        anchors = torch.cat(all_anchor, dim=0)
        positives = torch.cat(all_pos, dim=0)
        sim = anchors @ positives.T
        N = sim.size(0)

        result = {}
        for k in k_list:
            topk_indices = sim.topk(min(k, N), dim=1).indices
            hits = sum(1 for i in range(N) if i in topk_indices[i].tolist())
            result[f"recall@{k}"] = hits / N
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embedding_trainer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/trainer.py tests/test_embedding_trainer.py
git commit -m "feat(embedding): add gradient accumulation to ContrastiveTrainer"
```

---

### Task 6: Wire grad_accum in embedding script + create configs

**Files:**
- Modify: `scripts/02_train_embedding.py:54-66`
- Create: `configs/embedding_v2.yaml`
- Create: `configs/retrieval_v2.yaml`

- [ ] **Step 1: Create embedding_v2 config**

Create `configs/embedding_v2.yaml`:

```yaml
model_name: microsoft/deberta-v3-base
proj_dim: 256
tau: 0.12
batch_size: 16
epochs: 15
lr: 2.0e-5
weight_decay: 0.01
warmup_ratio: 0.1
max_seq_length: 128
grad_clip: 1.0
patience: 5
seed: 42
triplets_path: data/processed/contrastive_triplets.jsonl
val_ratio: 0.1
ckpt_dir: checkpoints/embedding
log_path: logs/embedding_v2.jsonl
grad_accum_steps: 4
```

- [ ] **Step 2: Create retrieval_v2 config**

Create `configs/retrieval_v2.yaml`:

```yaml
top_k: 2
threshold: 0.3
index_dir: indexes
```

- [ ] **Step 3: Wire grad_accum_steps in 02_train_embedding.py**

In `scripts/02_train_embedding.py`, modify the scheduler calculation and trainer creation.

Replace lines 54-66 (from `optimizer = ...` to end of trainer creation):

```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    epochs = args.epochs if args.epochs else cfg["epochs"]
    grad_accum = cfg.get("grad_accum_steps", 1)
    total_steps = (len(train_loader) * epochs) // grad_accum
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    trainer = ContrastiveTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        tau=cfg["tau"], device=device,
        log_path=cfg["log_path"], grad_clip=cfg["grad_clip"],
        use_fp16=device == "cuda",
        grad_accum_steps=grad_accum,
    )
```

- [ ] **Step 4: Smoke test**

Run: `python scripts/02_train_embedding.py --config configs/embedding_v2.yaml --limit 16 --epochs 1`
Expected: completes without error

- [ ] **Step 5: Commit**

```bash
git add configs/embedding_v2.yaml configs/retrieval_v2.yaml scripts/02_train_embedding.py
git commit -m "feat(embedding): wire grad_accum in script, add v2 configs (tau=0.12, accum=4)"
```

---

### Kaggle Session 2: Improved Retrieval

After Task 6, create 4 Kaggle notebooks:
1. `nb1_embedding_v2.ipynb` — retrain embedding with `configs/embedding_v2.yaml`
2. `nb2_index_v2.ipynb` — rebuild FAISS index
3. `nb3_absa_crf_ret.ipynb` — train CRF ABSA with retrieval, using `configs/absa_crf.yaml` + `configs/retrieval_v2.yaml`
4. `nb4_eval_crf_ret.ipynb` — evaluate, compare CRF no-retrieval vs CRF retrieval

---

## Phase 3B-2: Hard Negative Mining

### Task 7: Add embedding-based hard negative mining

**Files:**
- Modify: `src/data/contrastive_builder.py`
- Modify: `tests/test_contrastive_builder.py`

- [ ] **Step 1: Write failing tests for hard negative mining**

Append to `tests/test_contrastive_builder.py`:

```python
import numpy as np
from src.data.contrastive_builder import build_hard_negative_triplets


def test_hard_neg_mining_returns_valid_triplets():
    recs = [
        {"id": "r0", "sentence": "s0", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r1", "sentence": "s1", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r2", "sentence": "s2", "aspect_category": "FOOD#QUALITY", "polarity": "negative"},
        {"id": "r3", "sentence": "s3", "aspect_category": "SERVICE#GENERAL", "polarity": "positive"},
    ]
    vectors = np.array([
        [1.0, 0.0], [0.5, 0.5], [0.9, 0.1], [0.3, 0.7],
    ], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    triplets = build_hard_negative_triplets(recs, vectors)
    assert len(triplets) > 0
    for t in triplets:
        assert t["anchor_aspect"] == t["positive_aspect"]
        assert t["anchor_polarity"] == t["positive_polarity"]
        assert t["anchor_aspect"] == t["neg1_aspect"]
        assert t["anchor_polarity"] != t["neg1_polarity"]
        assert t["anchor_aspect"] != t["neg2_aspect"]
        assert t["anchor_polarity"] == t["neg2_polarity"]


def test_hard_neg_picks_most_similar():
    recs = [
        {"id": "r0", "sentence": "s0", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r1", "sentence": "s1", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r2", "sentence": "s2", "aspect_category": "FOOD#QUALITY", "polarity": "negative"},
        {"id": "r3", "sentence": "s3", "aspect_category": "FOOD#QUALITY", "polarity": "negative"},
        {"id": "r4", "sentence": "s4", "aspect_category": "SERVICE#GENERAL", "polarity": "positive"},
    ]
    vectors = np.array([
        [1.0, 0.0],   # r0 anchor
        [0.5, 0.5],   # r1
        [0.95, 0.05], # r2 - very similar to r0, neg1 candidate (same asp, diff pol)
        [0.1, 0.9],   # r3 - dissimilar to r0, neg1 candidate
        [0.3, 0.7],   # r4
    ], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    triplets = build_hard_negative_triplets(recs, vectors)
    r0_triplet = next(t for t in triplets if t["anchor_id"] == "r0")
    assert r0_triplet["neg1_id"] == "r2"


def test_hard_neg_skip_when_no_candidates():
    recs = [
        {"id": "r0", "sentence": "s0", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r1", "sentence": "s1", "aspect_category": "FOOD#QUALITY", "polarity": "negative"},
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    triplets = build_hard_negative_triplets(recs, vectors)
    assert triplets == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contrastive_builder.py -v -k "hard_neg"`
Expected: FAIL — `ImportError: cannot import name 'build_hard_negative_triplets'`

- [ ] **Step 3: Implement hard negative mining**

Add to `src/data/contrastive_builder.py`:

```python
import numpy as np


def build_hard_negative_triplets(cls_records: list[dict],
                                  vectors: np.ndarray,
                                  seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    sim = vectors @ vectors.T

    by_asp_pol = defaultdict(list)
    by_asp = defaultdict(list)
    by_pol = defaultdict(list)
    for i, r in enumerate(cls_records):
        by_asp_pol[(r["aspect_category"], r["polarity"])].append(i)
        by_asp[r["aspect_category"]].append(i)
        by_pol[r["polarity"]].append(i)

    triplets = []
    for i, anchor in enumerate(cls_records):
        a_asp, a_pol, a_id = anchor["aspect_category"], anchor["polarity"], anchor["id"]

        pos_indices = [j for j in by_asp_pol[(a_asp, a_pol)] if j != i]
        if not pos_indices:
            continue

        neg1_indices = [j for j in by_asp[a_asp]
                        if cls_records[j]["polarity"] != a_pol and j != i]
        if not neg1_indices:
            continue

        neg2_indices = [j for j in by_pol[a_pol]
                        if cls_records[j]["aspect_category"] != a_asp and j != i]
        if not neg2_indices:
            continue

        pos_idx = rng.choice(pos_indices)

        neg1_sims = sim[i, neg1_indices]
        neg1_idx = neg1_indices[int(np.argmax(neg1_sims))]

        neg2_sims = sim[i, neg2_indices]
        neg2_idx = neg2_indices[int(np.argmax(neg2_sims))]

        pos = cls_records[pos_idx]
        neg1 = cls_records[neg1_idx]
        neg2 = cls_records[neg2_idx]

        triplets.append({
            "anchor_id": a_id, "anchor_sentence": anchor["sentence"],
            "anchor_aspect": a_asp, "anchor_polarity": a_pol,
            "positive_id": pos["id"], "positive_sentence": pos["sentence"],
            "positive_aspect": pos["aspect_category"], "positive_polarity": pos["polarity"],
            "neg1_id": neg1["id"], "neg1_sentence": neg1["sentence"],
            "neg1_aspect": neg1["aspect_category"], "neg1_polarity": neg1["polarity"],
            "neg2_id": neg2["id"], "neg2_sentence": neg2["sentence"],
            "neg2_aspect": neg2["aspect_category"], "neg2_polarity": neg2["polarity"],
        })

    return triplets
```

Add `import numpy as np` to the top of the file (alongside existing imports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contrastive_builder.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/contrastive_builder.py tests/test_contrastive_builder.py
git commit -m "feat(data): embedding-based hard negative mining for contrastive triplets"
```

---

### Task 8: Create hard triplets script + config

**Files:**
- Create: `scripts/build_hard_triplets.py`
- Create: `configs/embedding_v3.yaml`

- [ ] **Step 1: Create the script**

Create `scripts/build_hard_triplets.py`:

```python
import argparse
import logging
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.contrastive_builder import build_hard_negative_triplets
from src.embedding.model import ContrastiveEmbedder
from src.retrieval.encoder import encode_records
from src.utils.io import load_yaml, read_jsonl, write_jsonl
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_ckpt", required=True)
    parser.add_argument("--cls_path", default="data/processed/classification.jsonl")
    parser.add_argument("--out_path", default="data/processed/hard_contrastive_triplets.jsonl")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ContrastiveEmbedder(model_name=args.model_name, proj_dim=args.proj_dim)
    model.load_state_dict(torch.load(args.embedding_ckpt, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Loaded embedding: %s", args.embedding_ckpt)

    cls_records = read_jsonl(args.cls_path)
    train_records = [r for r in cls_records if r["split"] == "train"]
    logger.info("Train records: %d", len(train_records))

    vectors = encode_records(train_records, model, tokenizer,
                             max_length=args.max_length, device=device)
    logger.info("Encoded %d vectors, shape %s", len(vectors), vectors.shape)

    triplets = build_hard_negative_triplets(train_records, vectors, seed=args.seed)
    logger.info("Built %d hard negative triplets", len(triplets))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    write_jsonl(triplets, args.out_path)
    logger.info("Saved to %s", args.out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create embedding_v3 config**

Create `configs/embedding_v3.yaml`:

```yaml
model_name: microsoft/deberta-v3-base
proj_dim: 256
tau: 0.12
batch_size: 16
epochs: 15
lr: 2.0e-5
weight_decay: 0.01
warmup_ratio: 0.1
max_seq_length: 128
grad_clip: 1.0
patience: 5
seed: 42
triplets_path: data/processed/hard_contrastive_triplets.jsonl
val_ratio: 0.1
ckpt_dir: checkpoints/embedding
log_path: logs/embedding_v3.jsonl
grad_accum_steps: 4
```

- [ ] **Step 3: Run all tests to verify nothing is broken**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/build_hard_triplets.py configs/embedding_v3.yaml
git commit -m "feat(scripts): hard negative triplet mining script with embedding_v3 config"
```

---

### Kaggle Session 3: Hard Negatives

After Task 8, create 5 Kaggle notebooks:
1. `nb0_hard_triplets.ipynb` — run `scripts/build_hard_triplets.py` with embedding_v2 checkpoint
2. `nb1_embedding_v3.ipynb` — retrain embedding on hard triplets with `configs/embedding_v3.yaml`
3. `nb2_index_v3.ipynb` — rebuild FAISS index
4. `nb3_absa_crf_ret_v3.ipynb` — train CRF ABSA with retrieval + `configs/retrieval_v2.yaml`
5. `nb4_eval_v3.ipynb` — evaluate, compare all variants

---

## Expected Comparison Matrix

| Metric | GĐ2 no-ret | 3A: CRF no-ret | 3B-1: CRF+ret | 3B-2: CRF+ret+hard |
|--------|-----------|----------------|----------------|---------------------|
| Joint F1 | 0.6104 | ? (expect ↑) | ? | ? |
| Span F1 | 0.6489 | ? (expect ↑) | ? | ? |
| Sent Acc | 0.9243 | ~same | ? | ? |
| Sent MacF1 | 0.8234 | ~same | ? | ? |

**Success criteria:**
- 3A: Span F1 > 0.6489 (CRF improves BIO consistency)
- 3B-1: Retrieval delta < -6.4pp (retrieval less harmful with better params)
- 3B-2: Retrieval helps or breaks even vs no-retrieval

---

## Dependency Graph

```
Task 1 ─ Task 2 ─ Task 3 ─ Task 4 ─ [Kaggle Session 1]
                                           │
Task 5 ─ Task 6 ──────────────────── [Kaggle Session 2]
                                           │
Task 7 ─ Task 8 ──────────────────── [Kaggle Session 3]
```

Tasks 1-4 (CRF) and Tasks 5-6 (grad_accum) are independent and can be implemented in parallel.
Tasks 7-8 (hard negatives) are independent of Tasks 1-4 but can also be done in parallel.
Kaggle sessions must be sequential: Session 1 → Session 2 → Session 3.
