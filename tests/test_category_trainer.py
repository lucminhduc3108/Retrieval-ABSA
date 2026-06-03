import torch
from torch.utils.data import DataLoader, TensorDataset

from src.absa.category_trainer import (
    _tune_thresholds, _apply_thresholds, CategoryTrainer,
)
from src.data.category_builder import NUM_CATEGORIES, CATEGORY_LIST


def test_tune_thresholds_returns_correct_length():
    logits = torch.randn(20, NUM_CATEGORIES)
    labels = torch.zeros(20, NUM_CATEGORIES)
    labels[:10, 0] = 1.0
    thresholds = _tune_thresholds(logits, labels)
    assert len(thresholds) == NUM_CATEGORIES
    assert all(0.1 <= t <= 0.5 for t in thresholds)


def test_apply_thresholds_returns_sets():
    logits = torch.zeros(2, NUM_CATEGORIES)
    logits[0, 0] = 5.0
    logits[1, 5] = 5.0
    thresholds = [0.5] * NUM_CATEGORIES
    result = _apply_thresholds(logits, thresholds)
    assert len(result) == 2
    assert CATEGORY_LIST[0] in result[0]
    assert CATEGORY_LIST[5] in result[1]


def test_apply_thresholds_empty_when_all_below():
    logits = torch.full((2, NUM_CATEGORIES), -5.0)
    thresholds = [0.5] * NUM_CATEGORIES
    result = _apply_thresholds(logits, thresholds)
    assert all(len(s) == 0 for s in result)


def _make_loader(n=8, seq_len=16):
    input_ids = torch.randint(0, 100, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    labels = torch.zeros(n, NUM_CATEGORIES)
    labels[:, 0] = 1.0
    ds = TensorDataset(input_ids, attention_mask, labels)

    def collate(batch):
        ids, mask, lab = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(mask),
            "category_labels": torch.stack(lab),
        }
    return DataLoader(ds, batch_size=4, collate_fn=collate)


def test_trainer_evaluate():
    from src.absa.category_model import CategoryDetector
    model = CategoryDetector(num_categories=NUM_CATEGORIES)
    trainer = CategoryTrainer(
        model=model, optimizer=None, scheduler=None,
        device="cpu", log_path="",
    )
    loader = _make_loader()
    metrics = trainer.evaluate(loader)
    assert "category_f1" in metrics
    assert "thresholds" in metrics
    assert len(metrics["thresholds"]) == NUM_CATEGORIES
