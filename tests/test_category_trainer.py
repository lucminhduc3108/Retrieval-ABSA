import torch
from torch.utils.data import DataLoader, TensorDataset

from src.absa.category_trainer import (
    _tune_thresholds, _apply_thresholds,
    _tune_global_threshold, _apply_global_threshold,
    tune_topk, apply_topk,
    CategoryTrainer,
)
from src.data.category_builder import NUM_CATEGORIES, CATEGORY_LIST


def test_tune_thresholds_returns_correct_length():
    logits = torch.randn(20, NUM_CATEGORIES)
    labels = torch.zeros(20, NUM_CATEGORIES)
    labels[:10, 0] = 1.0
    thresholds = _tune_thresholds(logits, labels)
    assert len(thresholds) == NUM_CATEGORIES
    assert all(0.05 <= t <= 0.90 for t in thresholds)


def test_apply_thresholds_returns_sets():
    logits = torch.zeros(2, NUM_CATEGORIES)
    logits[0, 0] = 5.0
    logits[1, 2] = 5.0
    thresholds = [0.5] * NUM_CATEGORIES
    result = _apply_thresholds(logits, thresholds)
    assert len(result) == 2
    assert CATEGORY_LIST[0] in result[0]
    assert CATEGORY_LIST[2] in result[1]


def test_apply_thresholds_empty_when_all_below():
    logits = torch.full((2, NUM_CATEGORIES), -5.0)
    thresholds = [0.5] * NUM_CATEGORIES
    result = _apply_thresholds(logits, thresholds)
    assert all(len(s) == 0 for s in result)


def test_tune_global_threshold_returns_float():
    logits = torch.randn(20, NUM_CATEGORIES)
    labels = torch.zeros(20, NUM_CATEGORIES)
    labels[:10, 0] = 1.0
    t = _tune_global_threshold(logits, labels)
    assert isinstance(t, float)
    assert 0.05 <= t <= 0.90


def test_tune_global_threshold_avoids_low_precision():
    logits = torch.full((30, NUM_CATEGORIES), -5.0)
    labels = torch.zeros(30, NUM_CATEGORIES)
    for i in range(10):
        logits[i, 0] = 2.0
        labels[i, 0] = 1.0
    logits[:, 1] = 0.0  # sigmoid(0)=0.5 → low threshold fires 30 FPs
    t = _tune_global_threshold(logits, labels)
    assert t > 0.50


def test_apply_global_threshold_basic():
    logits = torch.full((2, NUM_CATEGORIES), -5.0)
    logits[0, 0] = 5.0
    logits[0, 1] = 5.0
    logits[1, 2] = 5.0
    result = _apply_global_threshold(logits, 0.5)
    assert CATEGORY_LIST[0] in result[0]
    assert CATEGORY_LIST[1] in result[0]
    assert len(result[0]) == 2
    assert CATEGORY_LIST[2] in result[1]
    assert len(result[1]) == 1


def test_apply_global_threshold_kmax_cap():
    logits = torch.full((1, NUM_CATEGORIES), 10.0)
    result = _apply_global_threshold(logits, 0.5, k_max=3)
    assert len(result[0]) == 3


def test_apply_global_threshold_can_return_empty():
    logits = torch.full((2, NUM_CATEGORIES), -10.0)
    result = _apply_global_threshold(logits, 0.5)
    assert all(len(s) == 0 for s in result)


def test_tune_topk_returns_int_in_range():
    logits = torch.randn(50, NUM_CATEGORIES)
    labels = torch.zeros(50, NUM_CATEGORIES)
    for i in range(50):
        labels[i, i % NUM_CATEGORIES] = 1.0
    k = tune_topk(logits, labels)
    assert isinstance(k, int)
    assert 1 <= k <= 4


def test_apply_topk_returns_exactly_k():
    logits = torch.randn(10, NUM_CATEGORIES)
    for k in [1, 2, 3]:
        result = apply_topk(logits, k)
        assert len(result) == 10
        assert all(len(s) == k for s in result)


def test_apply_topk_picks_highest_probs():
    logits = torch.full((1, NUM_CATEGORIES), -10.0)
    logits[0, 1] = 5.0
    logits[0, 3] = 3.0
    result = apply_topk(logits, 2)
    assert CATEGORY_LIST[1] in result[0]
    assert CATEGORY_LIST[3] in result[0]
    assert len(result[0]) == 2


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
    assert "threshold" in metrics
    assert isinstance(metrics["threshold"], float)
    assert "thresholds" in metrics
    assert len(metrics["thresholds"]) == NUM_CATEGORIES
