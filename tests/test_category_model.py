import torch

from src.absa.category_model import AsymmetricLoss, CategoryDetector
from src.data.category_builder import NUM_CATEGORIES


def _make_batch(batch_size=2, seq_len=16, vocab_size=128):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, attention_mask


def test_forward_shape():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=NUM_CATEGORIES)
    ids, mask = _make_batch()
    out = model(ids, mask)
    assert out["logits"].shape == (2, NUM_CATEGORIES)
    assert out["loss"] is None


def test_forward_with_labels():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=NUM_CATEGORIES)
    ids, mask = _make_batch()
    labels = torch.zeros(2, NUM_CATEGORIES)
    labels[0, 0] = 1.0
    labels[1, 2] = 1.0
    out = model(ids, mask, category_labels=labels)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_pos_weight():
    pw = torch.ones(NUM_CATEGORIES) * 2.0
    model = CategoryDetector(model_name="microsoft/deberta-v3-base",
                             num_categories=NUM_CATEGORIES, pos_weight=pw)
    ids, mask = _make_batch()
    labels = torch.zeros(2, NUM_CATEGORIES)
    labels[0, 0] = 1.0
    out = model(ids, mask, category_labels=labels)
    assert out["loss"] is not None


def test_gradient_flows():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=NUM_CATEGORIES)
    ids, mask = _make_batch()
    labels = torch.zeros(2, NUM_CATEGORIES)
    labels[0, 0] = 1.0
    out = model(ids, mask, category_labels=labels)
    out["loss"].backward()
    assert model.category_head.weight.grad is not None


def test_asl_loss_forward():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=NUM_CATEGORIES,
                             use_asl=True, asl_gamma_neg=4, asl_gamma_pos=0, asl_margin=0.05)
    ids, mask = _make_batch()
    labels = torch.zeros(2, NUM_CATEGORIES)
    labels[0, 0] = 1.0
    out = model(ids, mask, category_labels=labels)
    assert out["logits"].shape == (2, NUM_CATEGORIES)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_cat_attention_forward():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=NUM_CATEGORIES,
                             use_cat_attention=True)
    ids, mask = _make_batch()
    labels = torch.zeros(2, NUM_CATEGORIES)
    labels[0, 2] = 1.0
    out = model(ids, mask, category_labels=labels)
    assert out["logits"].shape == (2, NUM_CATEGORIES)
    assert out["loss"] is not None
