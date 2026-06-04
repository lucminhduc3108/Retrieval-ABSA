import torch

from src.absa.sentiment_model import SentimentPredictor


def _make_batch(batch_size=2, seq_len=32, vocab_size=128, k=2):
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "neighbor_polarities": torch.randint(0, 3, (batch_size, k)),
        "neighbor_scores": torch.rand(batch_size, k),
        "sentiment_label": torch.randint(0, 3, (batch_size,)),
    }


def test_forward_with_retrieval():
    model = SentimentPredictor(use_retrieval=True)
    batch = _make_batch()
    out = model(**batch)
    assert out["logits"].shape == (2, 3)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_forward_no_retrieval():
    model = SentimentPredictor(use_retrieval=False)
    batch = _make_batch()
    del batch["neighbor_polarities"]
    del batch["neighbor_scores"]
    out = model(**batch)
    assert out["logits"].shape == (2, 3)
    assert out["loss"] is not None


def test_forward_retrieval_model_without_neighbors():
    model = SentimentPredictor(use_retrieval=True)
    batch = _make_batch()
    batch["neighbor_polarities"] = None
    batch["neighbor_scores"] = None
    out = model(**batch)
    assert out["logits"].shape == (2, 3)


def test_no_loss_without_labels():
    model = SentimentPredictor(use_retrieval=False)
    ids = torch.randint(0, 128, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    out = model(ids, mask)
    assert out["loss"] is None


def test_gradient_flows_through_label_interp():
    model = SentimentPredictor(use_retrieval=True)
    batch = _make_batch()
    out = model(**batch)
    out["loss"].backward()
    assert model.label_interp.polarity_embedding.weight.grad is not None
    assert model.sentiment_head[0].weight.grad is not None


def test_forward_with_class_weights():
    weights = torch.tensor([1.0, 1.5, 4.0])
    model = SentimentPredictor(use_retrieval=False, class_weights=weights)
    ids = torch.randint(0, 128, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    label = torch.tensor([0, 2])
    out = model(ids, mask, sentiment_label=label)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0
