import torch
from torch.utils.data import DataLoader, TensorDataset

from src.absa.sentiment_trainer import SentimentTrainer
from src.absa.sentiment_model import SentimentPredictor


def _make_loader(n=8, seq_len=16, k=2, use_retrieval=True):
    input_ids = torch.randint(0, 100, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    sentiment_label = torch.randint(0, 3, (n,))

    if use_retrieval:
        nb_pol = torch.randint(0, 3, (n, k))
        nb_scores = torch.rand(n, k)
        ds = TensorDataset(input_ids, attention_mask, sentiment_label,
                           nb_pol, nb_scores)

        def collate(batch):
            ids, mask, lab, pol, sc = zip(*batch)
            return {
                "input_ids": torch.stack(ids),
                "attention_mask": torch.stack(mask),
                "sentiment_label": torch.stack(lab),
                "neighbor_polarities": torch.stack(pol),
                "neighbor_scores": torch.stack(sc),
            }
    else:
        ds = TensorDataset(input_ids, attention_mask, sentiment_label)

        def collate(batch):
            ids, mask, lab = zip(*batch)
            return {
                "input_ids": torch.stack(ids),
                "attention_mask": torch.stack(mask),
                "sentiment_label": torch.stack(lab),
            }

    return DataLoader(ds, batch_size=4, collate_fn=collate)


def test_evaluate_no_retrieval():
    model = SentimentPredictor(use_retrieval=False)
    trainer = SentimentTrainer(
        model=model, optimizer=None, scheduler=None,
        device="cpu", log_path="",
    )
    loader = _make_loader(use_retrieval=False)
    metrics = trainer.evaluate(loader)
    assert "sentiment_acc" in metrics
    assert "sentiment_macro_f1" in metrics
    assert 0.0 <= metrics["sentiment_acc"] <= 1.0


def test_evaluate_with_retrieval():
    model = SentimentPredictor(use_retrieval=True)
    trainer = SentimentTrainer(
        model=model, optimizer=None, scheduler=None,
        device="cpu", log_path="",
    )
    loader = _make_loader(use_retrieval=True)
    metrics = trainer.evaluate(loader)
    assert "sentiment_acc" in metrics
    assert "loss" in metrics
