import torch
from torch.utils.data import DataLoader, TensorDataset

from src.absa.model import RetrievalABSA
from src.absa.trainer import ABSATrainer


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


def test_evaluate_returns_expected_keys():
    torch.manual_seed(0)
    model = RetrievalABSA()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ABSATrainer(model, optimizer, scheduler=None,
                          device="cpu", log_path="")
    loader = _make_loader()
    result = trainer.evaluate(loader)
    expected_keys = {"loss", "bio_token_f1", "span_f1",
                     "sentiment_acc", "sentiment_macro_f1", "joint_f1"}
    assert expected_keys.issubset(set(result.keys()))
    assert 0 <= result["sentiment_acc"] <= 1


def test_train_returns_history():
    torch.manual_seed(0)
    model = RetrievalABSA()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ABSATrainer(model, optimizer, scheduler=None,
                          device="cpu", log_path="")
    loader = _make_loader()
    history = trainer.train(loader, loader, epochs=2)
    assert len(history) == 2
    assert "train_loss" in history[0]
    assert "span_f1" in history[0]


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
