import torch

from src.absa.model import RetrievalABSA


def test_forward_shapes():
    m = RetrievalABSA()
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones_like(ids)
    out = m(ids, mask)
    assert out["bio_logits"].shape == (2, 32, 3)
    assert out["sentiment_logits"].shape == (2, 3)
    assert out["loss"] is None


def test_forward_with_labels_returns_scalar_loss():
    m = RetrievalABSA()
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones_like(ids)
    bio = torch.zeros(2, 32, dtype=torch.long)
    sent = torch.zeros(2, dtype=torch.long)
    out = m(ids, mask, bio_labels=bio, sentiment_label=sent)
    assert out["loss"].dim() == 0
    assert out["loss"].item() > 0


def test_bio_ignore_index():
    m = RetrievalABSA()
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones_like(ids)
    bio = torch.full((2, 32), -100, dtype=torch.long)
    sent = torch.zeros(2, dtype=torch.long)
    out = m(ids, mask, bio_labels=bio, sentiment_label=sent)
    assert out["loss_bio"].item() == 0.0


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
