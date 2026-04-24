import torch
from src.embedding.model import ContrastiveEmbedder


def test_encode_returns_normalized_vectors():
    m = ContrastiveEmbedder(proj_dim=256)
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    out = m.encode(ids, mask)
    assert out.shape == (2, 256)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4)


def test_forward_returns_all_vector_keys():
    m = ContrastiveEmbedder(proj_dim=256)
    B, L = 2, 16
    ids = torch.randint(0, 1000, (B, L))
    mask = torch.ones_like(ids)
    out = m(ids, mask, ids, mask, ids, mask, ids, mask)
    assert out["anchor_vecs"].shape == (B, 256)
    assert out["pos_vecs"].shape == (B, 256)
    assert out["neg1_vecs"].shape == (B, 256)
    assert out["neg2_vecs"].shape == (B, 256)


def test_forward_without_negatives():
    m = ContrastiveEmbedder(proj_dim=256)
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    out = m(ids, mask, ids, mask)
    assert out["anchor_vecs"].shape == (2, 256)
    assert out["pos_vecs"].shape == (2, 256)
    assert out["neg1_vecs"] is None
    assert out["neg2_vecs"] is None
