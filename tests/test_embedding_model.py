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


def test_model_with_cls_head_returns_logits():
    m = ContrastiveEmbedder(proj_dim=32, num_polarities=3)
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    out = m(ids, mask, ids, mask)
    assert "cls_logits" in out
    assert out["cls_logits"].shape == (2, 3)


def test_model_without_cls_head_returns_none_logits():
    m = ContrastiveEmbedder(proj_dim=32, num_polarities=0)
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    out = m(ids, mask, ids, mask)
    assert out["cls_logits"] is None


def test_model_with_proj_head_returns_proj_logits():
    m = ContrastiveEmbedder(proj_dim=32, proj_num_polarities=3)
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    out = m(ids, mask, ids, mask)
    assert "proj_logits" in out
    assert out["proj_logits"].shape == (2, 3)


def test_model_without_proj_head_returns_none_proj_logits():
    m = ContrastiveEmbedder(proj_dim=32, proj_num_polarities=0)
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    out = m(ids, mask, ids, mask)
    assert out["proj_logits"] is None


def test_model_with_attention_pool_encode_shape():
    m = ContrastiveEmbedder(proj_dim=32, use_attention_pool=True)
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    out = m.encode(ids, mask)
    assert out.shape == (2, 32)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4)


def test_model_attention_pool_differs_from_cls():
    torch.manual_seed(42)
    m_cls = ContrastiveEmbedder(proj_dim=32, use_attention_pool=False)
    m_attn = ContrastiveEmbedder(proj_dim=32, use_attention_pool=True)
    # copy weights so only pooling differs
    m_attn.encoder.load_state_dict(m_cls.encoder.state_dict())
    m_attn.projection.load_state_dict(m_cls.projection.state_dict())
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones_like(ids)
    out_cls = m_cls.encode(ids, mask)
    out_attn = m_attn.encode(ids, mask)
    # outputs should differ because pooling is different
    assert not torch.allclose(out_cls, out_attn, atol=1e-4)
