import torch
import torch.nn.functional as F

from src.absa.learnable_retriever import LearnableRetriever


def _make_inputs(B=2, K=2, D=256, embed_dim=64):
    query_vec = F.normalize(torch.randn(B, D), dim=-1)
    neighbor_vecs = F.normalize(torch.randn(B, K, D), dim=-1)
    neighbor_polarities = torch.randint(0, 3, (B, K))
    query_polarity = torch.randint(0, 3, (B,))
    return query_vec, neighbor_vecs, neighbor_polarities, query_polarity


def test_forward_output_shapes():
    lr = LearnableRetriever()
    qv, nv, np_, qp = _make_inputs()
    label_repr, scores = lr(qv, nv, np_)
    assert label_repr.shape == (2, 64)
    assert scores.shape == (2, 2)


def test_identity_W_scores_equal_cosine():
    lr = LearnableRetriever()
    # W is initialized to identity — W(q) = q
    qv, nv, np_, _ = _make_inputs(B=3, K=4)
    label_repr, scores = lr(qv, nv, np_)
    # cosine scores: (B, K) = bmm(nv, qv.unsqueeze(-1)).squeeze(-1)
    cosine = torch.bmm(nv, qv.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(scores, cosine, atol=1e-5)


def test_ranking_loss_positive_when_wrong_order():
    lr = LearnableRetriever(margin=0.1)
    B, K, D = 2, 2, 256
    # Construct: diff-polarity neighbor scores higher than same-polarity
    # query_polarity = 0 for both samples
    query_polarity = torch.zeros(B, dtype=torch.long)
    # neighbor_polarities: first neighbor is different (1), second is same (0)
    neighbor_polarities = torch.tensor([[1, 0], [1, 0]])
    # scores: diff-pol (idx 0) > same-pol (idx 1)
    scores = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    loss = lr.ranking_loss(scores, neighbor_polarities, query_polarity)
    assert loss.item() > 0


def test_ranking_loss_zero_when_correct():
    lr = LearnableRetriever(margin=0.1)
    B, K = 2, 2
    query_polarity = torch.zeros(B, dtype=torch.long)
    neighbor_polarities = torch.tensor([[0, 1], [0, 1]])
    # same-pol (idx 0) > diff-pol (idx 1) by more than margin
    scores = torch.tensor([[0.9, 0.0], [0.8, 0.0]])
    loss = lr.ranking_loss(scores, neighbor_polarities, query_polarity)
    assert loss.item() == 0.0


def test_ranking_loss_zero_when_no_diff_polarity_neighbors():
    lr = LearnableRetriever()
    B, K = 2, 2
    query_polarity = torch.zeros(B, dtype=torch.long)
    # All neighbors are same polarity as query
    neighbor_polarities = torch.zeros(B, K, dtype=torch.long)
    scores = torch.randn(B, K)
    loss = lr.ranking_loss(scores, neighbor_polarities, query_polarity)
    assert loss.item() == 0.0
    # Zero fallback must stay on computation graph for backward()
    assert loss.requires_grad


def test_gradient_flows_through_W():
    lr = LearnableRetriever()
    qv, nv, np_, qp = _make_inputs()
    label_repr, scores = lr(qv, nv, np_)
    loss = label_repr.sum() + scores.sum()
    loss.backward()
    assert lr.W.weight.grad is not None
    assert lr.polarity_embedding.weight.grad is not None


def test_ranking_loss_no_nan_with_fp16():
    lr = LearnableRetriever().half()
    B, K = 4, 3
    query_polarity = torch.zeros(B, dtype=torch.long)
    neighbor_polarities = torch.tensor([[0, 1, 2], [0, 0, 0], [1, 1, 0], [2, 0, 1]])
    scores = torch.randn(B, K, dtype=torch.float16)
    loss = lr.ranking_loss(scores, neighbor_polarities, query_polarity)
    assert not torch.isnan(loss), f"ranking_loss is NaN: {loss}"
    assert not torch.isinf(loss), f"ranking_loss is inf: {loss}"


def test_forward_no_nan_with_fp16():
    lr = LearnableRetriever().half()
    qv = torch.randn(2, 256, dtype=torch.float16)
    nv = torch.randn(2, 3, 256, dtype=torch.float16)
    np_ = torch.randint(0, 3, (2, 3))
    label_repr, scores = lr(qv, nv, np_)
    assert not torch.isnan(label_repr).any(), "label_repr contains NaN"
    assert not torch.isnan(scores).any(), "scores contains NaN"


def test_tau_affects_alpha_concentration():
    lr_sharp = LearnableRetriever(tau=0.01)
    lr_flat = LearnableRetriever(tau=2.0)
    # Share same W and polarity_embedding for fair comparison
    lr_flat.W = lr_sharp.W
    lr_flat.polarity_embedding = lr_sharp.polarity_embedding

    qv, nv, np_, _ = _make_inputs(B=1, K=3)
    np_ = torch.tensor([[0, 1, 2]], dtype=torch.long)
    repr_sharp, _ = lr_sharp(qv, nv, np_)
    repr_flat, _ = lr_flat(qv, nv, np_)
    # Sharp tau concentrates mass → different output from flat tau
    assert not torch.allclose(repr_sharp, repr_flat, atol=1e-3)


def test_diagonal_mode_output_shapes():
    lr = LearnableRetriever(w_mode="diagonal")
    qv, nv, np_, _ = _make_inputs()
    label_repr, scores = lr(qv, nv, np_)
    assert label_repr.shape == (2, 64)
    assert scores.shape == (2, 2)


def test_diagonal_mode_gradient_flows():
    lr = LearnableRetriever(w_mode="diagonal")
    qv, nv, np_, _ = _make_inputs()
    label_repr, scores = lr(qv, nv, np_)
    (label_repr.sum() + scores.sum()).backward()
    assert lr.W_diag.grad is not None
    assert lr.polarity_embedding.weight.grad is not None


def test_low_rank_mode_output_shapes():
    lr = LearnableRetriever(w_mode="low_rank", w_rank=8)
    qv, nv, np_, _ = _make_inputs()
    label_repr, scores = lr(qv, nv, np_)
    assert label_repr.shape == (2, 64)
    assert scores.shape == (2, 2)


def test_low_rank_mode_gradient_flows():
    lr = LearnableRetriever(w_mode="low_rank", w_rank=8)
    qv, nv, np_, _ = _make_inputs()
    label_repr, scores = lr(qv, nv, np_)
    (label_repr.sum() + scores.sum()).backward()
    assert lr.W_A.grad is not None
    assert lr.W_B.grad is not None


def test_padding_gets_zero_weight():
    lr = LearnableRetriever()
    qv = F.normalize(torch.randn(1, 256), dim=-1)
    nv = torch.zeros(1, 2, 256)
    nv[0, 0] = F.normalize(torch.randn(1, 256), dim=-1)  # real neighbor
    # nv[0, 1] stays zero → padding
    np_ = torch.tensor([[0, 1]])
    label_repr, scores = lr(qv, nv, np_)
    # Padded neighbor should have -inf score
    assert scores[0, 1].item() == float("-inf")
    # label_repr should only reflect the real neighbor's polarity
    assert not torch.isnan(label_repr).any()


def test_ranking_loss_ignores_padding():
    lr = LearnableRetriever(margin=0.1)
    B, K = 1, 3
    query_polarity = torch.tensor([1])  # negative
    # neighbor 0: same polarity (1), neighbor 1: diff (0), neighbor 2: padding
    neighbor_polarities = torch.tensor([[1, 0, 0]])
    scores = torch.tensor([[0.5, 0.2, 0.0]])
    # neighbor_vecs: first two real, third is zero (padding)
    nv = torch.randn(1, 3, 256)
    nv[0, 2] = 0.0  # padding

    loss_with_pad = lr.ranking_loss(scores, neighbor_polarities,
                                    query_polarity, neighbor_vecs=nv)
    # Without padding info, neighbor 2 (polarity=0, score=0.0) would count as
    # diff-polarity and increase best_diff. With padding masked, it's excluded.
    loss_without = lr.ranking_loss(scores[:, :2],
                                   neighbor_polarities[:, :2],
                                   query_polarity)
    assert torch.allclose(loss_with_pad, loss_without, atol=1e-6)
