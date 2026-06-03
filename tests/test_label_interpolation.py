import torch

from src.absa.label_interpolation import LabelInterpolation


def test_output_shape():
    li = LabelInterpolation(num_labels=3, embed_dim=64, tau=0.05)
    polarities = torch.tensor([[0, 1], [2, 0]])
    scores = torch.tensor([[0.99, 0.98], [0.97, 0.96]])
    out = li(polarities, scores)
    assert out.shape == (2, 64)


def test_zero_vector_when_no_neighbors():
    li = LabelInterpolation(num_labels=3, embed_dim=64, tau=0.05)
    polarities = torch.zeros(3, 0, dtype=torch.long)
    scores = torch.zeros(3, 0)
    out = li(polarities, scores)
    assert out.shape == (3, 64)
    assert (out == 0).all()


def test_single_neighbor_returns_embedding():
    li = LabelInterpolation(num_labels=3, embed_dim=64, tau=0.05)
    polarities = torch.tensor([[1]])
    scores = torch.tensor([[0.99]])
    out = li(polarities, scores)
    expected = li.polarity_embedding(torch.tensor([1]))
    assert torch.allclose(out.squeeze(0), expected.squeeze(0))


def test_gradient_flows():
    li = LabelInterpolation(num_labels=3, embed_dim=64, tau=0.05)
    polarities = torch.tensor([[0, 1]])
    scores = torch.tensor([[0.99, 0.98]])
    out = li(polarities, scores)
    loss = out.sum()
    loss.backward()
    assert li.polarity_embedding.weight.grad is not None


def test_tau_affects_distribution():
    li_sharp = LabelInterpolation(num_labels=3, embed_dim=64, tau=0.01)
    li_flat = LabelInterpolation(num_labels=3, embed_dim=64, tau=1.0)
    li_flat.polarity_embedding = li_sharp.polarity_embedding

    polarities = torch.tensor([[0, 1]])
    scores = torch.tensor([[0.99, 0.90]])

    out_sharp = li_sharp(polarities, scores)
    out_flat = li_flat(polarities, scores)
    assert not torch.allclose(out_sharp, out_flat, atol=1e-3)
