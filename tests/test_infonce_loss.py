import torch
import torch.nn.functional as F
from src.embedding.loss import infonce_loss


def test_loss_is_positive_scalar():
    torch.manual_seed(0)
    a = F.normalize(torch.randn(4, 8), dim=-1)
    p = F.normalize(torch.randn(4, 8), dim=-1)
    loss = infonce_loss(a, p, tau=0.07)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_identical_inputs_give_low_loss():
    torch.manual_seed(0)
    a = F.normalize(torch.randn(4, 8), dim=-1)
    loss_same = infonce_loss(a, a, tau=0.07)
    loss_rand = infonce_loss(a, F.normalize(torch.randn(4, 8), dim=-1), tau=0.07)
    assert loss_same < loss_rand


def test_hard_negatives_increase_loss():
    torch.manual_seed(0)
    a = F.normalize(torch.randn(4, 8), dim=-1)
    p = a.clone()
    n1 = F.normalize(torch.randn(4, 8), dim=-1)
    n2 = F.normalize(torch.randn(4, 8), dim=-1)
    loss_no_neg = infonce_loss(a, p, tau=0.07)
    loss_one_neg = infonce_loss(a, p, negatives=[n1], tau=0.07)
    loss_two_neg = infonce_loss(a, p, negatives=[n1, n2], tau=0.07)
    assert loss_one_neg >= loss_no_neg
    assert loss_two_neg >= loss_one_neg


def test_loss_is_differentiable():
    raw = torch.randn(4, 8, requires_grad=True)
    a = F.normalize(raw, dim=-1)
    p = F.normalize(torch.randn(4, 8), dim=-1)
    loss = infonce_loss(a, p, tau=0.07)
    loss.backward()
    assert raw.grad is not None
