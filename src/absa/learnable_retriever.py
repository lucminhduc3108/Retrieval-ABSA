import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableRetriever(nn.Module):
    def __init__(self, vec_dim: int = 256, num_labels: int = 3,
                 embed_dim: int = 64, tau: float = 0.5,
                 margin: float = 0.1):
        super().__init__()
        self.W = nn.Linear(vec_dim, vec_dim, bias=False)
        nn.init.eye_(self.W.weight)  # identity init → cosine at epoch 0
        self.polarity_embedding = nn.Embedding(num_labels, embed_dim)
        self.tau = tau
        self.embed_dim = embed_dim
        self.margin = margin

    def forward(self, query_vec: torch.Tensor,
                neighbor_vecs: torch.Tensor,
                neighbor_polarities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Wq = self.W(query_vec)
        scores = torch.bmm(neighbor_vecs, Wq.unsqueeze(-1)).squeeze(-1)
        alpha = F.softmax(scores.float() / self.tau, dim=1).to(scores.dtype)
        embeds = self.polarity_embedding(neighbor_polarities)
        label_repr = (alpha.unsqueeze(-1) * embeds).sum(dim=1)
        return label_repr, scores

    def ranking_loss(self, scores: torch.Tensor,
                     neighbor_polarities: torch.Tensor,
                     query_polarity: torch.Tensor) -> torch.Tensor:
        same_mask = neighbor_polarities == query_polarity.unsqueeze(1)
        diff_mask = ~same_mask

        has_same = same_mask.any(dim=1)
        has_diff = diff_mask.any(dim=1)
        has_both = has_same & has_diff

        if not has_both.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

        scores_f = scores[has_both].float()
        same_m = same_mask[has_both]
        diff_m = diff_mask[has_both]

        BIG = 1e9
        worst_same = scores_f.masked_fill(~same_m, BIG).min(dim=1).values
        best_diff = scores_f.masked_fill(~diff_m, -BIG).max(dim=1).values

        triplet = F.relu(self.margin - worst_same + best_diff)
        return triplet.mean()
