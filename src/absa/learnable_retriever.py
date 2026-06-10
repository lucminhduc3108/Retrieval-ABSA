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
        # query_vec: (B, D), neighbor_vecs: (B, K, D), neighbor_polarities: (B, K)
        Wq = self.W(query_vec)  # (B, D)
        scores = torch.bmm(neighbor_vecs, Wq.unsqueeze(-1)).squeeze(-1)  # (B, K)
        alpha = F.softmax(scores / self.tau, dim=1)  # (B, K)
        embeds = self.polarity_embedding(neighbor_polarities)  # (B, K, embed_dim)
        label_repr = (alpha.unsqueeze(-1) * embeds).sum(dim=1)  # (B, embed_dim)
        return label_repr, scores

    def ranking_loss(self, scores: torch.Tensor,
                     neighbor_polarities: torch.Tensor,
                     query_polarity: torch.Tensor) -> torch.Tensor:
        # scores: (B, K), neighbor_polarities: (B, K), query_polarity: (B,)
        same_mask = neighbor_polarities == query_polarity.unsqueeze(1)  # (B, K)
        diff_mask = ~same_mask

        has_same = same_mask.any(dim=1)
        has_diff = diff_mask.any(dim=1)
        has_both = has_same & has_diff

        if not has_both.any():
            return torch.tensor(0.0, device=scores.device)

        INF = float("inf")
        same_scores = scores.masked_fill(~same_mask, -INF)
        diff_scores = scores.masked_fill(~diff_mask, -INF)

        best_same = same_scores.max(dim=1).values  # (B,)
        best_diff = diff_scores.max(dim=1).values  # (B,)

        triplet = F.relu(self.margin - best_same + best_diff)
        triplet = triplet * has_both.float()
        return triplet.sum() / has_both.float().sum()
