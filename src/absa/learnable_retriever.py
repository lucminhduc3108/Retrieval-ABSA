import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableRetriever(nn.Module):
    def __init__(self, vec_dim: int = 256, num_labels: int = 3,
                 embed_dim: int = 64, tau: float = 0.5,
                 margin: float = 0.1, w_mode: str = "full",
                 w_rank: int = 16):
        super().__init__()
        self._w_mode = w_mode
        if w_mode == "full":
            self.W = nn.Linear(vec_dim, vec_dim, bias=False)
            nn.init.eye_(self.W.weight)
        elif w_mode == "diagonal":
            self.W_diag = nn.Parameter(torch.ones(vec_dim))
        elif w_mode == "low_rank":
            self.W_A = nn.Parameter(torch.empty(vec_dim, w_rank))
            self.W_B = nn.Parameter(torch.empty(w_rank, vec_dim))
            nn.init.orthogonal_(self.W_A)
            self.W_B.data.copy_(self.W_A.data.T[:w_rank])
        else:
            raise ValueError(f"Unknown w_mode: {w_mode}")
        self.polarity_embedding = nn.Embedding(num_labels, embed_dim)
        self.tau = tau
        self.embed_dim = embed_dim
        self.margin = margin

    def forward(self, query_vec: torch.Tensor,
                neighbor_vecs: torch.Tensor,
                neighbor_polarities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_mode == "full":
            Wq = self.W(query_vec)
        elif self._w_mode == "diagonal":
            Wq = self.W_diag * query_vec
        else:
            Wq = query_vec @ (self.W_A @ self.W_B).T
        scores = torch.bmm(neighbor_vecs, Wq.unsqueeze(-1)).squeeze(-1)

        is_padding = (neighbor_vecs.abs().sum(dim=-1) == 0)
        scores = scores.masked_fill(is_padding, float("-inf"))

        alpha = F.softmax(scores.float() / self.tau, dim=1).to(scores.dtype)
        all_padding = is_padding.all(dim=1)
        alpha = alpha.masked_fill(all_padding.unsqueeze(-1), 0.0)

        embeds = self.polarity_embedding(neighbor_polarities)
        label_repr = (alpha.unsqueeze(-1) * embeds).sum(dim=1)
        return label_repr, scores

    def ranking_loss(self, scores: torch.Tensor,
                     neighbor_polarities: torch.Tensor,
                     query_polarity: torch.Tensor,
                     neighbor_vecs: torch.Tensor | None = None) -> torch.Tensor:
        if neighbor_vecs is not None:
            valid_mask = (neighbor_vecs.abs().sum(dim=-1) > 0)
        else:
            valid_mask = torch.ones_like(neighbor_polarities, dtype=torch.bool)

        same_mask = (neighbor_polarities == query_polarity.unsqueeze(1)) & valid_mask
        diff_mask = (neighbor_polarities != query_polarity.unsqueeze(1)) & valid_mask

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
