import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelInterpolation(nn.Module):
    def __init__(self, num_labels: int = 3, embed_dim: int = 64,
                 tau: float = 0.05):
        super().__init__()
        self.polarity_embedding = nn.Embedding(num_labels, embed_dim)
        self.tau = tau
        self.embed_dim = embed_dim

    def forward(self, neighbor_polarities: torch.Tensor,
                neighbor_scores: torch.Tensor) -> torch.Tensor:
        if neighbor_polarities.size(1) == 0:
            batch = neighbor_polarities.size(0)
            return torch.zeros(batch, self.embed_dim,
                               device=neighbor_polarities.device)

        embeds = self.polarity_embedding(neighbor_polarities)
        alpha = F.softmax(neighbor_scores / self.tau, dim=1)
        label_repr = (alpha.unsqueeze(-1) * embeds).sum(dim=1)
        return label_repr
