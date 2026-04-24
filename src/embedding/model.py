import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ContrastiveEmbedder(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0]
        projected = self.projection(cls_vec)
        return F.normalize(projected, p=2, dim=-1)

    def forward(self, anchor_ids, anchor_mask, pos_ids, pos_mask,
                neg1_ids=None, neg1_mask=None,
                neg2_ids=None, neg2_mask=None) -> dict:
        anchor_vecs = self.encode(anchor_ids, anchor_mask)
        pos_vecs = self.encode(pos_ids, pos_mask)
        neg1_vecs = self.encode(neg1_ids, neg1_mask) if neg1_ids is not None else None
        neg2_vecs = self.encode(neg2_ids, neg2_mask) if neg2_ids is not None else None
        return {
            "anchor_vecs": anchor_vecs,
            "pos_vecs": pos_vecs,
            "neg1_vecs": neg1_vecs,
            "neg2_vecs": neg2_vecs,
        }
