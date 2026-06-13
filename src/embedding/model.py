import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SentimentAttentionPool(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.scale = hidden_size ** -0.5

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = (hidden_states @ self.query) * self.scale
        scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = F.softmax(scores, dim=1)
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)


class ContrastiveEmbedder(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 proj_dim: int = 256, dropout: float = 0.1,
                 gradient_checkpointing: bool = False,
                 num_polarities: int = 0,
                 proj_num_polarities: int = 0,
                 use_attention_pool: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
        )
        self.cls_head = nn.Linear(hidden, num_polarities) if num_polarities > 0 else None
        self.proj_head = nn.Linear(proj_dim, proj_num_polarities) if proj_num_polarities > 0 else None
        self.pool = SentimentAttentionPool(hidden) if use_attention_pool else None

    def _pool(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            return self.pool(outputs.last_hidden_state, attention_mask)
        return outputs.last_hidden_state[:, 0]

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs, attention_mask)
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def _encode_with_heads(self, input_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> tuple:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0]
        cls_logits = self.cls_head(cls_vec) if self.cls_head is not None else None
        pooled = self._pool(outputs, attention_mask)
        projected = self.projection(pooled)
        proj_logits = self.proj_head(projected) if self.proj_head is not None else None
        return F.normalize(projected, p=2, dim=-1), cls_logits, proj_logits

    def forward(self, anchor_ids, anchor_mask, pos_ids, pos_mask,
                neg1_ids=None, neg1_mask=None,
                neg2_ids=None, neg2_mask=None) -> dict:
        anchor_vecs, cls_logits, proj_logits = self._encode_with_heads(anchor_ids, anchor_mask)
        pos_vecs = self.encode(pos_ids, pos_mask)
        neg1_vecs = self.encode(neg1_ids, neg1_mask) if neg1_ids is not None else None
        neg2_vecs = self.encode(neg2_ids, neg2_mask) if neg2_ids is not None else None
        return {
            "anchor_vecs": anchor_vecs,
            "pos_vecs": pos_vecs,
            "neg1_vecs": neg1_vecs,
            "neg2_vecs": neg2_vecs,
            "cls_logits": cls_logits,
            "proj_logits": proj_logits,
        }
