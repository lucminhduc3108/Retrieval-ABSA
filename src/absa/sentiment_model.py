import torch
import torch.nn as nn
from transformers import AutoModel

from src.absa.label_interpolation import LabelInterpolation
from src.absa.learnable_retriever import LearnableRetriever


class RetrievalGate(nn.Module):
    def __init__(self, hidden: int = 768, embed_dim: int = 64):
        super().__init__()
        self.gate = nn.Linear(hidden + embed_dim + 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.847)

    def forward(self, cls_output: torch.Tensor,
                label_repr: torch.Tensor,
                mean_score: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat(
            [cls_output, label_repr, mean_score.unsqueeze(-1)], dim=-1)
        return torch.sigmoid(self.gate(gate_input))


class SentimentPredictor(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_sent_labels: int = 3,
                 embed_dim: int = 64, tau: float = 0.05,
                 dropout: float = 0.1, use_retrieval: bool = True,
                 use_learnable_retriever: bool = False,
                 class_weights: torch.Tensor | None = None,
                 margin: float = 0.1, w_mode: str = "full",
                 w_rank: int = 16,
                 embedding_model: "nn.Module | None" = None,
                 aux_label_repr_weight: float = 0.0,
                 retrieval_dropout: float = 0.0,
                 use_gate: bool = False):
        super().__init__()
        self.embedding_model = embedding_model
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.use_retrieval = use_retrieval
        self.retrieval_dropout = retrieval_dropout

        if use_retrieval:
            if use_learnable_retriever:
                self.learnable_retriever = LearnableRetriever(
                    vec_dim=256, num_labels=num_sent_labels,
                    embed_dim=embed_dim, tau=tau,
                    margin=margin, w_mode=w_mode, w_rank=w_rank)
                self.label_interp = None
            else:
                self.label_interp = LabelInterpolation(
                    num_labels=num_sent_labels, embed_dim=embed_dim, tau=tau)
                self.learnable_retriever = None
            input_dim = hidden + embed_dim
        else:
            self.label_interp = None
            self.learnable_retriever = None
            input_dim = hidden

        self.sentiment_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_sent_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self.aux_polarity_head = None
        if use_retrieval and aux_label_repr_weight > 0:
            self.aux_polarity_head = nn.Linear(embed_dim, num_sent_labels)

        self.use_gate = use_gate and use_retrieval
        self.retrieval_gate = None
        if self.use_gate:
            self.retrieval_gate = RetrievalGate(
                hidden=self.encoder.config.hidden_size, embed_dim=embed_dim)

    def forward(self, input_ids, attention_mask,
                neighbor_polarities=None, neighbor_scores=None,
                query_vec=None, neighbor_vecs=None, query_polarity=None,
                sentiment_label=None,
                embed_input_ids=None, embed_attention_mask=None) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]

        emb_cls_logits = None
        if (self.embedding_model is not None
                and embed_input_ids is not None
                and embed_attention_mask is not None):
            query_vec, emb_cls_logits, _ = self.embedding_model._encode_with_heads(
                embed_input_ids, embed_attention_mask)

        ranking_loss = None
        gate_alpha = None

        if self.use_retrieval:
            if self.learnable_retriever is not None:
                if query_vec is not None and neighbor_vecs is not None and neighbor_polarities is not None:
                    label_repr, scores = self.learnable_retriever(
                        query_vec, neighbor_vecs, neighbor_polarities)
                    if query_polarity is not None:
                        ranking_loss = self.learnable_retriever.ranking_loss(
                            scores, neighbor_polarities, query_polarity,
                            neighbor_vecs=neighbor_vecs)
                else:
                    label_repr = torch.zeros(
                        cls_output.size(0), self.learnable_retriever.embed_dim,
                        device=cls_output.device)
            elif self.label_interp is not None:
                if neighbor_polarities is not None:
                    label_repr = self.label_interp(neighbor_polarities, neighbor_scores)
                else:
                    label_repr = torch.zeros(
                        cls_output.size(0), self.label_interp.embed_dim,
                        device=cls_output.device)
            else:
                label_repr = torch.zeros(cls_output.size(0), 64, device=cls_output.device)

            aux_logits = None
            if self.aux_polarity_head is not None:
                aux_logits = self.aux_polarity_head(label_repr)

            if self.retrieval_gate is not None:
                if neighbor_scores is not None:
                    is_pad = torch.isinf(neighbor_scores) & (neighbor_scores < 0)
                    safe = neighbor_scores.masked_fill(is_pad, 0.0)
                    valid = (~is_pad).float()
                    count = valid.sum(dim=1).clamp(min=1.0)
                    mean_score = (safe * valid).sum(dim=1) / count
                else:
                    mean_score = torch.zeros(
                        cls_output.size(0), device=cls_output.device)
                gate_alpha = self.retrieval_gate(cls_output, label_repr, mean_score)
                label_repr = gate_alpha * label_repr
            elif self.training and self.retrieval_dropout > 0.0:
                mask = torch.bernoulli(
                    torch.full((label_repr.size(0), 1), 1.0 - self.retrieval_dropout,
                               device=label_repr.device)
                )
                label_repr = label_repr * mask

            final = torch.cat([cls_output, label_repr], dim=-1)
        else:
            final = cls_output
            aux_logits = None

        logits = self.sentiment_head(final)

        loss = None
        if sentiment_label is not None:
            loss = self.loss_fn(logits, sentiment_label)

        return {"logits": logits, "loss": loss, "ranking_loss": ranking_loss,
                "emb_cls_logits": emb_cls_logits, "aux_logits": aux_logits,
                "gate_alpha": gate_alpha}
