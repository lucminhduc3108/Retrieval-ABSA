import torch
import torch.nn as nn
from transformers import AutoModel

from src.absa.label_interpolation import LabelInterpolation
from src.absa.learnable_retriever import LearnableRetriever


class SentimentPredictor(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_sent_labels: int = 3,
                 embed_dim: int = 64, tau: float = 0.05,
                 dropout: float = 0.1, use_retrieval: bool = True,
                 use_learnable_retriever: bool = False,
                 class_weights: torch.Tensor | None = None,
                 margin: float = 0.1, w_mode: str = "full",
                 w_rank: int = 16):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.use_retrieval = use_retrieval

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

    def forward(self, input_ids, attention_mask,
                neighbor_polarities=None, neighbor_scores=None,
                query_vec=None, neighbor_vecs=None, query_polarity=None,
                sentiment_label=None) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]

        ranking_loss = None

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
            final = torch.cat([cls_output, label_repr], dim=-1)
        else:
            final = cls_output

        logits = self.sentiment_head(final)

        loss = None
        if sentiment_label is not None:
            loss = self.loss_fn(logits, sentiment_label)

        return {"logits": logits, "loss": loss, "ranking_loss": ranking_loss}
