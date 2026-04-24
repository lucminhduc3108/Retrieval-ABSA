import torch
import torch.nn as nn
from transformers import AutoModel


class RetrievalABSA(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_bio_labels: int = 3, num_sent_labels: int = 3,
                 lambda_cls: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.bio_head = nn.Linear(hidden, num_bio_labels)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_sent_labels),
        )
        self.lambda_cls = lambda_cls
        self.bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask,
                bio_labels=None, sentiment_label=None) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]

        bio_logits = self.bio_head(sequence_output)
        sentiment_logits = self.sentiment_head(cls_output)

        loss = None
        loss_bio = None
        loss_cls = None

        if bio_labels is not None and sentiment_label is not None:
            loss_bio = self.bio_loss_fn(
                bio_logits.view(-1, bio_logits.size(-1)), bio_labels.view(-1))
            if torch.isnan(loss_bio):
                loss_bio = torch.tensor(0.0, device=input_ids.device)
            loss_cls = self.cls_loss_fn(sentiment_logits, sentiment_label)
            loss = loss_bio + self.lambda_cls * loss_cls

        return {
            "bio_logits": bio_logits,
            "sentiment_logits": sentiment_logits,
            "loss": loss,
            "loss_bio": loss_bio,
            "loss_cls": loss_cls,
        }
