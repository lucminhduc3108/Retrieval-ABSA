import torch
import torch.nn as nn
from transformers import AutoModel


class CategoryDetector(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_categories: int = 12,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        
        # ContextPooler to add regularization and complexity
        self.pooler = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        self.category_head = nn.Linear(hidden, num_categories)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight if pos_weight is not None else None)

    def forward(self, input_ids, attention_mask,
                category_labels=None) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.pooler(cls_output)
        logits = self.category_head(pooled_output)

        loss = None
        if category_labels is not None:
            loss = self.loss_fn(logits, category_labels)

        return {"logits": logits, "loss": loss}
