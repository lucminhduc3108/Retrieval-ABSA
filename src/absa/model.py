import torch
import torch.nn as nn
from transformers import AutoModel


class RetrievalABSA(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_bio_labels: int = 3, num_sent_labels: int = 3,
                 lambda_cls: float = 0.5, dropout: float = 0.1,
                 cls_class_weights: list[float] | None = None,
                 use_crf: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.bio_head = nn.Linear(hidden, num_bio_labels)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_sent_labels),
        )
        self.lambda_cls = lambda_cls
        self.use_crf = use_crf
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_bio_labels, batch_first=True)
        self.bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        cls_weight = torch.tensor(cls_class_weights, dtype=torch.float32) if cls_class_weights else None
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=cls_weight)

    def forward(self, input_ids, attention_mask,
                bio_input_ids=None, bio_attention_mask=None,
                bio_labels=None, sentiment_label=None,
                crf_mask=None) -> dict:
        if bio_input_ids is not None:
            bio_out = self.encoder(input_ids=bio_input_ids,
                                   attention_mask=bio_attention_mask)
            bio_sequence = bio_out.last_hidden_state
            sent_out = self.encoder(input_ids=input_ids,
                                    attention_mask=attention_mask)
            cls_output = sent_out.last_hidden_state[:, 0]
        else:
            outputs = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask)
            bio_sequence = outputs.last_hidden_state
            cls_output = bio_sequence[:, 0]

        bio_logits = self.bio_head(bio_sequence)
        sentiment_logits = self.sentiment_head(cls_output)

        loss = None
        loss_bio = None
        loss_cls = None

        if bio_labels is not None and sentiment_label is not None:
            if bio_input_ids is not None and bio_labels.size(1) > bio_logits.size(1):
                bio_labels = bio_labels[:, :bio_logits.size(1)]
                if crf_mask is not None:
                    crf_mask = crf_mask[:, :bio_logits.size(1)]

            if self.use_crf and crf_mask is not None and crf_mask.any():
                safe_labels = bio_labels.clone()
                safe_labels[safe_labels == -100] = 0
                crf_mask_fixed = crf_mask.clone()
                crf_mask_fixed[:, 0] = True
                _nll = -self.crf(bio_logits.float(), safe_labels,
                                 mask=crf_mask_fixed, reduction='none')
                _tokens = crf_mask_fixed.sum(dim=1).float()
                loss_bio = (_nll / _tokens).mean()
            elif self.use_crf and (crf_mask is None or not crf_mask.any()):
                loss_bio = torch.tensor(0.0, device=input_ids.device)
            else:
                loss_bio = self.bio_loss_fn(
                    bio_logits.reshape(-1, bio_logits.size(-1)), bio_labels.reshape(-1))
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
