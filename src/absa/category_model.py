import torch
import torch.nn as nn
from transformers import AutoModel

from src.data.category_builder import (
    ENTITY_LIST, ENT2IDX, ENTITY2ATTRS, MULTI_ATTR_ENTITIES, NUM_ENTITIES,
)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, margin=0.05,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits, targets):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.margin > 0:
            xs_neg = (xs_neg + self.margin).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        if self.pos_weight is not None:
            los_pos = los_pos * self.pos_weight
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_neg * (1 - targets)
            pt1 = xs_pos * targets
            pt = pt0 + pt1
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss = loss * (1 - pt).pow(gamma)
        return -loss.mean()


class CategoryDetector(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_categories: int = 12,
                 pos_weight: torch.Tensor | None = None,
                 use_asl: bool = False,
                 asl_gamma_neg: int = 4,
                 asl_gamma_pos: int = 0,
                 asl_margin: float = 0.05,
                 use_cat_attention: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.use_cat_attention = use_cat_attention

        if use_cat_attention:
            self.cat_queries = nn.Embedding(num_categories, hidden)
            self.cat_attention = nn.MultiheadAttention(hidden, num_heads=8,
                                                       batch_first=True, dropout=0.1)
            self.cat_norm = nn.LayerNorm(hidden)
        else:
            self.pooler = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Dropout(0.2)
            )

        self.category_head = nn.Linear(hidden, num_categories)

        if use_asl:
            self.loss_fn = AsymmetricLoss(gamma_neg=asl_gamma_neg,
                                          gamma_pos=asl_gamma_pos,
                                          margin=asl_margin,
                                          pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask,
                category_labels=None) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.use_cat_attention:
            last_hidden = outputs.last_hidden_state
            B = last_hidden.size(0)
            q = self.cat_queries.weight.unsqueeze(0).expand(B, -1, -1)
            key_padding_mask = ~attention_mask.bool()
            attn_out, _ = self.cat_attention(q, last_hidden, last_hidden,
                                              key_padding_mask=key_padding_mask)
            attn_out = self.cat_norm(attn_out)
            logits = (attn_out * self.category_head.weight.unsqueeze(0)).sum(-1) + self.category_head.bias
        else:
            cls_output = outputs.last_hidden_state[:, 0]
            pooled_output = self.pooler(cls_output)
            logits = self.category_head(pooled_output)

        loss = None
        if category_labels is not None:
            loss = self.loss_fn(logits, category_labels)

        return {"logits": logits, "loss": loss}


class HierarchicalCategoryDetector(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 num_entities: int = NUM_ENTITIES,
                 pos_weight_entity: torch.Tensor | None = None,
                 pos_weight_food: torch.Tensor | None = None,
                 pos_weight_drinks: torch.Tensor | None = None,
                 pos_weight_restaurant: torch.Tensor | None = None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        hidden = self.encoder.config.hidden_size

        self.pooler = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(0.2),
        )

        self.entity_head = nn.Linear(hidden, num_entities)
        self.entity_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_entity)

        self.food_attr_head = nn.Linear(hidden, len(ENTITY2ATTRS["FOOD"]))
        self.drinks_attr_head = nn.Linear(hidden, len(ENTITY2ATTRS["DRINKS"]))
        self.restaurant_attr_head = nn.Linear(hidden, len(ENTITY2ATTRS["RESTAURANT"]))

        self.food_attr_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_food)
        self.drinks_attr_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_drinks)
        self.restaurant_attr_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_restaurant)

    def forward(self, input_ids, attention_mask,
                entity_labels=None,
                food_attr_labels=None,
                drinks_attr_labels=None,
                restaurant_attr_labels=None) -> dict:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(outputs.last_hidden_state[:, 0])

        entity_logits = self.entity_head(pooled)
        food_logits = self.food_attr_head(pooled)
        drinks_logits = self.drinks_attr_head(pooled)
        restaurant_logits = self.restaurant_attr_head(pooled)

        loss = None
        if entity_labels is not None:
            loss = self.entity_loss_fn(entity_logits, entity_labels)
            for ent_idx, attr_logits, attr_labels, loss_fn in [
                (ENT2IDX["FOOD"], food_logits, food_attr_labels, self.food_attr_loss_fn),
                (ENT2IDX["DRINKS"], drinks_logits, drinks_attr_labels, self.drinks_attr_loss_fn),
                (ENT2IDX["RESTAURANT"], restaurant_logits, restaurant_attr_labels, self.restaurant_attr_loss_fn),
            ]:
                if attr_labels is not None:
                    mask = entity_labels[:, ent_idx] > 0
                    if mask.any():
                        loss = loss + loss_fn(attr_logits[mask], attr_labels[mask])

        return {
            "entity_logits": entity_logits,
            "food_attr_logits": food_logits,
            "drinks_attr_logits": drinks_logits,
            "restaurant_attr_logits": restaurant_logits,
            "loss": loss,
        }
