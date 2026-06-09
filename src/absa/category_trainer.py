import json
import logging
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from src.data.category_builder import (
    CATEGORY_LIST, NUM_CATEGORIES,
    ENTITY_LIST, ENT2IDX, ENTITY2ATTRS, MULTI_ATTR_ENTITIES, NUM_ENTITIES,
)
from src.evaluation.category_metrics import category_f1, per_category_f1

_TOPK_RANGE = range(1, 5)

logger = logging.getLogger(__name__)

THRESHOLD_GRID = [round(0.05 + i * 0.05, 2) for i in range(18)]  # 0.05 to 0.90


def _tune_thresholds(all_logits: torch.Tensor,
                     all_labels: torch.Tensor) -> list[float]:
    probs = torch.sigmoid(all_logits)
    best_thresholds = [0.5] * NUM_CATEGORIES
    for cat_idx in range(NUM_CATEGORIES):
        best_f1 = -1.0
        gold_col = all_labels[:, cat_idx]
        pred_col = probs[:, cat_idx]
        for t in THRESHOLD_GRID:
            preds = (pred_col >= t).long()
            tp = ((preds == 1) & (gold_col == 1)).sum().item()
            fp = ((preds == 1) & (gold_col == 0)).sum().item()
            fn = ((preds == 0) & (gold_col == 1)).sum().item()
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            if f > best_f1:
                best_f1 = f
                best_thresholds[cat_idx] = t
    return best_thresholds


def _apply_thresholds(logits: torch.Tensor,
                      thresholds: list[float]) -> list[set[str]]:
    probs = torch.sigmoid(logits)
    result = []
    for i in range(probs.size(0)):
        cats = set()
        for j in range(NUM_CATEGORIES):
            if probs[i, j] >= thresholds[j]:
                cats.add(CATEGORY_LIST[j])
        result.append(cats)
    return result


def _tune_global_threshold(all_logits: torch.Tensor,
                           all_labels: torch.Tensor) -> float:
    probs = torch.sigmoid(all_logits)
    best_threshold = 0.5
    best_f1 = -1.0
    for t in THRESHOLD_GRID:
        preds = (probs >= t).long()
        tp = ((preds == 1) & (all_labels == 1)).sum().item()
        fp = ((preds == 1) & (all_labels == 0)).sum().item()
        fn = ((preds == 0) & (all_labels == 1)).sum().item()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        if f > best_f1:
            best_f1 = f
            best_threshold = t
    return best_threshold


def _apply_global_threshold(logits: torch.Tensor,
                            threshold: float,
                            k_max: int = 5) -> list[set[str]]:
    probs = torch.sigmoid(logits)
    result = []
    for i in range(probs.size(0)):
        above = [(j, probs[i, j].item()) for j in range(NUM_CATEGORIES)
                 if probs[i, j] >= threshold]
        if len(above) > k_max:
            above.sort(key=lambda x: x[1], reverse=True)
            above = above[:k_max]
        result.append({CATEGORY_LIST[j] for j, _ in above})
    return result


def tune_topk(all_logits: torch.Tensor,
              all_labels: torch.Tensor,
              k_range: range = _TOPK_RANGE) -> int:
    probs = torch.sigmoid(all_logits)
    gold_cats = []
    for i in range(all_labels.size(0)):
        gold_cats.append({CATEGORY_LIST[j]
                          for j in range(NUM_CATEGORIES)
                          if all_labels[i, j] == 1})
    best_k = 1
    best_f1 = -1.0
    for k in k_range:
        pred_cats = apply_topk(all_logits, k)
        m = category_f1(pred_cats, gold_cats)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_k = k
    return best_k


def apply_topk(logits: torch.Tensor, k: int) -> list[set[str]]:
    probs = torch.sigmoid(logits)
    topk_indices = probs.topk(min(k, probs.size(1)), dim=1).indices
    return [{CATEGORY_LIST[j] for j in topk_indices[i].tolist()}
            for i in range(probs.size(0))]


class CategoryTrainer:
    def __init__(self, model, optimizer, scheduler, device,
                 patience: int = 5, grad_clip: float = 1.0,
                 log_path: str = "", use_fp16: bool = False,
                 grad_accum_steps: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.grad_clip = grad_clip
        self.log_path = log_path
        self.grad_accum_steps = grad_accum_steps
        self.use_fp16 = use_fp16 and device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_fp16 else None

    def _run_batch(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast("cuda", enabled=self.use_fp16):
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                category_labels=batch["category_labels"],
            )
        return out

    def train(self, train_loader, val_loader, epochs: int,
              ckpt_path: str | None = None) -> list[dict]:
        history = []
        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            self.optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                out = self._run_batch(batch)
                loss = out["loss"] / self.grad_accum_steps
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += out["loss"].item()

                is_accum_step = (step + 1) % self.grad_accum_steps == 0
                is_last_step = (step + 1) == len(train_loader)
                if is_accum_step or is_last_step:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        if self.grad_clip > 0:
                            clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.grad_clip > 0:
                            clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader)
            record = {"epoch": epoch, "train_loss": avg_loss, **val_metrics}
            history.append(record)
            logger.info("Epoch %d: loss=%.4f cat_f1=%.4f (threshold=%.2f)",
                        epoch, avg_loss, val_metrics["category_f1"],
                        val_metrics["threshold"])

            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            if val_metrics["category_f1"] > best_f1:
                best_f1 = val_metrics["category_f1"]
                patience_counter = 0
                if ckpt_path:
                    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state": self.model.state_dict(),
                        "threshold": val_metrics["threshold"],
                        "thresholds": val_metrics["thresholds"],
                    }, ckpt_path)
                    logger.info("Saved best model (cat_f1=%.4f)", best_f1)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return history

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        total_loss = 0
        all_logits = []
        all_labels = []
        num_batches = 0

        for batch in loader:
            out = self._run_batch(batch)
            if out["loss"] is not None:
                total_loss += out["loss"].item()
            num_batches += 1
            all_logits.append(out["logits"].cpu())
            all_labels.append(batch["category_labels"])

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        thresholds = _tune_thresholds(all_logits, all_labels)
        pred_cats = _apply_thresholds(all_logits, thresholds)
        
        gold_cats = []
        for i in range(all_labels.size(0)):
            cats = {CATEGORY_LIST[j] for j in range(NUM_CATEGORIES) if all_labels[i, j] == 1}
            gold_cats.append(cats)

        cat_m = category_f1(pred_cats, gold_cats)
        avg_threshold = sum(thresholds) / len(thresholds)

        return {
            "loss": total_loss / max(num_batches, 1),
            "category_f1": cat_m["f1"],
            "category_precision": cat_m["precision"],
            "category_recall": cat_m["recall"],
            "threshold": avg_threshold,
            "thresholds": thresholds,
        }


# ---------------------------------------------------------------------------
# Hierarchical threshold tuning / decode
# ---------------------------------------------------------------------------

def _tune_thresholds_generic(logits: torch.Tensor,
                             labels: torch.Tensor) -> list[float]:
    probs = torch.sigmoid(logits)
    num_cols = logits.size(1)
    best_thresholds = [0.5] * num_cols
    for col in range(num_cols):
        best_f1 = -1.0
        gold_col = labels[:, col]
        pred_col = probs[:, col]
        for t in THRESHOLD_GRID:
            preds = (pred_col >= t).long()
            tp = ((preds == 1) & (gold_col == 1)).sum().item()
            fp = ((preds == 1) & (gold_col == 0)).sum().item()
            fn = ((preds == 0) & (gold_col == 1)).sum().item()
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            if f > best_f1:
                best_f1 = f
                best_thresholds[col] = t
    return best_thresholds


def tune_entity_thresholds(entity_logits: torch.Tensor,
                           entity_labels: torch.Tensor) -> list[float]:
    return _tune_thresholds_generic(entity_logits, entity_labels)


def tune_attr_thresholds(attr_logits: torch.Tensor,
                         attr_labels: torch.Tensor,
                         entity_labels: torch.Tensor,
                         entity_idx: int) -> list[float]:
    mask = entity_labels[:, entity_idx] > 0
    if not mask.any():
        return [0.5] * attr_logits.size(1)
    return _tune_thresholds_generic(attr_logits[mask], attr_labels[mask])


def hierarchical_decode(entity_logits: torch.Tensor,
                        food_logits: torch.Tensor,
                        drinks_logits: torch.Tensor,
                        restaurant_logits: torch.Tensor,
                        entity_thresholds: list[float],
                        food_attr_thresholds: list[float],
                        drinks_attr_thresholds: list[float],
                        restaurant_attr_thresholds: list[float]) -> list[set[str]]:
    entity_probs = torch.sigmoid(entity_logits)
    attr_logits_map = {
        "FOOD": food_logits, "DRINKS": drinks_logits, "RESTAURANT": restaurant_logits,
    }
    attr_thresh_map = {
        "FOOD": food_attr_thresholds,
        "DRINKS": drinks_attr_thresholds,
        "RESTAURANT": restaurant_attr_thresholds,
    }
    results = []
    for i in range(entity_probs.size(0)):
        cats = set()
        for ent_idx, entity in enumerate(ENTITY_LIST):
            if entity_probs[i, ent_idx] < entity_thresholds[ent_idx]:
                continue
            if entity not in MULTI_ATTR_ENTITIES:
                cats.add(f"{entity}#GENERAL")
            else:
                attr_probs = torch.sigmoid(attr_logits_map[entity][i])
                thresholds = attr_thresh_map[entity]
                for attr_idx, attr in enumerate(ENTITY2ATTRS[entity]):
                    if attr_probs[attr_idx] >= thresholds[attr_idx]:
                        cats.add(f"{entity}#{attr}")
        results.append(cats)
    return results


# ---------------------------------------------------------------------------
# HierarchicalCategoryTrainer
# ---------------------------------------------------------------------------

class HierarchicalCategoryTrainer:
    def __init__(self, model, optimizer, scheduler, device,
                 patience: int = 5, grad_clip: float = 1.0,
                 log_path: str = "", use_fp16: bool = False,
                 grad_accum_steps: int = 1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.grad_clip = grad_clip
        self.log_path = log_path
        self.grad_accum_steps = grad_accum_steps
        self.use_fp16 = use_fp16 and device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_fp16 else None

    def _run_batch(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast("cuda", enabled=self.use_fp16):
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                entity_labels=batch["entity_labels"],
                food_attr_labels=batch["food_attr_labels"],
                drinks_attr_labels=batch["drinks_attr_labels"],
                restaurant_attr_labels=batch["restaurant_attr_labels"],
            )
        return out

    def train(self, train_loader, val_loader, epochs: int,
              ckpt_path: str | None = None) -> list[dict]:
        history = []
        best_f1 = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            self.optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                out = self._run_batch(batch)
                loss = out["loss"] / self.grad_accum_steps
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += out["loss"].item()

                is_accum_step = (step + 1) % self.grad_accum_steps == 0
                is_last_step = (step + 1) == len(train_loader)
                if is_accum_step or is_last_step:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        if self.grad_clip > 0:
                            clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.grad_clip > 0:
                            clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            val_metrics = self.evaluate(val_loader)
            record = {"epoch": epoch, "train_loss": avg_loss, **val_metrics}
            history.append(record)
            logger.info("Epoch %d: loss=%.4f cat_f1=%.4f",
                        epoch, avg_loss, val_metrics["category_f1"])

            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            if val_metrics["category_f1"] > best_f1:
                best_f1 = val_metrics["category_f1"]
                patience_counter = 0
                if ckpt_path:
                    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state": self.model.state_dict(),
                        "entity_thresholds": val_metrics["entity_thresholds"],
                        "food_attr_thresholds": val_metrics["food_attr_thresholds"],
                        "drinks_attr_thresholds": val_metrics["drinks_attr_thresholds"],
                        "restaurant_attr_thresholds": val_metrics["restaurant_attr_thresholds"],
                    }, ckpt_path)
                    logger.info("Saved best model (cat_f1=%.4f)", best_f1)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return history

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        total_loss = 0
        all_entity_logits = []
        all_food_logits = []
        all_drinks_logits = []
        all_restaurant_logits = []
        all_entity_labels = []
        all_food_labels = []
        all_drinks_labels = []
        all_restaurant_labels = []
        num_batches = 0

        for batch in loader:
            out = self._run_batch(batch)
            if out["loss"] is not None:
                total_loss += out["loss"].item()
            num_batches += 1
            all_entity_logits.append(out["entity_logits"].cpu())
            all_food_logits.append(out["food_attr_logits"].cpu())
            all_drinks_logits.append(out["drinks_attr_logits"].cpu())
            all_restaurant_logits.append(out["restaurant_attr_logits"].cpu())
            all_entity_labels.append(batch["entity_labels"])
            all_food_labels.append(batch["food_attr_labels"])
            all_drinks_labels.append(batch["drinks_attr_labels"])
            all_restaurant_labels.append(batch["restaurant_attr_labels"])

        ent_logits = torch.cat(all_entity_logits, dim=0)
        food_logits = torch.cat(all_food_logits, dim=0)
        drinks_logits = torch.cat(all_drinks_logits, dim=0)
        rest_logits = torch.cat(all_restaurant_logits, dim=0)
        ent_labels = torch.cat(all_entity_labels, dim=0)
        food_labels = torch.cat(all_food_labels, dim=0)
        drinks_labels = torch.cat(all_drinks_labels, dim=0)
        rest_labels = torch.cat(all_restaurant_labels, dim=0)

        entity_thresholds = tune_entity_thresholds(ent_logits, ent_labels)
        food_attr_thresholds = tune_attr_thresholds(
            food_logits, food_labels, ent_labels, ENT2IDX["FOOD"])
        drinks_attr_thresholds = tune_attr_thresholds(
            drinks_logits, drinks_labels, ent_labels, ENT2IDX["DRINKS"])
        restaurant_attr_thresholds = tune_attr_thresholds(
            rest_logits, rest_labels, ent_labels, ENT2IDX["RESTAURANT"])

        pred_cats = hierarchical_decode(
            ent_logits, food_logits, drinks_logits, rest_logits,
            entity_thresholds, food_attr_thresholds,
            drinks_attr_thresholds, restaurant_attr_thresholds,
        )

        gold_cats = []
        for i in range(ent_labels.size(0)):
            cats = set()
            for ent_idx, entity in enumerate(ENTITY_LIST):
                if ent_labels[i, ent_idx] == 0:
                    continue
                if entity not in MULTI_ATTR_ENTITIES:
                    cats.add(f"{entity}#GENERAL")
                else:
                    attr_labels = {"FOOD": food_labels, "DRINKS": drinks_labels,
                                   "RESTAURANT": rest_labels}[entity]
                    for attr_idx, attr in enumerate(ENTITY2ATTRS[entity]):
                        if attr_labels[i, attr_idx] == 1:
                            cats.add(f"{entity}#{attr}")
            gold_cats.append(cats)

        cat_m = category_f1(pred_cats, gold_cats)

        return {
            "loss": total_loss / max(num_batches, 1),
            "category_f1": cat_m["f1"],
            "category_precision": cat_m["precision"],
            "category_recall": cat_m["recall"],
            "entity_thresholds": entity_thresholds,
            "food_attr_thresholds": food_attr_thresholds,
            "drinks_attr_thresholds": drinks_attr_thresholds,
            "restaurant_attr_thresholds": restaurant_attr_thresholds,
        }
