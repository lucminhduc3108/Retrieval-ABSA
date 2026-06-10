import json
import logging
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from src.data.category_builder import POL2ID

logger = logging.getLogger(__name__)

_POL_LABELS = ["positive", "negative", "neutral"]
_POL_IDS = [POL2ID[p] for p in _POL_LABELS]


class SentimentTrainer:
    def __init__(self, model, optimizer, scheduler, device,
                 patience: int = 5, grad_clip: float = 1.0,
                 log_path: str = "", use_fp16: bool = False,
                 grad_accum_steps: int = 1, lambda_rank: float = 0.1):
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
        self.lambda_rank = lambda_rank

    def _run_batch(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast("cuda", enabled=self.use_fp16):
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                neighbor_polarities=batch.get("neighbor_polarities"),
                neighbor_scores=batch.get("neighbor_scores"),
                query_vec=batch.get("query_vec"),
                neighbor_vecs=batch.get("neighbor_vecs"),
                query_polarity=batch.get("query_polarity"),
                sentiment_label=batch["sentiment_label"],
            )
        # Combined loss: sentiment + lambda_rank * ranking
        ranking_loss = out.get("ranking_loss")
        if out["loss"] is not None and ranking_loss is not None:
            out["combined_loss"] = out["loss"] + self.lambda_rank * ranking_loss
        else:
            out["combined_loss"] = out["loss"]
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
                loss = out["combined_loss"] / self.grad_accum_steps
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += out["combined_loss"].item()

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
            logger.info(
                "Epoch %d: loss=%.4f sent_acc=%.4f macro_f1=%.4f "
                "pos=%.3f neg=%.3f neu=%.3f",
                epoch, avg_loss, val_metrics["sentiment_acc"],
                val_metrics["sentiment_macro_f1"],
                val_metrics["f1_positive"], val_metrics["f1_negative"],
                val_metrics["f1_neutral"])

            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            if val_metrics["sentiment_macro_f1"] > best_f1:
                best_f1 = val_metrics["sentiment_macro_f1"]
                patience_counter = 0
                if ckpt_path:
                    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info("Saved best model (macro_f1=%.4f)", best_f1)
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
        all_preds = []
        all_golds = []
        num_batches = 0

        for batch in loader:
            out = self._run_batch(batch)
            if out["combined_loss"] is not None:
                total_loss += out["combined_loss"].item()
            num_batches += 1

            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            golds = batch["sentiment_label"].tolist()
            all_preds.extend(preds)
            all_golds.extend(golds)

        acc = accuracy_score(all_golds, all_preds)
        macro_f1 = f1_score(all_golds, all_preds, average="macro", zero_division=0)
        per_class = f1_score(all_golds, all_preds, average=None,
                             labels=_POL_IDS, zero_division=0)

        return {
            "loss": total_loss / max(num_batches, 1),
            "sentiment_acc": float(acc),
            "sentiment_macro_f1": float(macro_f1),
            "f1_positive": float(per_class[0]),
            "f1_negative": float(per_class[1]),
            "f1_neutral": float(per_class[2]),
        }
