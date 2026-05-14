import json
import logging
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from src.evaluation.metrics import (
    bio_token_metrics, constrained_bio_decode, extract_spans, span_f1,
    sentiment_metrics, joint_f1,
)

logger = logging.getLogger(__name__)


class ABSATrainer:
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
        if self.use_fp16:
            logger.info("fp16 mixed precision enabled")
        if self.grad_accum_steps > 1:
            logger.info("Gradient accumulation: %d steps", self.grad_accum_steps)

    def _run_batch(self, batch):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast("cuda", enabled=self.use_fp16):
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bio_labels=batch["bio_labels"],
                sentiment_label=batch["sentiment_label"],
                crf_mask=batch.get("crf_mask"),
            )
        return out

    def train(self, train_loader, val_loader, epochs: int,
              ckpt_path: str | None = None) -> list[dict]:
        history = []
        best_joint_f1 = -1
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
            logger.info("Epoch %d: loss=%.4f span_f1=%.4f sent_acc=%.4f joint_f1=%.4f",
                        epoch, avg_loss, val_metrics["span_f1"],
                        val_metrics["sentiment_acc"], val_metrics["joint_f1"])

            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            if val_metrics["joint_f1"] > best_joint_f1:
                best_joint_f1 = val_metrics["joint_f1"]
                patience_counter = 0
                if ckpt_path:
                    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info("Saved best model (joint_f1=%.4f)", best_joint_f1)
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
        all_bio_preds = []
        all_bio_golds = []
        all_sent_preds = []
        all_sent_golds = []
        all_pred_spans_with_pol = []
        all_gold_spans_with_pol = []
        num_batches = 0

        use_crf = getattr(self.model, 'use_crf', False)

        for batch in loader:
            out = self._run_batch(batch)
            if out["loss"] is not None:
                total_loss += out["loss"].item()
            num_batches += 1

            bio_logits = out["bio_logits"]
            bio_golds = batch["bio_labels"].to(self.device)

            if use_crf:
                crf_mask = batch["crf_mask"].to(self.device)
                # torchcrf requires mask[:, 0] all True; force it on then strip
                # position 0 from decoded output to match gold_seq length
                crf_mask_fixed = crf_mask.clone()
                crf_mask_fixed[:, 0] = True
                decoded = self.model.crf.decode(bio_logits.float(), mask=crf_mask_fixed)
            else:
                decoded = None

            sent_logits = out["sentiment_logits"]
            sent_preds = sent_logits.argmax(dim=-1)
            sent_golds = batch["sentiment_label"].to(self.device)

            for i in range(bio_golds.size(0)):
                mask = bio_golds[i] != -100
                gold_seq = bio_golds[i][mask].cpu().tolist()

                if decoded is not None:
                    # decoded[i] has length = sum(crf_mask_fixed[i]) = sum(mask) + 1
                    # strip the forced position-0 element to align with gold_seq
                    pred_seq = decoded[i][1:]
                else:
                    pred_seq = bio_logits[i].argmax(dim=-1)[mask].cpu().tolist()
                    pred_seq = constrained_bio_decode(pred_seq)

                all_bio_preds.append(pred_seq)
                all_bio_golds.append(gold_seq)

                pred_sp = extract_spans(pred_seq)
                gold_sp = extract_spans(gold_seq)

                sp = sent_preds[i].item()
                sg = sent_golds[i].item()
                all_sent_preds.append(sp)
                all_sent_golds.append(sg)

                all_pred_spans_with_pol.append(
                    [(s, e, sp) for s, e in pred_sp])
                all_gold_spans_with_pol.append(
                    [(s, e, sg) for s, e in gold_sp])

        avg_loss = total_loss / max(num_batches, 1)
        bio_m = bio_token_metrics(all_bio_preds, all_bio_golds)
        span_m = span_f1(
            [extract_spans(s) for s in all_bio_preds],
            [extract_spans(s) for s in all_bio_golds],
        )
        sent_m = sentiment_metrics(all_sent_preds, all_sent_golds)
        joint = joint_f1(all_pred_spans_with_pol, all_gold_spans_with_pol)

        return {
            "loss": avg_loss,
            "bio_token_f1": bio_m["f1"],
            "span_f1": span_m["f1"],
            "sentiment_acc": sent_m["accuracy"],
            "sentiment_macro_f1": sent_m["macro_f1"],
            "joint_f1": joint,
        }
