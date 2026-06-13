import json
import logging
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from src.embedding.loss import infonce_loss

logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    def __init__(self, model, optimizer, scheduler, tau, device,
                 log_path, grad_clip=1.0, use_fp16=False,
                 grad_accum_steps=1,
                 loss_mode="combined", loss_alpha=1.0, loss_beta=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tau = tau
        self.device = device
        self.log_path = log_path
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.loss_mode = loss_mode
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.use_fp16 = use_fp16 and device == "cuda"
        self.scaler = GradScaler("cuda") if self.use_fp16 else None
        if self.use_fp16:
            logger.info("fp16 mixed precision enabled")
        if self.grad_accum_steps > 1:
            logger.info("Gradient accumulation: %d steps", self.grad_accum_steps)
        if self.loss_mode == "split":
            logger.info("Split loss: alpha=%.2f (polarity), beta=%.2f (category)",
                        self.loss_alpha, self.loss_beta)

    def _run_batch(self, batch):
        keys = ["anchor_input_ids", "anchor_attention_mask",
                "pos_input_ids", "pos_attention_mask",
                "neg1_input_ids", "neg1_attention_mask",
                "neg2_input_ids", "neg2_attention_mask"]
        batch = {k: batch[k].to(self.device) for k in keys}
        with autocast("cuda", enabled=self.use_fp16):
            out = self.model(
                batch["anchor_input_ids"], batch["anchor_attention_mask"],
                batch["pos_input_ids"], batch["pos_attention_mask"],
                batch["neg1_input_ids"], batch["neg1_attention_mask"],
                batch["neg2_input_ids"], batch["neg2_attention_mask"],
            )
            anchor = out["anchor_vecs"]
            pos = out["pos_vecs"]
            neg1 = out["neg1_vecs"]
            neg2 = out["neg2_vecs"]

            if self.loss_mode == "split" and neg1 is not None and neg2 is not None:
                loss_pol = infonce_loss(anchor, pos, negatives=[neg1], tau=self.tau)
                loss_cat = infonce_loss(anchor, pos, negatives=[neg2], tau=self.tau)
                loss = self.loss_alpha * loss_pol + self.loss_beta * loss_cat
            else:
                negatives = []
                if neg1 is not None:
                    negatives.append(neg1)
                if neg2 is not None:
                    negatives.append(neg2)
                loss = infonce_loss(anchor, pos,
                                    negatives=negatives if negatives else None,
                                    tau=self.tau)
                loss_pol = loss_cat = None
        return loss, out, loss_pol, loss_cat

    def train(self, train_loader, val_loader, epochs, patience=3,
              ckpt_path=None) -> list[dict]:
        history = []
        best_recall = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            total_loss_pol = 0.0
            total_loss_cat = 0.0
            self.optimizer.zero_grad()
            epoch_pos_sim = 0.0
            epoch_neg1_sim = 0.0
            epoch_neg2_sim = 0.0
            for step, batch in enumerate(train_loader):
                loss, out, loss_pol, loss_cat = self._run_batch(batch)
                scaled_loss = loss / self.grad_accum_steps
                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                total_loss += loss.item()
                if loss_pol is not None:
                    total_loss_pol += loss_pol.item()
                    total_loss_cat += loss_cat.item()

                with torch.no_grad():
                    a = out["anchor_vecs"].detach()
                    p = out["pos_vecs"].detach()
                    epoch_pos_sim += (a * p).sum(dim=1).mean().item()
                    n1 = out.get("neg1_vecs")
                    n2 = out.get("neg2_vecs")
                    epoch_neg1_sim += (a * n1.detach()).sum(dim=1).mean().item() if n1 is not None else 0.0
                    epoch_neg2_sim += (a * n2.detach()).sum(dim=1).mean().item() if n2 is not None else 0.0

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
            n_steps = len(train_loader)
            avg_pos_sim = epoch_pos_sim / n_steps
            avg_neg1_sim = epoch_neg1_sim / n_steps
            avg_neg2_sim = epoch_neg2_sim / n_steps
            margin1 = avg_pos_sim - avg_neg1_sim
            margin2 = avg_pos_sim - avg_neg2_sim

            recall = self.evaluate_recall(val_loader)
            record = {
                "epoch": epoch, "train_loss": avg_loss,
                "avg_pos_sim": avg_pos_sim, "avg_neg1_sim": avg_neg1_sim,
                "avg_neg2_sim": avg_neg2_sim,
                "margin_neg1": margin1, "margin_neg2": margin2,
                **recall,
            }
            if self.loss_mode == "split":
                record["loss_polarity"] = total_loss_pol / n_steps
                record["loss_category"] = total_loss_cat / n_steps
            history.append(record)
            logger.info(
                "Epoch %d | loss=%.4f | pos=%.3f neg1=%.3f neg2=%.3f "
                "| m1=%.3f m2=%.3f | R@1=%.3f R@3=%.3f R@5=%.3f",
                epoch, avg_loss, avg_pos_sim, avg_neg1_sim, avg_neg2_sim,
                margin1, margin2,
                recall.get("recall@1", 0.0), recall["recall@3"],
                recall.get("recall@5", 0.0),
            )

            if self.log_path:
                Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            if recall["recall@3"] > best_recall:
                best_recall = recall["recall@3"]
                patience_counter = 0
                if ckpt_path:
                    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), ckpt_path)
                    logger.info("Saved best model (recall@3=%.4f)", best_recall)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        return history

    @torch.no_grad()
    def evaluate_recall(self, val_loader, k_list=(1, 3, 5)) -> dict:
        self.model.eval()
        all_anchor = []
        all_pos = []
        for batch in val_loader:
            batch_dev = {k: batch[k].to(self.device) for k in batch}
            a_vec = self.model.encode(batch_dev["anchor_input_ids"],
                                      batch_dev["anchor_attention_mask"])
            p_vec = self.model.encode(batch_dev["pos_input_ids"],
                                      batch_dev["pos_attention_mask"])
            all_anchor.append(a_vec.cpu())
            all_pos.append(p_vec.cpu())

        anchors = torch.cat(all_anchor, dim=0)
        positives = torch.cat(all_pos, dim=0)
        sim = anchors @ positives.T
        N = sim.size(0)

        result = {}
        for k in k_list:
            topk_indices = sim.topk(min(k, N), dim=1).indices
            hits = sum(1 for i in range(N) if i in topk_indices[i].tolist())
            result[f"recall@{k}"] = hits / N

        intra_mean = sim.diagonal().mean().item()
        if N > 1:
            mask = ~torch.eye(N, dtype=torch.bool)
            inter_mean = sim[mask].mean().item()
        else:
            inter_mean = 0.0
        ratio = intra_mean / (inter_mean + 1e-8)
        result["intra_sim"] = intra_mean
        result["inter_sim"] = inter_mean
        result["intra_inter_ratio"] = ratio
        logger.info("Eval | intra=%.4f inter=%.4f ratio=%.4f",
                     intra_mean, inter_mean, ratio)

        return result
