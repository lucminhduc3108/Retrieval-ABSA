import json
import logging
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from src.embedding.loss import infonce_loss

logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    def __init__(self, model, optimizer, scheduler, tau, device,
                 log_path, grad_clip=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tau = tau
        self.device = device
        self.log_path = log_path
        self.grad_clip = grad_clip

    def _run_batch(self, batch):
        keys = ["anchor_input_ids", "anchor_attention_mask",
                "pos_input_ids", "pos_attention_mask",
                "neg1_input_ids", "neg1_attention_mask",
                "neg2_input_ids", "neg2_attention_mask"]
        batch = {k: batch[k].to(self.device) for k in keys}
        out = self.model(
            batch["anchor_input_ids"], batch["anchor_attention_mask"],
            batch["pos_input_ids"], batch["pos_attention_mask"],
            batch["neg1_input_ids"], batch["neg1_attention_mask"],
            batch["neg2_input_ids"], batch["neg2_attention_mask"],
        )
        negatives = []
        if out["neg1_vecs"] is not None:
            negatives.append(out["neg1_vecs"])
        if out["neg2_vecs"] is not None:
            negatives.append(out["neg2_vecs"])
        loss = infonce_loss(out["anchor_vecs"], out["pos_vecs"],
                            negatives=negatives if negatives else None,
                            tau=self.tau)
        return loss, out

    def train(self, train_loader, val_loader, epochs, patience=3,
              ckpt_path=None) -> list[dict]:
        history = []
        best_recall = -1
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                loss, _ = self._run_batch(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            recall = self.evaluate_recall(val_loader)
            record = {"epoch": epoch, "train_loss": avg_loss, **recall}
            history.append(record)
            logger.info("Epoch %d: loss=%.4f recall@3=%.4f", epoch, avg_loss, recall["recall@3"])

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
        return result
