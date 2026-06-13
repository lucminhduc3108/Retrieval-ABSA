import argparse
import logging
import math
import os
import sys

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.absa.category_dataset import CategoryDataset
from src.absa.category_model import CategoryDetector
from src.absa.category_trainer import CategoryTrainer
from src.data.category_builder import CATEGORY_LIST, NUM_CATEGORIES
from src.utils.io import load_yaml, read_jsonl
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_pos_weight(records: list[dict],
                       cap: float | None = None) -> torch.Tensor:
    n = len(records)
    counts = [0] * NUM_CATEGORIES
    for r in records:
        for i, v in enumerate(r["category_vector"]):
            counts[i] += v
    weights = []
    for c in counts:
        if c > 0:
            w = math.sqrt((n - c) / c)
            if cap is not None:
                w = min(w, cap)
            weights.append(w)
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ckpt_path", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    records = read_jsonl(cfg["category_path"])
    train_records = [r for r in records if r["split"] == "train"]

    if args.limit:
        train_records = train_records[:args.limit]

    try:
        import numpy as np
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        label_matrix = np.array([r["category_vector"] for r in train_records])
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=cfg["val_ratio"],
                                               random_state=cfg["seed"])
        for train_idx, val_idx in msss.split(label_matrix, label_matrix):
            train_records, val_records = ([train_records[i] for i in train_idx],
                                          [train_records[i] for i in val_idx])
        logger.info("Multi-label stratified split: Train %d, Val %d",
                    len(train_records), len(val_records))
    except ImportError:
        stratify_key = [min(sum(r["category_vector"]), 2) for r in train_records]
        train_records, val_records = train_test_split(
            train_records, test_size=cfg["val_ratio"],
            random_state=cfg["seed"], stratify=stratify_key,
        )
        logger.info("Train: %d, Val: %d", len(train_records), len(val_records))

    pw_cap = cfg.get("pos_weight_cap", 5.0)
    encoder_lr = float(cfg.get("encoder_lr", 2e-5))
    head_lr = float(cfg.get("head_lr", 1e-4))
    epochs = args.epochs if args.epochs else cfg["epochs"]
    grad_accum = cfg.get("grad_accum_steps", 1)
    use_fp16 = cfg.get("use_fp16", device == "cuda")
    ckpt_path = args.ckpt_path or os.path.join(cfg["ckpt_dir"], "best.pt")

    use_asl = cfg.get("use_asl", False)
    if pw_cap is None:
        pos_weight = None
        logger.info("pos_weight: disabled")
    else:
        pos_weight = compute_pos_weight(train_records, cap=pw_cap).to(device)
        logger.info("pos_weight (cap=%.1f): %s",
                    pw_cap, [f"{w:.2f}" for w in pos_weight.tolist()])
    if use_asl:
        logger.info("ASL enabled (gamma_neg=%s, margin=%s)",
                    cfg.get("asl_gamma_neg", 4), cfg.get("asl_margin", 0.05))

    train_ds = CategoryDataset(train_records, tokenizer_name=cfg["model_name"],
                               max_length=cfg["max_seq_length"])
    val_ds = CategoryDataset(val_records, tokenizer_name=cfg["model_name"],
                             max_length=cfg["max_seq_length"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    use_cat_attention = cfg.get("use_cat_attention", False)
    model = CategoryDetector(
        model_name=cfg["model_name"],
        num_categories=cfg["num_categories"],
        pos_weight=pos_weight,
        use_asl=use_asl,
        asl_gamma_neg=cfg.get("asl_gamma_neg", 4),
        asl_gamma_pos=cfg.get("asl_gamma_pos", 0),
        asl_margin=cfg.get("asl_margin", 0.05),
        use_cat_attention=use_cat_attention,
    ).to(device)

    if use_cat_attention:
        head_params = (list(model.category_head.parameters()) +
                       list(model.cat_queries.parameters()) +
                       list(model.cat_attention.parameters()) +
                       list(model.cat_norm.parameters()))
    else:
        head_params = (list(model.category_head.parameters()) +
                       list(model.pooler.parameters()))
    param_groups = [
        {"params": list(model.encoder.parameters()), "lr": encoder_lr},
        {"params": head_params, "lr": head_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])

    total_steps = (len(train_loader) * epochs) // grad_accum
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    trainer = CategoryTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        device=device, patience=cfg["patience"],
        grad_clip=cfg["grad_clip"], log_path=cfg["log_path"],
        use_fp16=use_fp16, grad_accum_steps=grad_accum,
    )
    trainer.train(train_loader, val_loader, epochs=epochs, ckpt_path=ckpt_path)

    logger.info("Training complete. Best checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
