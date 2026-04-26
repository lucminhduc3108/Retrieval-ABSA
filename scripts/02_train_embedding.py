import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.datasets import ContrastiveTripletDataset
from src.embedding.model import ContrastiveEmbedder
from src.embedding.trainer import ContrastiveTrainer
from src.utils.io import load_yaml
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/embedding.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit dataset size for smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for smoke test")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = ContrastiveTripletDataset(cfg["triplets_path"], tokenizer,
                                        max_length=cfg["max_seq_length"])

    if args.limit:
        dataset.triplets = dataset.triplets[:args.limit]

    val_size = max(1, int(len(dataset) * cfg["val_ratio"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(cfg["seed"]))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    model = ContrastiveEmbedder(
        model_name=cfg["model_name"],
        proj_dim=cfg["proj_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    epochs = args.epochs if args.epochs else cfg["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    trainer = ContrastiveTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        tau=cfg["tau"], device=device,
        log_path=cfg["log_path"], grad_clip=cfg["grad_clip"],
        use_fp16=device == "cuda",
    )

    ckpt_path = os.path.join(cfg["ckpt_dir"], "best.pt")
    trainer.train(train_loader, val_loader, epochs=epochs,
                  patience=cfg["patience"], ckpt_path=ckpt_path)

    logger.info("Training complete. Best checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
