import argparse
import logging
import math
import os
import sys
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.absa.sentiment_dataset import SentimentDataset
from src.absa.sentiment_model import SentimentPredictor
from src.absa.sentiment_trainer import SentimentTrainer
from src.embedding.model import ContrastiveEmbedder
from src.retrieval.index import load_index
from src.retrieval.retriever import Retriever
from src.utils.io import load_yaml, read_jsonl
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2.yaml")
    parser.add_argument("--embedding_ckpt", default=None)
    parser.add_argument("--index_dir", default="indexes/")
    parser.add_argument("--retrieval_config", default="configs/retrieval_v2.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_retrieval", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--ckpt_path", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    use_retrieval = cfg.get("use_retrieval", True) and not args.no_retrieval
    ret_cfg = load_yaml(args.retrieval_config) if use_retrieval else {"top_k": 0, "threshold": 0.0}
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s, retrieval: %s", device, use_retrieval)

    embedding_model = None
    retriever = None
    if use_retrieval:
        if not args.embedding_ckpt:
            parser.error("--embedding_ckpt required when using retrieval")
        embedding_model = ContrastiveEmbedder(
            model_name=cfg["model_name"], proj_dim=256)
        embedding_model.load_state_dict(
            torch.load(args.embedding_ckpt, map_location=device))
        embedding_model.to(device)
        embedding_model.eval()
        logger.info("Loaded embedding model: %s", args.embedding_ckpt)

        index, metadata, _ = load_index(args.index_dir)
        retriever = Retriever(index, metadata,
                              top_k=ret_cfg["top_k"],
                              threshold=ret_cfg["threshold"])
        logger.info("Loaded FAISS index (%d vectors)", index.ntotal)
    else:
        logger.info("Running WITHOUT retrieval")

    records = read_jsonl(cfg["sentiment_path"])
    train_records = [r for r in records if r["split"] == "train"]

    if args.limit:
        train_records = train_records[:args.limit]

    sentences = [r["sentence"] for r in train_records]
    unique_sents = list(set(sentences))
    from sklearn.model_selection import train_test_split as tts
    train_sents, val_sents = tts(
        unique_sents, test_size=cfg["val_ratio"], random_state=cfg["seed"])
    val_sents_set = set(val_sents)
    train_recs = [r for r in train_records if r["sentence"] not in val_sents_set]
    val_recs = [r for r in train_records if r["sentence"] in val_sents_set]

    logger.info("Train: %d records, Val: %d records", len(train_recs), len(val_recs))

    pol_counts = Counter(r["polarity"] for r in train_recs)
    pol_order = ["positive", "negative", "neutral"]
    freq = [pol_counts.get(p, 1) for p in pol_order]
    raw = [math.sqrt(len(train_recs) / f) for f in freq]
    norm = raw[0]
    cls_weights = [w / norm for w in raw]
    logger.info("Sqrt class weights: pos=%.3f neg=%.3f neu=%.3f",
                cls_weights[0], cls_weights[1], cls_weights[2])
    class_weights_tensor = torch.tensor(cls_weights, dtype=torch.float32).to(device)

    ds_kwargs = dict(
        retriever=retriever,
        tokenizer_name=cfg["model_name"],
        embedding_model=embedding_model,
        max_length=cfg["max_seq_length"],
        top_k=ret_cfg.get("top_k", 0) if use_retrieval else 0,
        device=device,
        use_retrieval=use_retrieval,
    )

    train_ds = SentimentDataset(train_recs, **ds_kwargs)
    val_ds = SentimentDataset(val_recs, **ds_kwargs)

    if embedding_model is not None:
        embedding_model.cpu()
        train_ds.device = "cpu"
        train_ds.embedding_model = embedding_model
        val_ds.device = "cpu"
        val_ds.embedding_model = embedding_model
        torch.cuda.empty_cache()
        logger.info("Moved embedding model to CPU to free GPU memory")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    model = SentimentPredictor(
        model_name=cfg["model_name"],
        num_sent_labels=cfg["num_sent_labels"],
        embed_dim=cfg.get("embed_dim", 64),
        tau=cfg.get("tau", 0.05),
        dropout=cfg.get("dropout", 0.1),
        use_retrieval=use_retrieval,
        class_weights=class_weights_tensor,
    ).to(device)

    encoder_lr = cfg.get("encoder_lr", cfg.get("lr", 2e-5))
    head_lr = cfg.get("head_lr", cfg.get("lr", 2e-4))
    param_groups = [
        {"params": list(model.encoder.parameters()), "lr": encoder_lr},
        {"params": list(model.sentiment_head.parameters()), "lr": head_lr},
    ]
    if model.label_interp is not None:
        param_groups.append(
            {"params": list(model.label_interp.parameters()), "lr": head_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["weight_decay"])

    epochs = args.epochs if args.epochs else cfg["epochs"]
    grad_accum = args.grad_accum_steps or cfg.get("grad_accum_steps", 1)
    total_steps = (len(train_loader) * epochs) // grad_accum
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    trainer = SentimentTrainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        device=device, patience=cfg["patience"],
        grad_clip=cfg["grad_clip"], log_path=cfg["log_path"],
        use_fp16=device == "cuda",
        grad_accum_steps=grad_accum,
    )

    ckpt_path = args.ckpt_path or os.path.join(cfg["ckpt_dir"], "best.pt")
    trainer.train(train_loader, val_loader, epochs=epochs, ckpt_path=ckpt_path)
    logger.info("Training complete. Best checkpoint: %s", ckpt_path)


if __name__ == "__main__":
    main()
