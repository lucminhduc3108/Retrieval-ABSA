import argparse
import logging
import os
import sys

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.absa.dataset import RetrievalABSADataset
from src.absa.model import RetrievalABSA
from src.absa.trainer import ABSATrainer
from src.embedding.model import ContrastiveEmbedder
from src.retrieval.index import load_index
from src.retrieval.retriever import Retriever
from src.utils.io import load_yaml, read_jsonl
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/absa.yaml")
    parser.add_argument("--embedding_ckpt", default=None)
    parser.add_argument("--index_dir", default="indexes/")
    parser.add_argument("--retrieval_config", default="configs/retrieval.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no_retrieval", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--ckpt_path", default=None,
                        help="Override checkpoint save path")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ret_cfg = load_yaml(args.retrieval_config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    embedding_model = None
    retriever = None
    if not args.no_retrieval:
        if not args.embedding_ckpt:
            parser.error("--embedding_ckpt is required when not using --no_retrieval")
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
        logger.info("Loaded FAISS index (%d vectors) from %s",
                    index.ntotal, args.index_dir)
    else:
        logger.info("Running WITHOUT retrieval (--no_retrieval)")

    bio_records = read_jsonl(cfg["bio_path"])
    train_records = [r for r in bio_records if r["split"] == "train"]
    test_records = [r for r in bio_records if r["split"] == "test"]

    if args.limit:
        train_records = train_records[:args.limit]

    polarities = [r.get("polarity", "positive") for r in train_records]
    train_records, val_records = train_test_split(
        train_records, test_size=cfg["val_ratio"],
        random_state=cfg["seed"], stratify=polarities,
    )

    logger.info("Train: %d, Val: %d, Test: %d",
                len(train_records), len(val_records), len(test_records))

    ds_kwargs = dict(
        retriever=retriever,
        tokenizer_name=cfg["model_name"],
        embedding_model=embedding_model,
        max_length=cfg["max_seq_length"],
        query_budget=cfg["query_budget"],
        top_k=ret_cfg["top_k"] if not args.no_retrieval else 0,
        device=device,
    )

    train_ds = RetrievalABSADataset(train_records, **ds_kwargs)
    val_ds = RetrievalABSADataset(val_records, **ds_kwargs)

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

    model = RetrievalABSA(
        model_name=cfg["model_name"],
        num_bio_labels=cfg["num_bio_labels"],
        num_sent_labels=cfg["num_sent_labels"],
        lambda_cls=cfg["lambda_cls"],
        dropout=cfg["dropout"],
        cls_class_weights=cfg.get("cls_class_weights"),
    ).to(device)

    if hasattr(train_ds, 'tokenizer') and len(train_ds.tokenizer) > model.encoder.config.vocab_size:
        model.encoder.resize_token_embeddings(len(train_ds.tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    epochs = args.epochs if args.epochs else cfg["epochs"]
    grad_accum = args.grad_accum_steps or cfg.get("grad_accum_steps", 1)
    total_steps = (len(train_loader) * epochs) // grad_accum
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    trainer = ABSATrainer(
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
