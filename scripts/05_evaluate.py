import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding_ckpt", required=True)
    parser.add_argument("--index_dir", default="indexes/")
    parser.add_argument("--retrieval_config", default="configs/retrieval.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ret_cfg = load_yaml(args.retrieval_config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    embedding_model = ContrastiveEmbedder(
        model_name=cfg["model_name"], proj_dim=256)
    embedding_model.load_state_dict(
        torch.load(args.embedding_ckpt, map_location=device))
    embedding_model.to(device)
    embedding_model.eval()

    index, metadata, _ = load_index(args.index_dir)
    retriever = Retriever(index, metadata,
                          top_k=ret_cfg["top_k"],
                          threshold=ret_cfg["threshold"])
    logger.info("Loaded FAISS index (%d vectors)", index.ntotal)

    bio_records = read_jsonl(cfg["bio_path"])
    test_records = [r for r in bio_records if r["split"] == "test"]
    logger.info("Test records: %d", len(test_records))

    test_ds = RetrievalABSADataset(
        test_records, retriever=retriever,
        tokenizer_name=cfg["model_name"],
        embedding_model=embedding_model,
        max_length=cfg["max_seq_length"],
        query_budget=cfg["query_budget"],
        top_k=ret_cfg["top_k"],
        device=device,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = RetrievalABSA(
        model_name=cfg["model_name"],
        num_bio_labels=cfg["num_bio_labels"],
        num_sent_labels=cfg["num_sent_labels"],
        lambda_cls=cfg["lambda_cls"],
        dropout=cfg["dropout"],
    ).to(device)

    if hasattr(test_ds, 'tokenizer') and len(test_ds.tokenizer) > model.encoder.config.vocab_size:
        model.encoder.resize_token_embeddings(len(test_ds.tokenizer))

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    logger.info("Loaded ABSA checkpoint: %s", args.checkpoint)

    trainer = ABSATrainer(
        model=model, optimizer=None, scheduler=None,
        device=device, log_path="",
    )
    results = trainer.evaluate(test_loader)

    table = (
        "| Metric           | Value  |\n"
        "|------------------|--------|\n"
        f"| BIO token F1     | {results['bio_token_f1']:.4f} |\n"
        f"| Span F1          | {results['span_f1']:.4f} |\n"
        f"| Sentiment Acc    | {results['sentiment_acc']:.4f} |\n"
        f"| Sentiment MacF1  | {results['sentiment_macro_f1']:.4f} |\n"
        f"| Joint F1         | {results['joint_f1']:.4f} |\n"
    )
    print("\n" + table)

    out_path = "logs/eval_results.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
