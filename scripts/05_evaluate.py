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


def make_table(results: dict, label: str = "") -> str:
    header = f" ({label})" if label else ""
    return (
        f"| Metric{header:16s} | Value  |\n"
        "|------------------|--------|\n"
        f"| BIO token F1     | {results['bio_token_f1']:.4f} |\n"
        f"| Span F1          | {results['span_f1']:.4f} |\n"
        f"| Sentiment Acc    | {results['sentiment_acc']:.4f} |\n"
        f"| Sentiment MacF1  | {results['sentiment_macro_f1']:.4f} |\n"
        f"| Joint F1         | {results['joint_f1']:.4f} |\n"
    )


def evaluate_subset(records, model, trainer, cfg, ret_cfg, retriever,
                    embedding_model, device, no_retrieval):
    ds = RetrievalABSADataset(
        records, retriever=retriever,
        tokenizer_name=cfg["model_name"],
        embedding_model=embedding_model,
        max_length=cfg["max_seq_length"],
        query_budget=cfg["query_budget"],
        top_k=ret_cfg["top_k"] if not no_retrieval else 0,
        device=device,
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"])
    return trainer.evaluate(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/absa.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding_ckpt", default=None)
    parser.add_argument("--index_dir", default="indexes/")
    parser.add_argument("--retrieval_config", default="configs/retrieval.yaml")
    parser.add_argument("--no_retrieval", action="store_true")
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

        index, metadata, _ = load_index(args.index_dir)
        retriever = Retriever(index, metadata,
                              top_k=ret_cfg["top_k"],
                              threshold=ret_cfg["threshold"])
        logger.info("Loaded FAISS index (%d vectors)", index.ntotal)
    else:
        logger.info("Running WITHOUT retrieval (--no_retrieval)")

    bio_records = read_jsonl(cfg["bio_path"])
    test_records = [r for r in bio_records if r["split"] == "test"]
    explicit_records = [r for r in test_records if not r.get("implicit", False)]
    implicit_records = [r for r in test_records if r.get("implicit", False)]
    logger.info("Test: %d (explicit: %d, implicit: %d)",
                len(test_records), len(explicit_records), len(implicit_records))

    model = RetrievalABSA(
        model_name=cfg["model_name"],
        num_bio_labels=cfg["num_bio_labels"],
        num_sent_labels=cfg["num_sent_labels"],
        lambda_cls=cfg["lambda_cls"],
        dropout=cfg["dropout"],
        cls_class_weights=cfg.get("cls_class_weights"),
    ).to(device)

    test_ds_tmp = RetrievalABSADataset(
        test_records[:1], retriever=retriever,
        tokenizer_name=cfg["model_name"],
        embedding_model=embedding_model,
        max_length=cfg["max_seq_length"],
        query_budget=cfg["query_budget"],
        top_k=ret_cfg["top_k"] if not args.no_retrieval else 0,
        device=device,
    )
    if hasattr(test_ds_tmp, 'tokenizer') and len(test_ds_tmp.tokenizer) > model.encoder.config.vocab_size:
        model.encoder.resize_token_embeddings(len(test_ds_tmp.tokenizer))

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    logger.info("Loaded ABSA checkpoint: %s", args.checkpoint)

    trainer = ABSATrainer(
        model=model, optimizer=None, scheduler=None,
        device=device, log_path="",
    )

    eval_kwargs = dict(
        model=model, trainer=trainer, cfg=cfg, ret_cfg=ret_cfg,
        retriever=retriever, embedding_model=embedding_model,
        device=device, no_retrieval=args.no_retrieval,
    )

    output_parts = []

    results_all = evaluate_subset(test_records, **eval_kwargs)
    table_all = make_table(results_all, "All")
    output_parts.append(table_all)
    print("\n" + table_all)

    if explicit_records:
        results_exp = evaluate_subset(explicit_records, **eval_kwargs)
        table_exp = make_table(results_exp, "Explicit")
        output_parts.append(table_exp)
        print(table_exp)

    if implicit_records:
        results_imp = evaluate_subset(implicit_records, **eval_kwargs)
        table_imp = make_table(results_imp, "Implicit")
        output_parts.append(table_imp)
        print(table_imp)

    out_path = "logs/eval_results.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(output_parts))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
