import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.absa.sentiment_dataset import SentimentDataset
from src.absa.sentiment_model import SentimentPredictor
from src.absa.sentiment_trainer import SentimentTrainer
from src.embedding.model import ContrastiveEmbedder
from src.retrieval.index import load_index
from src.retrieval.retriever import Retriever
from src.utils.io import load_yaml, read_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--stage2_config", required=True)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--no_retrieval", action="store_true")
    parser.add_argument("--embedding_ckpt", default=None)
    parser.add_argument("--index_dir", default="indexes/")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_yaml(args.stage2_config)
    use_retrieval = cfg.get("use_retrieval", True) and not args.no_retrieval
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
            torch.load(args.embedding_ckpt, map_location=device), strict=False)
        embedding_model.to(device)
        embedding_model.eval()

        logger.info("Loading index from %s", args.index_dir)
        index, metadata, store_vectors = load_index(args.index_dir)
        
        ret_cfg = load_yaml(cfg.get("retrieval_config", "configs/retrieval_v2.yaml"))
        retriever = Retriever(index, metadata, top_k=ret_cfg.get("top_k", 3), threshold=ret_cfg.get("threshold", 0.0))
    else:
        logger.info("Running WITHOUT retrieval")

    records = read_jsonl(os.path.join(args.data_dir, "sentiment_records.jsonl"))
    test_records = [r for r in records if r.get("split") == args.split]
    
    if not test_records:
        logger.warning("No records found for split '%s'", args.split)
        return
        
    logger.info("Evaluate %d records from split '%s'", len(test_records), args.split)

    ds_kwargs = dict(
        retriever=retriever,
        tokenizer_name=cfg["model_name"],
        embedding_model=embedding_model,
        store_vectors=None,
        max_length=cfg.get("max_seq_length", 128),
        top_k=ret_cfg.get("top_k", 0) if use_retrieval else 0,
        device=device,
        use_retrieval=use_retrieval,
        joint_training=False,
    )
    
    test_ds = SentimentDataset(test_records, **ds_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.get("batch_size", 32))

    use_learnable_retriever = cfg.get("use_learnable_retriever", False)
    rank_margin = cfg.get("rank_margin", 0.1)
    w_mode = cfg.get("w_mode", "full")
    w_rank = cfg.get("w_rank", 16)
    aux_label_repr_weight = cfg.get("aux_label_repr_weight", 0.0)
    retrieval_dropout = cfg.get("retrieval_dropout", 0.0)
    use_gate = cfg.get("use_gate", False)
    use_two_head = cfg.get("use_two_head", False)
    ret_lambda = cfg.get("ret_lambda", 0.2)
    
    model = SentimentPredictor(
        model_name=cfg["model_name"],
        num_sent_labels=cfg["num_sent_labels"],
        embed_dim=cfg.get("embed_dim", 64),
        tau=cfg.get("tau", 0.05),
        dropout=cfg.get("dropout", 0.1),
        use_retrieval=use_retrieval,
        use_learnable_retriever=use_learnable_retriever,
        class_weights=None,
        margin=rank_margin,
        w_mode=w_mode,
        w_rank=w_rank,
        embedding_model=None,
        aux_label_repr_weight=aux_label_repr_weight,
        retrieval_dropout=retrieval_dropout,
        use_gate=use_gate,
        use_two_head=use_two_head,
        ret_lambda=ret_lambda,
    ).to(device)
    
    logger.info("Loading Stage 2 checkpoint: %s", args.stage2_ckpt)
    model.load_state_dict(torch.load(args.stage2_ckpt, map_location=device), strict=False)
    
    trainer = SentimentTrainer(model=model, optimizer=None, scheduler=None, device=device)
    metrics = trainer.evaluate(test_loader)
    
    print("\n" + "="*60)
    print("=== Gold-Category Sentiment Accuracy (Test Set) ===")
    print(f"Total opinions: {len(test_records)}")
    print(f"Accuracy:       {metrics['sentiment_acc']:.4f} ({metrics['sentiment_acc']*100:.2f}%)")
    print(f"Macro F1:       {metrics['sentiment_macro_f1']:.4f}")
    print(f"Per-polarity:   pos={metrics['f1_positive']:.4f}  neg={metrics['f1_negative']:.4f}  neu={metrics['f1_neutral']:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
