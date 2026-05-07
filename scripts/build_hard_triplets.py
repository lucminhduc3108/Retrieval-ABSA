import argparse
import logging
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.contrastive_builder import build_hard_negative_triplets
from src.embedding.model import ContrastiveEmbedder
from src.retrieval.encoder import encode_records
from src.utils.io import load_yaml, read_jsonl, write_jsonl
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_ckpt", required=True)
    parser.add_argument("--cls_path", default="data/processed/classification.jsonl")
    parser.add_argument("--out_path", default="data/processed/hard_contrastive_triplets.jsonl")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ContrastiveEmbedder(model_name=args.model_name, proj_dim=args.proj_dim)
    model.load_state_dict(torch.load(args.embedding_ckpt, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Loaded embedding: %s", args.embedding_ckpt)

    cls_records = read_jsonl(args.cls_path)
    train_records = [r for r in cls_records if r["split"] == "train"]
    logger.info("Train records: %d", len(train_records))

    vectors = encode_records(train_records, model, tokenizer,
                             max_length=args.max_length, device=device)
    logger.info("Encoded %d vectors, shape %s", len(vectors), vectors.shape)

    triplets = build_hard_negative_triplets(train_records, vectors, seed=args.seed)
    logger.info("Built %d hard negative triplets", len(triplets))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    write_jsonl(triplets, args.out_path)
    logger.info("Saved to %s", args.out_path)


if __name__ == "__main__":
    main()
