import argparse
import logging
import os
import sys

import numpy as np
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
    parser.add_argument("--no_neg2", action="store_true",
                        help="Build polarity-only hard triplets (no neg2)")
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

    include_neg2 = not args.no_neg2
    triplets = build_hard_negative_triplets(train_records, vectors, seed=args.seed,
                                            include_neg2=include_neg2)
    logger.info("Built %d hard negative triplets (include_neg2=%s)", len(triplets), include_neg2)

    pos_sims = np.array([t["pos_sim"] for t in triplets])
    neg1_sims = np.array([t["neg1_sim"] for t in triplets])
    logger.info("=== Triplet similarity summary ===")
    logger.info("pos_sim  : mean=%.4f std=%.4f min=%.4f max=%.4f",
                pos_sims.mean(), pos_sims.std(), pos_sims.min(), pos_sims.max())
    logger.info("neg1_sim : mean=%.4f std=%.4f min=%.4f max=%.4f",
                neg1_sims.mean(), neg1_sims.std(), neg1_sims.min(), neg1_sims.max())

    violations = int(np.sum(neg1_sims >= pos_sims))
    total_pairs = len(triplets)
    if include_neg2:
        neg2_sims = np.array([t["neg2_sim"] for t in triplets])
        logger.info("neg2_sim : mean=%.4f std=%.4f min=%.4f max=%.4f",
                    neg2_sims.mean(), neg2_sims.std(), neg2_sims.min(), neg2_sims.max())
        violations += int(np.sum(neg2_sims >= pos_sims))
        total_pairs *= 2
    logger.info("Collapsed (neg_sim >= pos_sim): %d / %d (%.1f%%)",
                violations, total_pairs, 100 * violations / total_pairs)

    rng = np.random.default_rng(42)
    n = len(vectors)
    idx = rng.integers(0, n, size=(min(n, 500), 2))
    random_sims = np.array([float(vectors[a] @ vectors[b]) for a, b in idx if a != b])
    logger.info("Random pair sim: mean=%.4f std=%.4f", random_sims.mean(), random_sims.std())
    logger.info("Hard neg1 vs random: +%.4f", neg1_sims.mean() - random_sims.mean())
    if include_neg2:
        logger.info("Hard neg2 vs random: +%.4f", neg2_sims.mean() - random_sims.mean())

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    write_jsonl(triplets, args.out_path)
    logger.info("Saved to %s", args.out_path)


if __name__ == "__main__":
    main()
