import argparse
import logging
import os
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embedding.model import ContrastiveEmbedder
from src.retrieval.encoder import encode_records
from src.retrieval.index import build_index, save_index
from src.utils.io import read_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_ckpt", required=True)
    parser.add_argument("--input", default="data/processed/classification.jsonl")
    parser.add_argument("--bio_input", default="data/processed/bio_tagging.jsonl")
    parser.add_argument("--out_dir", default="indexes/")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ContrastiveEmbedder(model_name=args.model_name, proj_dim=args.proj_dim)
    model.load_state_dict(torch.load(args.embedding_ckpt, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Loaded embedding checkpoint: %s", args.embedding_ckpt)

    cls_records = [r for r in read_jsonl(args.input) if r.get("split") == "train"]
    logger.info("Classification records (train): %d", len(cls_records))

    bio_map = {}
    if os.path.exists(args.bio_input):
        for r in read_jsonl(args.bio_input):
            bio_map[r["id"]] = {"tokens": r["tokens"], "bio_tags": r["bio_tags"]}

    metadata = []
    for r in cls_records:
        bio_info = bio_map.get(r["id"], {})
        metadata.append({
            "id": r["id"],
            "sentence": r["sentence"],
            "aspect_category": r["aspect_category"],
            "polarity": r["polarity"],
            "tokens": bio_info.get("tokens"),
            "bio_tags": bio_info.get("bio_tags"),
        })

    t0 = time.time()
    vectors = encode_records(cls_records, model, tokenizer,
                             max_length=args.max_seq_length,
                             batch_size=args.batch_size, device=device)
    elapsed = time.time() - t0
    logger.info("Encoded %d vectors in %.1fs", len(vectors), elapsed)

    index = build_index(vectors)
    save_index(index, metadata, vectors, args.out_dir)

    faiss_path = os.path.join(args.out_dir, "train.faiss")
    npy_path = os.path.join(args.out_dir, "train_vectors.npy")
    logger.info("Saved index: %s (%.1f MB)", faiss_path,
                os.path.getsize(faiss_path) / 1e6)
    logger.info("Saved vectors: %s (%.1f MB)", npy_path,
                os.path.getsize(npy_path) / 1e6)
    logger.info("Metadata records: %d", len(metadata))
    logger.info("Done.")


if __name__ == "__main__":
    main()
