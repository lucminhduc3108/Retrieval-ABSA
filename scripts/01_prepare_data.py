import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.xml_parser import parse_semeval2014_xml
from src.data.dedup import deduplicate_opinions
from src.data.cls_builder import build_cls_records
from src.data.contrastive_builder import build_contrastive_triplets
from src.data.category_builder import build_category_records, build_sentiment_records
from src.utils.io import write_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEMEVAL_FILES = {
    "14_train": "SemEval-2014/Restaurants_Train.xml",
    "14_test": "SemEval-2014/Restaurants_Test_Gold.xml",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".", help="Project root containing SemEval-2014/")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    all_cls = []
    all_category = []
    all_sentiment = []
    total_sentences = 0
    total_opinions = 0

    for key, rel_path in SEMEVAL_FILES.items():
        split = "train" if "train" in key else "test"
        full_path = os.path.join(args.base_dir, rel_path)
        logger.info("Parsing %s -> split=%s", rel_path, split)

        parsed = parse_semeval2014_xml(full_path)
        total_sentences += len(parsed)
        total_opinions += sum(len(s["opinions"]) for s in parsed)

        parsed, dedup_stats = deduplicate_opinions(parsed)
        logger.info("  %s dedup: %d dups removed, %d conflicts resolved, %d conflicts dropped",
                     split, dedup_stats["duplicates_removed"],
                     dedup_stats["conflicts_resolved"], dedup_stats["conflicts_dropped"])

        cls = build_cls_records(parsed, split=split)
        all_cls.extend(cls)

        cat_records = build_category_records(parsed, split=split)
        sent_records = build_sentiment_records(parsed, split=split)
        all_category.extend(cat_records)
        all_sentiment.extend(sent_records)

    train_cls = [r for r in all_cls if r["split"] == "train"]

    # --- Contrastive triplets (from train only) ---
    triplets = build_contrastive_triplets(train_cls, seed=42)

    # --- Write outputs ---
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(all_cls, os.path.join(args.out_dir, "classification.jsonl"))
    write_jsonl(triplets, os.path.join(args.out_dir, "contrastive_triplets.jsonl"))
    write_jsonl(all_category, os.path.join(args.out_dir, "category_detection.jsonl"))
    write_jsonl(all_sentiment, os.path.join(args.out_dir, "sentiment_records.jsonl"))

    # --- Summary ---
    logger.info("=== Data Preparation Summary (SemEval 2014 Task 4 — Restaurant) ===")
    logger.info("Total sentences parsed: %d", total_sentences)
    logger.info("Total opinions (pre-filter): %d", total_opinions)
    logger.info("Train CLS: %d", len(train_cls))
    logger.info("Contrastive triplets: %d", len(triplets))
    logger.info("Category detection records: %d (train: %d, test: %d)",
                len(all_category),
                len([r for r in all_category if r["split"] == "train"]),
                len([r for r in all_category if r["split"] == "test"]))
    logger.info("Sentiment records: %d (train: %d, test: %d)",
                len(all_sentiment),
                len([r for r in all_sentiment if r["split"] == "train"]),
                len([r for r in all_sentiment if r["split"] == "test"]))


if __name__ == "__main__":
    main()
