import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.xml_parser import parse_mams_xml
from src.data.dedup import deduplicate_opinions
from src.data.cls_builder import build_cls_records
from src.data.contrastive_builder import build_contrastive_triplets
from src.data.category_builder import build_category_records, build_sentiment_records
from src.data.mams_mapping import MAMS_TRAIN_XML, MAMS_VAL_XML, MAMS_TEST_XML
from src.utils.io import write_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".", help="Project root containing data/mams/")
    parser.add_argument("--out_dir", default="data/processed_mams")
    args = parser.parse_args()

    all_cls = []
    all_category = []
    all_sentiment = []

    for mams_path, split in [(MAMS_TRAIN_XML, "train"),
                              (MAMS_VAL_XML, "train"),
                              (MAMS_TEST_XML, "test")]:
        full_path = os.path.join(args.base_dir, mams_path)
        if not os.path.exists(full_path):
            logger.error("MAMS file not found: %s", full_path)
            sys.exit(1)

        parsed = parse_mams_xml(full_path)
        parsed, stats = deduplicate_opinions(parsed)
        logger.info("MAMS %s (split=%s): %d sentences, %d opinions "
                     "(dedup: %d dup removed, %d conflicts dropped)",
                     os.path.basename(mams_path), split, len(parsed),
                     sum(len(s["opinions"]) for s in parsed),
                     stats["duplicates_removed"], stats["conflicts_dropped"])

        cls = build_cls_records(parsed, split=split)
        all_cls.extend(cls)

        cat_records = build_category_records(parsed, split=split)
        sent_records = build_sentiment_records(parsed, split=split)
        all_category.extend(cat_records)
        all_sentiment.extend(sent_records)

    train_cls = [r for r in all_cls if r["split"] == "train"]
    test_cls = [r for r in all_cls if r["split"] == "test"]

    triplets = build_contrastive_triplets(train_cls, seed=42, include_neg2=False)

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(all_cls, os.path.join(args.out_dir, "classification.jsonl"))
    write_jsonl(triplets, os.path.join(args.out_dir, "contrastive_triplets_polonly.jsonl"))
    write_jsonl(all_category, os.path.join(args.out_dir, "category_detection.jsonl"))
    write_jsonl(all_sentiment, os.path.join(args.out_dir, "sentiment_records.jsonl"))

    logger.info("=== MAMS Standalone Data Preparation ===")
    logger.info("Train CLS: %d (from train.xml + val.xml)", len(train_cls))
    logger.info("Test CLS: %d (from test.xml)", len(test_cls))
    logger.info("Contrastive triplets (polonly): %d", len(triplets))
    logger.info("Category detection: %d (train: %d, test: %d)",
                len(all_category),
                len([r for r in all_category if r["split"] == "train"]),
                len([r for r in all_category if r["split"] == "test"]))
    logger.info("Sentiment records: %d (train: %d, test: %d)",
                len(all_sentiment),
                len([r for r in all_sentiment if r["split"] == "train"]),
                len([r for r in all_sentiment if r["split"] == "test"]))
    logger.info("Output dir: %s", args.out_dir)


if __name__ == "__main__":
    main()
