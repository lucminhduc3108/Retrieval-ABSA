import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.xml_parser import parse_semeval_xml
from src.data.bio_builder import build_bio_records, build_implicit_records
from src.data.cls_builder import build_cls_records
from src.data.contrastive_builder import build_contrastive_triplets
from src.utils.io import write_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SEMEVAL_FILES = {
    "16_train": "SemEval-Dataset/SemEval 2016 Task 5/Restaurant Training/ABSA16_Restaurants_Train_SB1_v2.xml",
    "16_test": "SemEval-Dataset/SemEval 2016 Task 5/Phase B/Gold Annotation/Restaurant/EN_REST_SB1_TEST.xml.gold",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=".", help="Project root containing SemEval-Dataset/")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    all_bio = []
    all_cls = []
    total_sentences = 0
    total_opinions = 0

    for key, rel_path in SEMEVAL_FILES.items():
        split = "train" if "train" in key else "test"
        full_path = os.path.join(args.base_dir, rel_path)
        logger.info("Parsing %s -> split=%s", rel_path, split)

        parsed = parse_semeval_xml(full_path)
        total_sentences += len(parsed)
        total_opinions += sum(len(s["opinions"]) for s in parsed)

        bio = build_bio_records(parsed, split=split)
        implicit = build_implicit_records(parsed, split=split)
        cls = build_cls_records(parsed, split=split)

        all_bio.extend(bio)
        all_bio.extend(implicit)
        all_cls.extend(cls)

    train_bio = [r for r in all_bio if r["split"] == "train"]
    test_bio = [r for r in all_bio if r["split"] == "test"]
    train_cls = [r for r in all_cls if r["split"] == "train"]

    # --- Contrastive triplets (from train only) ---
    triplets = build_contrastive_triplets(train_cls, seed=42)

    # --- Write outputs ---
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(all_bio, os.path.join(args.out_dir, "bio_tagging.jsonl"))
    write_jsonl(all_cls, os.path.join(args.out_dir, "classification.jsonl"))
    write_jsonl(triplets, os.path.join(args.out_dir, "contrastive_triplets.jsonl"))

    # --- Summary ---
    logger.info("=== Data Preparation Summary (SemEval 2016 SB1) ===")
    logger.info("Total sentences parsed: %d", total_sentences)
    logger.info("Total opinions (pre-filter): %d", total_opinions)
    logger.info("Train BIO: %d, Test BIO: %d", len(train_bio), len(test_bio))
    logger.info("Train CLS: %d", len(train_cls))
    logger.info("Explicit BIO (train): %d", len([r for r in train_bio if not r.get("implicit")]))
    logger.info("Implicit BIO (train): %d", len([r for r in train_bio if r.get("implicit")]))
    logger.info("Contrastive triplets: %d", len(triplets))


if __name__ == "__main__":
    main()
