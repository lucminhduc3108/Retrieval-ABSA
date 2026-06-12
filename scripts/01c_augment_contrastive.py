import argparse
import json
import logging
import os
import random
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.contrastive_builder import build_contrastive_triplets
from src.data.mams_mapping import SAFE_MAP, MAMS_XML
from src.utils.io import read_jsonl, write_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_mams_neutral(mams_xml: str) -> list[dict]:
    tree = ET.parse(mams_xml)
    root = tree.getroot()
    candidates = []
    for sentence in root.findall("sentence"):
        text_node = sentence.find("text")
        if text_node is None or text_node.text is None:
            continue
        text = text_node.text.strip()
        aspects_node = sentence.find("aspectCategories")
        if aspects_node is None:
            continue
        for aspect in aspects_node:
            cat = aspect.get("category", "").lower()
            pol = aspect.get("polarity", "")
            if pol == "neutral" and cat in SAFE_MAP:
                candidates.append({
                    "sentence": text,
                    "aspect_category": SAFE_MAP[cat],
                    "polarity": "neutral",
                })
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Add MAMS neutral samples to classification records and rebuild contrastive triplets")
    parser.add_argument("--target", type=int, default=300,
                        help="Number of MAMS neutral samples to add")
    parser.add_argument("--cls_path", default="data/processed/classification.jsonl")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    candidates = extract_mams_neutral(MAMS_XML)
    logger.info("MAMS neutral candidates: %d", len(candidates))

    if len(candidates) < args.target:
        logger.warning("Only %d candidates available (requested %d)", len(candidates), args.target)
        sampled = candidates
    else:
        sampled = random.sample(candidates, args.target)
    logger.info("Sampled %d MAMS neutral records", len(sampled))

    cls_records = read_jsonl(args.cls_path)
    train_records = [r for r in cls_records if r["split"] == "train"]
    test_records = [r for r in cls_records if r["split"] == "test"]
    logger.info("Original train: %d, test: %d", len(train_records), len(test_records))

    for i, rec in enumerate(sampled, start=1):
        rec["id"] = f"mams_cls_{i:04d}"
        rec["split"] = "train"
        rec["source"] = "mams"
        train_records.append(rec)

    logger.info("Augmented train: %d (added %d neutral)", len(train_records), len(sampled))

    triplets = build_contrastive_triplets(train_records, seed=args.seed)
    logger.info("Built %d contrastive triplets", len(triplets))

    from collections import Counter
    pol_counts = Counter(t["anchor_polarity"] for t in triplets)
    logger.info("Triplet anchor polarity: %s", dict(pol_counts))

    os.makedirs(args.out_dir, exist_ok=True)
    aug_cls_path = os.path.join(args.out_dir, "classification_aug.jsonl")
    write_jsonl(train_records + test_records, aug_cls_path)
    logger.info("Saved %d records to %s", len(train_records) + len(test_records), aug_cls_path)

    aug_triplets_path = os.path.join(args.out_dir, "contrastive_triplets_aug.jsonl")
    write_jsonl(triplets, aug_triplets_path)
    logger.info("Saved %d triplets to %s", len(triplets), aug_triplets_path)


if __name__ == "__main__":
    main()
