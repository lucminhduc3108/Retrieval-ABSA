import argparse
import json
import logging
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.mams_mapping import SAFE_MAP, MAMS_XML

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ORIGINAL_JSONL = "data/processed/sentiment_records.jsonl"


def augment_neutral(target_samples: int, out_path: str, seed: int = 42):
    random.seed(seed)

    logger.info("Loading MAMS ACSA Train XML...")
    tree = ET.parse(MAMS_XML)
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
                    "category": SAFE_MAP[cat],
                    "polarity": "neutral",
                    "split": "train",
                    "source": "mams",
                })

    logger.info("Found %d valid neutral candidates in MAMS.", len(candidates))

    if len(candidates) < target_samples:
        logger.warning("Only %d candidates (requested %d)", len(candidates), target_samples)
        sampled = candidates
    else:
        sampled = random.sample(candidates, target_samples)
    logger.info("Sampled %d records.", len(sampled))

    records = []
    with open(ORIGINAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info("Loaded %d original SemEval records.", len(records))

    for i, rec in enumerate(sampled, start=1):
        rec["id"] = f"mams_neu_{i:04d}"
        records.append(rec)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Saved %d records (+%d MAMS) to %s", len(records), len(sampled), out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Augment sentiment records with MAMS neutral samples")
    parser.add_argument("--target", type=int, default=300,
                        help="Number of MAMS neutral samples to add")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = f"data/processed/sentiment_records_aug_{args.target}.jsonl"
    augment_neutral(args.target, out_path, seed=args.seed)


if __name__ == "__main__":
    main()
