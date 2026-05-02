import argparse
import json
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
    "15_train": "SemEval-Dataset/SemEval 2015 Task 12/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml",
    "15_test": "SemEval-Dataset/SemEval 2015 Task 12/Gold Annotation/ABSA15_Restaurants_Test.xml",
    "16_train": "SemEval-Dataset/SemEval 2016 Task 5/Restaurant Training/ABSA16_Restaurants_Train_SB1_v2.xml",
    "16_test": "SemEval-Dataset/SemEval 2016 Task 5/Phase B/Gold Annotation/Restaurant/EN_REST_SB1_TEST.xml.gold",
}


def _annotation_key(r: dict) -> str:
    """Unique key for deduplication: normalized sentence + aspect + polarity + bio_tags."""
    return json.dumps({
        "sentence": r["sentence"].strip().lower(),
        "aspect_category": r.get("aspect_category", ""),
        "polarity": r.get("polarity", ""),
        "bio_tags": r.get("bio_tags", []),
    }, sort_keys=True)


def _cls_key(r: dict) -> str:
    """Unique key for cls records: normalized sentence + aspect + polarity."""
    return json.dumps({
        "sentence": r["sentence"].strip().lower(),
        "aspect_category": r["aspect_category"],
        "polarity": r["polarity"],
    }, sort_keys=True)


def _dedup_train(train_records: list[dict], test_records: list[dict],
                 key_fn) -> list[dict]:
    """Remove duplicates within train and remove train records that match test."""
    test_keys = set(key_fn(r) for r in test_records)

    seen = set()
    deduped = []
    removed_internal = 0
    removed_leakage = 0

    for r in train_records:
        k = key_fn(r)
        if k in test_keys:
            removed_leakage += 1
            continue
        if k in seen:
            removed_internal += 1
            continue
        seen.add(k)
        deduped.append(r)

    logger.info("  Dedup: %d internal duplicates removed, %d leakage matches removed",
                removed_internal, removed_leakage)
    return deduped


def _mark_clean_test(test_records: list[dict], train_records: list[dict]) -> list[str]:
    """Return IDs of test records whose sentences don't appear in train."""
    train_sents = set(r["sentence"].strip().lower() for r in train_records)
    clean_ids = []
    for r in test_records:
        if r["sentence"].strip().lower() not in train_sents:
            clean_ids.append(r["id"])
    return clean_ids


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

    # --- Deduplication ---
    train_bio = [r for r in all_bio if r["split"] == "train"]
    test_bio = [r for r in all_bio if r["split"] == "test"]
    train_cls = [r for r in all_cls if r["split"] == "train"]
    test_cls = [r for r in all_cls if r["split"] == "test"]

    logger.info("Before dedup: train_bio=%d, train_cls=%d", len(train_bio), len(train_cls))

    logger.info("Deduplicating BIO records...")
    train_bio = _dedup_train(train_bio, test_bio, _annotation_key)
    logger.info("Deduplicating CLS records...")
    train_cls = _dedup_train(train_cls, test_cls, _cls_key)

    logger.info("After dedup: train_bio=%d, train_cls=%d", len(train_bio), len(train_cls))

    all_bio = train_bio + test_bio
    all_cls = train_cls + test_cls

    # --- Clean test IDs (sentences not seen in train) ---
    clean_test_ids = _mark_clean_test(test_bio, train_bio)
    logger.info("Clean test records (unseen sentences): %d / %d",
                len(clean_test_ids), len(test_bio))

    # --- Contrastive triplets (from deduped train only) ---
    triplets = build_contrastive_triplets(train_cls, seed=42)

    # --- Write outputs ---
    write_jsonl(all_bio, os.path.join(args.out_dir, "bio_tagging.jsonl"))
    write_jsonl(all_cls, os.path.join(args.out_dir, "classification.jsonl"))
    write_jsonl(triplets, os.path.join(args.out_dir, "contrastive_triplets.jsonl"))

    clean_test_path = os.path.join(args.out_dir, "clean_test_ids.json")
    with open(clean_test_path, "w") as f:
        json.dump(clean_test_ids, f)

    # --- Summary ---
    explicit_bio = [r for r in all_bio if not r.get("implicit", False)]
    implicit_bio = [r for r in all_bio if r.get("implicit", False)]
    train_final = [r for r in all_bio if r["split"] == "train"]
    test_final = [r for r in all_bio if r["split"] == "test"]

    logger.info("=== Data Preparation Summary ===")
    logger.info("Total sentences parsed: %d", total_sentences)
    logger.info("Total opinions (pre-filter): %d", total_opinions)
    logger.info("--- After dedup ---")
    logger.info("Train records: %d (bio), %d (cls)", len(train_bio), len(train_cls))
    logger.info("Test records: %d (unchanged)", len(test_bio))
    logger.info("Explicit BIO: %d", len([r for r in train_final if not r.get("implicit")]))
    logger.info("Implicit BIO: %d", len([r for r in train_final if r.get("implicit")]))
    logger.info("Contrastive triplets: %d", len(triplets))
    logger.info("Clean test IDs: %d / %d (%.1f%%)",
                len(clean_test_ids), len(test_bio),
                len(clean_test_ids) / len(test_bio) * 100 if test_bio else 0)


if __name__ == "__main__":
    main()
