"""Analyze data duplication between SemEval 2015 and 2016 datasets.

Reports sentence-level and annotation-level overlap. Does NOT modify data.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def read_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def normalize(text: str) -> str:
    return text.strip().lower()


def record_key(r: dict) -> str:
    return json.dumps({
        "sentence": normalize(r["sentence"]),
        "aspect_category": r.get("aspect_category", ""),
        "polarity": r.get("polarity", ""),
        "bio_tags": r.get("bio_tags", []),
    }, sort_keys=True)


def main():
    bio_path = "data/processed/bio_tagging.jsonl"
    if not os.path.exists(bio_path):
        print(f"File not found: {bio_path}")
        return

    records = read_jsonl(bio_path)
    print(f"Total records: {len(records)}")

    train_records = [r for r in records if r["split"] == "train"]
    test_records = [r for r in records if r["split"] == "test"]
    print(f"Train: {len(train_records)}, Test: {len(test_records)}")

    # --- Sentence-level analysis ---
    sent_to_splits = defaultdict(set)
    sent_to_records = defaultdict(list)
    for r in records:
        ns = normalize(r["sentence"])
        sent_to_splits[ns].add(r["split"])
        sent_to_records[ns].append(r)

    unique_sentences = len(sent_to_records)
    dup_sentences = sum(1 for recs in sent_to_records.values() if len(recs) > 1)
    cross_split = sum(1 for splits in sent_to_splits.values() if len(splits) > 1)

    print(f"\n--- Sentence-level ---")
    print(f"Unique sentences: {unique_sentences}")
    print(f"Sentences appearing >1 time: {dup_sentences}")
    print(f"Sentences in BOTH train and test: {cross_split}")

    if cross_split > 0:
        print(f"\n  WARNING: {cross_split} sentences appear in both train and test!")
        print(f"  This means test results may be inflated (data leakage).")

    # --- Annotation-level analysis ---
    full_keys = defaultdict(int)
    for r in records:
        full_keys[record_key(r)] += 1

    exact_dups = sum(1 for c in full_keys.values() if c > 1)
    total_dup_records = sum(c for c in full_keys.values() if c > 1)

    print(f"\n--- Annotation-level (sentence + aspect + polarity + BIO) ---")
    print(f"Unique annotations: {len(full_keys)}")
    print(f"Annotations appearing >1 time: {exact_dups}")
    print(f"Total records in duplicate groups: {total_dup_records}")

    # --- Same sentence, different annotations ---
    same_sent_diff_ann = 0
    for ns, recs in sent_to_records.items():
        if len(recs) > 1:
            keys = set(record_key(r) for r in recs)
            if len(keys) > 1:
                same_sent_diff_ann += 1

    print(f"\n--- Same sentence, different annotations ---")
    print(f"Sentences with multiple distinct annotations: {same_sent_diff_ann}")

    # --- Train/test overlap detail ---
    train_sents = set(normalize(r["sentence"]) for r in train_records)
    test_sents = set(normalize(r["sentence"]) for r in test_records)
    overlap_sents = train_sents & test_sents

    print(f"\n--- Train/Test overlap ---")
    print(f"Train unique sentences: {len(train_sents)}")
    print(f"Test unique sentences: {len(test_sents)}")
    print(f"Overlapping sentences: {len(overlap_sents)}")
    if test_sents:
        print(f"Test leakage rate: {len(overlap_sents)/len(test_sents)*100:.1f}%")

    train_keys = set()
    test_keys = set()
    for r in train_records:
        train_keys.add(record_key(r))
    for r in test_records:
        test_keys.add(record_key(r))
    overlap_keys = train_keys & test_keys
    print(f"Exact annotation overlap (train & test): {len(overlap_keys)}")

    # --- Polarity distribution per split ---
    print(f"\n--- Polarity distribution ---")
    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        pol_counts = defaultdict(int)
        for r in split_records:
            pol_counts[r.get("polarity", "unknown")] += 1
        total = len(split_records)
        print(f"  {split_name}:")
        for pol in sorted(pol_counts.keys()):
            pct = pol_counts[pol] / total * 100
            print(f"    {pol}: {pol_counts[pol]} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
