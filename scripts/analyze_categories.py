"""Analyze category distribution in category_detection.jsonl"""
import json
import math
from collections import Counter

recs = [json.loads(l) for l in open("data/processed/category_detection.jsonl", encoding="utf-8")]
train = [r for r in recs if r["split"] == "train"]
test = [r for r in recs if r["split"] == "test"]

cats = sorted(set(c for r in train + test for c in r["categories"]))

print(f"Train sentences: {len(train)}, Test sentences: {len(test)}")
print()

# Category distribution
print("=== Category Distribution (sentence-level) ===")
print(f"{'Category':<30} {'Train':>6} {'Train%':>7} {'Test':>6} {'Test%':>7}  {'pos_weight':>10}  {'ratio(max/this)':>15}")
print("-" * 100)

train_counts = {}
test_counts = {}
max_train = 0

for cat in cats:
    c_tr = sum(1 for r in train if cat in r["categories"])
    c_te = sum(1 for r in test if cat in r["categories"])
    train_counts[cat] = c_tr
    test_counts[cat] = c_te
    if c_tr > max_train:
        max_train = c_tr

for cat in cats:
    c_tr = train_counts[cat]
    c_te = test_counts[cat]
    n = len(train)
    pw = min(math.sqrt((n - c_tr) / c_tr), 5.0) if c_tr > 0 else 1.0
    ratio = max_train / c_tr if c_tr > 0 else float('inf')
    capped = " (CAPPED)" if pw >= 5.0 else ""
    print(f"{cat:<30} {c_tr:>6} {c_tr/n*100:>6.1f}% {c_te:>6} {c_te/n*100:>6.1f}%  pw={pw:>6.2f}{capped}  {ratio:>10.1f}x")

print("-" * 100)
total_tr_ops = sum(sum(r["category_vector"]) for r in train)
total_te_ops = sum(sum(r["category_vector"]) for r in test)
print(f"Total category occurrences: train={total_tr_ops:.0f}, test={total_te_ops:.0f}")
print(f"Avg categories per sentence: train={total_tr_ops/len(train):.2f}, test={total_te_ops/len(test):.2f}")

# Multi-label distribution
print()
print("=== Multi-label Distribution ===")
tr_nlabels = Counter(sum(r["category_vector"]) for r in train)
te_nlabels = Counter(sum(r["category_vector"]) for r in test)
print(f"{'#categories':>12} {'Train':>8} {'Train%':>8} {'Test':>8} {'Test%':>8}")
for k in sorted(set(list(tr_nlabels.keys()) + list(te_nlabels.keys()))):
    tr_c = tr_nlabels.get(k, 0)
    te_c = te_nlabels.get(k, 0)
    print(f"{int(k):>12} {tr_c:>8} {tr_c/len(train)*100:>7.1f}% {te_c:>8} {te_c/len(test)*100:>7.1f}%")

# Imbalance summary
print()
print("=== Imbalance Summary ===")
sorted_cats = sorted(cats, key=lambda c: train_counts[c], reverse=True)
print(f"Most frequent:  {sorted_cats[0]} ({train_counts[sorted_cats[0]]})")
print(f"Least frequent: {sorted_cats[-1]} ({train_counts[sorted_cats[-1]]})")
print(f"Imbalance ratio: {max_train / train_counts[sorted_cats[-1]]:.1f}x")
print(f"pos_weight range: {min(math.sqrt((len(train)-train_counts[c])/train_counts[c]) for c in cats if train_counts[c]>0):.2f} - {max(min(math.sqrt((len(train)-train_counts[c])/train_counts[c]),5.0) for c in cats if train_counts[c]>0):.2f}")
