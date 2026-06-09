"""Calculate pos_weight under different strategies for the plan."""
import json
import math
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

recs = [json.loads(l) for l in open("data/processed/category_detection.jsonl", encoding="utf-8")]
train = [r for r in recs if r["split"] == "train"]

# Simulate 80/20 split
n_train = int(len(train) * 0.8)  # ~1366
n = n_train

from src.data.category_builder import CATEGORY_LIST, NUM_CATEGORIES

# Count in full train (approximate for split)
counts = [0] * NUM_CATEGORIES
for r in train:
    for i, v in enumerate(r["category_vector"]):
        counts[i] += v

# Scale to 80% split
counts_80 = [int(c * 0.8) for c in counts]

print(f"Train sentences (80% split): ~{n_train}")
print()

strategies = {
    "Current (sqrt, cap=5.0)": lambda n, c: min(math.sqrt((n-c)/c), 5.0) if c > 0 else 1.0,
    "A: sqrt, NO cap":         lambda n, c: math.sqrt((n-c)/c) if c > 0 else 1.0,
    "B: sqrt, cap=3.0":        lambda n, c: min(math.sqrt((n-c)/c), 3.0) if c > 0 else 1.0,
    "C: log, no cap":          lambda n, c: math.log1p((n-c)/c) if c > 0 else 1.0,
    "D: no pos_weight":        lambda n, c: 1.0,
}

for name, fn in strategies.items():
    print(f"=== {name} ===")
    weights = []
    for i, cat in enumerate(CATEGORY_LIST):
        c = counts_80[i]
        w = fn(n_train, c)
        weights.append(w)
        capped = " (was CAPPED)" if name == "Current (sqrt, cap=5.0)" and w >= 5.0 else ""
        print(f"  {cat:<30} count={c:>4}  pw={w:>6.2f}{capped}")
    print(f"  Range: [{min(weights):.2f}, {max(weights):.2f}]")
    print(f"  Max/Min ratio: {max(weights)/min(weights):.1f}x")
    print()
