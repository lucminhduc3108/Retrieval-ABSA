"""Analyze impact of reducing categories: merge vs drop scenarios."""
import json
from collections import Counter

recs = [json.loads(l) for l in open("data/processed/category_detection.jsonl", encoding="utf-8")]
train = [r for r in recs if r["split"] == "train"]
test = [r for r in recs if r["split"] == "test"]

# Current 12 categories grouped by entity
GROUPS = {
    "AMBIENCE": ["AMBIENCE#GENERAL"],
    "DRINKS":   ["DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE_OPTIONS"],
    "FOOD":     ["FOOD#PRICES", "FOOD#QUALITY", "FOOD#STYLE_OPTIONS"],
    "LOCATION": ["LOCATION#GENERAL"],
    "RESTAURANT": ["RESTAURANT#GENERAL", "RESTAURANT#MISCELLANEOUS", "RESTAURANT#PRICES"],
    "SERVICE":  ["SERVICE#GENERAL"],
}

def count_cat(records, cat):
    return sum(1 for r in records if cat in r["categories"])

def count_group(records, group_cats):
    return sum(1 for r in records if any(c in r["categories"] for c in group_cats))

print("=" * 80)
print("SCENARIO 0: Current 12 categories")
print("=" * 80)
for group, cats in sorted(GROUPS.items()):
    for cat in cats:
        tr = count_cat(train, cat)
        te = count_cat(test, cat)
        print(f"  {cat:<30} train={tr:>4}  test={te:>4}")
    if len(cats) > 1:
        tr = count_group(train, cats)
        te = count_group(test, cats)
        print(f"  {'=> merged ' + group:<30} train={tr:>4}  test={te:>4}")
    print()

# Scenario 1: Merge to 6 coarse categories (entity-level)
print("=" * 80)
print("SCENARIO 1: Merge to 6 coarse categories (entity-level)")
print("  AMBIENCE, DRINKS, FOOD, LOCATION, RESTAURANT, SERVICE")
print("=" * 80)
coarse_counts_tr = {}
coarse_counts_te = {}
for group, cats in sorted(GROUPS.items()):
    coarse_counts_tr[group] = count_group(train, cats)
    coarse_counts_te[group] = count_group(test, cats)
    print(f"  {group:<20} train={coarse_counts_tr[group]:>4} ({coarse_counts_tr[group]/len(train)*100:>5.1f}%)  test={coarse_counts_te[group]:>4}")

max_c = max(coarse_counts_tr.values())
min_c = min(coarse_counts_tr.values())
print(f"\n  Imbalance ratio: {max_c/min_c:.1f}x (was 34.0x)")
print(f"  Min category: {min(coarse_counts_tr, key=coarse_counts_tr.get)} = {min_c}")

# Scenario 2: Drop 4 rare DRINKS+LOCATION, keep 8
print()
print("=" * 80)
print("SCENARIO 2: Drop 4 rare categories (DRINKS#*, LOCATION#GENERAL), keep 8")
print("=" * 80)
DROP = {"DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE_OPTIONS", "LOCATION#GENERAL"}
KEEP8 = [c for g in GROUPS.values() for c in g if c not in DROP]
lost_train = sum(1 for r in train if all(c in DROP for c in r["categories"]))
lost_test = sum(1 for r in test if all(c in DROP for c in r["categories"]))
lost_opinions_tr = sum(1 for r in train for c in r["categories"] if c in DROP)
lost_opinions_te = sum(1 for r in test for c in r["categories"] if c in DROP)
print(f"  Dropped categories: {sorted(DROP)}")
print(f"  Sentences lost (all cats dropped): train={lost_train}, test={lost_test}")
print(f"  Opinions lost: train={lost_opinions_tr}, test={lost_opinions_te}")
print(f"  Remaining categories ({len(KEEP8)}):")
for cat in sorted(KEEP8):
    tr = count_cat(train, cat)
    te = count_cat(test, cat)
    print(f"    {cat:<30} train={tr:>4}  test={te:>4}")
counts8 = [count_cat(train, c) for c in KEEP8]
print(f"\n  Imbalance ratio: {max(counts8)/min(counts8):.1f}x (was 34.0x)")

# Scenario 3: Merge DRINKS→FOOD (since both are consumables), drop LOCATION, keep 9→8
print()
print("=" * 80)
print("SCENARIO 3: Merge DRINKS#* into FOOD#* equivalents, drop LOCATION, keep 8")
print("  DRINKS#PRICES=>FOOD#PRICES, DRINKS#QUALITY=>FOOD#QUALITY, DRINKS#STYLE=>FOOD#STYLE")
print("=" * 80)
MERGE_MAP = {
    "DRINKS#PRICES": "FOOD#PRICES",
    "DRINKS#QUALITY": "FOOD#QUALITY",
    "DRINKS#STYLE_OPTIONS": "FOOD#STYLE_OPTIONS",
}
merged_cats_8 = [c for c in sorted(set(c for g in GROUPS.values() for c in g)) 
                 if c not in MERGE_MAP and c != "LOCATION#GENERAL"]

def merged_count(records, target_cat):
    sources = [k for k, v in MERGE_MAP.items() if v == target_cat]
    return sum(1 for r in records if target_cat in r["categories"] or any(s in r["categories"] for s in sources))

for cat in sorted(merged_cats_8):
    if cat in MERGE_MAP.values():
        tr = merged_count(train, cat)
        te = merged_count(test, cat)
        sources = [k for k,v in MERGE_MAP.items() if v == cat]
        print(f"  {cat:<30} train={tr:>4}  test={te:>4}  (includes {', '.join(sources)})")
    else:
        tr = count_cat(train, cat)
        te = count_cat(test, cat)
        print(f"  {cat:<30} train={tr:>4}  test={te:>4}")

counts_m = []
for cat in merged_cats_8:
    if cat in MERGE_MAP.values():
        counts_m.append(merged_count(train, cat))
    else:
        counts_m.append(count_cat(train, cat))
print(f"\n  Imbalance ratio: {max(counts_m)/min(counts_m):.1f}x (was 34.0x)")
lost_loc_tr = sum(1 for r in train if all(c == "LOCATION#GENERAL" for c in r["categories"]))
lost_loc_te = sum(1 for r in test if all(c == "LOCATION#GENERAL" for c in r["categories"]))
print(f"  Sentences lost (LOCATION only): train={lost_loc_tr}, test={lost_loc_te}")

# Co-occurrence analysis
print()
print("=" * 80)
print("CO-OCCURRENCE: How often do rare categories appear alone vs with common ones?")
print("=" * 80)
rare = {"DRINKS#PRICES", "DRINKS#QUALITY", "DRINKS#STYLE_OPTIONS", "LOCATION#GENERAL"}
for cat in sorted(rare):
    total = count_cat(train, cat)
    alone = sum(1 for r in train if cat in r["categories"] and len(r["categories"]) == 1)
    with_others = total - alone
    print(f"  {cat:<30} total={total:>3}, alone={alone:>3} ({alone/total*100:.0f}%), with_others={with_others:>3} ({with_others/total*100:.0f}%)")

# Impact on Stage 2
print()
print("=" * 80)
print("IMPACT ON STAGE 2 (Sentiment)")
print("=" * 80)
sent_recs = [json.loads(l) for l in open("data/processed/sentiment_records.jsonl", encoding="utf-8")]
sent_train = [r for r in sent_recs if r["split"] == "train"]
sent_test = [r for r in sent_recs if r["split"] == "test"]
print(f"Total sentiment records: train={len(sent_train)}, test={len(sent_test)}")
for cat in sorted(rare):
    tr_s = sum(1 for r in sent_train if r["category"] == cat)
    te_s = sum(1 for r in sent_test if r["category"] == cat)
    print(f"  {cat:<30} sentiment train={tr_s:>3}, test={te_s:>3}")
print(f"  Total rare cat sentiment: train={sum(1 for r in sent_train if r['category'] in rare)}, test={sum(1 for r in sent_test if r['category'] in rare)}")
print(f"  % of total: train={sum(1 for r in sent_train if r['category'] in rare)/len(sent_train)*100:.1f}%, test={sum(1 for r in sent_test if r['category'] in rare)/len(sent_test)*100:.1f}%")
