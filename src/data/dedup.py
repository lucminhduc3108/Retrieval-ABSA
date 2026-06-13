import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


def deduplicate_opinions(sentences: list[dict]) -> tuple[list[dict], dict]:
    stats = {"duplicates_removed": 0, "conflicts_resolved": 0, "conflicts_dropped": 0}
    cleaned = []

    for sent in sentences:
        by_cat = defaultdict(list)
        for op in sent["opinions"]:
            by_cat[op["category"]].append(op)

        kept = []
        for cat, ops in by_cat.items():
            polarities = [op["polarity"] for op in ops]
            unique = set(polarities)

            if len(ops) == 1:
                kept.append(ops[0])
            elif len(unique) == 1:
                kept.append(ops[0])
                stats["duplicates_removed"] += len(ops) - 1
            else:
                counts = Counter(polarities)
                top_pol, top_count = counts.most_common(1)[0]
                second_count = counts.most_common(2)[1][1]
                if top_count > second_count:
                    kept.append(next(op for op in ops if op["polarity"] == top_pol))
                    stats["conflicts_resolved"] += 1
                    stats["duplicates_removed"] += len(ops) - 1
                else:
                    stats["conflicts_dropped"] += 1
                    stats["duplicates_removed"] += len(ops)

        cleaned.append({
            "sentence_id": sent["sentence_id"],
            "text": sent["text"],
            "opinions": kept,
        })

    logger.info(
        "Dedup: %d duplicates removed, %d conflicts resolved (majority), %d conflicts dropped (tied)",
        stats["duplicates_removed"], stats["conflicts_resolved"], stats["conflicts_dropped"],
    )
    return cleaned, stats
