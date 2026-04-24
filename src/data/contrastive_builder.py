import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


def build_contrastive_triplets(cls_records: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    by_asp_pol = defaultdict(list)
    by_asp = defaultdict(list)
    by_pol = defaultdict(list)
    for r in cls_records:
        by_asp_pol[(r["aspect_category"], r["polarity"])].append(r)
        by_asp[r["aspect_category"]].append(r)
        by_pol[r["polarity"]].append(r)

    triplets = []
    for anchor in cls_records:
        a_asp, a_pol, a_id = anchor["aspect_category"], anchor["polarity"], anchor["id"]

        pos_candidates = [r for r in by_asp_pol[(a_asp, a_pol)] if r["id"] != a_id]
        if not pos_candidates:
            logger.warning("No positive candidate for anchor %s", a_id)
            continue

        neg1_candidates = [r for r in by_asp[a_asp]
                           if r["polarity"] != a_pol and r["id"] != a_id]
        if not neg1_candidates:
            logger.warning("No hard negative candidate for anchor %s", a_id)
            continue

        neg2_candidates = [r for r in by_pol[a_pol]
                           if r["aspect_category"] != a_asp and r["id"] != a_id]
        if not neg2_candidates:
            logger.warning("No semi-hard negative candidate for anchor %s", a_id)
            continue

        pos = rng.choice(pos_candidates)
        neg1 = rng.choice(neg1_candidates)
        neg2 = rng.choice(neg2_candidates)

        triplets.append({
            "anchor_id": a_id, "anchor_sentence": anchor["sentence"],
            "anchor_aspect": a_asp, "anchor_polarity": a_pol,
            "positive_id": pos["id"], "positive_sentence": pos["sentence"],
            "positive_aspect": pos["aspect_category"], "positive_polarity": pos["polarity"],
            "neg1_id": neg1["id"], "neg1_sentence": neg1["sentence"],
            "neg1_aspect": neg1["aspect_category"], "neg1_polarity": neg1["polarity"],
            "neg2_id": neg2["id"], "neg2_sentence": neg2["sentence"],
            "neg2_aspect": neg2["aspect_category"], "neg2_polarity": neg2["polarity"],
        })

    return triplets
