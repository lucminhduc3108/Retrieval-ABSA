import logging
import random
from collections import defaultdict

import numpy as np

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


def build_hard_negative_triplets(cls_records: list[dict],
                                  vectors: np.ndarray,
                                  seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    sim = vectors @ vectors.T

    by_asp_pol = defaultdict(list)
    by_asp = defaultdict(list)
    by_pol = defaultdict(list)
    for i, r in enumerate(cls_records):
        by_asp_pol[(r["aspect_category"], r["polarity"])].append(i)
        by_asp[r["aspect_category"]].append(i)
        by_pol[r["polarity"]].append(i)

    triplets = []
    for i, anchor in enumerate(cls_records):
        a_asp, a_pol, a_id = anchor["aspect_category"], anchor["polarity"], anchor["id"]

        pos_indices = [j for j in by_asp_pol[(a_asp, a_pol)] if j != i]
        if not pos_indices:
            continue

        neg1_indices = [j for j in by_asp[a_asp]
                        if cls_records[j]["polarity"] != a_pol and j != i]
        if not neg1_indices:
            continue

        neg2_indices = [j for j in by_pol[a_pol]
                        if cls_records[j]["aspect_category"] != a_asp and j != i]
        if not neg2_indices:
            continue

        pos_idx = rng.choice(pos_indices)

        neg1_sims = sim[i, neg1_indices]
        neg1_idx = neg1_indices[int(np.argmax(neg1_sims))]

        neg2_sims = sim[i, neg2_indices]
        neg2_idx = neg2_indices[int(np.argmax(neg2_sims))]

        pos = cls_records[pos_idx]
        neg1 = cls_records[neg1_idx]
        neg2 = cls_records[neg2_idx]

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
