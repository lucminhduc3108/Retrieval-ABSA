from src.data.contrastive_builder import build_contrastive_triplets


def _mk(i, asp, pol):
    return {"id": f"r{i}", "sentence": f"s{i}", "aspect_category": asp,
            "polarity": pol, "split": "train"}


def test_positive_shares_aspect_and_polarity():
    recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
            _mk(2, "FOOD#QUALITY", "negative"), _mk(3, "SERVICE#GENERAL", "positive")]
    triplets = build_contrastive_triplets(recs, seed=0)
    for t in triplets:
        assert t["anchor_aspect"] == t["positive_aspect"]
        assert t["anchor_polarity"] == t["positive_polarity"]
        assert t["anchor_id"] != t["positive_id"]


def test_hard_neg_same_aspect_different_polarity():
    recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
            _mk(2, "FOOD#QUALITY", "negative"), _mk(3, "SERVICE#GENERAL", "positive")]
    triplets = build_contrastive_triplets(recs, seed=0)
    for t in triplets:
        assert t["anchor_aspect"] == t["neg1_aspect"]
        assert t["anchor_polarity"] != t["neg1_polarity"]


def test_semi_hard_neg_different_aspect_same_polarity():
    recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
            _mk(2, "FOOD#QUALITY", "negative"), _mk(3, "SERVICE#GENERAL", "positive")]
    triplets = build_contrastive_triplets(recs, seed=0)
    for t in triplets:
        assert t["anchor_aspect"] != t["neg2_aspect"]
        assert t["anchor_polarity"] == t["neg2_polarity"]


def test_skip_when_no_positive_candidate():
    recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "negative")]
    assert build_contrastive_triplets(recs, seed=0) == []


def test_skip_when_no_hard_neg_candidate():
    recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
            _mk(2, "SERVICE#GENERAL", "positive")]
    assert build_contrastive_triplets(recs, seed=0) == []


def test_skip_when_no_semi_hard_neg_candidate():
    recs = [_mk(0, "FOOD#QUALITY", "positive"), _mk(1, "FOOD#QUALITY", "positive"),
            _mk(2, "FOOD#QUALITY", "negative")]
    assert build_contrastive_triplets(recs, seed=0) == []


import numpy as np
from src.data.contrastive_builder import build_hard_negative_triplets


def test_hard_neg_mining_returns_valid_triplets():
    recs = [
        {"id": "r0", "sentence": "s0", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r1", "sentence": "s1", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r2", "sentence": "s2", "aspect_category": "FOOD#QUALITY", "polarity": "negative"},
        {"id": "r3", "sentence": "s3", "aspect_category": "SERVICE#GENERAL", "polarity": "positive"},
    ]
    vectors = np.array([
        [1.0, 0.0], [0.5, 0.5], [0.9, 0.1], [0.3, 0.7],
    ], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    triplets = build_hard_negative_triplets(recs, vectors)
    assert len(triplets) > 0
    for t in triplets:
        assert t["anchor_aspect"] == t["positive_aspect"]
        assert t["anchor_polarity"] == t["positive_polarity"]
        assert t["anchor_aspect"] == t["neg1_aspect"]
        assert t["anchor_polarity"] != t["neg1_polarity"]
        assert t["anchor_aspect"] != t["neg2_aspect"]
        assert t["anchor_polarity"] == t["neg2_polarity"]


def test_hard_neg_picks_most_similar():
    recs = [
        {"id": "r0", "sentence": "s0", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r1", "sentence": "s1", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r2", "sentence": "s2", "aspect_category": "FOOD#QUALITY", "polarity": "negative"},
        {"id": "r3", "sentence": "s3", "aspect_category": "FOOD#QUALITY", "polarity": "negative"},
        {"id": "r4", "sentence": "s4", "aspect_category": "SERVICE#GENERAL", "polarity": "positive"},
    ]
    vectors = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.95, 0.05],
        [0.1, 0.9],
        [0.3, 0.7],
    ], dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    triplets = build_hard_negative_triplets(recs, vectors)
    r0_triplet = next(t for t in triplets if t["anchor_id"] == "r0")
    assert r0_triplet["neg1_id"] == "r2"


def test_hard_neg_skip_when_no_candidates():
    recs = [
        {"id": "r0", "sentence": "s0", "aspect_category": "FOOD#QUALITY", "polarity": "positive"},
        {"id": "r1", "sentence": "s1", "aspect_category": "FOOD#QUALITY", "polarity": "negative"},
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    triplets = build_hard_negative_triplets(recs, vectors)
    assert triplets == []
