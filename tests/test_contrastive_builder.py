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
