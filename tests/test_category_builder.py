from src.data.category_builder import (
    CATEGORY_LIST, CAT2IDX, NUM_CATEGORIES, POL2ID,
    build_category_records, build_sentiment_records,
)


SAMPLE_PARSED = [
    {
        "sentence_id": "s1",
        "text": "The food was great but service was slow.",
        "opinions": [
            {"target": "food", "category": "food", "polarity": "positive",
             "from_char": 4, "to_char": 8},
            {"target": "service", "category": "service", "polarity": "negative",
             "from_char": 23, "to_char": 30},
        ],
    },
    {
        "sentence_id": "s2",
        "text": "Overpriced.",
        "opinions": [
            {"target": None, "category": "price", "polarity": "negative",
             "from_char": 0, "to_char": 0},
        ],
    },
]


def test_category_list_has_5():
    assert NUM_CATEGORIES == 5
    assert len(CATEGORY_LIST) == 5
    assert CATEGORY_LIST == sorted(CATEGORY_LIST)


def test_cat2idx_consistent():
    for i, cat in enumerate(CATEGORY_LIST):
        assert CAT2IDX[cat] == i


def test_pol2id():
    assert set(POL2ID.keys()) == {"positive", "negative", "neutral"}


def test_build_category_records_groups_by_sentence():
    records = build_category_records(SAMPLE_PARSED, split="train")
    assert len(records) == 2

    r1 = records[0]
    assert r1["sentence_id"] == "s1"
    assert set(r1["categories"]) == {"food", "service"}
    assert r1["category_vector"][CAT2IDX["food"]] == 1
    assert r1["category_vector"][CAT2IDX["service"]] == 1
    assert sum(r1["category_vector"]) == 2
    assert r1["split"] == "train"

    r2 = records[1]
    assert sum(r2["category_vector"]) == 1
    assert r2["category_vector"][CAT2IDX["price"]] == 1


def test_build_category_records_skips_unknown_category():
    parsed = [{
        "sentence_id": "s3",
        "text": "test",
        "opinions": [
            {"target": None, "category": "UNKNOWN#THING", "polarity": "positive",
             "from_char": 0, "to_char": 0},
        ],
    }]
    records = build_category_records(parsed, split="train")
    assert len(records) == 0


def test_build_sentiment_records_one_per_opinion():
    records = build_sentiment_records(SAMPLE_PARSED, split="train")
    assert len(records) == 3

    food_rec = [r for r in records if r["category"] == "food"]
    assert len(food_rec) == 1
    assert food_rec[0]["polarity"] == "positive"
    assert food_rec[0]["sentence"] == "The food was great but service was slow."


def test_build_sentiment_records_multiple_per_category():
    parsed = [{
        "sentence_id": "s4",
        "text": "Service was slow but people were friendly.",
        "opinions": [
            {"target": "Service", "category": "service", "polarity": "negative",
             "from_char": 0, "to_char": 7},
            {"target": "people", "category": "service", "polarity": "positive",
             "from_char": 24, "to_char": 30},
        ],
    }]
    records = build_sentiment_records(parsed, split="train")
    service_recs = [r for r in records if r["category"] == "service"]
    assert len(service_recs) == 2
    polarities = {r["polarity"] for r in service_recs}
    assert polarities == {"positive", "negative"}


def test_build_sentiment_records_skips_unknown():
    parsed = [{
        "sentence_id": "s5",
        "text": "test",
        "opinions": [
            {"target": None, "category": "TOTALLY_UNKNOWN", "polarity": "positive",
             "from_char": 0, "to_char": 0},
        ],
    }]
    records = build_sentiment_records(parsed, split="test")
    assert len(records) == 0


def test_category_vector_length():
    records = build_category_records(SAMPLE_PARSED, split="train")
    for r in records:
        assert len(r["category_vector"]) == NUM_CATEGORIES


def test_record_has_no_hierarchical_fields():
    records = build_category_records(SAMPLE_PARSED, split="train")
    for r in records:
        assert "entity_vector" not in r
        assert "food_attr_vector" not in r
        assert "drinks_attr_vector" not in r
        assert "restaurant_attr_vector" not in r
