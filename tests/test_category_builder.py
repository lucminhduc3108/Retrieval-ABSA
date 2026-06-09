from src.data.category_builder import (
    CATEGORY_LIST, CAT2IDX, NUM_CATEGORIES, POL2ID,
    ENTITY_LIST, ENT2IDX, NUM_ENTITIES, ENTITY2ATTRS,
    MULTI_ATTR_ENTITIES, ATTR2IDX,
    build_category_records, build_sentiment_records,
)


SAMPLE_PARSED = [
    {
        "sentence_id": "s1",
        "text": "The food was great but service was slow.",
        "opinions": [
            {"target": "food", "category": "FOOD#QUALITY", "polarity": "positive",
             "from_char": 4, "to_char": 8},
            {"target": "service", "category": "SERVICE#GENERAL", "polarity": "negative",
             "from_char": 23, "to_char": 30},
        ],
    },
    {
        "sentence_id": "s2",
        "text": "Overpriced.",
        "opinions": [
            {"target": None, "category": "RESTAURANT#PRICES", "polarity": "negative",
             "from_char": 0, "to_char": 0},
        ],
    },
]


def test_category_list_has_12():
    assert NUM_CATEGORIES == 12
    assert len(CATEGORY_LIST) == 12
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
    assert set(r1["categories"]) == {"FOOD#QUALITY", "SERVICE#GENERAL"}
    assert r1["category_vector"][CAT2IDX["FOOD#QUALITY"]] == 1
    assert r1["category_vector"][CAT2IDX["SERVICE#GENERAL"]] == 1
    assert sum(r1["category_vector"]) == 2
    assert r1["split"] == "train"

    r2 = records[1]
    assert sum(r2["category_vector"]) == 1
    assert r2["category_vector"][CAT2IDX["RESTAURANT#PRICES"]] == 1


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

    food_rec = [r for r in records if r["category"] == "FOOD#QUALITY"]
    assert len(food_rec) == 1
    assert food_rec[0]["polarity"] == "positive"
    assert food_rec[0]["sentence"] == "The food was great but service was slow."


def test_build_sentiment_records_conflict_kept():
    parsed = [{
        "sentence_id": "s4",
        "text": "Service was slow but people were friendly.",
        "opinions": [
            {"target": "Service", "category": "SERVICE#GENERAL", "polarity": "negative",
             "from_char": 0, "to_char": 7},
            {"target": "people", "category": "SERVICE#GENERAL", "polarity": "positive",
             "from_char": 24, "to_char": 30},
        ],
    }]
    records = build_sentiment_records(parsed, split="train")
    service_recs = [r for r in records if r["category"] == "SERVICE#GENERAL"]
    assert len(service_recs) == 2
    polarities = {r["polarity"] for r in service_recs}
    assert polarities == {"positive", "negative"}


def test_build_sentiment_records_skips_unknown():
    parsed = [{
        "sentence_id": "s5",
        "text": "test",
        "opinions": [
            {"target": None, "category": "FOOD#GENERAL", "polarity": "positive",
             "from_char": 0, "to_char": 0},
        ],
    }]
    records = build_sentiment_records(parsed, split="test")
    assert len(records) == 0


def test_category_vector_length():
    records = build_category_records(SAMPLE_PARSED, split="train")
    for r in records:
        assert len(r["category_vector"]) == NUM_CATEGORIES


# --- Hierarchical constants ---

def test_entity_list_has_6():
    assert NUM_ENTITIES == 6
    assert len(ENTITY_LIST) == 6
    assert ENTITY_LIST == sorted(ENTITY_LIST)


def test_ent2idx_consistent():
    for i, ent in enumerate(ENTITY_LIST):
        assert ENT2IDX[ent] == i


def test_entity2attrs_covers_all_categories():
    reconstructed = set()
    for entity, attrs in ENTITY2ATTRS.items():
        for attr in attrs:
            reconstructed.add(f"{entity}#{attr}")
    assert reconstructed == set(CATEGORY_LIST)


def test_multi_attr_entities():
    for ent in MULTI_ATTR_ENTITIES:
        assert len(ENTITY2ATTRS[ent]) > 1
    for ent in ENTITY_LIST:
        if ent not in MULTI_ATTR_ENTITIES:
            assert ENTITY2ATTRS[ent] == ["GENERAL"]


# --- Hierarchical record fields ---

def test_build_category_records_hierarchical_fields():
    records = build_category_records(SAMPLE_PARSED, split="train")
    r1 = records[0]  # FOOD#QUALITY + SERVICE#GENERAL
    assert len(r1["entity_vector"]) == NUM_ENTITIES
    assert r1["entity_vector"][ENT2IDX["FOOD"]] == 1
    assert r1["entity_vector"][ENT2IDX["SERVICE"]] == 1
    assert sum(r1["entity_vector"]) == 2
    assert r1["food_attr_vector"][ATTR2IDX["FOOD"]["QUALITY"]] == 1
    assert sum(r1["food_attr_vector"]) == 1
    assert sum(r1["drinks_attr_vector"]) == 0
    assert sum(r1["restaurant_attr_vector"]) == 0

    r2 = records[1]  # RESTAURANT#PRICES
    assert r2["entity_vector"][ENT2IDX["RESTAURANT"]] == 1
    assert sum(r2["entity_vector"]) == 1
    assert r2["restaurant_attr_vector"][ATTR2IDX["RESTAURANT"]["PRICES"]] == 1
    assert sum(r2["restaurant_attr_vector"]) == 1
