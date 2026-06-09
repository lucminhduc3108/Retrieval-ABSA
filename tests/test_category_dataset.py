from src.absa.category_dataset import CategoryDataset, HierarchicalCategoryDataset
from src.data.category_builder import (
    CAT2IDX, NUM_CATEGORIES, NUM_ENTITIES,
    ENT2IDX, ATTR2IDX,
)


SAMPLE_RECORDS = [
    {
        "sentence_id": "s1",
        "sentence": "The food was great.",
        "categories": ["FOOD#QUALITY"],
        "category_vector": [int(c == "FOOD#QUALITY") for c in sorted(CAT2IDX)],
        "entity_vector": [int(e == "FOOD") for e in ["AMBIENCE", "DRINKS", "FOOD", "LOCATION", "RESTAURANT", "SERVICE"]],
        "food_attr_vector": [0, 1, 0],  # PRICES=0, QUALITY=1, STYLE_OPTIONS=0
        "drinks_attr_vector": [0, 0, 0],
        "restaurant_attr_vector": [0, 0, 0],
        "split": "train",
    },
    {
        "sentence_id": "s2",
        "sentence": "Nice ambience and good drinks.",
        "categories": ["AMBIENCE#GENERAL", "DRINKS#QUALITY"],
        "category_vector": [
            int(c in ("AMBIENCE#GENERAL", "DRINKS#QUALITY")) for c in sorted(CAT2IDX)
        ],
        "entity_vector": [1, 1, 0, 0, 0, 0],  # AMBIENCE, DRINKS
        "food_attr_vector": [0, 0, 0],
        "drinks_attr_vector": [0, 1, 0],  # QUALITY=1
        "restaurant_attr_vector": [0, 0, 0],
        "split": "train",
    },
]


def test_dataset_length():
    ds = CategoryDataset(SAMPLE_RECORDS)
    assert len(ds) == 2


def test_item_shapes():
    ds = CategoryDataset(SAMPLE_RECORDS, max_length=32)
    item = ds[0]
    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].shape == (32,)
    assert item["category_labels"].shape == (NUM_CATEGORIES,)


def test_labels_correct():
    ds = CategoryDataset(SAMPLE_RECORDS)
    item = ds[1]
    labels = item["category_labels"]
    assert labels[CAT2IDX["AMBIENCE#GENERAL"]] == 1.0
    assert labels[CAT2IDX["DRINKS#QUALITY"]] == 1.0
    assert labels[CAT2IDX["FOOD#QUALITY"]] == 0.0
    assert labels.sum() == 2.0


def test_single_label():
    ds = CategoryDataset(SAMPLE_RECORDS)
    item = ds[0]
    labels = item["category_labels"]
    assert labels[CAT2IDX["FOOD#QUALITY"]] == 1.0
    assert labels.sum() == 1.0


# --- Hierarchical dataset tests ---

def test_hierarchical_dataset_shapes():
    ds = HierarchicalCategoryDataset(SAMPLE_RECORDS, max_length=32)
    item = ds[0]
    assert item["input_ids"].shape == (32,)
    assert item["attention_mask"].shape == (32,)
    assert item["entity_labels"].shape == (NUM_ENTITIES,)
    assert item["food_attr_labels"].shape == (3,)
    assert item["drinks_attr_labels"].shape == (3,)
    assert item["restaurant_attr_labels"].shape == (3,)


def test_hierarchical_dataset_values():
    ds = HierarchicalCategoryDataset(SAMPLE_RECORDS)
    item0 = ds[0]  # FOOD#QUALITY
    assert item0["entity_labels"][ENT2IDX["FOOD"]] == 1.0
    assert item0["entity_labels"].sum() == 1.0
    assert item0["food_attr_labels"][ATTR2IDX["FOOD"]["QUALITY"]] == 1.0
    assert item0["food_attr_labels"].sum() == 1.0

    item1 = ds[1]  # AMBIENCE#GENERAL + DRINKS#QUALITY
    assert item1["entity_labels"][ENT2IDX["AMBIENCE"]] == 1.0
    assert item1["entity_labels"][ENT2IDX["DRINKS"]] == 1.0
    assert item1["entity_labels"].sum() == 2.0
    assert item1["drinks_attr_labels"][ATTR2IDX["DRINKS"]["QUALITY"]] == 1.0
