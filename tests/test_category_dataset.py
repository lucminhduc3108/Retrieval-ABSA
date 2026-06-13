from src.absa.category_dataset import CategoryDataset
from src.data.category_builder import CAT2IDX, NUM_CATEGORIES


SAMPLE_RECORDS = [
    {
        "sentence_id": "s1",
        "sentence": "The food was great.",
        "categories": ["food"],
        "category_vector": [int(c == "food") for c in sorted(CAT2IDX)],
        "split": "train",
    },
    {
        "sentence_id": "s2",
        "sentence": "Nice ambience and good drinks.",
        "categories": ["ambience", "food"],
        "category_vector": [int(c in ("ambience", "food")) for c in sorted(CAT2IDX)],
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
    assert labels[CAT2IDX["ambience"]] == 1.0
    assert labels[CAT2IDX["food"]] == 1.0
    assert labels[CAT2IDX["service"]] == 0.0
    assert labels.sum() == 2.0


def test_single_label():
    ds = CategoryDataset(SAMPLE_RECORDS)
    item = ds[0]
    labels = item["category_labels"]
    assert labels[CAT2IDX["food"]] == 1.0
    assert labels.sum() == 1.0
