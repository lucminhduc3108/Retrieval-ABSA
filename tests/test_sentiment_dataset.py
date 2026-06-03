import torch

from src.absa.sentiment_dataset import SentimentDataset
from src.data.category_builder import POL2ID


SAMPLE_RECORDS = [
    {
        "id": "s1_FOOD#QUALITY",
        "sentence": "The food was great.",
        "category": "FOOD#QUALITY",
        "polarity": "positive",
        "split": "train",
    },
    {
        "id": "s2_SERVICE#GENERAL",
        "sentence": "Service was slow.",
        "category": "SERVICE#GENERAL",
        "polarity": "negative",
        "split": "train",
    },
]


def test_dataset_length():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=False)
    assert len(ds) == 2


def test_item_shapes_no_retrieval():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=False, max_length=64)
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["attention_mask"].shape == (64,)
    assert item["sentiment_label"].ndim == 0
    assert "neighbor_polarities" not in item


def test_item_shapes_with_retrieval_no_model():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=True,
                          top_k=2, max_length=64)
    item = ds[0]
    assert item["input_ids"].shape == (64,)
    assert item["neighbor_polarities"].shape == (2,)
    assert item["neighbor_scores"].shape == (2,)


def test_sentiment_label_mapping():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=False)
    item0 = ds[0]
    item1 = ds[1]
    assert item0["sentiment_label"].item() == POL2ID["positive"]
    assert item1["sentiment_label"].item() == POL2ID["negative"]


def test_input_contains_sentence_and_category():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=False, max_length=64)
    item = ds[0]
    decoded = ds.tokenizer.decode(item["input_ids"], skip_special_tokens=True)
    assert "food" in decoded.lower()
    assert "quality" in decoded.lower()


def test_padded_neighbors_when_no_retriever():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=True,
                          top_k=3, max_length=64)
    item = ds[0]
    assert item["neighbor_polarities"].shape == (3,)
    assert (item["neighbor_scores"] == 0.0).all()


def test_no_retrieval_keys_when_disabled():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=False)
    item = ds[0]
    assert "neighbor_polarities" not in item
    assert "neighbor_scores" not in item
