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
    # Padding uses -inf so softmax assigns zero weight
    assert torch.all(torch.isinf(item["neighbor_scores"]) & (item["neighbor_scores"] < 0))


def test_no_retrieval_keys_when_disabled():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=False)
    item = ds[0]
    assert "neighbor_polarities" not in item
    assert "neighbor_scores" not in item


def test_query_polarity_present_when_retrieval_enabled():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=True, top_k=2, max_length=64)
    item0 = ds[0]
    item1 = ds[1]
    assert "query_polarity" in item0
    assert item0["query_polarity"].item() == POL2ID["positive"]
    assert item1["query_polarity"].item() == POL2ID["negative"]


def test_query_polarity_absent_when_retrieval_disabled():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=False)
    assert "query_polarity" not in ds[0]


def test_query_vec_shape_when_retrieval_enabled():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=True, top_k=2, max_length=64)
    item = ds[0]
    assert "query_vec" in item
    assert item["query_vec"].shape == (256,)
    assert item["query_vec"].dtype == torch.float32


def test_neighbor_vecs_shape_when_retrieval_enabled():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=True, top_k=2, max_length=64)
    item = ds[0]
    assert "neighbor_vecs" in item
    assert item["neighbor_vecs"].shape == (2, 256)
    assert item["neighbor_vecs"].dtype == torch.float32


def test_neighbor_vecs_use_store_vectors():
    import numpy as np
    store = np.random.randn(10, 256).astype("float32")
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=True,
                          top_k=2, max_length=64, store_vectors=store)
    item = ds[0]
    assert item["neighbor_vecs"].shape == (2, 256)


def test_new_retrieval_fields_absent_when_retrieval_disabled():
    ds = SentimentDataset(SAMPLE_RECORDS, use_retrieval=False)
    item = ds[0]
    assert "query_vec" not in item
    assert "neighbor_vecs" not in item
    assert "query_polarity" not in item
