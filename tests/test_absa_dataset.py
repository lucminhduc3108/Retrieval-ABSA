import torch
from unittest.mock import MagicMock

from src.absa.dataset import RetrievalABSADataset, POL2ID, BIO2ID


def _make_bio_record(rid="r0", sentence="The food is great", tokens=None,
                     bio_tags=None, aspect="FOOD#QUALITY", polarity="positive",
                     split="train", implicit=False):
    if tokens is None:
        tokens = sentence.split()
    if bio_tags is None:
        bio_tags = ["O"] * len(tokens)
    return {
        "id": rid, "sentence": sentence, "tokens": tokens,
        "bio_tags": bio_tags, "aspect_category": aspect,
        "polarity": polarity, "split": split, "implicit": implicit,
    }


def test_no_retriever_returns_expected_keys():
    rec = _make_bio_record(bio_tags=["O", "B-ASP", "O", "O"])
    ds = RetrievalABSADataset([rec], retriever=None, tokenizer_name="microsoft/deberta-v3-base",
                               embedding_model=None, max_length=64, top_k=0)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "bio_labels" in item
    assert "sentiment_label" in item
    assert item["input_ids"].shape == (64,)
    assert item["bio_labels"].shape == (64,)


def test_sentiment_label_mapping():
    for pol, expected_id in POL2ID.items():
        rec = _make_bio_record(polarity=pol)
        ds = RetrievalABSADataset([rec], retriever=None,
                                   tokenizer_name="microsoft/deberta-v3-base",
                                   embedding_model=None, max_length=64, top_k=0)
        item = ds[0]
        assert item["sentiment_label"].item() == expected_id


def test_special_tokens_have_ignore_label():
    rec = _make_bio_record(bio_tags=["O", "B-ASP", "O", "O"])
    ds = RetrievalABSADataset([rec], retriever=None,
                               tokenizer_name="microsoft/deberta-v3-base",
                               embedding_model=None, max_length=64, top_k=0)
    item = ds[0]
    assert item["bio_labels"][0].item() == -100  # [CLS]


def test_implicit_record_all_bio_ignored():
    rec = _make_bio_record(implicit=True, bio_tags=["O", "O", "O", "O"])
    ds = RetrievalABSADataset([rec], retriever=None,
                               tokenizer_name="microsoft/deberta-v3-base",
                               embedding_model=None, max_length=64, top_k=0)
    item = ds[0]
    assert (item["bio_labels"] == -100).all()


def test_retrieved_tokens_have_ignore_label():
    rec = _make_bio_record(bio_tags=["O", "B-ASP", "O", "O"])
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"sentence": "Nice place", "aspect_category": "AMBIENCE#GENERAL",
         "polarity": "positive", "tokens": ["Nice", "place"],
         "bio_tags": ["O", "O"]},
    ]
    mock_model = MagicMock()
    mock_model.encode.return_value = torch.randn(1, 256)

    ds = RetrievalABSADataset([rec], retriever=mock_retriever,
                               tokenizer_name="microsoft/deberta-v3-base",
                               embedding_model=mock_model,
                               max_length=128, top_k=1)
    item = ds[0]
    seq_len = (item["attention_mask"] == 1).sum().item()
    query_end = item["bio_labels"].tolist().index(-100, 1)
    for i in range(query_end, seq_len):
        assert item["bio_labels"][i].item() == -100


def test_query_id_passed_to_retriever():
    rec = _make_bio_record(rid="test_id_123")
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_model = MagicMock()
    mock_model.encode.return_value = torch.randn(1, 256)

    ds = RetrievalABSADataset([rec], retriever=mock_retriever,
                               tokenizer_name="microsoft/deberta-v3-base",
                               embedding_model=mock_model,
                               max_length=64, top_k=3)
    _ = ds[0]
    mock_retriever.retrieve.assert_called_once()
    call_args = mock_retriever.retrieve.call_args
    assert call_args[1].get("query_id") == "test_id_123" or \
           (len(call_args[0]) > 1 and call_args[0][1] == "test_id_123")
