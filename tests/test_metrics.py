from src.evaluation.metrics import (
    bio_token_metrics, extract_spans, span_f1, sentiment_metrics, joint_f1,
)


def test_extract_spans_basic():
    assert extract_spans([0, 1, 2, 0, 1, 0]) == [(1, 3), (4, 5)]


def test_extract_spans_consecutive_b():
    assert extract_spans([1, 1, 0]) == [(0, 1), (1, 2)]


def test_extract_spans_trailing_i():
    assert extract_spans([1, 2, 2]) == [(0, 3)]


def test_bio_token_perfect():
    m = bio_token_metrics([[0, 1, 2, 0]], [[0, 1, 2, 0]])
    assert m["f1"] == 1.0


def test_bio_token_ignore_index():
    m = bio_token_metrics([[0, 1, 2, 0]], [[-100, 1, 2, 0]])
    assert m["f1"] == 1.0


def test_bio_token_all_o():
    m = bio_token_metrics([[0, 0, 0]], [[0, 0, 0]])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_span_f1_perfect():
    m = span_f1([[(1, 3), (4, 5)]], [[(1, 3), (4, 5)]])
    assert m["f1"] == 1.0


def test_span_f1_partial():
    m = span_f1([[(1, 3), (4, 5)]], [[(1, 3), (4, 6)]])
    assert 0 < m["f1"] < 1


def test_sentiment_metrics_perfect():
    m = sentiment_metrics([0, 1, 2], [0, 1, 2])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_joint_f1_requires_both():
    assert joint_f1([[(1, 3, 0)]], [[(1, 3, 1)]]) == 0.0
    assert joint_f1([[(1, 3, 1)]], [[(1, 3, 1)]]) == 1.0


def test_joint_f1_empty():
    assert joint_f1([[]], [[]]) == 0.0
