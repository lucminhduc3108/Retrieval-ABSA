import numpy as np

from src.retrieval.index import build_index
from src.retrieval.retriever import Retriever


def _setup():
    v = np.eye(4, dtype="float32")
    meta = [{"id": f"r{i}", "sentence": f"s{i}"} for i in range(4)]
    return Retriever(build_index(v), meta, top_k=2)


def test_self_excluded():
    r = _setup()
    out = r.retrieve(np.eye(4, dtype="float32")[0:1], query_id="r0")
    ids = [m["id"] for m in out]
    assert "r0" not in ids


def test_returns_up_to_top_k():
    r = _setup()
    out = r.retrieve(np.eye(4, dtype="float32")[0:1], query_id="r0")
    assert len(out) <= 2


def test_no_exclusion_when_id_none():
    r = _setup()
    out = r.retrieve(np.eye(4, dtype="float32")[0:1], query_id=None)
    assert any(m["id"] == "r0" for m in out)


def test_threshold_filters_all():
    r = Retriever(build_index(np.eye(4, dtype="float32")),
                  [{"id": f"r{i}"} for i in range(4)],
                  top_k=2, threshold=1.01)
    assert r.retrieve(np.eye(4, dtype="float32")[0:1], query_id=None) == []


def test_results_include_score():
    r = _setup()
    out = r.retrieve(np.eye(4, dtype="float32")[0:1], query_id=None)
    assert "score" in out[0]


def test_exclude_sentence_removes_all_records_with_same_sentence():
    v = np.random.randn(5, 4).astype("float32")
    meta = [
        {"id": "r0_op0", "sentence": "same sentence"},
        {"id": "r0_op1", "sentence": "same sentence"},
        {"id": "r1_op0", "sentence": "different one"},
        {"id": "r2_op0", "sentence": "another one"},
        {"id": "r3_op0", "sentence": "yet another"},
    ]
    r = Retriever(build_index(v), meta, top_k=3)
    out = r.retrieve(v[0:1], exclude_sentence="same sentence")
    returned_sents = {m["sentence"] for m in out}
    assert "same sentence" not in returned_sents
    assert len(out) <= 3


def test_exclude_sentence_backward_compatible_with_query_id():
    r = _setup()
    out = r.retrieve(np.eye(4, dtype="float32")[0:1], query_id="r0")
    assert "r0" not in [m["id"] for m in out]
