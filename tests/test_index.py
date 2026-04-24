import numpy as np

from src.retrieval.index import build_index, save_index, load_index


def test_self_retrieval():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((10, 8)).astype("float32")
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    idx = build_index(v)
    D, I = idx.search(v[:1], 1)
    assert I[0, 0] == 0


def test_roundtrip(tmp_path):
    v = np.eye(4, dtype="float32")
    idx = build_index(v)
    meta = [{"id": f"r{i}"} for i in range(4)]
    save_index(idx, meta, v, str(tmp_path))
    idx2, meta2, vecs2 = load_index(str(tmp_path))
    assert meta2 == meta
    assert idx2.ntotal == 4
    np.testing.assert_array_equal(vecs2, v)
