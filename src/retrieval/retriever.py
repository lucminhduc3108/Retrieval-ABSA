import faiss
import numpy as np


class Retriever:
    def __init__(self, index: faiss.IndexFlatIP, metadata: list[dict],
                 top_k: int = 3, threshold: float = 0.0):
        self.index = index
        self.metadata = metadata
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query_vec: np.ndarray,
                 query_id: str | None = None,
                 exclude_sentence: str | None = None) -> list[dict]:
        extra = 1 if query_id else 0
        if exclude_sentence:
            extra = 10
        query_vec = np.ascontiguousarray(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)
        D, I = self.index.search(query_vec, self.top_k + extra)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            if score < self.threshold:
                continue
            meta = self.metadata[idx]
            if query_id is not None and meta.get("id") == query_id:
                continue
            if exclude_sentence is not None and meta.get("sentence") == exclude_sentence:
                continue
            results.append({**meta, "score": float(score), "faiss_idx": int(idx)})
            if len(results) >= self.top_k:
                break
        return results
