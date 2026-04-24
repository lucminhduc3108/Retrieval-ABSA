import os

import faiss
import numpy as np

from src.utils.io import read_jsonl, write_jsonl


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    faiss.normalize_L2(vectors)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors)
    return idx


def save_index(index: faiss.IndexFlatIP, metadata: list[dict],
               vectors: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "train.faiss"))
    write_jsonl(metadata, os.path.join(out_dir, "train_metadata.jsonl"))
    np.save(os.path.join(out_dir, "train_vectors.npy"), vectors)


def load_index(out_dir: str) -> tuple[faiss.IndexFlatIP, list[dict], np.ndarray]:
    index = faiss.read_index(os.path.join(out_dir, "train.faiss"))
    metadata = read_jsonl(os.path.join(out_dir, "train_metadata.jsonl"))
    vectors = np.load(os.path.join(out_dir, "train_vectors.npy"))
    return index, metadata, vectors
