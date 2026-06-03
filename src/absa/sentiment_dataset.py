import logging

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.category_builder import POL2ID

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    def __init__(self, records: list[dict], retriever=None,
                 tokenizer_name: str = "microsoft/deberta-v3-base",
                 embedding_model=None,
                 max_length: int = 256, top_k: int = 2,
                 device: str = "cpu", use_retrieval: bool = True):
        self.records = records
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.top_k = top_k
        self.device = device
        self.use_retrieval = use_retrieval

    def __len__(self):
        return len(self.records)

    def _retrieve_neighbors(self, sentence: str,
                            category: str) -> list[dict]:
        if self.retriever is None or self.embedding_model is None:
            return []

        tok_enc = self.tokenizer(
            sentence, category,
            max_length=128, padding=False, truncation=True,
            return_tensors="pt")
        with torch.no_grad():
            query_vec = self.embedding_model.encode(
                tok_enc["input_ids"].to(self.device),
                tok_enc["attention_mask"].to(self.device))
        query_np = query_vec.cpu().numpy().astype("float32")
        return self.retriever.retrieve(
            query_np, exclude_sentence=sentence)

    def __getitem__(self, idx) -> dict:
        record = self.records[idx]
        sentence = record["sentence"]
        category = record["category"]

        neighbors = []
        if self.use_retrieval and self.top_k > 0:
            neighbors = self._retrieve_neighbors(sentence, category)

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        sent_enc = self.tokenizer(sentence, add_special_tokens=False)
        cat_enc = self.tokenizer(category, add_special_tokens=False)

        token_ids = [cls_id] + sent_enc["input_ids"] + [sep_id] + \
                    cat_enc["input_ids"] + [sep_id]

        if neighbors:
            remaining = self.max_length - len(token_ids)
            per_nb = max(1, remaining // len(neighbors))
            for nb in neighbors:
                nb_enc = self.tokenizer(nb["sentence"], add_special_tokens=False)
                nb_ids = nb_enc["input_ids"]
                if len(nb_ids) > per_nb - 1:
                    nb_ids = nb_ids[:per_nb - 1]
                token_ids.extend(nb_ids + [sep_id])

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        pad_len = self.max_length - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len

        sentiment_label = POL2ID[record["polarity"]]

        result = {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
        }

        if self.use_retrieval:
            if neighbors:
                pol_ids = [POL2ID[nb["polarity"]] for nb in neighbors]
                scores = [nb["score"] for nb in neighbors]
            else:
                pol_ids = []
                scores = []

            while len(pol_ids) < self.top_k:
                pol_ids.append(0)
                scores.append(0.0)

            result["neighbor_polarities"] = torch.tensor(
                pol_ids[:self.top_k], dtype=torch.long)
            result["neighbor_scores"] = torch.tensor(
                scores[:self.top_k], dtype=torch.float32)

        return result
