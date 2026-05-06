import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

POL2ID = {"positive": 0, "negative": 1, "neutral": 2}
BIO2ID = {"O": 0, "B-ASP": 1, "I-ASP": 2}

IGNORE_INDEX = -100


class RetrievalABSADataset(Dataset):
    def __init__(self, bio_records: list[dict], retriever,
                 tokenizer_name: str = "microsoft/deberta-v3-base",
                 embedding_model=None,
                 max_length: int = 512, query_budget: int = 100,
                 top_k: int = 3, device: str = "cpu"):
        self.records = bio_records
        self.retriever = retriever
        self.embedding_model = embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.query_budget = query_budget
        self.top_k = top_k
        self.device = device

        if "[ASP]" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["[ASP]", "[POL]"]})

    def __len__(self):
        return len(self.records)

    def _align_bio_labels(self, sentence: str, tokens: list[str],
                          bio_tags: list[str], implicit: bool) -> tuple[list[int], list[int]]:
        enc = self.tokenizer(sentence, add_special_tokens=False,
                             return_offsets_mapping=True)
        subword_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        if implicit:
            return subword_ids, [IGNORE_INDEX] * len(subword_ids)

        char_labels = [0] * len(sentence)
        pos = 0
        for token, tag in zip(tokens, bio_tags):
            idx = sentence.find(token, pos)
            if idx == -1:
                pos += len(token) + 1
                continue
            label = BIO2ID.get(tag, 0)
            for c in range(idx, idx + len(token)):
                if c < len(char_labels):
                    char_labels[c] = label
            pos = idx + len(token)

        subword_labels = []
        for start, end in offsets:
            if start == end:
                subword_labels.append(IGNORE_INDEX)
            else:
                subword_labels.append(char_labels[start])
        return subword_ids, subword_labels

    def _retrieve_neighbors(self, record: dict) -> list[dict]:
        if self.retriever is None or self.embedding_model is None or self.top_k == 0:
            return []

        tok_enc = self.tokenizer(
            record["sentence"], record["aspect_category"],
            max_length=128, padding=False, truncation=True,
            return_tensors="pt")
        with torch.no_grad():
            query_vec = self.embedding_model.encode(
                tok_enc["input_ids"].to(self.device),
                tok_enc["attention_mask"].to(self.device))
        query_np = query_vec.cpu().numpy().astype("float32")
        return self.retriever.retrieve(query_np, query_id=record["id"])

    def __getitem__(self, idx) -> dict:
        record = self.records[idx]
        implicit = record.get("implicit", False)

        query_ids, query_labels = self._align_bio_labels(
            record["sentence"], record["tokens"],
            record["bio_tags"], implicit)

        aspect_enc = self.tokenizer(record["aspect_category"],
                                    add_special_tokens=False)
        aspect_ids = aspect_enc["input_ids"]

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id

        query_part_ids = [cls_id] + query_ids + [sep_id] + aspect_ids + [sep_id]
        query_part_labels = [IGNORE_INDEX] + query_labels + \
                            [IGNORE_INDEX] * (1 + len(aspect_ids) + 1)

        if len(query_ids) > self.query_budget:
            logger.warning("Query %s has %d tokens (budget=%d), shrinking retrieved budget",
                           record["id"], len(query_ids), self.query_budget)

        neighbors = self._retrieve_neighbors(record)

        retrieved_ids = []
        if neighbors:
            remaining = self.max_length - len(query_part_ids)
            per_neighbor = max(1, remaining // len(neighbors))

            asp_token_id = self.tokenizer.convert_tokens_to_ids("[ASP]")
            pol_token_id = self.tokenizer.convert_tokens_to_ids("[POL]")

            for nb in neighbors:
                sent_enc = self.tokenizer(nb["sentence"], add_special_tokens=False)
                asp_enc = self.tokenizer(nb["aspect_category"], add_special_tokens=False)
                pol_enc = self.tokenizer(nb["polarity"], add_special_tokens=False)

                nb_ids = sent_enc["input_ids"] + [asp_token_id] + asp_enc["input_ids"] + \
                         [pol_token_id] + pol_enc["input_ids"] + [sep_id]

                if len(nb_ids) > per_neighbor:
                    nb_ids = nb_ids[:per_neighbor - 1] + [sep_id]
                retrieved_ids.extend(nb_ids)

        all_ids = query_part_ids + retrieved_ids
        all_labels = query_part_labels + [IGNORE_INDEX] * len(retrieved_ids)

        if len(all_ids) > self.max_length:
            all_ids = all_ids[:self.max_length]
            all_labels = all_labels[:self.max_length]

        pad_len = self.max_length - len(all_ids)
        attention_mask = [1] * len(all_ids) + [0] * pad_len
        all_ids = all_ids + [self.tokenizer.pad_token_id] * pad_len
        all_labels = all_labels + [IGNORE_INDEX] * pad_len

        sentiment_label = POL2ID[record["polarity"]]

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bio_labels": torch.tensor(all_labels, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "query_id": record["id"],
            "crf_mask": torch.tensor([l != IGNORE_INDEX for l in all_labels], dtype=torch.bool),
        }
