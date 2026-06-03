import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data.category_builder import NUM_CATEGORIES


class CategoryDataset(Dataset):
    def __init__(self, records: list[dict],
                 tokenizer_name: str = "microsoft/deberta-v3-base",
                 max_length: int = 128):
        self.records = records
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx) -> dict:
        record = self.records[idx]
        enc = self.tokenizer(
            record["sentence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        category_labels = torch.tensor(
            record["category_vector"], dtype=torch.float32)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "category_labels": category_labels,
        }
