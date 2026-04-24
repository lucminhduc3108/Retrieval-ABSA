import torch
from torch.utils.data import Dataset

from src.utils.io import read_jsonl


class ContrastiveTripletDataset(Dataset):
    def __init__(self, triplets_path: str, tokenizer, max_length: int = 128):
        self.triplets = read_jsonl(triplets_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def _tokenize(self, sentence: str, aspect: str) -> dict:
        enc = self.tokenizer(
            sentence, aspect,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx) -> dict:
        t = self.triplets[idx]
        anchor = self._tokenize(t["anchor_sentence"], t["anchor_aspect"])
        pos = self._tokenize(t["positive_sentence"], t["positive_aspect"])
        neg1 = self._tokenize(t["neg1_sentence"], t["neg1_aspect"])
        neg2 = self._tokenize(t["neg2_sentence"], t["neg2_aspect"])
        return {
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor["attention_mask"],
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg1_input_ids": neg1["input_ids"],
            "neg1_attention_mask": neg1["attention_mask"],
            "neg2_input_ids": neg2["input_ids"],
            "neg2_attention_mask": neg2["attention_mask"],
        }
