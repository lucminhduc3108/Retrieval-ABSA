import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class _RecordDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        enc = self.tokenizer(
            r["sentence"], r["aspect_category"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def encode_records(records: list[dict], model, tokenizer,
                   max_length: int = 128, batch_size: int = 64,
                   device: str = "cuda") -> np.ndarray:
    model.eval()
    ds = _RecordDataset(records, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size)
    all_vecs = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        vecs = model.encode(ids, mask)
        all_vecs.append(vecs.cpu().numpy())
    return np.concatenate(all_vecs, axis=0).astype("float32")
