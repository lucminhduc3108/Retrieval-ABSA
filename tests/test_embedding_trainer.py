import torch
from transformers import AutoTokenizer

from src.data.datasets import ContrastiveTripletDataset
from src.embedding.model import ContrastiveEmbedder
from src.embedding.loss import infonce_loss
from src.embedding.trainer import ContrastiveTrainer
from src.utils.io import write_jsonl


def _make_triplet(i):
    return {
        "anchor_id": f"a{i}", "anchor_sentence": f"food is good {i}",
        "anchor_aspect": "FOOD#QUALITY", "anchor_polarity": "positive",
        "positive_id": f"p{i}", "positive_sentence": f"great food {i}",
        "positive_aspect": "FOOD#QUALITY", "positive_polarity": "positive",
        "neg1_id": f"n1_{i}", "neg1_sentence": f"food was bad {i}",
        "neg1_aspect": "FOOD#QUALITY", "neg1_polarity": "negative",
        "neg2_id": f"n2_{i}", "neg2_sentence": f"great service {i}",
        "neg2_aspect": "SERVICE#GENERAL", "neg2_polarity": "positive",
    }


def test_dataset_item_has_expected_keys(tmp_path):
    p = tmp_path / "triplets.jsonl"
    write_jsonl([_make_triplet(0)], str(p))
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    ds = ContrastiveTripletDataset(str(p), tok, max_length=32)
    item = ds[0]
    assert set(item.keys()) == {
        "anchor_input_ids", "anchor_attention_mask",
        "pos_input_ids", "pos_attention_mask",
        "neg1_input_ids", "neg1_attention_mask",
        "neg2_input_ids", "neg2_attention_mask",
    }
    assert item["anchor_input_ids"].shape == (32,)


def test_one_step_decreases_loss(tmp_path):
    p = tmp_path / "triplets.jsonl"
    write_jsonl([_make_triplet(i) for i in range(4)], str(p))
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    ds = ContrastiveTripletDataset(str(p), tok, max_length=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)

    model = ContrastiveEmbedder(proj_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    batch = next(iter(loader))
    out = model(batch["anchor_input_ids"], batch["anchor_attention_mask"],
                batch["pos_input_ids"], batch["pos_attention_mask"],
                batch["neg1_input_ids"], batch["neg1_attention_mask"],
                batch["neg2_input_ids"], batch["neg2_attention_mask"])
    loss1 = infonce_loss(out["anchor_vecs"], out["pos_vecs"],
                         negatives=[out["neg1_vecs"], out["neg2_vecs"]], tau=0.07)

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    out2 = model(batch["anchor_input_ids"], batch["anchor_attention_mask"],
                 batch["pos_input_ids"], batch["pos_attention_mask"],
                 batch["neg1_input_ids"], batch["neg1_attention_mask"],
                 batch["neg2_input_ids"], batch["neg2_attention_mask"])
    loss2 = infonce_loss(out2["anchor_vecs"], out2["pos_vecs"],
                         negatives=[out2["neg1_vecs"], out2["neg2_vecs"]], tau=0.07)
    assert loss2.item() < loss1.item()


def test_evaluate_recall_returns_expected_keys(tmp_path):
    p = tmp_path / "triplets.jsonl"
    write_jsonl([_make_triplet(i) for i in range(8)], str(p))
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    ds = ContrastiveTripletDataset(str(p), tok, max_length=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)

    model = ContrastiveEmbedder(proj_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ContrastiveTrainer(model, optimizer, scheduler=None,
                                 tau=0.07, device="cpu", log_path="")
    result = trainer.evaluate_recall(loader, k_list=(1, 3, 5))
    assert "recall@1" in result
    assert "recall@3" in result
    assert "recall@5" in result
    assert 0 <= result["recall@1"] <= 1
