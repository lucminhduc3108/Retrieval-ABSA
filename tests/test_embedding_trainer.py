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


def test_training_decreases_loss(tmp_path):
    torch.manual_seed(0)
    triplets = [
        {
            "anchor_id": "a0", "anchor_sentence": "The pizza was absolutely delicious",
            "anchor_aspect": "FOOD#QUALITY", "anchor_polarity": "positive",
            "positive_id": "p0", "positive_sentence": "Amazing pasta and great taste",
            "positive_aspect": "FOOD#QUALITY", "positive_polarity": "positive",
            "neg1_id": "n1_0", "neg1_sentence": "The steak was terrible and raw",
            "neg1_aspect": "FOOD#QUALITY", "neg1_polarity": "negative",
            "neg2_id": "n2_0", "neg2_sentence": "Waiters were friendly and fast",
            "neg2_aspect": "SERVICE#GENERAL", "neg2_polarity": "positive",
        },
        {
            "anchor_id": "a1", "anchor_sentence": "Slow service and rude staff",
            "anchor_aspect": "SERVICE#GENERAL", "anchor_polarity": "negative",
            "positive_id": "p1", "positive_sentence": "The waiter ignored us completely",
            "positive_aspect": "SERVICE#GENERAL", "positive_polarity": "negative",
            "neg1_id": "n1_1", "neg1_sentence": "Excellent and attentive service",
            "neg1_aspect": "SERVICE#GENERAL", "neg1_polarity": "positive",
            "neg2_id": "n2_1", "neg2_sentence": "The soup was bland and cold",
            "neg2_aspect": "FOOD#QUALITY", "neg2_polarity": "negative",
        },
    ]
    p = tmp_path / "triplets.jsonl"
    write_jsonl(triplets, str(p))
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    ds = ContrastiveTripletDataset(str(p), tok, max_length=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)

    model = ContrastiveEmbedder(proj_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    batch = next(iter(loader))

    def compute_loss():
        out = model(batch["anchor_input_ids"], batch["anchor_attention_mask"],
                    batch["pos_input_ids"], batch["pos_attention_mask"],
                    batch["neg1_input_ids"], batch["neg1_attention_mask"],
                    batch["neg2_input_ids"], batch["neg2_attention_mask"])
        return infonce_loss(out["anchor_vecs"], out["pos_vecs"],
                            negatives=[out["neg1_vecs"], out["neg2_vecs"]], tau=0.07)

    loss_initial = compute_loss().item()
    for _ in range(5):
        loss = compute_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_final = compute_loss().item()
    assert loss_final < loss_initial


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


def test_trainer_with_grad_accum(tmp_path):
    torch.manual_seed(0)
    p = tmp_path / "triplets.jsonl"
    write_jsonl([_make_triplet(i) for i in range(8)], str(p))
    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    ds = ContrastiveTripletDataset(str(p), tok, max_length=32)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)

    model = ContrastiveEmbedder(proj_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = ContrastiveTrainer(model, optimizer, scheduler=None,
                                 tau=0.07, device="cpu", log_path="",
                                 grad_accum_steps=2)
    result = trainer.evaluate_recall(loader, k_list=(1, 3))
    assert "recall@1" in result
