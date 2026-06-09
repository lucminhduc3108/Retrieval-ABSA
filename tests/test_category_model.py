import torch

from src.absa.category_model import AsymmetricLoss, CategoryDetector, HierarchicalCategoryDetector
from src.data.category_builder import NUM_ENTITIES, ENT2IDX


def _make_batch(batch_size=2, seq_len=16, vocab_size=128):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, attention_mask


def test_forward_shape():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=12)
    ids, mask = _make_batch()
    out = model(ids, mask)
    assert out["logits"].shape == (2, 12)
    assert out["loss"] is None


def test_forward_with_labels():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=12)
    ids, mask = _make_batch()
    labels = torch.zeros(2, 12)
    labels[0, 0] = 1.0
    labels[1, 5] = 1.0
    out = model(ids, mask, category_labels=labels)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_pos_weight():
    pw = torch.ones(12) * 2.0
    model = CategoryDetector(model_name="microsoft/deberta-v3-base",
                             num_categories=12, pos_weight=pw)
    ids, mask = _make_batch()
    labels = torch.zeros(2, 12)
    labels[0, 0] = 1.0
    out = model(ids, mask, category_labels=labels)
    assert out["loss"] is not None


def test_gradient_flows():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=12)
    ids, mask = _make_batch()
    labels = torch.zeros(2, 12)
    labels[0, 0] = 1.0
    out = model(ids, mask, category_labels=labels)
    out["loss"].backward()
    assert model.category_head.weight.grad is not None


def test_asl_loss_forward():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=12,
                             use_asl=True, asl_gamma_neg=4, asl_gamma_pos=0, asl_margin=0.05)
    ids, mask = _make_batch()
    labels = torch.zeros(2, 12)
    labels[0, 0] = 1.0
    out = model(ids, mask, category_labels=labels)
    assert out["logits"].shape == (2, 12)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_cat_attention_forward():
    model = CategoryDetector(model_name="microsoft/deberta-v3-base", num_categories=12,
                             use_cat_attention=True)
    ids, mask = _make_batch()
    labels = torch.zeros(2, 12)
    labels[0, 3] = 1.0
    out = model(ids, mask, category_labels=labels)
    assert out["logits"].shape == (2, 12)
    assert out["loss"] is not None


# --- Hierarchical model tests ---

def test_hierarchical_forward_shape():
    model = HierarchicalCategoryDetector()
    ids, mask = _make_batch()
    out = model(ids, mask)
    assert out["entity_logits"].shape == (2, NUM_ENTITIES)
    assert out["food_attr_logits"].shape == (2, 3)
    assert out["drinks_attr_logits"].shape == (2, 3)
    assert out["restaurant_attr_logits"].shape == (2, 3)
    assert out["loss"] is None


def test_hierarchical_forward_with_labels():
    model = HierarchicalCategoryDetector()
    ids, mask = _make_batch()
    ent_labels = torch.zeros(2, NUM_ENTITIES)
    ent_labels[0, ENT2IDX["FOOD"]] = 1.0
    food_labels = torch.zeros(2, 3)
    food_labels[0, 1] = 1.0  # QUALITY
    drinks_labels = torch.zeros(2, 3)
    rest_labels = torch.zeros(2, 3)
    out = model(ids, mask, entity_labels=ent_labels,
                food_attr_labels=food_labels,
                drinks_attr_labels=drinks_labels,
                restaurant_attr_labels=rest_labels)
    assert out["loss"] is not None
    assert out["loss"].ndim == 0


def test_hierarchical_attr_loss_masking():
    model = HierarchicalCategoryDetector()
    ids, mask = _make_batch()
    ent_labels = torch.zeros(2, NUM_ENTITIES)
    food_labels = torch.zeros(2, 3)
    drinks_labels = torch.zeros(2, 3)
    rest_labels = torch.zeros(2, 3)
    out_no_entity = model(ids, mask, entity_labels=ent_labels,
                          food_attr_labels=food_labels,
                          drinks_attr_labels=drinks_labels,
                          restaurant_attr_labels=rest_labels)
    ent_labels2 = torch.zeros(2, NUM_ENTITIES)
    ent_labels2[0, ENT2IDX["FOOD"]] = 1.0
    food_labels2 = torch.zeros(2, 3)
    food_labels2[0, 1] = 1.0
    out_with_entity = model(ids, mask, entity_labels=ent_labels2,
                            food_attr_labels=food_labels2,
                            drinks_attr_labels=drinks_labels,
                            restaurant_attr_labels=rest_labels)
    assert out_with_entity["loss"].item() != out_no_entity["loss"].item()


def test_hierarchical_gradient_flows():
    model = HierarchicalCategoryDetector()
    ids, mask = _make_batch()
    ent_labels = torch.zeros(2, NUM_ENTITIES)
    ent_labels[0, ENT2IDX["FOOD"]] = 1.0
    food_labels = torch.zeros(2, 3)
    food_labels[0, 0] = 1.0
    out = model(ids, mask, entity_labels=ent_labels,
                food_attr_labels=food_labels,
                drinks_attr_labels=torch.zeros(2, 3),
                restaurant_attr_labels=torch.zeros(2, 3))
    out["loss"].backward()
    assert model.entity_head.weight.grad is not None
    assert model.food_attr_head.weight.grad is not None


def test_hierarchical_pos_weight():
    pw_ent = torch.ones(NUM_ENTITIES) * 2.0
    pw_food = torch.ones(3) * 1.5
    model = HierarchicalCategoryDetector(
        pos_weight_entity=pw_ent, pos_weight_food=pw_food)
    ids, mask = _make_batch()
    ent_labels = torch.zeros(2, NUM_ENTITIES)
    ent_labels[0, ENT2IDX["FOOD"]] = 1.0
    out = model(ids, mask, entity_labels=ent_labels,
                food_attr_labels=torch.zeros(2, 3),
                drinks_attr_labels=torch.zeros(2, 3),
                restaurant_attr_labels=torch.zeros(2, 3))
    assert out["loss"] is not None
