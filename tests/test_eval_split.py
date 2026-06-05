from sklearn.model_selection import train_test_split


def _make_records(n=200, num_cats=12):
    records = []
    for i in range(n):
        vec = [0] * num_cats
        vec[i % num_cats] = 1
        if i % 7 == 0:
            vec[(i + 3) % num_cats] = 1
        records.append({"category_vector": vec, "idx": i})
    return records


def test_stratify_mismatch_produces_different_splits():
    """Without stratify, same seed gives different split — confirms the bug."""
    records = _make_records()
    stratify_key = [sum(r["category_vector"]) for r in records]

    _, val_with = train_test_split(
        records, test_size=0.2, random_state=42, stratify=stratify_key)
    _, val_without = train_test_split(
        records, test_size=0.2, random_state=42)

    ids_with = {r["idx"] for r in val_with}
    ids_without = {r["idx"] for r in val_without}
    assert ids_with != ids_without


def test_stratify_split_is_deterministic():
    """Same params produce identical split every time."""
    records = _make_records()
    stratify_key = [sum(r["category_vector"]) for r in records]

    _, val_a = train_test_split(
        records, test_size=0.2, random_state=42, stratify=stratify_key)
    _, val_b = train_test_split(
        records, test_size=0.2, random_state=42, stratify=stratify_key)

    assert {r["idx"] for r in val_a} == {r["idx"] for r in val_b}
