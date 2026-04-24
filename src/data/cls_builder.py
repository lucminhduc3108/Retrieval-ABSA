def build_cls_records(parsed: list[dict], split: str) -> list[dict]:
    records = []
    for sent in parsed:
        for op_idx, op in enumerate(sent["opinions"]):
            records.append({
                "id": f"{sent['sentence_id']}_op{op_idx}",
                "sentence": sent["text"],
                "aspect_category": op["category"],
                "polarity": op["polarity"],
                "split": split,
            })
    return records
