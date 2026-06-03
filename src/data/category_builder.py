CATEGORY_LIST = sorted([
    "AMBIENCE#GENERAL",
    "DRINKS#PRICES",
    "DRINKS#QUALITY",
    "DRINKS#STYLE_OPTIONS",
    "FOOD#PRICES",
    "FOOD#QUALITY",
    "FOOD#STYLE_OPTIONS",
    "LOCATION#GENERAL",
    "RESTAURANT#GENERAL",
    "RESTAURANT#MISCELLANEOUS",
    "RESTAURANT#PRICES",
    "SERVICE#GENERAL",
])

CAT2IDX = {c: i for i, c in enumerate(CATEGORY_LIST)}
NUM_CATEGORIES = len(CATEGORY_LIST)

POL2ID = {"positive": 0, "negative": 1, "neutral": 2}


def build_category_records(parsed: list[dict], split: str) -> list[dict]:
    records = []
    for sent in parsed:
        cats = set()
        for op in sent["opinions"]:
            if op["category"] in CAT2IDX:
                cats.add(op["category"])
        if not cats:
            continue
        vec = [0] * NUM_CATEGORIES
        for c in cats:
            vec[CAT2IDX[c]] = 1
        records.append({
            "sentence_id": sent["sentence_id"],
            "sentence": sent["text"],
            "categories": sorted(cats),
            "category_vector": vec,
            "split": split,
        })
    return records


def build_sentiment_records(parsed: list[dict], split: str) -> list[dict]:
    records = []
    for sent in parsed:
        for op in sent["opinions"]:
            if op["category"] not in CAT2IDX:
                continue
            if op["polarity"] not in POL2ID:
                continue
            record_id = f"{sent['sentence_id']}_{op['category']}"
            records.append({
                "id": record_id,
                "sentence": sent["text"],
                "category": op["category"],
                "polarity": op["polarity"],
                "split": split,
            })
    return records
