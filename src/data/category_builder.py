CATEGORY_LIST = sorted([
    "ambience",
    "anecdotes/miscellaneous",
    "food",
    "price",
    "service",
])

MAMS_CATEGORY_LIST = sorted([
    "ambience",
    "food",
    "menu",
    "miscellaneous",
    "place",
    "price",
    "service",
    "staff",
])

CAT2IDX = {c: i for i, c in enumerate(CATEGORY_LIST)}
NUM_CATEGORIES = len(CATEGORY_LIST)

MAMS_CAT2IDX = {c: i for i, c in enumerate(MAMS_CATEGORY_LIST)}
MAMS_NUM_CATEGORIES = len(MAMS_CATEGORY_LIST)

CATEGORY_CONFIGS = {
    "semeval2014": (CATEGORY_LIST, CAT2IDX, NUM_CATEGORIES),
    "mams": (MAMS_CATEGORY_LIST, MAMS_CAT2IDX, MAMS_NUM_CATEGORIES),
}

POL2ID = {"positive": 0, "negative": 1, "neutral": 2}


def get_category_config(name: str = "semeval2014"):
    return CATEGORY_CONFIGS[name]


def build_category_records(parsed: list[dict], split: str,
                           category_list: list[str] | None = None) -> list[dict]:
    if category_list is None:
        category_list = CATEGORY_LIST
    cat2idx = {c: i for i, c in enumerate(category_list)}
    num_cats = len(category_list)
    records = []
    for sent in parsed:
        cats = set()
        for op in sent["opinions"]:
            if op["category"] in cat2idx:
                cats.add(op["category"])
        if not cats:
            continue
        vec = [0] * num_cats
        for c in cats:
            vec[cat2idx[c]] = 1
        records.append({
            "sentence_id": sent["sentence_id"],
            "sentence": sent["text"],
            "categories": sorted(cats),
            "category_vector": vec,
            "split": split,
        })
    return records


def build_sentiment_records(parsed: list[dict], split: str,
                            category_list: list[str] | None = None) -> list[dict]:
    if category_list is None:
        category_list = CATEGORY_LIST
    cat2idx = {c: i for i, c in enumerate(category_list)}
    records = []
    for sent in parsed:
        for op in sent["opinions"]:
            if op["category"] not in cat2idx:
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
