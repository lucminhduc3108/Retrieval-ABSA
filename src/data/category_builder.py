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

ENTITY_LIST = ["AMBIENCE", "DRINKS", "FOOD", "LOCATION", "RESTAURANT", "SERVICE"]
ENT2IDX = {e: i for i, e in enumerate(ENTITY_LIST)}
NUM_ENTITIES = len(ENTITY_LIST)

ENTITY2ATTRS = {
    "AMBIENCE": ["GENERAL"],
    "DRINKS": ["PRICES", "QUALITY", "STYLE_OPTIONS"],
    "FOOD": ["PRICES", "QUALITY", "STYLE_OPTIONS"],
    "LOCATION": ["GENERAL"],
    "RESTAURANT": ["GENERAL", "MISCELLANEOUS", "PRICES"],
    "SERVICE": ["GENERAL"],
}
MULTI_ATTR_ENTITIES = ["DRINKS", "FOOD", "RESTAURANT"]
ATTR2IDX = {ent: {a: i for i, a in enumerate(attrs)}
            for ent, attrs in ENTITY2ATTRS.items()}

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
        entity_vec = [0] * NUM_ENTITIES
        attr_vecs = {ent: [0] * len(ENTITY2ATTRS[ent])
                     for ent in MULTI_ATTR_ENTITIES}
        for c in cats:
            entity, attribute = c.split("#")
            entity_vec[ENT2IDX[entity]] = 1
            if entity in MULTI_ATTR_ENTITIES:
                attr_vecs[entity][ATTR2IDX[entity][attribute]] = 1
        records.append({
            "sentence_id": sent["sentence_id"],
            "sentence": sent["text"],
            "categories": sorted(cats),
            "category_vector": vec,
            "entity_vector": entity_vec,
            "food_attr_vector": attr_vecs["FOOD"],
            "drinks_attr_vector": attr_vecs["DRINKS"],
            "restaurant_attr_vector": attr_vecs["RESTAURANT"],
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
