import logging

logger = logging.getLogger(__name__)


def _whitespace_tokenize(text: str) -> list[tuple[str, int, int]]:
    tokens = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        tokens.append((text[i:j], i, j))
        i = j
    return tokens


def build_bio_records(parsed: list[dict], split: str) -> list[dict]:
    records = []
    for sent in parsed:
        token_spans = _whitespace_tokenize(sent["text"])
        tokens = [t for t, _, _ in token_spans]
        for op_idx, op in enumerate(sent["opinions"]):
            if op["target"] is None:
                continue
            bio = ["O"] * len(tokens)
            from_c, to_c = op["from_char"], op["to_char"]
            first = True
            for i, (_, start, end) in enumerate(token_spans):
                if start < to_c and end > from_c:
                    bio[i] = "B-ASP" if first else "I-ASP"
                    first = False
            records.append({
                "id": f"{sent['sentence_id']}_op{op_idx}",
                "sentence": sent["text"],
                "tokens": tokens,
                "bio_tags": bio,
                "aspect_category": op["category"],
                "polarity": op["polarity"],
                "split": split,
            })
    return records


def build_implicit_records(parsed: list[dict], split: str) -> list[dict]:
    records = []
    for sent in parsed:
        token_spans = _whitespace_tokenize(sent["text"])
        tokens = [t for t, _, _ in token_spans]
        for op_idx, op in enumerate(sent["opinions"]):
            if op["target"] is not None:
                continue
            bio = ["O"] * len(tokens)
            records.append({
                "id": f"{sent['sentence_id']}_op{op_idx}",
                "sentence": sent["text"],
                "tokens": tokens,
                "bio_tags": bio,
                "aspect_category": op["category"],
                "polarity": op["polarity"],
                "split": split,
                "implicit": True,
            })
    return records
