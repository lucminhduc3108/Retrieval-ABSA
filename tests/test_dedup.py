from src.data.dedup import deduplicate_opinions


def _sent(sid, opinions):
    return {"sentence_id": sid, "text": f"sentence {sid}", "opinions": opinions}


def _op(category, polarity, target="t"):
    return {"target": target, "category": category, "polarity": polarity, "from_char": 0, "to_char": 1}


def test_no_duplicates():
    sentences = [_sent("1", [_op("FOOD#QUALITY", "positive")])]
    cleaned, stats = deduplicate_opinions(sentences)
    assert len(cleaned[0]["opinions"]) == 1
    assert stats["duplicates_removed"] == 0
    assert stats["conflicts_resolved"] == 0
    assert stats["conflicts_dropped"] == 0


def test_empty_opinions():
    sentences = [_sent("1", [])]
    cleaned, stats = deduplicate_opinions(sentences)
    assert len(cleaned[0]["opinions"]) == 0
    assert stats["duplicates_removed"] == 0


def test_exact_duplicate_removed():
    sentences = [_sent("1", [
        _op("FOOD#QUALITY", "positive", "pizza"),
        _op("FOOD#QUALITY", "positive", "pasta"),
    ])]
    cleaned, stats = deduplicate_opinions(sentences)
    assert len(cleaned[0]["opinions"]) == 1
    assert cleaned[0]["opinions"][0]["polarity"] == "positive"
    assert stats["duplicates_removed"] == 1


def test_majority_vote_resolves_conflict():
    sentences = [_sent("1", [
        _op("FOOD#QUALITY", "positive", "pizza"),
        _op("FOOD#QUALITY", "positive", "pasta"),
        _op("FOOD#QUALITY", "negative", "sauce"),
    ])]
    cleaned, stats = deduplicate_opinions(sentences)
    assert len(cleaned[0]["opinions"]) == 1
    assert cleaned[0]["opinions"][0]["polarity"] == "positive"
    assert stats["conflicts_resolved"] == 1


def test_tied_conflict_drops_all():
    sentences = [_sent("1", [
        _op("FOOD#QUALITY", "positive", "pizza"),
        _op("FOOD#QUALITY", "negative", "sauce"),
    ])]
    cleaned, stats = deduplicate_opinions(sentences)
    assert len(cleaned[0]["opinions"]) == 0
    assert stats["conflicts_dropped"] == 1


def test_mixed_categories():
    sentences = [_sent("1", [
        _op("FOOD#QUALITY", "positive", "pizza"),
        _op("FOOD#QUALITY", "positive", "pasta"),
        _op("SERVICE#GENERAL", "negative", "waiter"),
        _op("AMBIENCE#GENERAL", "positive", "decor"),
        _op("AMBIENCE#GENERAL", "negative", "noise"),
    ])]
    cleaned, stats = deduplicate_opinions(sentences)
    cats = {op["category"]: op["polarity"] for op in cleaned[0]["opinions"]}
    assert cats["FOOD#QUALITY"] == "positive"
    assert cats["SERVICE#GENERAL"] == "negative"
    assert "AMBIENCE#GENERAL" not in cats
    assert stats["duplicates_removed"] == 1 + 2  # 1 food dup + 2 ambience dropped
    assert stats["conflicts_dropped"] == 1


def test_preserves_sentence_metadata():
    sentences = [_sent("abc:1", [_op("FOOD#QUALITY", "positive")])]
    cleaned, _ = deduplicate_opinions(sentences)
    assert cleaned[0]["sentence_id"] == "abc:1"
    assert cleaned[0]["text"] == "sentence abc:1"


def test_multiple_sentences():
    sentences = [
        _sent("1", [_op("FOOD#QUALITY", "positive"), _op("FOOD#QUALITY", "positive")]),
        _sent("2", [_op("SERVICE#GENERAL", "negative")]),
    ]
    cleaned, stats = deduplicate_opinions(sentences)
    assert len(cleaned) == 2
    assert len(cleaned[0]["opinions"]) == 1
    assert len(cleaned[1]["opinions"]) == 1
    assert stats["duplicates_removed"] == 1
