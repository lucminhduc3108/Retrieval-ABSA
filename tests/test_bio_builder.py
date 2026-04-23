from src.data.bio_builder import build_bio_records, build_implicit_records


def test_single_word_span():
    parsed = [{"sentence_id": "s1", "text": "The food is great",
               "opinions": [{"target": "food", "category": "FOOD#QUALITY",
                             "polarity": "positive", "from_char": 4, "to_char": 8}]}]
    recs = build_bio_records(parsed, split="train")
    assert len(recs) == 1
    assert recs[0]["tokens"] == ["The", "food", "is", "great"]
    assert recs[0]["bio_tags"] == ["O", "B-ASP", "O", "O"]
    assert recs[0]["aspect_category"] == "FOOD#QUALITY"
    assert recs[0]["polarity"] == "positive"
    assert recs[0]["split"] == "train"
    assert recs[0]["id"] == "s1_op0"


def test_multi_word_span():
    parsed = [{"sentence_id": "s2", "text": "The pad thai is nice",
               "opinions": [{"target": "pad thai", "category": "FOOD#QUALITY",
                             "polarity": "positive", "from_char": 4, "to_char": 12}]}]
    recs = build_bio_records(parsed, split="train")
    assert recs[0]["bio_tags"] == ["O", "B-ASP", "I-ASP", "O", "O"]


def test_null_target_skipped():
    parsed = [{"sentence_id": "s3", "text": "The restaurant is fine",
               "opinions": [{"target": None, "category": "RESTAURANT#GENERAL",
                             "polarity": "positive", "from_char": 0, "to_char": 0}]}]
    assert build_bio_records(parsed, split="train") == []


def test_two_opinions_same_sentence():
    parsed = [{"sentence_id": "s4", "text": "The food was lousy and the portions tiny",
               "opinions": [
                   {"target": "food", "category": "FOOD#QUALITY",
                    "polarity": "negative", "from_char": 4, "to_char": 8},
                   {"target": "portions", "category": "FOOD#STYLE_OPTIONS",
                    "polarity": "negative", "from_char": 27, "to_char": 35},
               ]}]
    recs = build_bio_records(parsed, split="train")
    assert len(recs) == 2
    assert recs[0]["bio_tags"] == ["O", "B-ASP", "O", "O", "O", "O", "O", "O"]
    assert recs[1]["bio_tags"] == ["O", "O", "O", "O", "O", "O", "B-ASP", "O"]


def test_implicit_records_for_null_target():
    parsed = [{"sentence_id": "s5", "text": "Overpriced and underwhelming",
               "opinions": [{"target": None, "category": "RESTAURANT#PRICES",
                             "polarity": "negative", "from_char": 0, "to_char": 0}]}]
    recs = build_implicit_records(parsed, split="train")
    assert len(recs) == 1
    assert recs[0]["tokens"] == ["Overpriced", "and", "underwhelming"]
    assert recs[0]["bio_tags"] == ["O", "O", "O"]
    assert recs[0]["implicit"] is True
    assert recs[0]["aspect_category"] == "RESTAURANT#PRICES"
    assert recs[0]["polarity"] == "negative"


def test_implicit_skips_explicit_targets():
    parsed = [{"sentence_id": "s6", "text": "The food is great",
               "opinions": [{"target": "food", "category": "FOOD#QUALITY",
                             "polarity": "positive", "from_char": 4, "to_char": 8}]}]
    assert build_implicit_records(parsed, split="train") == []
