from src.data.cls_builder import build_cls_records


def test_one_record_per_opinion_including_null():
    parsed = [{"sentence_id": "s1", "text": "Great food and fast service",
               "opinions": [
                   {"target": "food", "category": "FOOD#QUALITY", "polarity": "positive",
                    "from_char": 6, "to_char": 10},
                   {"target": None, "category": "SERVICE#GENERAL", "polarity": "positive",
                    "from_char": 0, "to_char": 0},
               ]}]
    recs = build_cls_records(parsed, split="train")
    assert len(recs) == 2
    assert recs[0]["aspect_category"] == "FOOD#QUALITY"
    assert recs[1]["aspect_category"] == "SERVICE#GENERAL"
    assert all(r["split"] == "train" for r in recs)


def test_sentence_with_no_opinions_skipped():
    parsed = [{"sentence_id": "s2", "text": "We went on Tuesday", "opinions": []}]
    assert build_cls_records(parsed, split="train") == []


def test_record_has_expected_fields():
    parsed = [{"sentence_id": "s1", "text": "Great food",
               "opinions": [{"target": "food", "category": "FOOD#QUALITY",
                             "polarity": "positive", "from_char": 6, "to_char": 10}]}]
    rec = build_cls_records(parsed, split="test")[0]
    assert set(rec.keys()) == {"id", "sentence", "aspect_category", "polarity", "split"}
