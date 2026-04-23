from src.data.xml_parser import parse_semeval_xml

FIXTURE = "tests/fixtures/toy_restaurant.xml"


def test_parse_returns_four_sentences():
    out = parse_semeval_xml(FIXTURE)
    assert len(out) == 4


def test_explicit_target_has_correct_offsets():
    out = parse_semeval_xml(FIXTURE)
    explicit = next(s for s in out if s["sentence_id"] == "explicit_case")
    op = explicit["opinions"][0]
    assert op["target"] == "food"
    assert op["category"] == "FOOD#QUALITY"
    assert op["polarity"] == "positive"
    assert op["from_char"] == 4
    assert op["to_char"] == 8


def test_null_target_is_none():
    out = parse_semeval_xml(FIXTURE)
    null_sent = next(s for s in out if s["sentence_id"] == "null_case")
    assert null_sent["opinions"][0]["target"] is None


def test_conflict_is_dropped():
    out = parse_semeval_xml(FIXTURE)
    conflict_sent = next(s for s in out if s["sentence_id"] == "conflict_case")
    assert conflict_sent["opinions"] == []


def test_no_opinion_sentence_has_empty_list():
    out = parse_semeval_xml(FIXTURE)
    no_op = next(s for s in out if s["sentence_id"] == "no_opinion_case")
    assert no_op["opinions"] == []
