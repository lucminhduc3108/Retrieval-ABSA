import pytest

from src.evaluation.category_metrics import (
    category_f1,
    per_category_f1,
    joint_category_sentiment_f1,
    sentiment_acc_given_correct_category,
)


def test_category_f1_perfect():
    pred = [{"FOOD#QUALITY", "SERVICE#GENERAL"}]
    gold = [{"FOOD#QUALITY", "SERVICE#GENERAL"}]
    m = category_f1(pred, gold)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0


def test_category_f1_partial():
    pred = [{"FOOD#QUALITY", "DRINKS#PRICES"}]
    gold = [{"FOOD#QUALITY", "SERVICE#GENERAL"}]
    m = category_f1(pred, gold)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5


def test_category_f1_empty_pred():
    pred = [set()]
    gold = [{"FOOD#QUALITY"}]
    m = category_f1(pred, gold)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_category_f1_multi_sentence():
    pred = [{"FOOD#QUALITY"}, {"SERVICE#GENERAL", "AMBIENCE#GENERAL"}]
    gold = [{"FOOD#QUALITY"}, {"SERVICE#GENERAL"}]
    m = category_f1(pred, gold)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == 1.0


def test_per_category_f1():
    cats = ["FOOD#QUALITY", "SERVICE#GENERAL"]
    pred = [{"FOOD#QUALITY"}, {"FOOD#QUALITY", "SERVICE#GENERAL"}]
    gold = [{"FOOD#QUALITY", "SERVICE#GENERAL"}, {"SERVICE#GENERAL"}]
    result = per_category_f1(pred, gold, cats)
    assert result["FOOD#QUALITY"]["support"] == 1
    assert result["SERVICE#GENERAL"]["support"] == 2
    assert result["SERVICE#GENERAL"]["recall"] == 0.5


def test_joint_f1_perfect():
    pred = [{("FOOD#QUALITY", "positive"), ("SERVICE#GENERAL", "negative")}]
    gold = [{("FOOD#QUALITY", "positive"), ("SERVICE#GENERAL", "negative")}]
    m = joint_category_sentiment_f1(pred, gold)
    assert m["f1"] == 1.0


def test_joint_f1_wrong_polarity():
    pred = [{("FOOD#QUALITY", "positive")}]
    gold = [{("FOOD#QUALITY", "negative")}]
    m = joint_category_sentiment_f1(pred, gold)
    assert m["f1"] == 0.0


def test_joint_f1_conflict_gold():
    pred = [{("SERVICE#GENERAL", "negative")}]
    gold = [{("SERVICE#GENERAL", "negative"), ("SERVICE#GENERAL", "positive")}]
    m = joint_category_sentiment_f1(pred, gold)
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5


def test_sentiment_acc_given_correct_cat():
    pred_pairs = [("FOOD#QUALITY", "positive"), ("SERVICE#GENERAL", "negative")]
    gold_by_cat = {
        "FOOD#QUALITY": {"positive"},
        "SERVICE#GENERAL": {"positive"},
    }
    m = sentiment_acc_given_correct_category(pred_pairs, gold_by_cat)
    assert m["total"] == 2
    assert m["correct"] == 1
    assert m["accuracy"] == 0.5


def test_sentiment_acc_skips_wrong_category():
    pred_pairs = [("DRINKS#PRICES", "positive")]
    gold_by_cat = {"FOOD#QUALITY": {"positive"}}
    m = sentiment_acc_given_correct_category(pred_pairs, gold_by_cat)
    assert m["total"] == 0
    assert m["accuracy"] == 0.0


def test_sentiment_acc_conflict_gold_accepts_either():
    pred_pairs = [("SERVICE#GENERAL", "negative")]
    gold_by_cat = {"SERVICE#GENERAL": {"negative", "positive"}}
    m = sentiment_acc_given_correct_category(pred_pairs, gold_by_cat)
    assert m["correct"] == 1
