from sklearn.metrics import accuracy_score, f1_score, recall_score


def category_f1(pred_cats_list: list[set[str]],
                gold_cats_list: list[set[str]]) -> dict:
    tp = fp = fn = 0
    for preds, golds in zip(pred_cats_list, gold_cats_list):
        tp += len(preds & golds)
        fp += len(preds - golds)
        fn += len(golds - preds)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def per_category_f1(pred_cats_list: list[set[str]],
                    gold_cats_list: list[set[str]],
                    category_list: list[str]) -> dict[str, dict]:
    results = {}
    for cat in category_list:
        tp = fp = fn = 0
        for preds, golds in zip(pred_cats_list, gold_cats_list):
            if cat in preds and cat in golds:
                tp += 1
            elif cat in preds:
                fp += 1
            elif cat in golds:
                fn += 1
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        results[cat] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
    return results


def joint_category_sentiment_f1(
    pred_pairs_list: list[set[tuple[str, str]]],
    gold_pairs_list: list[set[tuple[str, str]]],
) -> dict:
    tp = fp = fn = 0
    for preds, golds in zip(pred_pairs_list, gold_pairs_list):
        tp += len(preds & golds)
        fp += len(preds - golds)
        fn += len(golds - preds)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def sentiment_acc_given_correct_category(
    pred_pairs_list: list[set[tuple[str, str]]],
    gold_pairs_list: list[set[tuple[str, str]]],
) -> dict:
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    for pred_pairs, gold_pairs in zip(pred_pairs_list, gold_pairs_list):
        gold_by_cat: dict[str, set[str]] = {}
        for gc, gp in gold_pairs:
            gold_by_cat.setdefault(gc, set()).add(gp)
        for cat, pol in pred_pairs:
            if cat in gold_by_cat:
                total += 1
                gold_pols = gold_by_cat[cat]
                gold_pol = list(gold_pols)[0]
                y_true.append(gold_pol)
                y_pred.append(pol)
                if pol in gold_pols:
                    correct += 1
    acc = correct / total if total else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) if total else 0.0
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0) if total else 0.0
    return {"accuracy": acc, "macro_f1": float(macro_f1), "macro_recall": float(macro_recall), 
            "correct": correct, "total": total}
