from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score


def extract_spans(bio_seq: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, tag in enumerate(bio_seq):
        if tag == 1:  # B-ASP
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == 2:  # I-ASP
            if start is None:
                start = i
        else:  # O
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(bio_seq)))
    return spans


def bio_token_metrics(pred_seqs: list[list[int]],
                      gold_seqs: list[list[int]]) -> dict:
    all_pred = []
    all_gold = []
    for pred, gold in zip(pred_seqs, gold_seqs):
        for p, g in zip(pred, gold):
            if g == -100:
                continue
            all_pred.append(p)
            all_gold.append(g)

    if not all_pred or not any(g in (1, 2) for g in all_gold):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    p, r, f, _ = precision_recall_fscore_support(
        all_gold, all_pred, labels=[1, 2], average="micro", zero_division=0,
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def span_f1(pred_spans_list: list[list[tuple]],
            gold_spans_list: list[list[tuple]]) -> dict:
    tp = 0
    total_pred = 0
    total_gold = 0
    for i, (preds, golds) in enumerate(zip(pred_spans_list, gold_spans_list)):
        pred_set = set((i, s, e) for s, e in preds)
        gold_set = set((i, s, e) for s, e in golds)
        tp += len(pred_set & gold_set)
        total_pred += len(pred_set)
        total_gold += len(gold_set)

    if total_pred + total_gold == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gold if total_gold else 0.0
    f = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f}


def sentiment_metrics(preds: list[int], golds: list[int]) -> dict:
    acc = accuracy_score(golds, preds)
    macro_f = f1_score(golds, preds, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(macro_f)}


def joint_f1(pred_with_pol: list[list[tuple]],
             gold_with_pol: list[list[tuple]]) -> float:
    tp = 0
    total_pred = 0
    total_gold = 0
    for i, (preds, golds) in enumerate(zip(pred_with_pol, gold_with_pol)):
        pred_set = set((i, s, e, p) for s, e, p in preds)
        gold_set = set((i, s, e, p) for s, e, p in golds)
        tp += len(pred_set & gold_set)
        total_pred += len(pred_set)
        total_gold += len(gold_set)

    if total_pred + total_gold == 0:
        return 0.0
    precision = tp / total_pred if total_pred else 0.0
    recall = tp / total_gold if total_gold else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
