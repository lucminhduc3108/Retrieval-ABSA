import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.absa.category_dataset import CategoryDataset
from src.absa.category_model import CategoryDetector
from sklearn.model_selection import train_test_split
from src.absa.category_trainer import (
    _apply_global_threshold, _apply_thresholds,
    _tune_global_threshold, _tune_thresholds,
    tune_topk, apply_topk,
)
from src.absa.sentiment_dataset import SentimentDataset
from src.absa.sentiment_model import SentimentPredictor
from src.data.category_builder import (
    CATEGORY_LIST, CAT2IDX, NUM_CATEGORIES, POL2ID,
)
from src.embedding.model import ContrastiveEmbedder
from src.evaluation.category_metrics import (
    category_f1, per_category_f1,
    joint_category_sentiment_f1,
    sentiment_acc_given_correct_category,
)
from src.retrieval.index import load_index
from src.retrieval.retriever import Retriever
from src.utils.io import load_yaml, read_jsonl
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ID2POL = {v: k for k, v in POL2ID.items()}

STRATEGIES = ["per_category", "global", "topk"]


def collect_logits(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size)
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            all_logits.append(out["logits"].cpu())
    return torch.cat(all_logits, dim=0)


def decode_categories(logits, strategy, val_logits, val_labels):
    if strategy == "per_category":
        thresholds = _tune_thresholds(val_logits, val_labels)
        pred_cats = _apply_thresholds(logits, thresholds)
        info = f"per-cat thresholds: {[f'{t:.2f}' for t in thresholds]}"
    elif strategy == "global":
        threshold = _tune_global_threshold(val_logits, val_labels)
        pred_cats = _apply_global_threshold(logits, threshold)
        info = f"global threshold: {threshold:.2f}"
    elif strategy == "topk":
        k = tune_topk(val_logits, val_labels)
        pred_cats = apply_topk(logits, k)
        info = f"k={k}"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return pred_cats, info


def predict_sentiment(model, records, retriever, embedding_model,
                      tokenizer_name, max_length, top_k, device,
                      use_retrieval, batch_size=16):
    ds = SentimentDataset(
        records, retriever=retriever,
        tokenizer_name=tokenizer_name,
        embedding_model=embedding_model,
        max_length=max_length,
        top_k=top_k, device="cpu",
        use_retrieval=use_retrieval,
    )
    loader = DataLoader(ds, batch_size=batch_size)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            out = model(
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
                neighbor_polarities=batch_gpu.get("neighbor_polarities"),
                neighbor_scores=batch_gpu.get("neighbor_scores"),
            )
            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
    return all_preds


def build_gold_pairs(cat_records, sent_records):
    gold_by_sent = {}
    for r in sent_records:
        sid = r.get("sentence_id", r.get("id", "").rsplit("_", 1)[0])
        sent = r["sentence"]
        if sent not in gold_by_sent:
            gold_by_sent[sent] = set()
        gold_by_sent[sent].add((r["category"], r["polarity"]))

    gold_cats_list = []
    gold_pairs_list = []
    sentences = []
    for cr in cat_records:
        sent = cr["sentence"]
        sentences.append(sent)
        gold_cats_list.append(set(cr["categories"]))
        gold_pairs_list.append(gold_by_sent.get(sent, set()))

    return sentences, gold_cats_list, gold_pairs_list


def run_joint_eval(pred_cats_list, test_cat, gold_cats_list, gold_pairs_list,
                   s2_model, retriever, embedding_model, s2_cfg, ret_cfg,
                   use_retrieval, device):
    stage2_records = []
    for cr, pred_cats in zip(test_cat, pred_cats_list):
        for cat in sorted(pred_cats):
            stage2_records.append({
                "id": f"{cr['sentence_id']}_{cat}",
                "sentence": cr["sentence"],
                "category": cat,
                "polarity": "positive",
                "split": "test",
            })

    if stage2_records:
        sent_preds = predict_sentiment(
            s2_model, stage2_records, retriever, embedding_model,
            tokenizer_name=s2_cfg["model_name"],
            max_length=s2_cfg["max_seq_length"],
            top_k=ret_cfg.get("top_k", 0) if use_retrieval else 0,
            device=device, use_retrieval=use_retrieval,
        )
        for rec, pred_idx in zip(stage2_records, sent_preds):
            rec["predicted_polarity"] = ID2POL[pred_idx]

    pred_pairs_list = []
    rec_idx = 0
    for pred_cats in pred_cats_list:
        pairs = set()
        for cat in sorted(pred_cats):
            if rec_idx < len(stage2_records):
                pol = stage2_records[rec_idx].get("predicted_polarity", "positive")
                pairs.add((cat, pol))
                rec_idx += 1
        pred_pairs_list.append(pairs)

    cat_m = category_f1(pred_cats_list, gold_cats_list)
    joint_m = joint_category_sentiment_f1(pred_pairs_list, gold_pairs_list)
    sent_cond = sentiment_acc_given_correct_category(pred_pairs_list, gold_pairs_list)
    per_cat = per_category_f1(pred_cats_list, gold_cats_list, CATEGORY_LIST)
    return cat_m, joint_m, sent_cond, per_cat


def format_report(cat_m, joint_m, sent_cond, per_cat, strategy=None):
    report = []
    header = "# Joint Evaluation Results"
    if strategy:
        header += f" ({strategy})"
    report.append(header + "\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Category P | {cat_m['precision']:.4f} |")
    report.append(f"| Category R | {cat_m['recall']:.4f} |")
    report.append(f"| Category F1 | {cat_m['f1']:.4f} |")
    report.append(f"| Joint P | {joint_m['precision']:.4f} |")
    report.append(f"| Joint R | {joint_m['recall']:.4f} |")
    report.append(f"| Joint F1 | {joint_m['f1']:.4f} |")
    report.append(f"| Sent Acc|Correct Cat | {sent_cond['accuracy']:.4f} ({sent_cond['correct']}/{sent_cond['total']}) |")
    report.append("")
    report.append("## Per-Category F1\n")
    report.append("| Category | P | R | F1 | Support |")
    report.append("|----------|---|---|----|----|")
    for cat in CATEGORY_LIST:
        m = per_cat[cat]
        report.append(f"| {cat} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    return "\n".join(report)


def log_sigmoid_stats(logits, label, threshold_info=""):
    probs = torch.sigmoid(logits)
    logger.info("[%s] Sigmoid stats — mean=%.4f, std=%.4f, p50=%.4f, p90=%.4f, p95=%.4f %s",
                label,
                probs.mean().item(), probs.std().item(),
                probs.quantile(0.5).item(), probs.quantile(0.9).item(),
                probs.quantile(0.95).item(), threshold_info)
    for j, cat in enumerate(CATEGORY_LIST):
        col = probs[:, j]
        logger.info("  %s: mean=%.4f, std=%.4f, max=%.4f",
                     cat, col.mean().item(), col.std().item(), col.max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt", required=True)
    parser.add_argument("--stage2_ckpt", required=True)
    parser.add_argument("--stage1_config", default="configs/stage1.yaml")
    parser.add_argument("--stage2_config", default="configs/stage2.yaml")
    parser.add_argument("--embedding_ckpt", default=None)
    parser.add_argument("--index_dir", default="indexes/")
    parser.add_argument("--retrieval_config", default="configs/retrieval_v2.yaml")
    parser.add_argument("--no_retrieval", action="store_true")
    parser.add_argument("--pred_strategy", default="all",
                        choices=["all"] + STRATEGIES)
    args = parser.parse_args()

    s1_cfg = load_yaml(args.stage1_config)
    s2_cfg = load_yaml(args.stage2_config)
    use_retrieval = s2_cfg.get("use_retrieval", True) and not args.no_retrieval
    ret_cfg = load_yaml(args.retrieval_config) if use_retrieval else {"top_k": 0, "threshold": 0.0}
    set_seed(s1_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s, retrieval: %s", device, use_retrieval)

    # --- Load Stage 1 ---
    s1_ckpt = torch.load(args.stage1_ckpt, map_location=device)
    s1_model = CategoryDetector(
        model_name=s1_cfg["model_name"],
        num_categories=s1_cfg["num_categories"],
    ).to(device)
    s1_model.load_state_dict(s1_ckpt["model_state"], strict=False)
    ckpt_threshold = s1_ckpt.get("threshold")
    if ckpt_threshold is not None:
        logger.info("Checkpoint global threshold: %.2f", ckpt_threshold)

    # --- Reconstruct val split (must match training script) ---
    train_records = [r for r in read_jsonl(s1_cfg["category_path"])
                     if r["split"] == "train"]
    stratify_key = [sum(r["category_vector"]) for r in train_records]
    _, val_records = train_test_split(
        train_records, test_size=s1_cfg["val_ratio"],
        random_state=s1_cfg["seed"],
        stratify=stratify_key)
    val_ds = CategoryDataset(val_records, tokenizer_name=s1_cfg["model_name"],
                             max_length=s1_cfg["max_seq_length"])
    val_loader = DataLoader(val_ds, batch_size=32)
    val_logits = collect_logits(s1_model, val_ds, device)
    val_labels = torch.stack([torch.tensor(r["category_vector"], dtype=torch.float32)
                              for r in val_records])
    logger.info("Val set: %d samples", len(val_records))

    # Consistency check: re-tuned threshold should match checkpoint
    retune_threshold = _tune_global_threshold(val_logits, val_labels)
    logger.info("Re-tuned global threshold: %.2f", retune_threshold)
    if ckpt_threshold is not None and abs(retune_threshold - ckpt_threshold) > 0.05:
        logger.warning(
            "Re-tuned threshold (%.2f) differs from checkpoint (%.2f) — possible split mismatch",
            retune_threshold, ckpt_threshold)

    log_sigmoid_stats(val_logits, "val")

    # --- Load Stage 2 + retrieval ---
    embedding_model = None
    retriever = None
    if use_retrieval:
        if not args.embedding_ckpt:
            parser.error("--embedding_ckpt required for retrieval")
        embedding_model = ContrastiveEmbedder(
            model_name=s2_cfg["model_name"], proj_dim=256)
        embedding_model.load_state_dict(
            torch.load(args.embedding_ckpt, map_location="cpu"))
        embedding_model.eval()
        logger.info("Loaded embedding model")

        index, metadata, _ = load_index(args.index_dir)
        retriever = Retriever(index, metadata,
                              top_k=ret_cfg["top_k"],
                              threshold=ret_cfg["threshold"])
        logger.info("Loaded FAISS index (%d vectors)", index.ntotal)

    s2_model = SentimentPredictor(
        model_name=s2_cfg["model_name"],
        num_sent_labels=s2_cfg["num_sent_labels"],
        embed_dim=s2_cfg.get("embed_dim", 64),
        tau=s2_cfg.get("tau", 0.05),
        dropout=s2_cfg.get("dropout", 0.1),
        use_retrieval=use_retrieval,
    ).to(device)
    s2_state = torch.load(args.stage2_ckpt, map_location=device)
    s2_model.load_state_dict(s2_state, strict=False)
    logger.info("Stage 2 loaded")

    # --- Collect test data and logits ---
    cat_records = read_jsonl(s1_cfg["category_path"])
    test_cat = [r for r in cat_records if r["split"] == "test"]
    sent_records = read_jsonl(s2_cfg["sentiment_path"])
    test_sent = [r for r in sent_records if r["split"] == "test"]

    cat_ds = CategoryDataset(test_cat, tokenizer_name=s1_cfg["model_name"],
                             max_length=s1_cfg["max_seq_length"])
    test_logits = collect_logits(s1_model, cat_ds, device)
    log_sigmoid_stats(test_logits, "test")

    sentences, gold_cats_list, gold_pairs_list = build_gold_pairs(
        test_cat, test_sent)

    # --- Run strategies ---
    strategies = STRATEGIES if args.pred_strategy == "all" else [args.pred_strategy]
    all_results = {}
    all_reports = []

    for strat in strategies:
        logger.info("--- Strategy: %s ---", strat)
        pred_cats, info = decode_categories(
            test_logits, strat, val_logits, val_labels)
        avg_preds = sum(len(s) for s in pred_cats) / max(len(pred_cats), 1)
        logger.info("[%s] %s, avg predicted cats/sentence: %.2f", strat, info, avg_preds)

        cat_m, joint_m, sent_cond, per_cat = run_joint_eval(
            pred_cats, test_cat, gold_cats_list, gold_pairs_list,
            s2_model, retriever, embedding_model, s2_cfg, ret_cfg,
            use_retrieval, device)

        all_results[strat] = {
            "cat_f1": cat_m["f1"], "cat_p": cat_m["precision"], "cat_r": cat_m["recall"],
            "joint_f1": joint_m["f1"], "joint_p": joint_m["precision"], "joint_r": joint_m["recall"],
            "sent_acc": sent_cond["accuracy"],
            "sent_correct": sent_cond["correct"], "sent_total": sent_cond["total"],
            "avg_preds": avg_preds, "info": info,
        }
        report = format_report(cat_m, joint_m, sent_cond, per_cat, strategy=strat)
        all_reports.append(report)
        logger.info("[%s] Cat F1=%.4f, Joint F1=%.4f, Sent Acc|CC=%.4f",
                    strat, cat_m["f1"], joint_m["f1"], sent_cond["accuracy"])

    # --- Print comparison table if multiple strategies ---
    if len(strategies) > 1:
        comp = ["\n# Strategy Comparison\n"]
        comp.append("| Strategy | Cat P | Cat R | Cat F1 | Joint F1 | Sent Acc|CC | Avg Preds | Config |")
        comp.append("|----------|-------|-------|--------|----------|-----------|-----------|--------|")
        for strat in strategies:
            r = all_results[strat]
            comp.append(f"| {strat} | {r['cat_p']:.4f} | {r['cat_r']:.4f} | {r['cat_f1']:.4f} "
                        f"| {r['joint_f1']:.4f} | {r['sent_acc']:.4f} ({r['sent_correct']}/{r['sent_total']}) "
                        f"| {r['avg_preds']:.2f} | {r['info']} |")
        comp_text = "\n".join(comp)
        print(comp_text)
        all_reports.insert(0, comp_text)

    for report in all_reports:
        if not report.startswith("\n#"):
            print("\n" + report)

    tag = "noret" if args.no_retrieval else "retrieval"
    out_path = f"logs/joint_eval_{tag}.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n\n".join(all_reports))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
