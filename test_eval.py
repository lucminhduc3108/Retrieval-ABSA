import torch
import json
from src.data.category_builder import CATEGORY_LIST, NUM_CATEGORIES
from src.absa.category_model import CategoryDetector
from src.evaluation.category_metrics import category_f1

def evaluate_local():
    # Load model
    print("Loading model...")
    checkpoint = torch.load("kaggle_upload/outputs_p5_nb1/stage1_r2_best.pt", map_location="cpu", weights_only=True)
    model = CategoryDetector()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    thresholds = checkpoint.get("thresholds", [0.5] * NUM_CATEGORIES)
    print("Thresholds:", thresholds)
    
    # We will simulate a forward pass manually, or just use the trainer?
    # Actually we just want to look at a few examples from val set
    with open("data/processed/category_detection.jsonl") as f:
        records = [json.loads(line) for line in f]
    
    # Use same seed to get val set
    from sklearn.model_selection import train_test_split
    stratify_key = [",".join(sorted(r["categories"])) for r in records]
    _, val_records = train_test_split(
        records, test_size=0.2, random_state=42, stratify=stratify_key
    )
    
    print(f"Val records: {len(val_records)}")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    
    pred_cats_list = []
    gold_cats_list = []
    
    for i, r in enumerate(val_records[:20]):
        text = r["text"]
        golds = set(r["categories"])
        gold_cats_list.append(golds)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out = model(**inputs)
            logits = out["logits"][0]
            probs = torch.sigmoid(logits)
        
        preds = set()
        for j, p in enumerate(probs):
            if p.item() >= thresholds[j]:
                preds.add(CATEGORY_LIST[j])
        pred_cats_list.append(preds)
        
        print(f"Text: {text}")
        print(f"  Gold: {golds}")
        print(f"  Pred: {preds}")
        # print(f"  Probs: {probs.tolist()}")
        print()
        
    m = category_f1(pred_cats_list, gold_cats_list)
    print("Subset Metric:", m)

if __name__ == "__main__":
    evaluate_local()
