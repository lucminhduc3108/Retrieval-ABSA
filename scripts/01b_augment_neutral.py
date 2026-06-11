import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Fix the seed for reproducibility
random.seed(42)

MAMS_XML = "data/mams/data/MAMS-ACSA/raw/train.xml"
ORIGINAL_JSONL = "data/processed/sentiment_records.jsonl"
OUTPUT_JSONL = "data/processed/sentiment_records_aug.jsonl"

# 1-to-1 Mapping for safe categories
SAFE_MAP = {
    'place': 'LOCATION#GENERAL',
    'miscellaneous': 'RESTAURANT#MISCELLANEOUS',
    'ambience': 'AMBIENCE#GENERAL',
    'service': 'SERVICE#GENERAL',
    'staff': 'SERVICE#GENERAL'
}

def augment_neutral(target_samples=650):
    print("Loading MAMS ACSA Train XML...")
    tree = ET.parse(MAMS_XML)
    root = tree.getroot()
    
    candidates = []
    
    # Extract neutral samples from safe categories
    for sentence in root.findall('sentence'):
        text_node = sentence.find('text')
        if text_node is None or text_node.text is None:
            continue
        text = text_node.text.strip()
        
        aspects_node = sentence.find('aspectCategories')
        if aspects_node is not None:
            for aspect in aspects_node:
                cat = aspect.get('category')
                pol = aspect.get('polarity')
                
                if pol == 'neutral' and cat in SAFE_MAP:
                    mapped_cat = SAFE_MAP[cat]
                    candidates.append({
                        "sentence": text,
                        "category": mapped_cat,
                        "polarity": "neutral",
                        "split": "train",
                        "source": "mams"
                    })

    print(f"Found {len(candidates)} valid neutral candidates in MAMS.")
    
    # Sample exactly `target_samples`
    if len(candidates) < target_samples:
        print(f"Warning: Only found {len(candidates)} candidates, which is less than requested {target_samples}.")
        sampled = candidates
    else:
        sampled = random.sample(candidates, target_samples)
        
    print(f"Sampled {len(sampled)} records.")
    
    # Read original SemEval records
    records = []
    with open(ORIGINAL_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    original_count = len(records)
    print(f"Loaded {original_count} original SemEval records.")
    
    # Assign IDs to new records and append
    start_id = 1
    for rec in sampled:
        rec["id"] = f"mams_neu_{start_id:04d}"
        start_id += 1
        records.append(rec)
        
    # Write to new augmented JSONL
    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Successfully saved {len(records)} records (+{len(sampled)} from MAMS) to {OUTPUT_JSONL}")

if __name__ == "__main__":
    augment_neutral(150)
