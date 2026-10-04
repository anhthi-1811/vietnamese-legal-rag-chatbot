import os
import sys
import json
import numpy as np
import pandas as pd
import ast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.evaluation.complementarity import analyze_complementarity_at_k

def parse_cids(val):
    if isinstance(val, (list, np.ndarray)): return [str(x) for x in val]
    if isinstance(val, str):
        try: return [str(x) for x in ast.literal_eval(val)]
        except: return [str(val)]
    return [str(val)]

def main():
    print("\n=== ANALYZING RETRIEVER COMPLEMENTARITY ===")
    
    bm25_path = os.path.join(PROJECT_ROOT, "experiments/results/bm25_train_rankings.parquet")
    bge_path = os.path.join(PROJECT_ROOT, "experiments/results/bge_m3_train_rankings.parquet")
    
    print(f"Loading BM25: {bm25_path}")
    print(f"Loading BGE: {bge_path}")
    
    df_bm25 = pd.read_parquet(bm25_path)
    df_bge = pd.read_parquet(bge_path)
    
    # Merge
    df = pd.merge(df_bm25, df_bge, on=['qid', 'question'], suffixes=('_bm25', '_bge'))
    
    # Preprocess CIDs
    golds = [set(parse_cids(x)) for x in df['gold_cids_bm25'].tolist()]
    preds_bm25 = [parse_cids(x) for x in df['retrieved_cids_bm25'].tolist()]
    preds_bge = [parse_cids(x) for x in df['retrieved_cids_bge'].tolist()]
    
    k_values = [1, 5, 10, 20]
    report = {}
    
    for k in k_values:
        counts, percentages, categories = analyze_complementarity_at_k(golds, preds_bm25, preds_bge, k)
        report[f"Top-{k}"] = {
            "counts": counts,
            "percentages": percentages
        }
        df[f'category_k{k}'] = categories
        
        print(f"\nTop-{k} Complementarity")
        print(f"  Both correct:       {counts['BOTH_CORRECT']} ({percentages['BOTH_CORRECT']}%)")
        print(f"  BM25 only correct:  {counts['A_ONLY_CORRECT']} ({percentages['A_ONLY_CORRECT']}%)")
        print(f"  BGE only correct:   {counts['B_ONLY_CORRECT']} ({percentages['B_ONLY_CORRECT']}%)")
        print(f"  Both wrong:         {counts['BOTH_WRONG']} ({percentages['BOTH_WRONG']}%)")
        
    save_path = os.path.join(PROJECT_ROOT, "experiments/results/retriever_complementarity.json")
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\nSaved report to: {save_path}")

    # Extract examples for Top-10
    print("\nRepresentative Examples (Top-10):")
    for cat in ["A_ONLY_CORRECT", "B_ONLY_CORRECT", "BOTH_WRONG"]:
        sample = df[df['category_k10'] == cat].head(1)
        if not sample.empty:
            q = sample['question'].values[0]
            print(f"[{cat}] Q: {q}")

if __name__ == "__main__":
    main()
