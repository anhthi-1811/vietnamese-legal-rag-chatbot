import os
import sys
import json
import ast
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def parse_cids(val):
    if isinstance(val, (list, np.ndarray)): return [str(x) for x in val]
    if isinstance(val, str):
        try: return [str(x) for x in ast.literal_eval(val)]
        except: return [str(val)]
    return [str(val)]

def calculate_recall(golds_list, preds_list):
    hits = 0
    for golds, preds in zip(golds_list, preds_list):
        if len(set(golds).intersection(set(preds))) > 0:
            hits += 1
    return hits / len(golds_list)

def main():
    print("\n=== CANDIDATE POOL COVERAGE ANALYSIS ===")
    
    depths = [20, 50, 100]
    records = []
    
    for depth in depths:
        print(f"\nProcessing Top-{depth}...")
        
        # Determine file suffix
        suffix = "" if depth == 20 else f"_top{depth}"
        bm25_path = os.path.join(PROJECT_ROOT, f"experiments/results/bm25_train_rankings{suffix}.parquet")
        bge_path = os.path.join(PROJECT_ROOT, f"experiments/results/bge_m3_train_rankings{suffix}.parquet")
        
        if not os.path.exists(bm25_path) or not os.path.exists(bge_path):
            print(f"  -> Missing files for depth {depth}. Skipping.")
            continue
            
        df_bm25 = pd.read_parquet(bm25_path)
        df_bge = pd.read_parquet(bge_path)
        
        # Preprocess
        golds = [set(parse_cids(x)) for x in df_bm25['gold_cids'].tolist()]
        bm25_preds = [parse_cids(x)[:depth] for x in df_bm25['retrieved_cids'].tolist()]
        bge_preds = [parse_cids(x)[:depth] for x in df_bge['retrieved_cids'].tolist()]
        
        # Calculate individual recalls
        bm25_recall = calculate_recall(golds, bm25_preds)
        bge_recall = calculate_recall(golds, bge_preds)
        
        # Build Union Pools
        union_recalls = []
        pool_sizes = []
        bge_has_gold_count = 0
        bm25_adds_gold_count = 0
        neither_count = 0
        
        for i in range(len(golds)):
            g = golds[i]
            p_bm25 = set(bm25_preds[i])
            p_bge = set(bge_preds[i])
            p_union = p_bm25.union(p_bge)
            
            pool_sizes.append(len(p_union))
            
            # Recall tracking
            bge_hit = len(g.intersection(p_bge)) > 0
            bm25_hit = len(g.intersection(p_bm25)) > 0
            union_hit = len(g.intersection(p_union)) > 0
            
            union_recalls.append(1 if union_hit else 0)
            
            if bge_hit:
                bge_has_gold_count += 1
            elif bm25_hit and not bge_hit:
                bm25_adds_gold_count += 1
            else:
                neither_count += 1
                
        # Aggregate statistics
        total = len(golds)
        record = {
            "Depth": depth,
            "BM25 Recall": bm25_recall,
            "BGE Recall": bge_recall,
            "Union Recall": np.mean(union_recalls),
            "Avg Pool Size": np.mean(pool_sizes),
            "Median Pool Size": np.median(pool_sizes),
            "P95 Pool Size": np.percentile(pool_sizes, 95),
            "Max Pool Size": np.max(pool_sizes),
            "BGE Had Gold %": (bge_has_gold_count / total) * 100,
            "BM25 Added Gold %": (bm25_adds_gold_count / total) * 100,
            "Neither Had Gold %": (neither_count / total) * 100
        }
        records.append(record)
    
    # Save results
    df_results = pd.DataFrame(records)
    out_parquet = os.path.join(PROJECT_ROOT, "experiments/results/candidate_pool_analysis.parquet")
    out_json = os.path.join(PROJECT_ROOT, "experiments/results/candidate_pool_summary.json")
    
    df_results.to_parquet(out_parquet, index=False)
    df_results.to_json(out_json, orient="records", indent=4)
    
    # Print Table
    print("\n=== CANDIDATE POOL SUMMARY TABLE ===")
    header = f"{'Depth':<7} | {'BM25 Recall':<11} | {'BGE Recall':<10} | {'Union Recall':<12} | {'Avg Pool Size':<13} | {'Max Pool'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['Depth']:<7} | {r['BM25 Recall']:<11.4f} | {r['BGE Recall']:<10.4f} | {r['Union Recall']:<12.4f} | {r['Avg Pool Size']:<13.1f} | {r['Max Pool Size']}")
    print("-" * len(header))
    
    print("\n=== COMPLEMENTARITY INSIGHTS ===")
    for r in records:
        print(f"Depth {r['Depth']}:")
        print(f"  - BGE already had gold:      {r['BGE Had Gold %']:.2f}%")
        print(f"  - BM25 added missing gold:   {r['BM25 Added Gold %']:.2f}%")
        print(f"  - Neither had gold:          {r['Neither Had Gold %']:.2f}%")

if __name__ == "__main__":
    main()
