import os
import sys
import ast
import pandas as pd
import numpy as np  

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.retrieval.candidate_pool import build_candidate_pool

def parse_cids(val):
    if isinstance(val, (list, np.ndarray)): return [str(x) for x in val]
    if isinstance(val, str):
        try: return [str(x) for x in ast.literal_eval(val)]
        except: return [str(val)]
    return [str(val)]

def parse_scores(val):
    if isinstance(val, (list, np.ndarray)): return [float(x) for x in val] # <-- Bắt thêm np.ndarray
    if isinstance(val, str):
        try: return [float(x) for x in ast.literal_eval(val)]
        except: return [float(val)]
    return [float(val)]

def main():
    print("\n=== BUILDING RERANK CANDIDATE POOL (Top-50) ===")
    
    bm25_path = os.path.join(PROJECT_ROOT, "experiments/results/bm25_train_rankings_top50.parquet")
    bge_path = os.path.join(PROJECT_ROOT, "experiments/results/bge_m3_train_rankings_top50.parquet")
    
    print(f"Loading BM25 candidates from: {bm25_path}")
    print(f"Loading BGE-M3 candidates from: {bge_path}")
    
    df_bm25 = pd.read_parquet(bm25_path)
    df_bge = pd.read_parquet(bge_path)
    
    assert len(df_bm25) == len(df_bge), "Row count mismatch between BM25 and BGE datasets."
    
    qids = df_bm25['qid'].tolist()
    questions = df_bm25['question'].tolist()
    
    print("Parsing candidates...")
    golds = [parse_cids(x) for x in df_bm25['gold_cids'].tolist()]
    bm25_cids_list = [parse_cids(x) for x in df_bm25['retrieved_cids'].tolist()]
    bm25_scores_list = [parse_scores(x) for x in df_bm25['retrieved_scores'].tolist()]
    
    bge_cids_list = [parse_cids(x) for x in df_bge['retrieved_cids'].tolist()]
    bge_scores_list = [parse_scores(x) for x in df_bge['retrieved_scores'].tolist()]
    
    print(f"Merging candidate pools for {len(qids):,} queries...")
    results = []
    
    for i in range(len(qids)):
        pool = build_candidate_pool(
            bm25_cids_list[i], bm25_scores_list[i],
            bge_cids_list[i], bge_scores_list[i]
        )
        
        results.append({
            'qid': qids[i],
            'question': questions[i],
            'gold_cids': golds[i],
            'candidate_cids': [d['cid'] for d in pool],
            'bm25_ranks': [d['bm25_rank'] for d in pool],
            'bge_ranks': [d['bge_rank'] for d in pool],
            'bm25_scores': [d['bm25_score'] for d in pool],
            'bge_scores': [d['bge_score'] for d in pool]
        })
        
    df_out = pd.DataFrame(results)
    
    out_path = os.path.join(PROJECT_ROOT, "experiments/results/rerank_candidates_train.parquet")
    df_out.to_parquet(out_path, index=False)
    
    print(f"\n-> Successfully merged pools for {len(df_out):,} queries.")
    avg_pool_size = df_out['candidate_cids'].apply(len).mean()
    print(f"-> Average Candidate Pool Size: {avg_pool_size:.1f} documents/query.")
    print(f"-> Saved dataset to: {out_path}")

if __name__ == "__main__":
    main()