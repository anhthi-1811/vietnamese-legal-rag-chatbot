import os
import sys
import time
import yaml
import json
import ast
import pandas as pd
import numpy as np
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.loader import load_parquet_data
from src.retrieval.bm25 import BM25Retriever
from src.evaluation.metrics import evaluate_retrieval

def parse_cids(val):
    if isinstance(val, (list, np.ndarray)): return [str(x) for x in val]
    if isinstance(val, str):
        try: return [str(x) for x in ast.literal_eval(val)]
        except: return [str(val)]
    return [str(val)]

def main():
    parser = argparse.ArgumentParser(description="Run BM25 Retrieval and Evaluate")
    parser.add_argument("--index_dir", default=os.path.join(PROJECT_ROOT, "artifacts/bm25"))
    parser.add_argument("--data_path", default=os.path.join(PROJECT_ROOT, "data/raw/YuITC/train.parquet"))
    parser.add_argument("--results_dir", default=os.path.join(PROJECT_ROOT, "experiments/results"))
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/retrieval.yaml"))
    parser.add_argument("--top_k", type=int, default=None, help="Override top_k from config")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output files (e.g. _top50)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    top_k = args.top_k if args.top_k else config["bm25"]["top_k"]
    eval_ks = list(set(config["evaluation"]["recall_k"] + config["evaluation"]["mrr_k"] + config["evaluation"]["ndcg_k"]))
    eval_ks.sort()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"\n=== STARTING BM25 RETRIEVAL (Top-{top_k}) ===")
    
    print(f"[1/4] Loading benchmark queries...")
    df = load_parquet_data(args.data_path)
    qids = df["qid"].tolist()
    queries = df["question"].tolist()
    df["gold_cids"] = df["cid"].apply(parse_cids)
    
    print(f"[2/4] Loading BM25 Index...")
    retriever = BM25Retriever()
    retriever.load(args.index_dir)
    
    print(f"[3/4] Running batch search for {len(queries):,} queries...")
    start_time = time.time()
    batch_cids, batch_scores = retriever.batch_search(queries, k=top_k)
    search_time = time.time() - start_time
    print(f"      -> Search completed in {search_time:.2f} seconds.")

    print(f"[4/4] Evaluating and Saving Results...")
    df["retrieved_cids"] = batch_cids
    df["retrieved_scores"] = batch_scores
    
    rankings_path = os.path.join(args.results_dir, f"bm25_train_rankings{args.output_suffix}.parquet")
    df[["qid", "question", "gold_cids", "retrieved_cids", "retrieved_scores"]].to_parquet(rankings_path, index=False)
    
    qrels = dict(zip(df["qid"], df["gold_cids"]))
    results = dict(zip(df["qid"], df["retrieved_cids"]))
    
    metrics = evaluate_retrieval(qrels, results, k_values=eval_ks)
    metrics_path = os.path.join(args.results_dir, f"bm25_train_metrics{args.output_suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"      -> Saved to {rankings_path} and {metrics_path}")
    print("=== BM25 EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()
