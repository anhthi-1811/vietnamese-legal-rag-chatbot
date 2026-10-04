import os
import sys
import time
import yaml
import json
import ast
import numpy as np
import pandas as pd
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.loader import load_parquet_data
from src.retrieval.dense import BGEM3Encoder
from src.retrieval.faiss_index import FAISSRetriever
from src.evaluation.metrics import evaluate_retrieval

def parse_cids(val):
    if isinstance(val, (list, np.ndarray)): return [str(x) for x in val]
    if isinstance(val, str):
        try: return [str(x) for x in ast.literal_eval(val)]
        except: return [str(val)]
    return [str(val)]

def main():
    parser = argparse.ArgumentParser(description="Run BGE-M3 Retrieval and Evaluate")
    parser.add_argument("--index_dir", default=os.path.join(PROJECT_ROOT, "artifacts/dense"))
    parser.add_argument("--data_path", default=os.path.join(PROJECT_ROOT, "data/raw/YuITC/train.parquet"))
    parser.add_argument("--results_dir", default=os.path.join(PROJECT_ROOT, "experiments/results"))
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/retrieval.yaml"))
    parser.add_argument("--top_k", type=int, default=None, help="Override top_k from config")
    parser.add_argument("--output_suffix", type=str, default="", help="Suffix for output files (e.g. _top50)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    dense_cfg = config.get("dense", {})
    top_k = args.top_k if args.top_k else dense_cfg["top_k"]
    eval_ks = list(set(config["evaluation"]["recall_k"] + config["evaluation"]["mrr_k"] + config["evaluation"]["ndcg_k"]))
    eval_ks.sort()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"\n=== STARTING BGE-M3 RETRIEVAL (Top-{top_k}) ===")
    
    print(f"[1/4] Loading benchmark queries...")
    df = load_parquet_data(args.data_path)
    qids = df["qid"].tolist()
    queries = df["question"].tolist()
    df["gold_cids"] = df["cid"].apply(parse_cids)
    
    print(f"[2/4] Loading Encoder and FAISS Index...")
    encoder = BGEM3Encoder(model_name=dense_cfg["model_name"], use_fp16=dense_cfg["use_fp16"])
    retriever = FAISSRetriever()
    retriever.load(args.index_dir)
    
    print(f"[3/4] Encoding & Searching {len(queries):,} queries...")
    query_embeddings = encoder.encode_queries(queries, batch_size=dense_cfg["batch_size"], max_length=512)
    batch_cids, batch_scores = retriever.search(query_embeddings, k=top_k)

    print(f"[4/4] Evaluating and Saving Results...")
    df["retrieved_cids"] = batch_cids
    df["retrieved_scores"] = batch_scores
    
    rankings_path = os.path.join(args.results_dir, f"bge_m3_train_rankings{args.output_suffix}.parquet")
    df[["qid", "question", "gold_cids", "retrieved_cids", "retrieved_scores"]].to_parquet(rankings_path, index=False)
    
    qrels = dict(zip(df["qid"], df["gold_cids"]))
    results = dict(zip(df["qid"], df["retrieved_cids"]))
    
    metrics = evaluate_retrieval(qrels, results, k_values=eval_ks)
    metrics_path = os.path.join(args.results_dir, f"bge_m3_train_metrics{args.output_suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"      -> Saved to {rankings_path} and {metrics_path}")
    print("=== BGE-M3 EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()
