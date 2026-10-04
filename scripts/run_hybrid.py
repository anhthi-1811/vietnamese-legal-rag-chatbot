import os
import sys
import yaml
import json
import time
import numpy as np
import pandas as pd
import ast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.retrieval.hybrid import batch_rrf
from src.evaluation.metrics import evaluate_retrieval

def parse_cids(val):
    if isinstance(val, (list, np.ndarray)): return [str(x) for x in val]
    if isinstance(val, str):
        try: return [str(x) for x in ast.literal_eval(val)]
        except: return [str(val)]
    return [str(val)]

def main():
    print("\n=== STARTING HYBRID RETRIEVAL (RRF) ===")
    
    # 1. Load config
    config_path = os.path.join(PROJECT_ROOT, "configs/retrieval.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    hybrid_cfg = config.get('hybrid', {})
    eval_ks = list(set(config['evaluation']['recall_k'] + config['evaluation']['mrr_k'] + config['evaluation']['ndcg_k']))
    eval_ks.sort()
    
    input_top_k = hybrid_cfg.get('input_top_k', 20)
    output_top_k = hybrid_cfg.get('output_top_k', 20)
    rrf_k_values = hybrid_cfg.get('rrf_k_values', [10, 20, 60, 100])
    selection_metric = hybrid_cfg.get('selection_metric', 'Recall@10')
    
    # 2. Load Data
    bm25_path = os.path.join(PROJECT_ROOT, "experiments/results/bm25_train_rankings.parquet")
    bge_path = os.path.join(PROJECT_ROOT, "experiments/results/bge_m3_train_rankings.parquet")
    
    df_bm25 = pd.read_parquet(bm25_path)
    df_bge = pd.read_parquet(bge_path)
    
    # 3. Safety Checks
    print("[1/4] Running safety checks on inputs...")
    assert len(df_bm25) == len(df_bge), "Row count mismatch between BM25 and BGE files."
    assert (df_bm25['qid'].values == df_bge['qid'].values).all(), "QID order mismatch."
    
    golds_bm25 = [parse_cids(x) for x in df_bm25['gold_cids'].tolist()]
    golds_bge = [parse_cids(x) for x in df_bge['gold_cids'].tolist()]
    assert golds_bm25 == golds_bge, "Gold CIDs disagree between ranking files."
    
    qids = df_bm25['qid'].tolist()
    questions = df_bm25['question'].tolist()
    
    # Enforce input_top_k
    preds_bm25 = [parse_cids(x)[:input_top_k] for x in df_bm25['retrieved_cids'].tolist()]
    preds_bge = [parse_cids(x)[:input_top_k] for x in df_bge['retrieved_cids'].tolist()]
    
    # 4. Tune RRF
    print(f"[2/4] Tuning rrf_k in {rrf_k_values}...")
    tuning_records = []
    best_score = -1
    best_k = None
    best_cids = None
    best_scores = None
    best_metrics = None
    best_time = 0
    
    qrels = dict(zip(qids, golds_bm25))
    
    for r_k in rrf_k_values:
        start_time = time.time()
        
        # Apply Fusion
        batch_fused_cids, batch_fused_scores = batch_rrf(
            [preds_bm25, preds_bge], 
            rrf_k=r_k, 
            top_k=output_top_k
        )
        
        fuse_time = time.time() - start_time
        
        # Evaluate
        results = dict(zip(qids, batch_fused_cids))
        metrics = evaluate_retrieval(qrels, results, k_values=eval_ks)
        
        # Record
        record = {'rrf_k': r_k}
        record.update(metrics)
        tuning_records.append(record)
        
        current_score = metrics.get(selection_metric, 0)
        print(f"      -> rrf_k={r_k}: {selection_metric} = {current_score:.4f} (Time: {fuse_time:.2f}s)")
        
        if current_score > best_score:
            best_score = current_score
            best_k = r_k
            best_cids = batch_fused_cids
            best_scores = batch_fused_scores
            best_metrics = metrics
            best_time = fuse_time

    # Save tuning table
    tuning_df = pd.DataFrame(tuning_records)
    tuning_path = os.path.join(PROJECT_ROOT, "experiments/results/hybrid_rrf_tuning.parquet")
    tuning_df.to_parquet(tuning_path, index=False)
    
    print(f"[3/4] Selected best rrf_k = {best_k} (based on {selection_metric}).")
    
    # 5. Final Safety Checks
    print("      -> Verifying final fused outputs...")
    assert len(best_cids) == len(qids)
    for i in range(min(100, len(best_cids))):
        assert len(best_cids[i]) > 0, "Query received empty rankings."
        assert len(set(best_cids[i])) == len(best_cids[i]), "Duplicate CID in fused ranking."
        assert not any(np.isnan(s) for s in best_scores[i]), "NaN in RRF scores."
        
    # 6. Save Final Results
    print(f"[4/4] Saving Final Hybrid Results...")
    df_final = pd.DataFrame({
        'qid': qids,
        'question': questions,
        'gold_cids': golds_bm25,
        'retrieved_cids': best_cids,
        'retrieved_scores': best_scores
    })
    
    rankings_path = os.path.join(PROJECT_ROOT, "experiments/results/hybrid_train_rankings.parquet")
    df_final.to_parquet(rankings_path, index=False)
    
    metrics_path = os.path.join(PROJECT_ROOT, "experiments/results/hybrid_train_metrics.json")
    final_metrics_payload = {
        "metadata": {
            "method": "rrf",
            "rrf_k": best_k,
            "input_top_k": input_top_k,
            "output_top_k": output_top_k,
            "queries_processed": len(qids),
            "fusion_runtime_seconds": round(best_time, 2)
        },
        "metrics": best_metrics
    }
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics_payload, f, indent=4)
        
    # 7. Print Comparison Table
    print("\n=== BASELINE COMPARISON ===")
    
    # Extract baseline scores for comparison
    metrics_bm25 = json.load(open(os.path.join(PROJECT_ROOT, "experiments/results/bm25_train_metrics.json")))
    metrics_bge = json.load(open(os.path.join(PROJECT_ROOT, "experiments/results/bge_m3_train_metrics.json")))
    
    headers = ["Metric", "BM25", "BGE-M3", "Hybrid"]
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10}")
    
    compare_keys = ["Recall@1", "Recall@5", "Recall@10", "Recall@20", "MRR@10", "nDCG@10"]
    for k in compare_keys:
        v_bm25 = metrics_bm25.get(k, 0)
        v_bge = metrics_bge.get(k, 0)
        v_hyb = best_metrics.get(k, 0)
        print(f"{k:<15} {v_bm25:<10.4f} {v_bge:<10.4f} {v_hyb:<10.4f}")

if __name__ == "__main__":
    main()
