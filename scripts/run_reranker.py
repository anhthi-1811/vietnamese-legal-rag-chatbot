import os
import sys
import time
import yaml
import json
import torch
import numpy as np
import pandas as pd
import argparse
import glob
from tqdm import tqdm

# Bật tối ưu hóa TF32 trên Ampere (A100) để tăng tốc nhân ma trận
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.retrieval.reranker import CrossEncoderReranker
from src.evaluation.metrics import evaluate_retrieval

def main():
    parser = argparse.ArgumentParser(description="Run Cross-Encoder Reranker with Checkpointing & Smart Batching")
    parser.add_argument("--candidates_path", default=os.path.join(PROJECT_ROOT, "experiments/results/rerank_candidates_train.parquet"))
    parser.add_argument("--corpus_path", default=os.path.join(PROJECT_ROOT, "data/processed/documents_clean.parquet"))
    parser.add_argument("--results_dir", default=os.path.join(PROJECT_ROOT, "experiments/results"))
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/retrieval.yaml"))
    parser.add_argument("--chunk_size", type=int, default=5000, help="Number of queries per chunk")
    args = parser.parse_args()

    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    rerank_cfg = config.get('reranker', {})
    top_k_output = rerank_cfg.get('top_k_output', 20)
    eval_ks = list(set(config['evaluation']['recall_k'] + config['evaluation']['mrr_k'] + config['evaluation']['ndcg_k']))
    eval_ks.sort()
    
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n=== STARTING CROSS-ENCODER RERANKER (FULL RUN WITH CHECKPOINTS) ===")
    
    # 2. Load Corpus
    print("[1/5] Loading corpus for document text lookup...")
    corpus = pd.read_parquet(args.corpus_path, columns=["cid", "text_clean"])
    doc_lookup = dict(zip(corpus['cid'].astype(str), corpus['text_clean']))
    
    # 3. Load Candidates
    print(f"[2/5] Loading candidate pools...")
    df_candidates = pd.read_parquet(args.candidates_path)
    total_queries = len(df_candidates)
    
    # 4. Initialize Reranker
    print(f"[3/5] Initializing Reranker model...")
    reranker = CrossEncoderReranker(
        model_name=rerank_cfg['model_name'], 
        use_fp16=rerank_cfg['use_fp16']
    )

    # 5. Process in Chunks
    print(f"[4/5] Processing {total_queries:,} queries in chunks of {args.chunk_size}...")
    
    num_chunks = (total_queries + args.chunk_size - 1) // args.chunk_size
    
    for chunk_idx in range(num_chunks):
        chunk_file = os.path.join(checkpoint_dir, f"reranker_chunk_{chunk_idx:04d}.parquet")
        if os.path.exists(chunk_file):
            print(f"      -> Chunk {chunk_idx+1}/{num_chunks} already processed. Skipping.")
            continue
            
        print(f"\n--- Processing Chunk {chunk_idx+1}/{num_chunks} ---")
        start_idx = chunk_idx * args.chunk_size
        end_idx = min((chunk_idx + 1) * args.chunk_size, total_queries)
        
        df_chunk = df_candidates.iloc[start_idx:end_idx]
        qids = df_chunk['qid'].tolist()
        questions = df_chunk['question'].tolist()
        golds = df_chunk['gold_cids'].tolist()
        cids_lists = df_chunk['candidate_cids'].tolist()
        
        # Flatten and prepare for SMART BATCHING
        indexed_pairs = [] # Store: (original_idx, approx_length, query_text, doc_text, qid_idx, cid)
        flat_idx = 0
        
        for i in range(len(qids)):
            q = questions[i]
            for cid in cids_lists[i]:
                if cid in doc_lookup:
                    doc = doc_lookup[cid]
                    approx_len = len(q) + len(doc)
                    indexed_pairs.append([flat_idx, approx_len, q, doc, i, cid])
                    flat_idx += 1
                    
        if not indexed_pairs:
            continue
            
        # SMART BATCHING: Sort by length to minimize padding tokens inside the batch
        indexed_pairs.sort(key=lambda x: x[1])
        
        model_input = [[x[2], x[3]] for x in indexed_pairs]
        
        chunk_start_time = time.time()
        
        # Inference
        scores = reranker.compute_scores(
            model_input, 
            batch_size=rerank_cfg['batch_size'], 
            max_length=rerank_cfg['max_length']
        )
        
        if isinstance(scores, float):
            scores = [scores]
            
        chunk_time = time.time() - chunk_start_time
        print(f"      -> Inference time: {chunk_time:.2f}s | Pairs/sec: {len(indexed_pairs)/chunk_time:.1f}")
        
        # Un-sort to restore original flattened order
        for i, score in enumerate(scores):
            indexed_pairs[i].append(float(score)) # Append score at index 6
            
        # Re-sort by original flat_idx
        indexed_pairs.sort(key=lambda x: x[0])
        
        # Re-group by query
        query_results = {i: [] for i in range(len(qids))}
        for item in indexed_pairs:
            q_idx = item[4]
            cid = item[5]
            score = item[6]
            query_results[q_idx].append((cid, score))
            
        final_cids = []
        final_scores = []
        for i in range(len(qids)):
            ranked = sorted(query_results[i], key=lambda x: x[1], reverse=True)[:top_k_output]
            final_cids.append([x[0] for x in ranked])
            final_scores.append([x[1] for x in ranked])
            
        df_chunk_results = pd.DataFrame({
            'qid': qids,
            'question': questions,
            'gold_cids': golds,
            'retrieved_cids': final_cids,
            'retrieved_scores': final_scores
        })
        
        df_chunk_results.to_parquet(chunk_file, index=False)
        print(f"      -> Saved checkpoint: {os.path.basename(chunk_file)}")
        
        # Free memory
        del indexed_pairs, model_input, scores, query_results
        torch.cuda.empty_cache()

    # 6. Merge Checkpoints and Evaluate
    print("\n[5/5] Merging chunks and evaluating full dataset...")
    all_chunks = sorted(glob.glob(os.path.join(checkpoint_dir, "reranker_chunk_*.parquet")))
    
    if not all_chunks:
        print("No chunks found. Exiting.")
        sys.exit(0)
        
    df_final = pd.concat([pd.read_parquet(f) for f in all_chunks], ignore_index=True)
    
    final_rankings_path = os.path.join(args.results_dir, "reranker_train_rankings.parquet")
    df_final.to_parquet(final_rankings_path, index=False)
    
    qrels = dict(zip(df_final['qid'], df_final['gold_cids']))
    results_dict = dict(zip(df_final['qid'], df_final['retrieved_cids']))
    
    metrics = evaluate_retrieval(qrels, results_dict, k_values=eval_ks)
    
    metrics_path = os.path.join(args.results_dir, "reranker_train_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"      -> Final results saved to {final_rankings_path}")
    print("\n=== RERANKER EVALUATION SUMMARY (FULL TRAIN) ===")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")

if __name__ == "__main__":
    main() 