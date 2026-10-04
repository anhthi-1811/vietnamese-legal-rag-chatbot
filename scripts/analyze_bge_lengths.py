import os
import sys
import time
import json
import argparse
import numpy as np
from transformers import AutoTokenizer

# Dynamically set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.loader import load_parquet_data

def main():
    parser = argparse.ArgumentParser(description="Analyze Token Lengths using BGE-M3 Tokenizer")
    parser.add_argument("--corpus_path", default=os.path.join(PROJECT_ROOT, "data/processed/documents_clean.parquet"))
    parser.add_argument("--save_path", default=os.path.join(PROJECT_ROOT, "experiments/results/bge_token_length_stats.json"))
    parser.add_argument("--model_name", default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for tokenization to save RAM")
    args = parser.parse_args()

    print(f"\n=== STARTING BGE-M3 TOKEN LENGTH ANALYSIS ===")
    
    # 1. Load data
    print(f"[1/4] Loading processed corpus from: {args.corpus_path}")
    corpus = load_parquet_data(args.corpus_path)
    texts = corpus['text_clean'].tolist()
    print(f"      -> Loaded {len(texts):,} documents.")

    # 2. Load tokenizer
    print(f"[2/4] Loading tokenizer: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer. Please run 'pip install transformers'. Details: {e}")
        sys.exit(1)

    # 3. Measure lengths in batches (to avoid RAM spikes)
    print(f"[3/4] Measuring token lengths (without truncation)...")
    start_time = time.time()
    
    token_lengths = []
    total_docs = len(texts)
    
    for i in range(0, total_docs, args.batch_size):
        batch_texts = texts[i : i + args.batch_size]
        # return_length=True gives us exactly what we need efficiently
        batch_encodings = tokenizer(
            batch_texts, 
            add_special_tokens=True, 
            truncation=False, 
            padding=False,
            return_length=True
        )
        token_lengths.extend(batch_encodings['length'])
        
        # Progress log
        if (i + args.batch_size) % 50000 == 0 or (i + args.batch_size) >= total_docs:
            print(f"      -> Processed {min(i + args.batch_size, total_docs):,} / {total_docs:,} documents")

    elapsed_time = time.time() - start_time
    print(f"      -> Tokenization complete in {elapsed_time:.2f} seconds.")

    # 4. Calculate Statistics
    print(f"[4/4] Calculating distribution statistics...")
    lengths_arr = np.array(token_lengths)
    
    stats = {
        "model_name": args.model_name,
        "total_documents": int(len(lengths_arr)),
        "min_tokens": int(np.min(lengths_arr)),
        "median_tokens": int(np.median(lengths_arr)),
        "mean_tokens": float(np.mean(lengths_arr)),
        "p90_tokens": int(np.percentile(lengths_arr, 90)),
        "p95_tokens": int(np.percentile(lengths_arr, 95)),
        "p99_tokens": int(np.percentile(lengths_arr, 99)),
        "max_tokens": int(np.max(lengths_arr)),
        "counts_above_thresholds": {
            "> 512": int(np.sum(lengths_arr > 512)),
            "> 1024": int(np.sum(lengths_arr > 1024)),
            "> 2048": int(np.sum(lengths_arr > 2048)),
            "> 4096": int(np.sum(lengths_arr > 4096)),
            "> 8192": int(np.sum(lengths_arr > 8192))
        }
    }

    # 5. Save and Display
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(stats, f, indent=4)

    print("\n=== TOKEN LENGTH SUMMARY ===")
    print(json.dumps(stats, indent=4))
    print(f"\nSummary saved to: {args.save_path}")

if __name__ == "__main__":
    main()  