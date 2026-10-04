import os
import sys
import time
import yaml
import argparse

# Dynamically set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.loader import load_parquet_data
from src.retrieval.bm25 import BM25Retriever

def main():
    parser = argparse.ArgumentParser(description="Build BM25 Index")
    parser.add_argument("--corpus_path", default=os.path.join(PROJECT_ROOT, "data/processed/documents_clean.parquet"))
    parser.add_argument("--save_dir", default=os.path.join(PROJECT_ROOT, "artifacts/bm25"))
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/retrieval.yaml"))
    args = parser.parse_args()

    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    bm25_config = config.get('bm25', {})
    
    print("\n=== STARTING BM25 INDEX BUILD ===")
    
    # 2. Load Corpus
    print(f"[1/4] Loading processed corpus from: {args.corpus_path}")
    corpus = load_parquet_data(args.corpus_path)
    
    cids = corpus['cid'].astype(str).tolist()
    texts = corpus['text_clean'].tolist()
    print(f"      -> Loaded {len(cids):,} documents.")

    # 3. Initialize & Fit BM25
    print(f"[2/4] Initializing BM25 Retriever (lowercase={bm25_config.get('lowercase', True)})...")
    start_time = time.time()
    
    retriever = BM25Retriever(
        lowercase=bm25_config.get('lowercase', True),
        k1=bm25_config.get('k1', 1.5),
        b=bm25_config.get('b', 0.75)
    )
    
    print("[3/4] Tokenizing and building index (this will take time)...")
    retriever.fit(cids, texts)
    
    build_time = time.time() - start_time
    print(f"      -> Build completed in {build_time:.2f} seconds.")

    # 4. Save Index
    print(f"[4/4] Saving index to: {args.save_dir}")
    retriever.save(args.save_dir)
    print("=== BM25 BUILD COMPLETE ===")

if __name__ == "__main__":
    main() 