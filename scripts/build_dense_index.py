import os
import sys
import time
import yaml
import argparse
import numpy as np

# Dynamically set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.loader import load_parquet_data
from src.retrieval.dense import BGEM3Encoder
from src.retrieval.faiss_index import FAISSRetriever

def main():
    parser = argparse.ArgumentParser(description="Build BGE-M3 Dense Index (FAISS)")
    parser.add_argument("--corpus_path", default=os.path.join(PROJECT_ROOT, "data/processed/documents_clean.parquet"))
    parser.add_argument("--save_dir", default=os.path.join(PROJECT_ROOT, "artifacts/dense"))
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/retrieval.yaml"))
    args = parser.parse_args()

    # 1. Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    dense_cfg = config.get('dense', {})

    print("\n=== STARTING BGE-M3 INDEX BUILD ===")
    
    # 2. Load Corpus
    print(f"[1/5] Loading processed corpus from: {args.corpus_path}")
    corpus = load_parquet_data(args.corpus_path)
    cids = corpus['cid'].astype(str).tolist()
    texts = corpus['text_clean'].tolist()
    print(f"      -> Loaded {len(cids):,} documents.")

    # 3. Initialize Encoder
    print(f"[2/5] Initializing Encoder ({dense_cfg['model_name']}) on GPU...")
    encoder = BGEM3Encoder(model_name=dense_cfg['model_name'], use_fp16=dense_cfg['use_fp16'])
    
    # 4. Encoding with progress tracking
    print(f"[3/5] Encoding {len(texts):,} documents (Batch: {dense_cfg['batch_size']}, MaxLen: {dense_cfg['max_length']})...")
    start_encode = time.time()
    
    all_embeddings = []
    chunk_size = 50000  # Process in chunks to free up GPU cache and show progress
    
    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i:i + chunk_size]
        print(f"      -> Encoding chunk {i:,} to {i + len(chunk_texts):,}...")
        chunk_emb = encoder.encode_documents(
            chunk_texts, 
            batch_size=dense_cfg['batch_size'], 
            max_length=dense_cfg['max_length']
        )
        all_embeddings.append(chunk_emb)
        
    final_embeddings = np.vstack(all_embeddings)
    encode_time = time.time() - start_encode
    print(f"      -> Encoding completed in {encode_time:.2f} seconds.")
    print(f"      -> Final embedding matrix shape: {final_embeddings.shape}")

    # 5. Build FAISS Index
    print("[4/5] Building FAISS IndexFlatIP...")
    start_faiss = time.time()
    retriever = FAISSRetriever()
    retriever.build(cids, final_embeddings)
    faiss_time = time.time() - start_faiss
    print(f"      -> FAISS build completed in {faiss_time:.2f} seconds.")

    # 6. Save Artifacts
    print(f"[5/5] Saving index to: {args.save_dir}")
    retriever.save(args.save_dir)
    print("=== BGE-M3 BUILD COMPLETE ===")

if __name__ == "__main__":
    main() 