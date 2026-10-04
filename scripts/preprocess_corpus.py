import os
import sys
import time
import re
import pandas as pd
import argparse

# 1. Dynamically set project root to ensure it runs from any terminal location
# __file__ points to src/data/preprocess_corpus.py, so root is 2 levels up
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.data_loader import load_parquet_data
from src.data.text_cleaner import normalize_vietnamese_text

def is_punct_only(text: str) -> bool:
    """Helper function to determine if a string contains only punctuation/symbols."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return True
    cleaned = re.sub(r'[\W_]+', '', text) 
    return len(cleaned) == 0

def clean_html_artifacts(text_series: pd.Series) -> pd.Series:
    """Safely removes basic HTML artifacts without destroying text structure."""
    print("   -> Checking for HTML artifacts...")
    html_mask = text_series.str.contains(r'<[a-z/][^>]*>|&[a-z]+;', regex=True, case=False)
    html_count = html_mask.sum()
    
    if html_count > 0:
        print(f"   -> Found {html_count} documents with HTML. Cleaning...")
        text_series = text_series.str.replace(r'<br\s*/?>', '\n', regex=True, flags=re.IGNORECASE)
        text_series = text_series.str.replace(r'&nbsp;', ' ', regex=True, flags=re.IGNORECASE)
        text_series = text_series.str.replace(r'<[^>]+>', '', regex=True)
        # Re-normalize just in case tag removal caused weird spacing
        text_series = text_series.apply(normalize_vietnamese_text)
    else:
        print("   -> No HTML artifacts found.")
        
    return text_series

def run_preprocessing_pipeline(raw_path: str, output_dir: str):
    """Executes the full defensive preprocessing pipeline."""
    start_time = time.time()
    print(f"=== STARTING CORPUS PREPROCESSING ===")
    
    # Step 1: Load Data
    print(f"[1/5] Loading raw corpus from: {raw_path}")
    try:
        corpus = load_parquet_data(raw_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load raw corpus. {e}")
        sys.exit(1)
        
    corpus['cid'] = corpus['cid'].astype(str)
    
    # Step 2: Text Normalization
    print(f"[2/5] Applying Unicode (NFC) & Whitespace normalization to {len(corpus):,} documents...")
    corpus['text_clean'] = corpus['text'].fillna("").apply(normalize_vietnamese_text)
    
    # Step 3: HTML Artifact Cleaning
    print("[3/5] Handling Artifacts...")
    corpus['text_clean'] = clean_html_artifacts(corpus['text_clean'])
    
    # Step 4: Feature Engineering & Quality Flags
    print("[4/5] Generating quality flags and calculating lengths...")
    corpus['char_count'] = corpus['text_clean'].str.len()
    corpus['word_count'] = corpus['text_clean'].str.split().str.len()
    
    corpus['is_very_short'] = corpus['word_count'] < 5
    corpus['needs_local_passage_selection'] = corpus['word_count'] > 1000
    corpus['is_punctuation_only'] = corpus['text_clean'].apply(is_punct_only)
    
    # Log flags
    print(f"   -> Very short (< 5 words) : {corpus['is_very_short'].sum():,}")
    print(f"   -> Needs passage selection: {corpus['needs_local_passage_selection'].sum():,}")
    print(f"   -> Punctuation only       : {corpus['is_punctuation_only'].sum():,}")

    # Step 5: Save Processed Data
    print("[5/5] Saving processed corpus...")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "documents_clean.parquet")
    
    # Select final columns (excluding raw text to save space)
    cols_to_save = [
        'cid', 'text_clean', 'char_count', 'word_count', 
        'is_very_short', 'needs_local_passage_selection', 'is_punctuation_only'
    ]
    
    corpus_final = corpus[cols_to_save].copy()
    corpus_final.to_parquet(save_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"=== PREPROCESSING COMPLETE in {elapsed:.2f} seconds ===")
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    # Setup argparse for flexible execution from terminal
    parser = argparse.ArgumentParser(description="Vietnamese Legal Corpus Preprocessing")
    parser.add_argument(
        "--raw_path", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "data/raw/YuITC/corpus.parquet"),
        help="Path to the raw corpus.parquet file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=os.path.join(PROJECT_ROOT, "data/processed"),
        help="Directory to save the processed documents_clean.parquet"
    )
    
    args = parser.parse_args()
    
    run_preprocessing_pipeline(raw_path=args.raw_path, output_dir=args.output_dir)