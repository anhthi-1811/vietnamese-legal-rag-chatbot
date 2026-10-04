import os
import sys
import argparse

# 1. Dynamically set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.loader import load_parquet_data
from src.data.validator import ProcessedDataValidator

def main():
    parser = argparse.ArgumentParser(description="Strict Post-Preprocessing Validation")
    parser.add_argument("--raw_corpus", default=os.path.join(PROJECT_ROOT, "data/raw/YuITC/corpus.parquet"))
    parser.add_argument("--clean_corpus", default=os.path.join(PROJECT_ROOT, "data/processed/documents_clean.parquet"))
    parser.add_argument("--train", default=os.path.join(PROJECT_ROOT, "data/raw/YuITC/train.parquet"))
    parser.add_argument("--test", default=os.path.join(PROJECT_ROOT, "data/raw/YuITC/test.parquet"))
    args = parser.parse_args()

    print("\n=== [1/3] LOADING DATA FOR VALIDATION ===")
    try:
        raw = load_parquet_data(args.raw_corpus)
        clean = load_parquet_data(args.clean_corpus)
        train = load_parquet_data(args.train)
        test = load_parquet_data(args.test)
        
        raw['cid'] = raw['cid'].astype(str)
        clean['cid'] = clean['cid'].astype(str)
        print("All datasets loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load data. {e}") 
        sys.exit(1)

    print("\n=== [2/3] RUNNING ASSERTIONS ===")
    validator = ProcessedDataValidator(raw, clean, train, test)
    summary_df, is_passed = validator.run_all_validations()

    print("\n=== [3/3] VALIDATION SUMMARY ===")
    # Print the DataFrame nicely in the console
    print(summary_df.to_string(index=False))

    print("\n" + "="*40)
    if is_passed:
        print("FINAL DECISION: PASS")
        print("The processed corpus is validated and ready for Retrieval Experiments.")
        sys.exit(0)
    else:
        print("FINAL DECISION: FAIL")
        print("One or more critical checks failed. Do not proceed to BM25.")
        sys.exit(1)

if __name__ == "__main__":
    main() 