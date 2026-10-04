import os
import sys
import json
import argparse
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.evaluation.rag_metrics import summarize_rag_results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate End-to-End RAG Outputs")
    parser.add_argument("--input", required=True, help="Path to RAG results parquet")
    parser.add_argument("--output-json", default=None, help="Save metrics to JSON")
    parser.add_argument("--errors-output", default=None, help="Save error rows to parquet")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"Lỗi: Không tìm thấy file {args.input}")
        sys.exit(1)

    print(f"Reading RAG results from {args.input}...")
    df = pd.read_parquet(args.input)
    
    try:
        metrics = summarize_rag_results(df)
    except Exception as e:
        print(f"Lỗi khi đánh giá metrics: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print(" SUMMARY: LEGAL RAG STRUCTURAL EVALUATION")
    print("="*50)
    print(f"Total Queries:                {metrics['total_queries']}")
    print(f"Successful Queries (ok):      {metrics['successful_queries']}")
    print(f"Pipeline Success Rate:        {metrics['pipeline_success_rate']:.2%}")
    print("-" * 50)
    print(f"Answered Rate:                {metrics['answered_rate']:.2%}")
    print(f"Insufficient Evidence Rate:   {metrics['insufficient_evidence_rate']:.2%}")
    print("-" * 50)
    print(f"Valid Citation Rate:          {metrics['citation_valid_rate']:.2%}")
    print(f"Hallucinated CID Rate:        {metrics['invalid_citation_query_rate']:.2%}")
    print(f"Inline/Declared Mismatch:     {metrics['missing_declared_query_rate']:.2%}")
    print(f"Declared-but-Unused Rate:     {metrics['uncited_declared_query_rate']:.2%}")
    print(f"Answered w/o Citation Rate:   {metrics['answered_without_citation_rate']:.2%}")
    print("-" * 50)
    print(f"Avg Context Sources:          {metrics['avg_context_sources']:.1f}")
    print(f"Avg Context Characters:       {metrics['avg_context_chars']:,.0f}")
    print(f"Avg Generation Latency:       {metrics['avg_generation_latency_ms']/1000:.2f}s")
    print("="*50 + "\n")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved metrics JSON to {args.output_json}")

    if args.errors_output:
        # Lọc ra các bản ghi có lỗi logic pipeline hoặc lỗi citation
        error_mask = (df["pipeline_status"] != "ok") | (df.get("citation_valid", False) != True)
        df_errors = df[error_mask]
        
        if not df_errors.empty:
            os.makedirs(os.path.dirname(os.path.abspath(args.errors_output)), exist_ok=True)
            df_errors.to_parquet(args.errors_output, index=False)
            print(f"Saved {len(df_errors)} error records to {args.errors_output}")
        else:
            print("No errors found. Error output file not generated.")

if __name__ == "__main__":
    main() 