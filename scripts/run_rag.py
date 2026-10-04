import os
import sys
import time
import yaml
import argparse
import signal
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.rag.passage_retriever import PassageRetriever
from src.rag.context_builder import ContextBuilder
from src.rag.generator import LegalAnswerGenerator
from src.rag.citation_checker import CitationChecker

# Global flag for graceful shutdown
_shutdown_requested = False

def handle_interrupt(sig, frame):
    global _shutdown_requested
    print("\n[CẢNH BÁO] Nhận tín hiệu ngắt (Ctrl+C). Sẽ lưu checkpoint ở mẫu hiện tại và dừng...")
    _shutdown_requested = True

signal.signal(signal.SIGINT, handle_interrupt)

def atomic_save_parquet(df: pd.DataFrame, path: str):
    """Save dataframe atomically to prevent data corruption during interrupt."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = path + ".tmp.parquet"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)

def load_yaml(path: str, default: dict) -> dict:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return default

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Legal RAG Pipeline Orchestrator")
    parser.add_argument("--rankings", required=True, help="Path to reranker output parquet")
    parser.add_argument("--corpus", default=os.path.join(PROJECT_ROOT, "data/processed/documents_clean.parquet"))
    # Gộp chung cấu hình vào 1 file duy nhất rag.yaml
    parser.add_argument("--rag-config", default=os.path.join(PROJECT_ROOT, "configs/rag.yaml"))
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "experiments/results/rag_results.parquet"))
    parser.add_argument("--limit", type=int, default=0, help="Max queries to process")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K docs to pass to context builder")
    parser.add_argument("--prompt-version", type=str, default="v1", choices=["v1", "v2"], help="Version of generator system prompt (v1=baseline, v2=calibrated)")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Save checkpoint frequency")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--save-context", action="store_true", help="Save full context string to output")
    return parser.parse_args()

def main():
    global _shutdown_requested
    args = parse_args()
    
    # 1. Load Configurations từ 1 file duy nhất
    full_cfg = load_yaml(args.rag_config, {})
    rag_cfg = full_cfg.get("rag", {})
    gen_cfg = full_cfg.get("generation", {})
    run_cfg = full_cfg.get("runtime", {})
    
    top_k = args.top_k or rag_cfg.get("top_k_documents", 5)
    checkpoint_every = args.checkpoint_every or run_cfg.get("checkpoint_every", 20)
    req_sleep = 1.0

    # 2. Init Modules
    retriever_cfg = rag_cfg.get("passage", {})
    passage_retriever = PassageRetriever(
        long_doc_threshold_chars=retriever_cfg.get("long_doc_threshold_chars", 6000),
        max_passage_chars=retriever_cfg.get("max_passage_chars", 2500),
        top_passages_per_doc=retriever_cfg.get("top_passages_per_doc", 2),
        overlap_paragraphs=retriever_cfg.get("overlap_paragraphs", 0)
    )
    
    context_cfg = rag_cfg.get("context", {})
    context_builder = ContextBuilder(
        max_context_chars=context_cfg.get("max_context_chars", 12000)
    )
    
    # Truyền thêm prompt_version vào LegalAnswerGenerator
    generator = LegalAnswerGenerator(
        model_name=gen_cfg.get("model_name", "gemini-2.5-flash"),  
        prompt_version=args.prompt_version,
        temperature=gen_cfg.get("temperature", 0.0),
        max_output_tokens=gen_cfg.get("max_output_tokens", 1024),
        api_retries=gen_cfg.get("max_retries", 3)
    )
    
    citation_checker = CitationChecker()

    # 3. Load Data
    print(f"Loading corpus from {args.corpus}...")
    df_corpus = pd.read_parquet(args.corpus, columns=["cid", "text_clean"])
    doc_lookup = dict(zip(df_corpus["cid"].astype(str), df_corpus["text_clean"]))

    print(f"Loading rankings from {args.rankings}...")
    df_rankings = pd.read_parquet(args.rankings)
    
    # 4. Handle Resume Logic
    processed_qids = set()
    results = []
    if args.resume and os.path.exists(args.output):
        try:
            df_existing = pd.read_parquet(args.output)
            if "qid" in df_existing.columns:
                processed_qids = set(df_existing["qid"].tolist())
                results = df_existing.to_dict("records")
                print(f"Resuming... found {len(processed_qids)} processed queries.")
        except Exception as e:
            print(f"Failed to read existing output for resume: {e}")

    queries_to_run = df_rankings[~df_rankings["qid"].isin(processed_qids)].copy()
    if args.limit > 0:
        queries_to_run = queries_to_run.head(args.limit)

    print(f"Starting generation for {len(queries_to_run)} queries (prompt_version={args.prompt_version})...")
    
    processed_count = 0
    for _, row in tqdm(queries_to_run.iterrows(), total=len(queries_to_run)):
        if _shutdown_requested:
            break
            
        t0_pipeline = time.time()
        qid = row["qid"]
        question = row["question"]
        gold_cids = row.get("gold_cids", [])
        
        # Safely convert and extract CIDs
        retrieved_cids_raw = row.get("retrieved_cids", [])
        retrieved_scores = row.get("retrieved_scores", [])
        
        retrieved_cids = [str(c).strip() for c in retrieved_cids_raw][:top_k]
        scores = retrieved_scores[:top_k] if len(retrieved_scores) >= len(retrieved_cids) else [0.0]*len(retrieved_cids)

        row_result = {
            "qid": qid,
            "question": question,
            "gold_cids": gold_cids,
            "retrieved_cids": retrieved_cids,
            "retrieved_scores": scores,
            "prompt_version": args.prompt_version,
            "pipeline_status": "ok",
            "pipeline_error": None
        }

        try:
            # A. Prepare Docs Lookup
            docs_for_retrieval = []
            for cid, score in zip(retrieved_cids, scores):
                if cid in doc_lookup:
                    docs_for_retrieval.append({
                        "cid": cid,
                        "text": doc_lookup[cid],
                        "score": score
                    })

            # B. Passage Retrieval
            processed_docs = passage_retriever.process_documents(
                question=question, 
                documents=docs_for_retrieval
            )

            # C. Context Building
            context_data = context_builder.build_context(processed_docs)
            context_str = context_data.get("context_str", "")
            included_cids = context_data.get("included_cids", [])
            
            row_result.update({
                "included_cids": included_cids,
                "num_context_sources": context_data.get("num_sources", len(included_cids)),
                "context_chars": context_data.get("total_chars", len(context_str)),
                "context_budget_exhausted": context_data.get("budget_exhausted", False)
            })
            
            if args.save_context:
                row_result["context_str"] = context_str

            # D. Generation
            t0_gen = time.time()
            gen_output = generator.generate(
                question=question,
                context_str=context_str,
                allowed_cids=included_cids
            )
            gen_latency = (time.time() - t0_gen) * 1000
            
            usage = gen_output.get("usage") or {}
            row_result.update({
                "answer": gen_output.get("answer", ""),
                "answer_status": gen_output.get("answer_status", "unknown"),
                "declared_cited_cids": gen_output.get("cited_cids", []),
                "raw_response": gen_output.get("raw_response", ""),
                "model_name": gen_output.get("model_name", ""),
                "generation_input_tokens": usage.get("prompt_tokens", 0),
                "generation_output_tokens": usage.get("candidates_tokens", 0),
                "generation_latency_ms": gen_latency
            })

            # E. Citation Checking
            check_output = citation_checker.check(
                answer=row_result["answer"],
                declared_cids=row_result["declared_cited_cids"],
                allowed_cids=included_cids,
                answer_status=row_result["answer_status"]
            )
            
            row_result.update({
                "citation_valid": check_output.get("is_valid", False),
                "inline_cids": check_output.get("inline_cids", []),
                "invalid_cids": check_output.get("invalid_cids", []),
                "missing_declared_cids": check_output.get("missing_declared_cids", []),
                "uncited_declared_cids": check_output.get("uncited_declared_cids", []),
                "citation_warnings": check_output.get("warnings", [])
            })

        except Exception as e:
            row_result["pipeline_status"] = "error"
            row_result["pipeline_error"] = f"{type(e).__name__}: {str(e)}"

        # Finalize latency and record
        row_result["pipeline_latency_ms"] = (time.time() - t0_pipeline) * 1000
        results.append(row_result)
        processed_count += 1
        
        # Checkpoint logic
        if processed_count % checkpoint_every == 0: 
            atomic_save_parquet(pd.DataFrame(results), args.output)

        if req_sleep > 0:
            time.sleep(req_sleep)

    # Final Save
    if processed_count > 0 or _shutdown_requested:
        atomic_save_parquet(pd.DataFrame(results), args.output)
        print(f"\nSaved {len(results)} total records to {args.output}")

if __name__ == "__main__":
    main() 