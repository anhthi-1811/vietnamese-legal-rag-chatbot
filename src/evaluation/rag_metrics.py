import ast
import numpy as np
import pandas as pd

def _safe_list(val) -> list:
    """Safely convert various data types into a standard Python list."""
    if isinstance(val, list):
        return val
    if isinstance(val, (np.ndarray, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []

def _safe_len(val) -> int:
    """Safely get the length of a list-like object."""
    return len(_safe_list(val))

def _safe_divide(numerator: float, denominator: float) -> float:
    """Safe division returning 0.0 if denominator is 0."""
    return float(numerator) / float(denominator) if denominator else 0.0

def summarize_rag_results(df: pd.DataFrame) -> dict:
    """
    Xác định và tính toán các chỉ số RAG dựa trên cấu trúc trích dẫn và kết quả thực thi.
    """
    if df.empty:
        raise ValueError("DataFrame trống. Không có dữ liệu để tính toán metrics.")

    # 1. Pipeline Status
    total_queries = len(df)
    
    if "pipeline_status" in df.columns:
        successful_queries = len(df[df["pipeline_status"] == "ok"])
    else:
        successful_queries = total_queries
        
    failed_queries = total_queries - successful_queries
    pipeline_success_rate = _safe_divide(successful_queries, total_queries)

    # Lọc ra các dòng chạy thành công để tính metrics nội dung
    df_ok = df[df.get("pipeline_status", "ok") == "ok"].copy()
    total_ok = len(df_ok)

    # 2. Answer Status
    answered_count = 0
    insufficient_evidence_count = 0
    if "answer_status" in df_ok.columns:
        answered_count = len(df_ok[df_ok["answer_status"] == "answered"])
        insufficient_evidence_count = len(df_ok[df_ok["answer_status"] == "insufficient_evidence"])
    
    answered_rate = _safe_divide(answered_count, total_ok)
    insufficient_evidence_rate = _safe_divide(insufficient_evidence_count, total_ok)

    # 3. Citation Validation
    citation_valid_count = 0
    invalid_citation_queries = 0
    missing_declared_queries = 0
    uncited_declared_queries = 0
    answered_without_citation = 0

    if "citation_valid" in df_ok.columns:
        # Chuẩn hóa boolean
        df_ok["citation_valid"] = df_ok["citation_valid"].astype(bool)
        citation_valid_count = df_ok["citation_valid"].sum()
        
    if "invalid_cids" in df_ok.columns:
        invalid_citation_queries = sum(df_ok["invalid_cids"].apply(lambda x: _safe_len(x) > 0))
        
    if "missing_declared_cids" in df_ok.columns:
        missing_declared_queries = sum(df_ok["missing_declared_cids"].apply(lambda x: _safe_len(x) > 0))
        
    if "uncited_declared_cids" in df_ok.columns:
        uncited_declared_queries = sum(df_ok["uncited_declared_cids"].apply(lambda x: _safe_len(x) > 0))

    if "answer_status" in df_ok.columns and "inline_cids" in df_ok.columns:
        answered_without_citation = len(df_ok[
            (df_ok["answer_status"] == "answered") & 
            (df_ok["inline_cids"].apply(_safe_len) == 0)
        ])

    # 4. Aggregations / Averages
    avg_inline_citations_per_response = 0.0
    if "inline_cids" in df_ok.columns:
        avg_inline_citations_per_response = df_ok["inline_cids"].apply(_safe_len).mean()

    avg_declared_citations_per_response = 0.0
    if "declared_cited_cids" in df_ok.columns:
        avg_declared_citations_per_response = df_ok["declared_cited_cids"].apply(_safe_len).mean()

    avg_context_sources = df_ok["num_context_sources"].mean() if "num_context_sources" in df_ok.columns else 0.0
    avg_context_chars = df_ok["context_chars"].mean() if "context_chars" in df_ok.columns else 0.0

    avg_generation_latency_ms = df_ok["generation_latency_ms"].mean() if "generation_latency_ms" in df_ok.columns else 0.0
    avg_pipeline_latency_ms = df_ok["pipeline_latency_ms"].mean() if "pipeline_latency_ms" in df_ok.columns else 0.0

    total_generation_input_tokens = df_ok["generation_input_tokens"].sum() if "generation_input_tokens" in df_ok.columns else 0
    avg_generation_input_tokens = df_ok["generation_input_tokens"].mean() if "generation_input_tokens" in df_ok.columns else 0.0
    total_generation_output_tokens = df_ok["generation_output_tokens"].sum() if "generation_output_tokens" in df_ok.columns else 0
    avg_generation_output_tokens = df_ok["generation_output_tokens"].mean() if "generation_output_tokens" in df_ok.columns else 0.0

    return {
        "total_queries": int(total_queries),
        "successful_queries": int(successful_queries),
        "failed_queries": int(failed_queries),
        "pipeline_success_rate": float(pipeline_success_rate),
        
        "answered_count": int(answered_count),
        "answered_rate": float(answered_rate),
        "insufficient_evidence_count": int(insufficient_evidence_count),
        "insufficient_evidence_rate": float(insufficient_evidence_rate),
        
        "citation_valid_count": int(citation_valid_count),
        "citation_valid_rate": float(_safe_divide(citation_valid_count, total_ok)),
        
        "invalid_citation_queries": int(invalid_citation_queries),
        "invalid_citation_query_rate": float(_safe_divide(invalid_citation_queries, total_ok)),
        "missing_declared_queries": int(missing_declared_queries),
        "missing_declared_query_rate": float(_safe_divide(missing_declared_queries, total_ok)),
        "uncited_declared_queries": int(uncited_declared_queries),
        "uncited_declared_query_rate": float(_safe_divide(uncited_declared_queries, total_ok)),
        "answered_without_citation": int(answered_without_citation),
        "answered_without_citation_rate": float(_safe_divide(answered_without_citation, total_ok)),
        
        "avg_inline_citations_per_response": float(avg_inline_citations_per_response) if not pd.isna(avg_inline_citations_per_response) else 0.0,
        "avg_declared_citations_per_response": float(avg_declared_citations_per_response) if not pd.isna(avg_declared_citations_per_response) else 0.0,
        "avg_context_sources": float(avg_context_sources) if not pd.isna(avg_context_sources) else 0.0,
        "avg_context_chars": float(avg_context_chars) if not pd.isna(avg_context_chars) else 0.0,
        
        "avg_generation_latency_ms": float(avg_generation_latency_ms) if not pd.isna(avg_generation_latency_ms) else 0.0,
        "avg_pipeline_latency_ms": float(avg_pipeline_latency_ms) if not pd.isna(avg_pipeline_latency_ms) else 0.0,
        
        "total_generation_input_tokens": int(total_generation_input_tokens),
        "avg_generation_input_tokens": float(avg_generation_input_tokens) if not pd.isna(avg_generation_input_tokens) else 0.0,
        "total_generation_output_tokens": int(total_generation_output_tokens),
        "avg_generation_output_tokens": float(avg_generation_output_tokens) if not pd.isna(avg_generation_output_tokens) else 0.0
    } 