import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.evaluation.semantic_judge import (
    SemanticRAGJudge,
    normalize_and_clean_judge_data,
    validate_judge_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run or repair semantic evaluation using LLM-as-a-judge.")
    parser.add_argument("--input", default="experiments/results/rag_train_results_eval_100_v2.parquet", help="Path to source RAG results parquet")
    parser.add_argument("--repair-existing", default="", help="Path to existing semantic eval parquet to normalize locally and rerun only invalid rows")
    parser.add_argument("--corpus", default="/content/drive/MyDrive/Vietnamese_Legal_RAG/data/processed/documents_clean.parquet", help="Path to clean corpus parquet")
    parser.add_argument("--output", default="experiments/results/rag_train_semantic_eval_100_v2_clean.parquet", help="Path to save semantic eval output")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="Judge model name")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    parser.add_argument("--request-sleep", type=float, default=2.5, help="Sleep between judge API calls (seconds)")
    return parser.parse_args()


def extract_cids(cid_val):
    if cid_val is None:
        return []
    return re.findall(r"\d+", str(cid_val))


def build_gold_reference_text(gold_cids, corpus_map):
    cids = extract_cids(gold_cids)
    if not cids:
        return "Không có Gold CID cho câu hỏi này."
    texts = []
    for cid in cids:
        text = corpus_map.get(str(cid))
        if text:
            texts.append(f"[GOLD_CID={cid}]\n{text}")
    return "\n\n".join(texts) if texts else "Không tìm thấy nội dung văn bản tham chiếu trong corpus."


def row_to_judge_dict(row: dict) -> dict:
    """Reconstructs nested judge dict from flattened DataFrame row."""
    raw_tags = row.get("semantic_error_tags", [])
    if isinstance(raw_tags, (list, tuple, np.ndarray)):
        tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    else:
        tags = []

    return {
        "answer_correctness": {
            "score": row.get("correctness_score"),
            "label": row.get("correctness_label"),
            "reason": row.get("correctness_reason", ""),
        },
        "answer_completeness": {
            "score": row.get("completeness_score"),
            "label": row.get("completeness_label"),
            "reason": row.get("completeness_reason", ""),
        },
        "groundedness": {
            "score": row.get("groundedness_score"),
            "label": row.get("groundedness_label"),
            "reason": row.get("groundedness_reason", ""),
        },
        "citation_support": {
            "score": row.get("citation_support_score"),
            "label": row.get("citation_support_label"),
            "reason": row.get("citation_support_reason", ""),
        },
        "abstention": {
            "applicable": bool(row.get("abstention_applicable", False)),
            "appropriate": bool(row.get("abstention_appropriate", False)),
            "reason": str(row.get("abstention_reason", "")),
        },
        "error_tags": tags,
    }


def apply_judge_dict_to_row(rec: dict, eval_res: dict, model_name: str) -> dict:
    """Flattens validated judge dict back into row record."""
    for dim in ["answer_correctness", "answer_completeness", "groundedness", "citation_support"]:
        dim_data = eval_res.get(dim, {})
        prefix = dim.replace("answer_", "")
        rec[f"{prefix}_score"] = int(dim_data.get("score", 0))
        rec[f"{prefix}_label"] = str(dim_data.get("label", ""))
        rec[f"{prefix}_reason"] = str(dim_data.get("reason", ""))

    abstention = eval_res.get("abstention", {})
    rec["abstention_applicable"] = bool(abstention.get("applicable", False))
    rec["abstention_appropriate"] = bool(abstention.get("appropriate", False))
    rec["abstention_reason"] = str(abstention.get("reason", ""))

    rec["semantic_error_tags"] = list(eval_res.get("error_tags", []))
    rec["judge_model"] = model_name
    if "judge_raw_response" in eval_res:
        rec["judge_raw_response"] = eval_res["judge_raw_response"]
    if "judge_latency_ms" in eval_res:
        rec["judge_latency_ms"] = eval_res["judge_latency_ms"]
    rec["judge_status"] = eval_res.get("judge_status", "ok")
    rec["judge_error"] = eval_res.get("judge_error", "")
    return rec


def main():
    args = parse_args()
    print(f"Reading source RAG file: {args.input}")
    df_rag = pd.read_parquet(args.input)

    print(f"Loading corpus from: {args.corpus}")
    df_corpus = pd.read_parquet(args.corpus)
    id_col = "cid" if "cid" in df_corpus.columns else ("id" if "id" in df_corpus.columns else df_corpus.columns[0])
    text_col = "text_clean" if "text_clean" in df_corpus.columns else ("text" if "text" in df_corpus.columns else df_corpus.columns[1])
    corpus_map = dict(zip(df_corpus[id_col].astype(str), df_corpus[text_col].astype(str)))
    print(f"Corpus loaded & indexed: {len(corpus_map)} documents.")

    judge = SemanticRAGJudge(model_name=args.model)

    # Mode 1: Repair an existing semantic evaluation file (normalize locally + rerun only invalid rows)
    if args.repair_existing and os.path.exists(args.repair_existing):
        print(f"\nInspecting existing semantic evaluation file: {args.repair_existing}")
        df_existing = pd.read_parquet(args.repair_existing)
        existing_by_qid = {r["qid"]: r.to_dict() for _, r in df_existing.iterrows()}

        records = []
        locally_cleaned_count = 0
        rerun_qids = []

        for _, rag_row in df_rag.iterrows():
            qid = rag_row["qid"]
            rec = existing_by_qid.get(qid, rag_row.to_dict())
            answer_status = str(rag_row.get("answer_status", ""))

            # Ensure source RAG answer fields are untouched from df_rag
            for col in df_rag.columns:
                rec[col] = rag_row[col]

            if qid in existing_by_qid and rec.get("judge_status") == "ok":
                j_dict = row_to_judge_dict(rec)
                cleaned_dict, warnings = normalize_and_clean_judge_data(j_dict, answer_status)
                val_errors = validate_judge_data(cleaned_dict, answer_status)

                if not val_errors:
                    if warnings:
                        locally_cleaned_count += 1
                    cleaned_dict["judge_status"] = "ok"
                    cleaned_dict["judge_error"] = ""
                    rec = apply_judge_dict_to_row(rec, cleaned_dict, rec.get("judge_model", args.model))
                    records.append(rec)
                    continue
                else:
                    print(f"  [INVALID ROW -> SCHEDULED FOR RERUN] QID={qid}: {'; '.join(val_errors)}")
                    rerun_qids.append(qid)
            else:
                print(f"  [MISSING/ERROR ROW -> SCHEDULED FOR RERUN] QID={qid}")
                rerun_qids.append(qid)

            # Rerun SemanticRAGJudge ONLY for this invalid/missing row
            ref_context = build_gold_reference_text(rag_row.get("gold_cids"), corpus_map)
            retrieved_context = str(rag_row.get("context_str", ""))

            eval_res = judge.evaluate(
                question=str(rag_row.get("question", "")),
                answer=str(rag_row.get("answer", "")),
                answer_status=answer_status,
                reference_context=ref_context,
                retrieved_context=retrieved_context,
            )
            rec = apply_judge_dict_to_row(rec, eval_res, args.model)
            records.append(rec)
            if args.request_sleep > 0:
                time.sleep(args.request_sleep)

        df_out = pd.DataFrame(records)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df_out.to_parquet(args.output, index=False)
        print("\n==================================================")
        print(" REPAIR & TARGETED RERUN COMPLETE")
        print("==================================================")
        print(f"Total rows processed:            {len(df_out)}")
        print(f"Normalized/cleaned locally:      {locally_cleaned_count} (0 API calls)")
        print(f"Reran via SemanticRAGJudge API:  {len(rerun_qids)} (QIDs: {rerun_qids})")
        print(f"Saved clean output to:           {args.output}")
        return

    # Mode 2: Standard evaluation run with --resume support
    completed_qids = set()
    records = []
    if args.resume and os.path.exists(args.output):
        try:
            df_existing = pd.read_parquet(args.output)
            completed_qids = set(df_existing["qid"].tolist())
            records = df_existing.to_dict("records")
            print(f"Resuming: {len(completed_qids)} queries already evaluated.")
        except Exception as e:
            print(f"Could not resume from existing file ({e}). Starting fresh.")

    pbar = tqdm(total=len(df_rag), desc="Evaluating semantics")
    for _, row in df_rag.iterrows():
        qid = row["qid"]
        if qid in completed_qids:
            pbar.update(1)
            continue

        ref_context = build_gold_reference_text(row.get("gold_cids"), corpus_map)
        retrieved_context = str(row.get("context_str", ""))
        answer_status = str(row.get("answer_status", ""))

        eval_res = judge.evaluate(
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            answer_status=answer_status,
            reference_context=ref_context,
            retrieved_context=retrieved_context,
        )

        rec = apply_judge_dict_to_row(row.to_dict(), eval_res, args.model)
        records.append(rec)
        pbar.update(1)

        if len(records) % 10 == 0:
            pd.DataFrame(records).to_parquet(args.output, index=False)
        if args.request_sleep > 0:
            time.sleep(args.request_sleep)

    pbar.close()
    df_out = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df_out.to_parquet(args.output, index=False)
    print(f"\nSaved {len(df_out)} evaluated records to {args.output}")


if __name__ == "__main__":
    main()
