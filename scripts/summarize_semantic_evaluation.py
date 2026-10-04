import argparse
import ast
from collections import Counter
import numpy as np
import pandas as pd

def summarize(input_file, export_excel=None):
    df = pd.read_parquet(input_file)
    total = len(df)

    # 1. Phân loại theo GENERATOR OUTPUT gốc (không phụ thuộc Judge thành công hay thất bại)
    if "answer_status" in df.columns:
        is_abstain = df["answer_status"] == "insufficient_evidence"
    else:
        is_abstain = df["abstention_applicable"] == True

    gen_answered = int((~is_abstain).sum())
    gen_abstain = int(is_abstain.sum())

    # 2. Phân loại theo trạng thái SEMANTIC JUDGE
    judge_ok_mask = df["judge_status"] == "ok"
    judge_ok = int(judge_ok_mask.sum())
    judge_failed = total - judge_ok

    print("=" * 55)
    print(" SUMMARY: LEGAL RAG SEMANTIC EVALUATION")
    print("=" * 55)
    print("GENERATOR OUTPUT")
    print(f"- Total queries:         {total}")
    print(f"- Answered:              {gen_answered} ({gen_answered/total*100:.1f}%)")
    print(f"- Insufficient evidence: {gen_abstain} ({gen_abstain/total*100:.1f}%)")
    print("-" * 55)
    print("SEMANTIC JUDGE")
    print(f"- Successfully judged:   {judge_ok} / {total} ({judge_ok/total*100:.1f}%)")
    print(f"- Failed / missing:      {judge_failed} / {total} ({judge_failed/total*100:.1f}%)")

    # Kiểm tra schema trên các dòng đã chấm thành công
    df_ok = df[judge_ok_mask].copy()
    df_ans_ok = df_ok[~is_abstain.loc[df_ok.index]]
    df_abs_ok = df_ok[is_abstain.loc[df_ok.index]]

    print("-" * 55)
    print(f"Judged Breakdown: {len(df_ans_ok)} Answered | {len(df_abs_ok)} Insufficient Evidence")

    print(f"\n>>> METRICS FOR ANSWERED QUERIES (n = {len(df_ans_ok)}) <<<")
    for metric, valid_Order in [
        ("correctness", ["correct", "partial", "incorrect"]),
        ("completeness", ["complete", "partial", "incomplete"]),
        ("groundedness", ["supported", "partially_supported", "unsupported"]),
        ("citation_support", ["supported", "partially_supported", "unsupported"]),
    ]:
        s_col = f"{metric}_score"
        l_col = f"{metric}_label"
        mean_val = pd.to_numeric(df_ans_ok[s_col], errors="coerce").mean()
        print(f"\n--- {metric.upper()} ---")
        print(f"Mean Score: {mean_val:.2f} / 2.0")
        counts = df_ans_ok[l_col].value_counts(normalize=True) * 100
        for lbl in valid_Order:
            print(f"  % {lbl}: {counts.get(lbl, 0.0):.1f}%")

    print(f"\n>>> ABSTENTION METRICS (Insufficient Evidence, n = {len(df_abs_ok)}) <<<")
    print(f"Total Judged Abstentions: {len(df_abs_ok)} / {gen_abstain}")
    if len(df_abs_ok) > 0:
        valid_abs = int((df_abs_ok["abstention_appropriate"] == True).sum())
        unnec_abs = int((df_abs_ok["abstention_appropriate"] == False).sum())
        print(f"Appropriate (Valid) Abstention Rate:     {valid_abs/len(df_abs_ok)*100:.1f}% ({valid_abs}/{len(df_abs_ok)})")
        print(f"Unnecessary (Invalid) Abstention Rate:   {unnec_abs/len(df_abs_ok)*100:.1f}% ({unnec_abs}/{len(df_abs_ok)})")

    # Error tags
    all_tags = []
    for val in df_ok["semantic_error_tags"].dropna():
        if isinstance(val, (list, np.ndarray)):
            all_tags.extend([str(x) for x in val if str(x).strip()])
        elif isinstance(val, str) and val.startswith("["):
            try:
                all_tags.extend([str(x) for x in ast.literal_eval(val) if str(x).strip()])
            except Exception:
                all_tags.extend([t.strip(" '\"[]") for t in val.split() if t.strip(" '\"[]")])

    print("\n>>> ERROR TAGS FREQUENCY <<<")
    for tag, cnt in Counter(all_tags).most_common():
        print(f"- {tag}: {cnt}")

    print("=" * 55)
    if export_excel:
        df.to_excel(export_excel, index=False)
        print(f"Exported detailed results for manual review to: {export_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--export-excel", default=None)
    args = parser.parse_args()
    summarize(args.input, args.export_excel)
