import os
import re
import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Create a fixed random subset for RAG evaluation.")
    parser.add_argument("--input", required=True, help="Path to input reranker rankings parquet")
    parser.add_argument("--output", required=True, help="Path to save the subset parquet")
    parser.add_argument("--n", type=int, default=100, help="Number of queries to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def count_golds(gold_cids_val):
    """Trích xuất số lượng CID an toàn bất chấp kiểu numpy array, list hay string."""
    if gold_cids_val is None:
        return 0
    # Dùng regex quét mọi con số nguyên trong biểu diễn chuỗi
    cids = re.findall(r'\d+', str(gold_cids_val))
    return len(cids)

def main():
    args = parse_args()
    
    print(f"Reading input file: {args.input}")
    df = pd.read_parquet(args.input)
    total_rows = len(df)
    
    # Kiểm tra QID unique
    if df['qid'].nunique() != total_rows:
        raise ValueError("Input data contains duplicate QIDs!")
        
    print(f"Total unique queries in input: {total_rows}")
    
    # Tính số lượng gold CIDs an toàn
    df['gold_count'] = df['gold_cids'].apply(count_golds)
    df['gold_type'] = df['gold_count'].apply(lambda x: 'single' if x == 1 else ('multi' if x > 1 else 'zero'))
    
    # Lấy mẫu phân lớp (Stratified Sampling) giữ nguyên tỷ lệ Single/Multi
    if args.n >= total_rows:
        print(f"Warning: Requested {args.n} queries but only {total_rows} available. Using all.")
        sampled_df = df.copy()
    else:
        # Lấy mẫu theo từng nhóm gold_type
        sampled_groups = []
        for name, group in df.groupby('gold_type'):
            n_group = int(round(len(group) / total_rows * args.n))
            if n_group > 0:
                sampled_groups.append(group.sample(n=min(n_group, len(group)), random_state=args.seed))
                
        sampled_df = pd.concat(sampled_groups)
        
        # Bù trừ sai số làm tròn nếu chưa đủ hoặc vượt n
        if len(sampled_df) < args.n:
            shortfall = args.n - len(sampled_df)
            remaining = df.drop(sampled_df.index)
            extra = remaining.sample(n=shortfall, random_state=args.seed)
            sampled_df = pd.concat([sampled_df, extra])
        elif len(sampled_df) > args.n:
            sampled_df = sampled_df.sample(n=args.n, random_state=args.seed)
            
    # Sắp xếp lại theo index ban đầu
    sampled_df = sampled_df.sort_index()
    
    # Giữ nguyên toàn bộ schema gốc ban đầu
    sampled_df = sampled_df.drop(columns=['gold_count', 'gold_type'])
    
    # Lưu file parquet
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sampled_df.to_parquet(args.output, index=False)
    
    # Thống kê kết quả
    print("\n" + "="*50)
    print("SUBSET CREATION SUMMARY")
    print("="*50)
    print(f"Sampled Queries: {len(sampled_df)}")
    print(f"Seed used:       {args.seed}")
    
    gold_counts = sampled_df['gold_cids'].apply(count_golds)
    single_count = (gold_counts == 1).sum()
    multi_count = (gold_counts > 1).sum()
    zero_count = (gold_counts == 0).sum()
    
    print("\nGold Distribution in Subset:")
    print(f"- Single-gold: {single_count} ({(single_count/len(sampled_df))*100:.1f}%)")
    print(f"- Multi-gold:  {multi_count} ({(multi_count/len(sampled_df))*100:.1f}%)")
    if zero_count > 0:
        print(f"- Zero-gold:   {zero_count} ({(zero_count/len(sampled_df))*100:.1f}%)")
    print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main() 