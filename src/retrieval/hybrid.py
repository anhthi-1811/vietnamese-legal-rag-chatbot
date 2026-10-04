from typing import List, Tuple
from collections import defaultdict

def reciprocal_rank_fusion(rankings_list: List[List[str]], rrf_k: int = 60, top_k: int = 20) -> Tuple[List[str], List[float]]:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        rankings_list: A list of rankings (each ranking is a list of CIDs).
        rrf_k: The RRF constant.
        top_k: Number of final candidates to return.
        
    Returns:
        fused_cids: Top K fused CIDs.
        fused_scores: Corresponding RRF scores.
    """
    rrf_scores = defaultdict(float)
    
    for ranking in rankings_list:
        for rank_idx, cid in enumerate(ranking):
            rank = rank_idx + 1 # Rank starts at 1
            rrf_scores[cid] += 1.0 / (rrf_k + rank)
            
    # Sort descending by fused score
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Truncate to top_k
    fused_cids = [cid for cid, score in sorted_candidates[:top_k]]
    fused_scores = [score for cid, score in sorted_candidates[:top_k]]
    
    return fused_cids, fused_scores

def batch_rrf(batch_rankings_list: List[List[List[str]]], rrf_k: int = 60, top_k: int = 20):
    """Applies RRF across a batch of queries."""
    batch_fused_cids = []
    batch_fused_scores = []
    
    # Transpose list of rankings to iterate per query
    num_queries = len(batch_rankings_list[0])
    for i in range(num_queries):
        query_rankings = [retriever_batch[i] for retriever_batch in batch_rankings_list]
        fused_cids, fused_scores = reciprocal_rank_fusion(query_rankings, rrf_k, top_k)
        batch_fused_cids.append(fused_cids)
        batch_fused_scores.append(fused_scores)
        
    return batch_fused_cids, batch_fused_scores
