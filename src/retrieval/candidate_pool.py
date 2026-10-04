from typing import List, Dict, Any

def build_candidate_pool(
    bm25_cids: List[str], bm25_scores: List[float],
    bge_cids: List[str], bge_scores: List[float]
) -> List[Dict[str, Any]]:
    """
    Merges BM25 and BGE-M3 candidate lists into a unified, deduplicated pool.
    Preserves source attribution, original ranks, and scores.
    """
    pool_dict = {}
    
    # Process BM25 candidates
    for rank_idx, (cid, score) in enumerate(zip(bm25_cids, bm25_scores)):
        pool_dict[cid] = {
            "cid": cid,
            "source": "bm25",
            "bm25_rank": rank_idx + 1,
            "bge_rank": None,
            "bm25_score": float(score),
            "bge_score": None
        }
        
    # Process BGE-M3 candidates
    for rank_idx, (cid, score) in enumerate(zip(bge_cids, bge_scores)):
        if cid in pool_dict:
            pool_dict[cid]["source"] = "both"
            pool_dict[cid]["bge_rank"] = rank_idx + 1
            pool_dict[cid]["bge_score"] = float(score)
        else:
            pool_dict[cid] = {
                "cid": cid,
                "source": "bge",
                "bm25_rank": None,
                "bge_rank": rank_idx + 1,
                "bm25_score": None,
                "bge_score": float(score)
            }
            
    # Return as a list (order is initially BM25 then remaining BGE, but order doesn't strictly matter for reranker input)
    return list(pool_dict.values())
