import numpy as np
from typing import Dict, List, Set, Union, Any

def recall_at_k(actual: Set[str], predicted: List[str], k: int) -> float:
    """
    Computes Recall@K.
    Proportion of relevant documents found in the top-K retrieved documents.
    """
    if not actual:
        return 0.0
    
    predicted_k = predicted[:k]
    # Count how many of the top-k predicted are in the actual set
    relevant_retrieved = len(actual.intersection(predicted_k))
    
    return relevant_retrieved / len(actual)

def mrr_at_k(actual: Set[str], predicted: List[str], k: int) -> float:
    """
    Computes Mean Reciprocal Rank (MRR@K).
    Focuses on the rank of the *first* relevant document found.
    """
    if not actual:
        return 0.0
        
    predicted_k = predicted[:k]
    for i, p in enumerate(predicted_k):
        if p in actual:
            return 1.0 / (i + 1)  # Rank is 1-based index (i + 1)
            
    return 0.0

def ndcg_at_k(actual: Set[str], predicted: List[str], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@K).
    Accounts for both the presence and the position of relevant documents.
    """
    if not actual:
        return 0.0
        
    predicted_k = predicted[:k]
    
    # 1. Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, p in enumerate(predicted_k):
        if p in actual:
            # Relevance is binary (1) in this benchmark
            dcg += 1.0 / np.log2(i + 2) # i=0 corresponds to rank 1, log2(2) = 1
            
    # 2. Calculate IDCG (Ideal DCG)
    idcg = 0.0
    # The ideal scenario is finding all relevant documents at the very top
    ideal_len = min(len(actual), k)
    for i in range(ideal_len):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(qrels: Dict[Any, Union[List[str], Set[str]]], 
                       results: Dict[Any, List[str]], 
                       k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    Evaluates retrieval performance across an entire dataset.
    
    Args:
        qrels (dict): Ground truth mappings {qid: [gold_cids]}
        results (dict): Predicted mappings {qid: [predicted_cids]}
        k_values (list): List of K thresholds to evaluate.
        
    Returns:
        dict: Aggregated mean scores for Recall, MRR, and nDCG at each K.
    """
    # Initialize dictionary to hold all metric scores for all queries
    metrics_store = {f"{metric}@{k}": [] for metric in ["Recall", "MRR", "nDCG"] for k in k_values}
    
    for qid, actual_cids in qrels.items():
        # Force all actuals to be a set of strings for O(1) lookup and safety
        actual_set = set(str(cid) for cid in actual_cids)
        
        # Force all predictions to be a list of strings
        predicted_list = [str(cid) for cid in results.get(qid, [])]
        
        for k in k_values:
            metrics_store[f"Recall@{k}"].append(recall_at_k(actual_set, predicted_list, k))
            metrics_store[f"MRR@{k}"].append(mrr_at_k(actual_set, predicted_list, k))
            metrics_store[f"nDCG@{k}"].append(ndcg_at_k(actual_set, predicted_list, k))
            
    # Calculate the mean for each metric across all queries
    aggregated_metrics = {key: round(np.mean(val), 4) for key, val in metrics_store.items()}
    
    return aggregated_metrics

if __name__ == "__main__":
    # Quick Sanity Check
    print("Running sanity check for metrics...")
    
    dummy_qrels = {
        "q1": ["100", "101"], # Multi-gold
        "q2": ["200"]         # Single-gold
    }
    
    dummy_results = {
        "q1": ["99", "100", "102"], # Hits at rank 2
        "q2": ["200", "201", "202"] # Hits at rank 1
    }
    
    scores = evaluate_retrieval(dummy_qrels, dummy_results, k_values=[1, 3])
    
    print("\nSanity Check Results:")
    for metric, score in scores.items():
        print(f"{metric}: {score}") 