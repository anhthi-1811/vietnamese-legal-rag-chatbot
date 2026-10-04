from typing import List, Set

def is_correct_at_k(gold_cids: Set[str], retrieved_cids: List[str], k: int) -> bool:
    """Checks if at least one gold CID is in the top-K retrieved CIDs."""
    return len(gold_cids.intersection(set(retrieved_cids[:k]))) > 0

def analyze_complementarity_at_k(golds: List[Set[str]], preds_a: List[List[str]], preds_b: List[List[str]], k: int):
    """
    Categorizes queries into BOTH_CORRECT, A_ONLY_CORRECT, B_ONLY_CORRECT, BOTH_WRONG.
    """
    counts = {
        "BOTH_CORRECT": 0,
        "A_ONLY_CORRECT": 0,
        "B_ONLY_CORRECT": 0,
        "BOTH_WRONG": 0
    }
    
    categories = []
    
    for gold, p_a, p_b in zip(golds, preds_a, preds_b):
        correct_a = is_correct_at_k(gold, p_a, k)
        correct_b = is_correct_at_k(gold, p_b, k)
        
        if correct_a and correct_b:
            cat = "BOTH_CORRECT"
        elif correct_a and not correct_b:
            cat = "A_ONLY_CORRECT"
        elif not correct_a and correct_b:
            cat = "B_ONLY_CORRECT"
        else:
            cat = "BOTH_WRONG"
            
        counts[cat] += 1
        categories.append(cat)
        
    total = len(golds)
    percentages = {k: round((v / total) * 100, 2) for k, v in counts.items()}
    
    return counts, percentages, categories
