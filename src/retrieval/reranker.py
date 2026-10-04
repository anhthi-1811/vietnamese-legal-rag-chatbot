import numpy as np
from typing import List, Tuple

try:
    from FlagEmbedding import FlagReranker
except ImportError:
    raise ImportError("Vui lòng cài đặt thư viện: pip install -U FlagEmbedding")

class CrossEncoderReranker:
    """
    Wrapper for BAAI/bge-reranker-v2-m3.
    Uses Cross-Encoder architecture to compute relevance scores for (query, document) pairs.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        print(f"Loading Cross-Encoder Reranker ({model_name}) | FP16: {use_fp16} ...")
        # Load the official FlagReranker
        self.model = FlagReranker(model_name, use_fp16=use_fp16)
        print("Reranker loaded successfully.")

    def compute_scores(self, sentence_pairs: List[Tuple[str, str]], batch_size: int = 256, max_length: int = 1024) -> List[float]:
        """
        Computes relevance scores for a list of (query, document) pairs.
        FlagReranker automatically handles batching and tokenization internally.
        """
        if not sentence_pairs:
            return []

        # FlagReranker expects a list of [query, doc] lists or tuples
        scores = self.model.compute_score(
            sentence_pairs, 
            batch_size=batch_size, 
            max_length=max_length
        )
        
        # If only one pair is passed, FlagEmbedding might return a single float instead of a list
        if isinstance(scores, float):
            return [scores]
            
        return scores 