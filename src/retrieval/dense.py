import numpy as np
from typing import List
import torch

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    raise ImportError("Vui lòng cài đặt thư viện: pip install -U FlagEmbedding")

class BGEM3Encoder:
    """
    Wrapper for BAAI/bge-m3 dense embedding generation.
    Supports GPU, FP16, and safety checks for robust encoding.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        
        print(f"Loading BGEM3FlagModel ({model_name}) | FP16: {use_fp16} ...")
        # Load the official BGE-M3 model
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        print("Model loaded successfully.")

    def encode_documents(self, texts: List[str], batch_size: int = 16, max_length: int = 1024) -> np.ndarray:
        """
        Encodes a batch of documents into dense vectors.
        Includes automatic L2 normalization.
        """
        if not texts:
            return np.array([])

        # bge-m3 encode function handles batching and tokenization internally
        embeddings_dict = self.model.encode(
            texts, 
            batch_size=batch_size, 
            max_length=max_length, 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )
        
        dense_vecs = embeddings_dict['dense_vecs']
        
        # Defensive check
        self._validate_embeddings(dense_vecs)
        return dense_vecs

    def encode_queries(self, queries: List[str], batch_size: int = 16, max_length: int = 512) -> np.ndarray:
        """
        Encodes queries. max_length is typically shorter for queries.
        """
        if not queries:
            return np.array([])
            
        embeddings_dict = self.model.encode(
            queries, 
            batch_size=batch_size, 
            max_length=max_length, 
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )
        
        dense_vecs = embeddings_dict['dense_vecs']
        self._validate_embeddings(dense_vecs)
        return dense_vecs

    def _validate_embeddings(self, embeddings: np.ndarray):
        """Ensures vectors are not corrupted (NaN, Inf, or zero-vectors)."""
        if np.isnan(embeddings).any():
            raise ValueError("CRITICAL: Embeddings contain NaN values.")
        if np.isinf(embeddings).any():
            raise ValueError("CRITICAL: Embeddings contain Infinity values.")
        
        # Check for zero vectors (magnitude == 0)
        magnitudes = np.linalg.norm(embeddings, axis=1)
        if (magnitudes == 0).any():
            raise ValueError("CRITICAL: Found zero-magnitude vectors.") 