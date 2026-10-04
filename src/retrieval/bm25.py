import os
import re
import pickle
import numpy as np
from typing import List, Tuple, Union

try:
    import bm25s
except ImportError:
    raise ImportError("Vui lòng cài đặt bm25s: pip install bm25s")

class BM25Retriever:
    """
    A robust BM25 Retriever wrapper using the highly optimized `bm25s` library.
    Specifically designed for Vietnamese Legal documents.
    """
    def __init__(self, lowercase: bool = True, k1: float = 1.5, b: float = 0.75):
        self.lowercase = lowercase
        self.k1 = k1
        self.b = b
        self.model = None
        self.cids = [] # Maps row indices to actual CIDs
        
        # Regex pattern: Matches alphanumeric words, AND allows internal hyphens/slashes
        # Example matches: "Điều", "5", "123/2020/NĐ-CP", "khoản", "2a"
        self.token_pattern = re.compile(r"[\w]+(?:[-/][\w]+)*")

    def _legal_tokenize(self, text: str) -> List[str]:
        """
        Custom tokenizer for legal texts.
        Preserves identifiers like '123/2020/NĐ-CP' instead of breaking them apart.
        """
        if not isinstance(text, str):
            return []
        if self.lowercase:
            text = text.lower()
            
        return self.token_pattern.findall(text)

    def _batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Tokenizes a batch of documents/queries."""
        return [self._legal_tokenize(text) for text in texts]

    def fit(self, cids: List[str], texts: List[str]):
        """
        Builds the BM25 index from a list of CIDs and text documents.
        """
        if len(cids) != len(texts):
            raise ValueError("Độ dài của CIDs và texts không khớp nhau!")

        self.cids = cids
        print("Tokenizing corpus (this might take a minute for large datasets)...")
        corpus_tokens = self._batch_tokenize(texts)

        print(f"Building BM25 index for {len(corpus_tokens)} documents...")
        self.model = bm25s.BM25(k1=self.k1, b=self.b)
        self.model.index(corpus_tokens)
        print("BM25 index built successfully.")

    def search(self, query: str, k: int = 20) -> Tuple[List[str], List[float]]:
        """
        Searches the index for a single query.
        Returns: (List of CIDs, List of Scores)
        """
        if self.model is None:
            raise RuntimeError("Model chưa được load hoặc build. Hãy gọi fit() hoặc load().")

        query_tokens = self._legal_tokenize(query)
        
        # bm25s retrieve expects a list of tokenized queries
        results, scores = self.model.retrieve([query_tokens], k=k)
        
        # results and scores are returned as 2D arrays (batch_size, k)
        top_indices = results[0]
        top_scores = scores[0]

        # Map internal row indices back to the original CIDs
        retrieved_cids = [str(self.cids[idx]) for idx in top_indices]
        retrieved_scores = top_scores.tolist()

        return retrieved_cids, retrieved_scores

    def batch_search(self, queries: List[str], k: int = 20) -> Tuple[List[List[str]], List[List[float]]]:
        if self.model is None:
            raise RuntimeError("Model chưa được load hoặc build.")

        queries_tokens = self._batch_tokenize(queries)
         
        import os
        cpu_cores = os.cpu_count() or 2
        print(f"      -> Utilizing {cpu_cores} CPU cores for parallel searching...")
        
        # results shape: (num_queries, k)
        results, scores = self.model.retrieve(queries_tokens, k=k, n_threads=cpu_cores)

        batch_retrieved_cids = []
        batch_retrieved_scores = []

        for i in range(len(queries)):
            cids_for_q = [str(self.cids[idx]) for idx in results[i]]
            scores_for_q = scores[i].tolist()
            batch_retrieved_cids.append(cids_for_q)
            batch_retrieved_scores.append(scores_for_q)

        return batch_retrieved_cids, batch_retrieved_scores 

    def save(self, save_dir: str):
        """
        Saves the BM25 index and CID mappings to disk.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save bm25s model
        self.model.save(save_dir)
        
        # Save CID mapping
        cids_path = os.path.join(save_dir, "cids.pkl")
        with open(cids_path, 'wb') as f:
            pickle.dump(self.cids, f)
            
        print(f"Model and CIDs saved to: {save_dir}")

    def load(self, load_dir: str):
        """
        Loads the BM25 index and CID mappings from disk.
        """
        # Load bm25s model
        # set load_corpus=False to save RAM (we only need the index for retrieval, not the original texts)
        self.model = bm25s.BM25.load(load_dir, load_corpus=False)
        
        # Load CID mapping
        cids_path = os.path.join(load_dir, "cids.pkl")
        with open(cids_path, 'rb') as f:
            self.cids = pickle.load(f)
            
        # Manually reset k1 and b from loaded model parameters
        self.k1 = self.model.k1
        self.b = self.model.b
        
        print(f"Loaded BM25 index with {len(self.cids)} documents from {load_dir}") 