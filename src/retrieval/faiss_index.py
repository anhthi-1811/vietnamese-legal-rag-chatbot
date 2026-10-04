import os
import faiss
import numpy as np
import json
from typing import List, Tuple

class FAISSRetriever:
    """
    Wrapper for FAISS IndexFlatIP (Inner Product).
    Designed to work with L2-normalized embeddings for Cosine Similarity.
    """
    def __init__(self):
        self.index = None
        self.cids = np.array([])
        self.dim = 0

    def build(self, cids: List[str], embeddings: np.ndarray):
        """
        Initializes and populates the FAISS index.
        """
        if len(cids) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(cids)} CIDs vs {embeddings.shape[0]} embeddings.")
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.cids = np.array(cids, dtype=str)
        self.dim = embeddings.shape[1]
        
        print(f"Building FAISS IndexFlatIP (Dimension: {self.dim}) for {embeddings.shape[0]:,} documents...")
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        
        if self.index.ntotal != len(cids):
            raise RuntimeError(f"Index count ({self.index.ntotal}) != CID count ({len(cids)})")
        
        print("FAISS index built successfully.")

    def search(self, query_embeddings: np.ndarray, k: int = 20) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Searches the index using query embeddings.
        Returns mapped CIDs and scores.
        """
        if self.index is None:
            raise RuntimeError("FAISS Index is not built or loaded.")
            
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        # FAISS search
        scores, I = self.index.search(query_embeddings, k)

        batch_retrieved_cids = []
        batch_retrieved_scores = []

        for i in range(len(query_embeddings)):
            cids_for_q = self.cids[I[i]].tolist()
            scores_for_q = [float(x) for x in scores[i]] # typecast for JSON/Parquet safety 
            
            batch_retrieved_cids.append(cids_for_q)
            batch_retrieved_scores.append(scores_for_q)

        return batch_retrieved_cids, batch_retrieved_scores

    def save(self, save_dir: str):
        """Saves the FAISS index, CID mappings, and metadata."""
        os.makedirs(save_dir, exist_ok=True)
        
        index_path = os.path.join(save_dir, "bge_m3.index")
        faiss.write_index(self.index, index_path)
        
        cid_path = os.path.join(save_dir, "cid_mapping.npy")
        np.save(cid_path, self.cids)
        
        meta_path = os.path.join(save_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump({"ntotal": self.index.ntotal, "dim": self.dim}, f)
            
        print(f"Saved FAISS artifacts to: {save_dir}")

    def load(self, load_dir: str):
        """Loads the FAISS index and CID mappings from disk."""
        index_path = os.path.join(load_dir, "bge_m3.index")
        cid_path = os.path.join(load_dir, "cid_mapping.npy")
        meta_path = os.path.join(load_dir, "metadata.json")
        
        print(f"Loading FAISS artifacts from {load_dir}...")
        self.index = faiss.read_index(index_path)
        self.cids = np.load(cid_path)
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            self.dim = meta["dim"]
            
        if self.index.ntotal != len(self.cids):
            raise RuntimeError("Corrupted save: Index ntotal does not match cid_mapping length.")
        print(f"Successfully loaded {self.index.ntotal:,} vectors.") 