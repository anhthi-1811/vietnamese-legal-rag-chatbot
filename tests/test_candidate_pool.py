import unittest
from src.retrieval.candidate_pool import build_candidate_pool

class TestCandidatePool(unittest.TestCase):
    def test_deduplication_and_source(self):
        bm25_cids = ["doc1", "doc2", "doc3"]
        bm25_scores = [10.0, 9.0, 8.0]
        
        bge_cids = ["doc3", "doc4", "doc1"]
        bge_scores = [0.9, 0.8, 0.7]
        
        pool = build_candidate_pool(bm25_cids, bm25_scores, bge_cids, bge_scores)
        
        # Should contain exactly 4 unique docs: doc1, doc2, doc3, doc4
        self.assertEqual(len(pool), 4)
        
        # Check specific doc tracking
        doc1 = next(d for d in pool if d["cid"] == "doc1")
        self.assertEqual(doc1["source"], "both")
        self.assertEqual(doc1["bm25_rank"], 1)
        self.assertEqual(doc1["bge_rank"], 3)
        self.assertEqual(doc1["bm25_score"], 10.0)
        self.assertEqual(doc1["bge_score"], 0.7)
        
        doc2 = next(d for d in pool if d["cid"] == "doc2")
        self.assertEqual(doc2["source"], "bm25")
        self.assertIsNone(doc2["bge_rank"])
        
        doc4 = next(d for d in pool if d["cid"] == "doc4")
        self.assertEqual(doc4["source"], "bge")
        self.assertIsNone(doc4["bm25_rank"])

if __name__ == '__main__':
    unittest.main()
