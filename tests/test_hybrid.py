import unittest
from src.retrieval.hybrid import reciprocal_rank_fusion

class TestHybridRetrieval(unittest.TestCase):
    def test_rrf_scoring_logic(self):
        bm25 = ["A", "B", "C", "D"]
        bge = ["C", "A", "X", "B"]
        
        # rrf_k = 10
        # A: 1/(10+1) + 1/(10+2) = 1/11 + 1/12 = 0.0909 + 0.0833 = 0.1742
        # C: 1/(10+3) + 1/(10+1) = 1/13 + 1/11 = 0.0769 + 0.0909 = 0.1678
        cids, scores = reciprocal_rank_fusion([bm25, bge], rrf_k=10, top_k=5)
        
        self.assertEqual(len(cids), 5)
        self.assertEqual(len(set(cids)), 5) # Unique
        self.assertEqual(cids[0], "A") # A should be first
        self.assertEqual(cids[1], "C") # C should be second

    def test_single_retriever_contribution(self):
        # A document only in one ranking still gets retained
        r1 = ["A"]
        r2 = ["B"]
        cids, scores = reciprocal_rank_fusion([r1, r2], rrf_k=60, top_k=2)
        self.assertIn("A", cids)
        self.assertIn("B", cids)

    def test_top_k_truncation(self):
        cids, scores = reciprocal_rank_fusion([["A", "B", "C"], ["D", "E"]], rrf_k=60, top_k=2)
        self.assertEqual(len(cids), 2)
        
    def test_empty_rankings(self):
        cids, scores = reciprocal_rank_fusion([[], []], rrf_k=60, top_k=2)
        self.assertEqual(len(cids), 0)

if __name__ == '__main__':
    unittest.main()
