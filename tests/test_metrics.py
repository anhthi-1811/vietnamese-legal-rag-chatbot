import unittest
import numpy as np
from src.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k

class TestRetrievalMetrics(unittest.TestCase):

    def test_single_gold_at_rank_1(self):
        actual = {"100"}
        predicted = ["100", "101", "102"]
        self.assertEqual(recall_at_k(actual, predicted, k=3), 1.0)
        self.assertEqual(mrr_at_k(actual, predicted, k=3), 1.0)
        self.assertEqual(ndcg_at_k(actual, predicted, k=3), 1.0)

    def test_single_gold_at_rank_5(self):
        actual = {"100"}
        predicted = ["101", "102", "103", "104", "100", "105"]
        self.assertEqual(recall_at_k(actual, predicted, k=5), 1.0)
        self.assertEqual(mrr_at_k(actual, predicted, k=5), 1/5)
        expected_ndcg = (1.0 / np.log2(6)) / 1.0
        self.assertAlmostEqual(ndcg_at_k(actual, predicted, k=5), expected_ndcg, places=4)
        self.assertEqual(recall_at_k(actual, predicted, k=4), 0.0)
        self.assertEqual(mrr_at_k(actual, predicted, k=4), 0.0)

    def test_multi_gold(self):
        actual = {"100", "200"}
        predicted = ["100", "99", "200", "101"]
        self.assertEqual(recall_at_k(actual, predicted, k=3), 1.0)
        self.assertEqual(mrr_at_k(actual, predicted, k=3), 1.0)
        self.assertEqual(recall_at_k(actual, predicted, k=1), 0.5)

    def test_no_gold_retrieved(self):
        actual = {"100"}
        predicted = ["200", "201", "202"]
        self.assertEqual(recall_at_k(actual, predicted, k=3), 0.0)
        self.assertEqual(mrr_at_k(actual, predicted, k=3), 0.0)
        self.assertEqual(ndcg_at_k(actual, predicted, k=3), 0.0)

    def test_duplicate_gold_in_predictions(self):
        actual = {"100"}
        predicted = ["100", "200", "100"]
        self.assertEqual(recall_at_k(actual, predicted, k=3), 1.0)
        self.assertEqual(mrr_at_k(actual, predicted, k=3), 1.0)

    def test_empty_actual(self):
        actual = set()
        predicted = ["100", "101"]
        self.assertEqual(recall_at_k(actual, predicted, k=2), 0.0)
        self.assertEqual(mrr_at_k(actual, predicted, k=2), 0.0)
        self.assertEqual(ndcg_at_k(actual, predicted, k=2), 0.0)

if __name__ == '__main__':
    unittest.main()
