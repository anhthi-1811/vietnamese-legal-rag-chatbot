import unittest
from src.rag.citation_checker import CitationChecker

class TestCitationChecker(unittest.TestCase):
    
    def test_valid_citation(self):
        res = CitationChecker.check(
            answer="Luật bảo vệ rừng [CID=123] và đất đai [CID=456].",
            declared_cids=["123", "456"],
            allowed_cids=["123", "456", "789"],
            answer_status="answered"
        )
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["inline_cids"], ["123", "456"])
        self.assertFalse(res["invalid_cids"])
        
    def test_hallucinated_cid(self):
        res = CitationChecker.check(
            answer="Luật thuế [CID=999].",
            declared_cids=["999"],
            allowed_cids=["123", "456"],
            answer_status="answered"
        )
        self.assertFalse(res["is_valid"])
        self.assertIn("999", res["invalid_cids"])
        
    def test_inline_missing_from_declared(self):
        res = CitationChecker.check(
            answer="Theo luật [CID=123].",
            declared_cids=[],
            allowed_cids=["123"],
            answer_status="answered"
        )
        # Structural check có thể coi là Valid nhưng ném Warning
        self.assertTrue(res["is_valid"])
        self.assertIn("123", res["missing_declared_cids"])

    def test_declared_missing_from_inline(self):
        res = CitationChecker.check(
            answer="Tôi đã xem xét kỹ.",
            declared_cids=["123"],
            allowed_cids=["123"],
            answer_status="answered"
        )
        self.assertFalse(res["is_valid"]) # Answered nhưng không có inline -> Invalid
        self.assertIn("123", res["uncited_declared_cids"])

    def test_duplicate_inline_normalized(self):
        res = CitationChecker.check(
            answer="Đoạn một [CID=123]. Đoạn hai [CID=123].",
            declared_cids=["123"],
            allowed_cids=["123"],
            answer_status="answered"
        )
        self.assertEqual(res["inline_cids"], ["123"]) # Chỉ xuất hiện 1 lần
        
    def test_insufficient_evidence_with_no_citations(self):
        res = CitationChecker.check(
            answer="Các nguồn được cung cấp chưa đủ...",
            declared_cids=[],
            allowed_cids=["123"],
            answer_status="insufficient_evidence"
        )
        self.assertTrue(res["is_valid"]) # Hợp lệ vì không bắt buộc cite khi insufficient
        
    def test_integer_vs_string_normalization(self):
        res = CitationChecker.check(
            answer="Theo [CID=123]",
            declared_cids=[123], # Integer trong JSON
            allowed_cids=[123, 456], # Integer từ retrieval
            answer_status="answered"
        )
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["allowed_cids"], ["123", "456"]) 