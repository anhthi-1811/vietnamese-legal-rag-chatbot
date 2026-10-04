import unittest
from unittest.mock import MagicMock

from src.rag.passage_retriever import PassageRetriever 
from src.rag.context_builder import ContextBuilder
from src.rag.generator import LegalAnswerGenerator
from src.rag.citation_checker import CitationChecker

class TestRAGPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.retriever = PassageRetriever(long_doc_threshold_chars=50, max_passage_chars=100)
        self.builder = ContextBuilder(max_context_chars=1000)
        
        self.generator = LegalAnswerGenerator(model_name="mock-model")
        # Mock client của google-genai
        self.generator.client = MagicMock()

    def _mock_gemini_response(self, text_content):
        mock_resp = MagicMock()
        mock_resp.text = text_content
        mock_resp.usage_metadata.prompt_token_count = 10
        mock_resp.usage_metadata.candidates_token_count = 20
        mock_resp.usage_metadata.total_token_count = 30
        return mock_resp

    def test_end_to_end_pipeline_with_toy_data(self):
        question = "Mức phạt lỗi chạy xe quá tốc độ là bao nhiêu?"
        raw_documents = [
            {
                "cid": "DOC_01",
                "text": "Khoản 1: Chạy quá tốc độ từ 5-10km/h bị phạt 300.000 đồng.", 
                "score": 0.95
            },
            {
                "cid": "DOC_02",
                "text": "Chương 1 quy định chung. " * 5 + "Khoản 2: Chạy quá tốc độ trên 20km/h phạt 4 triệu đồng. " + "Chương 3 các điều khoản khác. " * 5,
                "score": 0.85
            }
        ]

        # 1. Local Passage Selection
        processed_docs = self.retriever.process_documents(question, raw_documents)
        self.assertEqual(len(processed_docs), 2)

        # 2. Context Builder
        context_result = self.builder.build_context(processed_docs)
        context_str = context_result["context_str"]
        allowed_cids = context_result["included_cids"]

        # 3. Generator (Mocked google-genai interface)
        mocked_json_output = """
        {
            "answer": "Chạy quá tốc độ từ 5-10km/h bị phạt 300.000 đồng [CID=DOC_01]. Nếu trên 20km/h phạt 4 triệu đồng [CID=DOC_02].",
            "cited_cids": ["DOC_01", "DOC_02"],
            "answer_status": "answered"
        }
        """
        self.generator.client.models.generate_content.return_value = self._mock_gemini_response(mocked_json_output)
        
        generation_result = self.generator.generate(
            question=question, 
            context_str=context_str, 
            allowed_cids=allowed_cids
        )

        self.assertEqual(generation_result["answer_status"], "answered")
        self.assertEqual(generation_result["cited_cids"], ["DOC_01", "DOC_02"])

        # 4. Citation Checker
        check_result = CitationChecker.check(
            answer=generation_result["answer"],
            declared_cids=generation_result["cited_cids"],
            allowed_cids=allowed_cids,
            answer_status=generation_result["answer_status"]
        )

        self.assertTrue(check_result["is_valid"])
        self.assertTrue(check_result["has_any_citation"])
        self.assertEqual(len(check_result["invalid_cids"]), 0)

if __name__ == '__main__':
    unittest.main() 