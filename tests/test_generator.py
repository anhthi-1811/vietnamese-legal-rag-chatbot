import unittest
from unittest.mock import MagicMock, patch
from src.rag.generator import LegalAnswerGenerator

class TestLegalAnswerGenerator(unittest.TestCase):
    def setUp(self):
        # Patch client API của Gemini để không tốn tiền/mạng khi test
        self.patcher = patch('src.rag.generator.genai.Client')
        self.MockClient = self.patcher.start()
        
        # Setup mock client
        self.mock_client_instance = self.MockClient.return_value
        
        # Khởi tạo generator với thông số test
        self.generator = LegalAnswerGenerator(
            model_name="gemini-test-model",
            temperature=0.0
        )
        
        # Ghi đè lại client bên trong generator bằng mock client
        self.generator.client = self.mock_client_instance

    def tearDown(self):
        self.patcher.stop()

    def _mock_response(self, text_content):
        """Hàm hỗ trợ giả lập kết quả trả về từ Gemini API."""
        mock_response = MagicMock()
        mock_response.text = text_content
        return mock_response

    def test_successful_generation(self):
        """Kiểm tra luồng sinh văn bản bình thường."""
        # Giả lập AI trả về text
        self.mock_client_instance.models.generate_content.return_value = self._mock_response(
            "Theo Điều 123 Bộ luật Hình sự [CID=123]"
        )
        
        # Chạy hàm generate (phiên bản V2 mới nhất)
        result = self.generator.generate(
            question="Tội giết người bị phạt thế nào?", 
            context_str="Văn bản luật...", 
            allowed_cids=["123"]
        )
        
        # Kiểm tra API có được gọi đúng không
        self.assertTrue(self.mock_client_instance.models.generate_content.called)
        
        # Kiểm tra kết quả đóng gói có chuẩn format V2 không
        self.assertEqual(result["answer"], "Theo Điều 123 Bộ luật Hình sự [CID=123]")
        self.assertEqual(result["answer_status"], "answered")
        self.assertEqual(result["cited_cids"], ["123"])

    def test_api_failure_handling(self):
        """Kiểm tra khả năng bắt lỗi tự động khi gọi API thất bại."""
        # Giả lập lỗi mất kết nối mạng hoặc sai API key
        self.mock_client_instance.models.generate_content.side_effect = Exception("API Timeout")
        
        result = self.generator.generate(
            question="Tội trộm cắp?", 
            context_str="...", 
            allowed_cids=["456"]
        )
        
        # Hệ thống phải bắt được exception và bọc lại an toàn
        self.assertIn("Lỗi khởi tạo Generator", result["answer"])
        self.assertIn("API Timeout", result["answer"])
        self.assertEqual(result["answer_status"], "error")
        self.assertEqual(result["cited_cids"], [])

if __name__ == "__main__":
    unittest.main() 