class ChatEvaluator:
    @staticmethod
    def evaluate_rewrite_accuracy(expected_keywords: list, rewritten_question: str) -> bool:
        """Kiểm tra xem câu viết lại có chứa đủ các từ khóa mong đợi (ngữ cảnh) hay không."""
        if not expected_keywords:
            return True
        
        if not rewritten_question:
            return False
            
        rq_lower = rewritten_question.lower()
        # Đạt điểm nếu chứa TẤT CẢ các từ khóa cốt lõi
        for kw in expected_keywords:
            if kw.lower() not in rq_lower:
                return False
        return True

    @staticmethod
    def evaluate_clarification(expected_status: str, actual_status: str) -> bool:
        """Kiểm tra xem AI có nhận diện đúng câu hỏi cần làm rõ (needs_clarification) hay không."""
        if expected_status:
            return expected_status == actual_status
        return True # Nếu JSON không yêu cầu test status thì auto Pass
