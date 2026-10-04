class LegalChatbot:
    def __init__(self, rag_pipeline, query_rewriter, session):
        self.rag_pipeline = rag_pipeline
        self.query_rewriter = query_rewriter
        self.session = session

    def chat(self, user_message: str) -> dict:
        # 1. Lấy lịch sử hội thoại gần nhất
        history = self.session.get_recent_messages()
        
        # 2. Phân tích và viết lại câu hỏi (nếu cần)
        rewrite_result = self.query_rewriter.rewrite(
            question=user_message,
            history=history
        )
        
        # 3. Xử lý Clarification (Giai đoạn 10)
        if rewrite_result["status"] == "needs_clarification":
            clarification_answer = "Bạn có thể cung cấp thêm ngữ cảnh hoặc nói rõ hơn bạn đang đề cập đến hành vi hay trường hợp nào không?"
            
            # Vẫn lưu lại câu hỏi mơ hồ và câu hỏi ngược của Bot để giữ liền mạch
            self.session.add_user_message(user_message)
            self.session.add_assistant_message(clarification_answer)
            
            return {
                "status": "needs_clarification",
                "answer": clarification_answer,
                "rewritten_info": rewrite_result
            }
            
        # 4. Nếu câu hỏi đã rõ ràng (unchanged hoặc rewritten)
        standalone_question = rewrite_result["standalone_question"]
        
        result = self.rag_pipeline.ask(
            original_question=user_message,
            retrieval_question=standalone_question
        )
        
        # 5. Lưu vào bộ nhớ (Memory)
        self.session.add_user_message(user_message)
        self.session.add_assistant_message(
            content=result["answer"],
            cited_cids=result["cited_cids"]
        )
        
        # Đính kèm thông tin rewrite để Debug ở UI
        result["rewritten_info"] = rewrite_result
        return result 
