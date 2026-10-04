import os
from google import genai
from google.genai import types

class GenerationParseError(Exception):
    """Ngoại lệ khi parse output của generator thất bại."""
    pass 

class LegalAnswerGenerator:
    def __init__(
        self,
        model_name="gemini-3.1-flash-lite",
        prompt_version="v2",
        temperature=0.0,
        max_output_tokens=1024,
        api_retries=3
    ):
        self.model_name = model_name
        self.prompt_version = prompt_version
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_retries = api_retries
        self.client = genai.Client()

    def generate(self, question: str, context_str: str, allowed_cids: list) -> dict:
        # GIAI ĐOẠN 12: Bổ sung chỉ thị cấm dùng History làm bằng chứng
        system_instruction = """You are a strictly grounded Vietnamese Legal AI Assistant.
        
CRITICAL RULES:
1. All legal claims MUST be supported by the retrieved legal evidence supplied in the context.
2. Conversation history (if provided) is ONLY to understand the user's conversational intent.
3. Do NOT treat the user's claims, numbers, or assumptions in the history as legal evidence.
4. If the retrieved context does not contain the answer, politely refuse to answer. Do not guess.
5. Always cite the exact [CID=xxx] at the end of the sentence when using information from that CID.
"""

        prompt = f"""Dựa vào các văn bản pháp luật sau đây, hãy trả lời câu hỏi của người dùng.
        
[CONTEXT START]
{context_str}
[CONTEXT END]

Câu hỏi: {question}

Lưu ý: 
- Chỉ dùng thông tin trong [CONTEXT].
- Nếu không có thông tin, hãy nói "Các nguồn được cung cấp chưa đủ để trả lời...".
- Ghi chú trích dẫn [CID=...] vào cuối mỗi ý."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
            )
            # Tạm thời trả về "answered" nếu có response để Pipeline xử lý tiếp
            return {
                "answer": response.text,
                "answer_status": "answered", 
                "cited_cids": allowed_cids 
            }
        except Exception as e:
            return {
                "answer": f"Lỗi khởi tạo Generator: {str(e)}",
                "answer_status": "error",
                "cited_cids": []
            }
