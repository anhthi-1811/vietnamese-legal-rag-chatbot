import os
import json
from google import genai
from google.genai import types

class QueryRewriter:
    def __init__(self, model_name="gemini-3.1-flash-lite", temperature=0.0):
        self.model_name = model_name
        self.temperature = temperature
        # Khởi tạo Gemini client (tự động đọc GEMINI_API_KEY từ môi trường)
        self.client = genai.Client()
        
        self.system_prompt = """You rewrite the latest Vietnamese legal user question into a standalone question using only the conversation history.

Rules:
1. Preserve the user's original intent.
2. Resolve references such as: "trường hợp đó", "khoản này", "người này", "mức phạt trên", "quy định đó".
3. Do not answer the legal question.
4. Do not introduce legal facts not present in the conversation.
5. If the question is already standalone, return status "unchanged" and the original question.
6. If the reference cannot be resolved confidently, return status "needs_clarification".

Return ONLY a JSON object with this exact schema:
{
    "status": "unchanged" | "rewritten" | "needs_clarification",
    "standalone_question": "string (the rewritten question, or original if unchanged, or empty if needs_clarification)",
    "reason": "string (brief explanation)"
}"""

    def rewrite(self, question: str, history: list) -> dict:
        # Nếu không có lịch sử hội thoại, mặc định là unchanged
        if not history:
            return {
                "status": "unchanged",
                "standalone_question": question,
                "reason": "No conversation history provided."
            }
        
        # Format lịch sử thành dạng text
        formatted_history = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n"
        
        prompt = f"Conversation History:\n{formatted_history}\nLatest Question: {question}\n\nRewrite the latest question as a standalone question."
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=self.temperature,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            # Fallback an toàn nếu API lỗi
            return {
                "status": "unchanged",
                "standalone_question": question,
                "reason": f"API Error: {str(e)}"
            } 
