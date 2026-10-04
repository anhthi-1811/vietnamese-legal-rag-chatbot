import os
import sys
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag.pipeline import LegalRAGPipeline
from src.chat.query_rewriter import QueryRewriter
from src.chat.session import ChatSession
from src.chat.chatbot import LegalChatbot

@pytest.fixture(scope="module")
def chatbot():
    rag_pipeline = LegalRAGPipeline(top_k=3)
    rewriter = QueryRewriter()
    session = ChatSession()
    return LegalChatbot(rag_pipeline, rewriter, session)

def test_multiturn_flow(chatbot):
    chatbot.session.clear()
    
    # Turn 1
    res1 = chatbot.chat("Tội giết người bị phạt thế nào?")
    assert res1["answer"] != ""
    assert len(res1["retrieved_docs"]) > 0
    
    # Turn 2 (Follow-up)
    res2 = chatbot.chat("Nếu không thuộc khoản 1 thì sao?")
    rewritten_q = res2["rewritten_info"]["standalone_question"].lower()
    
    # Kiểm tra xem AI có tự động nối chữ "giết người" vào câu 2 không
    assert "giết người" in rewritten_q
    assert res2["answer"] != "" 