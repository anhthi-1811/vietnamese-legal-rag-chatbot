import os
import sys

# Khai báo đường dẫn gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag.pipeline import LegalRAGPipeline
from src.chat.query_rewriter import QueryRewriter
from src.chat.session import ChatSession
from src.chat.chatbot import LegalChatbot

def main():
    # 1. Khởi tạo một lần duy nhất
    rag_pipeline = LegalRAGPipeline(top_k=5)
    rewriter = QueryRewriter()
    session = ChatSession(max_turns=6)
    
    chatbot = LegalChatbot(
        rag_pipeline=rag_pipeline,
        query_rewriter=rewriter,
        session=session
    )

    print("\n" + "="*60)
    print("VIETNAMESE LEGAL RAG — MULTI-TURN CHATBOT")
    print("Các lệnh đặc biệt:")
    print(" - 'new'    : Xóa lịch sử, bắt đầu chủ đề mới")
    print(" - 'history': Xem lại lịch sử chat hiện tại")
    print(" - 'exit'   : Thoát chương trình")
    print("="*60 + "\n")

    while True:
        try:
            question = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát.")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("Tạm biệt!")
            break
            
        if question.lower() == "new":
            chatbot.session.clear()
            print("Đã xóa bộ nhớ. Hãy bắt đầu một chủ đề mới!")
            print("-" * 60)
            continue
            
        if question.lower() == "history":
            print("\n--- LỊCH SỬ HỘI THOẠI (MEMORY) ---")
            history = chatbot.session.get_recent_messages()
            if not history:
                print("(Trống)")
            else:
                for msg in history:
                    icon = "👤" if msg["role"] == "user" else "🤖"
                    print(f"{icon} {msg['content']}")
            print("----------------------------------\n")
            continue
            
        if not question:
            continue

        print("Bot đang suy nghĩ...")
        result = chatbot.chat(question)

        # In câu trả lời
        print(f"\nBot: {result['answer']}")
        
        # Nếu không phải là câu hỏi ngược (clarification) thì in nguồn và debug
        if result.get("status") != "needs_clarification":
            cids_str = ", ".join(f"CID={c}" for c in result.get("cited_cids", []))
            print(f"\nNguồn: {cids_str or '(Không có)'}")
            
            # Hiển thị Debug cho thấy AI đã tự viết lại câu hỏi như thế nào
            rw_info = result.get("rewritten_info", {})
            if rw_info.get("status") == "rewritten":
                print(f"[Debug Rewrite]: '{rw_info.get('standalone_question')}'")
                
        print("-" * 60)

if __name__ == "__main__":
    main()
