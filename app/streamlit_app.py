import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag.pipeline import LegalRAGPipeline
from src.chat.query_rewriter import QueryRewriter
from src.chat.session import ChatSession
from src.chat.chatbot import LegalChatbot

st.set_page_config(page_title="Vietnamese Legal AI", page_icon="⚖️", layout="centered")

# --- Khởi tạo hệ thống (Chỉ chạy 1 lần) ---
@st.cache_resource
def load_chatbot():
    rag_pipeline = LegalRAGPipeline(top_k=5)
    rewriter = QueryRewriter()
    session = ChatSession(max_turns=6)
    return LegalChatbot(rag_pipeline, rewriter, session)

chatbot = load_chatbot()

# --- Giao diện Sidebar ---
with st.sidebar:
    st.title("Legal RAG Chatbot")
    st.markdown("Hệ thống tra cứu pháp luật Việt Nam thông minh.")
    if st.button("Bắt đầu cuộc trò chuyện mới", use_container_width=True):
        chatbot.session.clear()
        st.session_state.messages = []
        st.rerun()

# --- Khởi tạo Session State cho UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Hiển thị lịch sử chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Hiển thị các thông tin Debug (Nguồn, Rewrite) nếu có
        if "debug_info" in msg:
            with st.expander("Chi tiết truy xuất"):
                st.write(msg["debug_info"])

# --- Nhận input người dùng ---
user_query = st.chat_input("Nhập câu hỏi pháp luật của bạn...")

if user_query:
    # 1. In câu hỏi của user
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Xử lý câu trả lời của Bot
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu các quy định pháp luật..."):
            result = chatbot.chat(user_query)
            
            answer = result["answer"]
            status = result.get("status", "")
            rw_info = result.get("rewritten_info", {})
            cids = result.get("cited_cids", [])
            
            # Khung gộp thông tin Debug
            debug_text = f"**Câu hỏi đã xử lý ngữ cảnh:** {rw_info.get('standalone_question', 'N/A')}\n\n"
            debug_text += f"**Nguồn luật (CIDs):** {', '.join(cids) if cids else 'Không có'}"
            
            # In ra màn hình
            st.markdown(answer)
            if status != "needs_clarification":
                with st.expander("Chi tiết truy xuất"):
                    st.markdown(debug_text)
                    
            # Lưu lại vào Session UI
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "debug_info": debug_text if status != "needs_clarification" else None
            }) 
