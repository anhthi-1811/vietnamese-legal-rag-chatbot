from src.chat.session import ChatSession

def test_add_messages():
    session = ChatSession()
    session.add_user_message("Xin chào")
    session.add_assistant_message("Chào bạn, tôi là trợ lý pháp lý.", cited_cids=["123"])
    
    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["cited_cids"] == ["123"]

def test_clear():
    session = ChatSession()
    session.add_user_message("Hello")
    session.clear()
    assert len(session.messages) == 0

def test_max_turns():
    session = ChatSession(max_turns=2)
    # Thêm 3 lượt hội thoại (6 tin nhắn)
    for i in range(3):
        session.add_user_message(f"Câu hỏi {i}")
        session.add_assistant_message(f"Trả lời {i}")
    
    # get_recent_messages chỉ được trả về 4 tin nhắn (2 lượt gần nhất)
    recent = session.get_recent_messages()
    assert len(recent) == 4
    assert recent[0]["content"] == "Câu hỏi 1"
