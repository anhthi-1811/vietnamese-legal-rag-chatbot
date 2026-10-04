class ChatSession:
    def __init__(self, max_turns=6):
        self.messages = []
        self.max_turns = max_turns

    def add_user_message(self, content):
        self.messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant_message(self, content, cited_cids=None):
        self.messages.append({
            "role": "assistant",
            "content": content,
            "cited_cids": cited_cids or []
        })

    def get_recent_messages(self):
        # Mỗi turn gồm 1 user + 1 assistant, nên max_messages = max_turns * 2
        max_messages = self.max_turns * 2
        return self.messages[-max_messages:]

    def clear(self):
        self.messages = []
