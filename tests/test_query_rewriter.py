import os
import sys
import json

# Khai báo đường dẫn gốc của project để Python tìm thấy module 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chat.query_rewriter import QueryRewriter

def test_query_rewriter():
    rewriter = QueryRewriter()
    
    with open("tests/data/query_rewrite_cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
        
    for case in cases:
        result = rewriter.rewrite(case["question"], case["history"])
        
        # 1. Kiểm tra trạng thái phân loại
        assert result["status"] == case["expected_status"], \
            f"Failed {case['id']}: Expected {case['expected_status']}, got {result['status']}"
        
        # 2. Kiểm tra từ khóa ngữ nghĩa nếu được viết lại hoặc giữ nguyên
        if result["status"] in ["rewritten", "unchanged"]:
            standalone_q = result["standalone_question"].lower()
            for kw in case.get("expected_keywords", []):
                assert kw.lower() in standalone_q, \
                    f"Failed {case['id']}: Keyword '{kw}' missing in '{standalone_q}'"

if __name__ == "__main__":
    test_query_rewriter()
    print("All query rewrite tests passed!") 
