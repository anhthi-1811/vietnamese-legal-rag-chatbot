import os
import sys
import json
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag.pipeline import LegalRAGPipeline
from src.chat.query_rewriter import QueryRewriter
from src.chat.session import ChatSession
from src.chat.chatbot import LegalChatbot
from src.evaluation.chat_metrics import ChatEvaluator

def main():
    print("=" * 60)
    print("BẮT ĐẦU CHẠY EVALUATION CHO CHATBOT ĐA LƯỢT")
    print("=" * 60)

    # 1. Khởi tạo Chatbot
    rag_pipeline = LegalRAGPipeline(top_k=5)
    rewriter = QueryRewriter()
    session = ChatSession(max_turns=6)
    chatbot = LegalChatbot(rag_pipeline=rag_pipeline, query_rewriter=rewriter, session=session)

    # 2. Đọc file Benchmark
    benchmark_path = os.path.join(PROJECT_ROOT, "data/benchmark/chatbot_multiturn.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    results = []

    # 3. Duyệt qua từng cuộc hội thoại
    for conv in conversations:
        conv_id = conv["conversation_id"]
        print(f"\n💬 Đang xử lý Conversation: {conv_id} ({conv.get('description', '')})")
        
        # RESET lịch sử trước mỗi cuộc hội thoại mới! (Rất quan trọng)
        chatbot.session.clear()

        for turn in conv["turns"]:
            turn_id = turn["turn_id"]
            user_q = turn["user"]
            expected_kws = turn.get("expected_rewrite_keywords", [])
            expected_status = turn.get("expected_status", "")

            print(f"  -> Turn {turn_id}: {user_q}")

            # Đưa câu hỏi vào Chatbot
            t0 = time.time()
            bot_reply = chatbot.chat(user_q)
            latency_total = time.time() - t0

            # Lấy thông tin rewrite
            rw_info = bot_reply.get("rewritten_info", {})
            actual_status = rw_info.get("status", "unknown")
            rewritten_q = rw_info.get("standalone_question", "")

            # Chấm điểm
            rewrite_correct = ChatEvaluator.evaluate_rewrite_accuracy(expected_kws, rewritten_q)
            clarification_correct = ChatEvaluator.evaluate_clarification(expected_status, actual_status)

            # Lưu log
            results.append({
                "conversation_id": conv_id,
                "turn_id": turn_id,
                "original_question": user_q,
                "rewritten_question": rewritten_q,
                "rewrite_status": actual_status,
                "expected_status": expected_status,
                "rewrite_correct": rewrite_correct,
                "clarification_correct": clarification_correct,
                "answer_status": bot_reply.get("answer_status", actual_status),
                "cited_cids": str(bot_reply.get("cited_cids", [])),
                "latency_total_s": round(latency_total, 2)
            })

    # 4. Xuất kết quả ra DataFrame và lưu file Parquet
    df_results = pd.DataFrame(results)
    
    os.makedirs(os.path.join(PROJECT_ROOT, "experiments/results"), exist_ok=True)
    out_path = os.path.join(PROJECT_ROOT, "experiments/results/chatbot_multiturn_eval.parquet")
    df_results.to_parquet(out_path, index=False)

    print("\n" + "=" * 60)
    print("ĐÁNH GIÁ HOÀN TẤT!")
    print(f"Đã lưu báo cáo tại: {out_path}")
    print("=" * 60)
    
    # In ra một bảng thống kê nhỏ
    print("\nThống kê cơ bản:")
    print(f"- Tổng số lượt chat (Turns): {len(df_results)}")
    print(f"- Tỷ lệ Rewrite/Clarification đúng: {df_results['rewrite_correct'].mean():.1%} (Nếu False, cần Audit tay lại)")
    print(f"- Latency trung bình mỗi câu: {df_results['latency_total_s'].mean():.2f}s")

if __name__ == "__main__":
    main()
