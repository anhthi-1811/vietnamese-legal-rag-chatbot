import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag.pipeline import LegalRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Vietnamese Legal RAG — Single-Turn CLI")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K tai lieu dua vao context")
    parser.add_argument("--prompt-version", type=str, default="v2", choices=["v1", "v2"])
    parser.add_argument("--query", type=str, default=None, help="Chay 1 cau roi thoat")
    args = parser.parse_args()

    # Chỉ việc gọi "động cơ" RAG từ file pipeline.py
    pipeline = LegalRAGPipeline(top_k=args.top_k, prompt_version=args.prompt_version)

    if args.query:
        res = pipeline.ask(args.query)
        print("\nTra loi:", res["answer"])
        return

    print("\n" + "=" * 50)
    print("VIETNAMESE LEGAL RAG — SINGLE-TURN CLI")
    print("Gõ câu hỏi pháp luật và nhấn Enter. Gõ 'exit' để thoát.")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("Câu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã thoát chương trình.")
            break

        if question.lower() in {"exit", "quit", "q"}:
            print("Đã thoát chương trình.")
            break
        if not question:
            continue

        print("Đang xử lý...")
        result = pipeline.ask(question)

        print("\nTop retrieved evidence:")
        for i, doc in enumerate(result["retrieved_docs"], 1):
            snippet = doc["text"].replace("\n", " ").strip()[:110] + "..."
            print(f"  {i}. CID={doc['cid']} (score={doc['score']:.4f}) — {snippet}")

        print(f"\nTrả lời ({result['answer_status']}):\n{result['answer']}")
        print("\n Nguồn:", ", ".join(f"CID={c}" for c in result["cited_cids"]) or "(Không có)")
        
        print(f"Latency: Retrieval={result['retrieval_latency_s']}s | Gen={result['generation_latency_s']}s | Total={result['total_latency_s']}s")
        print("-" * 70 + "\n")

if __name__ == "__main__":
    main()
