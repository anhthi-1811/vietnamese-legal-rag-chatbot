import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag.pipeline import LegalRAGPipeline

def main():
    pipeline = LegalRAGPipeline()
    
    question = "Cố ý giết người thì bị phạt thế nào?"
    print(f"\nĐang profile truy vấn: '{question}'...")
    
    result = pipeline.ask(original_question=question)
    
    print("\n" + "="*40)
    print("BÁO CÁO LATENCY (GIÂY)")
    print("="*40)
    
    lat = result["latency"]
    print(f"BM25 Search:         {lat['bm25']:>7.3f}s")
    print(f"BGE Encode:          {lat['dense_encode']:>7.3f}s")
    print(f"FAISS Search:        {lat['faiss']:>7.3f}s")
    print(f"Candidate Union:     {lat['union']:>7.3f}s")
    print(f"Reranker:            {lat['reranker']:>7.3f}s")
    print(f"Passage & Context:   {lat['passage_context']:>7.3f}s")
    print(f"Gemini Generation:   {lat['generation']:>7.3f}s")
    print("-" * 40)
    print(f"TOTAL TIME:          {lat['total']:>7.3f}s")
    print("="*40)
    
    # Kiểm tra xác nhận dữ liệu output đã chuẩn hóa
    assert result["original_question"] == question
    assert result["retrieval_question"] == question
    print("\nOutput schema đã đạt chuẩn Giai đoạn 3.")

if __name__ == "__main__":
    main()
