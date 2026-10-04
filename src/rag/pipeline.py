import os
import sys
import time
import yaml
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

from src.data.cleaner import normalize_vietnamese_text
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import BGEM3Encoder
from src.retrieval.faiss_index import FAISSRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.rag.passage_retriever import PassageRetriever
from src.rag.context_builder import ContextBuilder
from src.rag.generator import LegalAnswerGenerator
from src.rag.citation_checker import CitationChecker

def load_yaml(path: str, default: dict = None) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return default or {}

class LegalRAGPipeline:
    def __init__(
        self,
        corpus_path: str = None,
        bm25_dir: str = None,
        dense_dir: str = None,
        top_k: int = 5,
        prompt_version: str = "v2",
    ):
        self.top_k = top_k
        self.prompt_version = prompt_version

        ret_cfg = load_yaml(os.path.join(PROJECT_ROOT, "configs/retrieval.yaml"))
        full_rag_cfg = load_yaml(os.path.join(PROJECT_ROOT, "configs/rag.yaml"))
        rag_cfg = full_rag_cfg.get("rag", {})
        gen_cfg = full_rag_cfg.get("generation", {})

        self.bm25_top_k = ret_cfg.get("bm25", {}).get("top_k", 50)
        self.dense_top_k = ret_cfg.get("dense", {}).get("top_k", 50)
        self.rerank_max_len = ret_cfg.get("reranker", {}).get("max_length", 512)
        self.rerank_batch_size = ret_cfg.get("reranker", {}).get("batch_size", 256)

        corpus_path = corpus_path or os.path.join(PROJECT_ROOT, "data/processed/documents_clean.parquet")
        bm25_dir = bm25_dir or os.path.join(PROJECT_ROOT, "artifacts/bm25")
        dense_dir = dense_dir or os.path.join(PROJECT_ROOT, "artifacts/dense")

        print("=" * 70)
        print("KHỞI ĐỘNG RAG ENGINE (PIPELINE)")
        print("=" * 70)
        
        df_corpus = pd.read_parquet(corpus_path, columns=["cid", "text_clean"])
        self.doc_lookup = dict(zip(df_corpus["cid"].astype(str), df_corpus["text_clean"].astype(str)))
        
        bm25_cfg = ret_cfg.get("bm25", {})
        self.bm25 = BM25Retriever(
            lowercase=bm25_cfg.get("lowercase", True),
            k1=bm25_cfg.get("k1", 1.5),
            b=bm25_cfg.get("b", 0.75),
        )
        self.bm25.load(bm25_dir)

        dense_cfg = ret_cfg.get("dense", {})
        self.encoder = BGEM3Encoder(
            model_name=dense_cfg.get("model_name", "BAAI/bge-m3"),
            use_fp16=dense_cfg.get("use_fp16", True),
        )
        self.faiss_index = FAISSRetriever()
        self.faiss_index.load(dense_dir)

        rerank_cfg = ret_cfg.get("reranker", {})
        rerank_model = rerank_cfg.get("model_name", "BAAI/bge-reranker-v2-m3")
        self.reranker = CrossEncoderReranker(
            model_name=rerank_model,
            use_fp16=rerank_cfg.get("use_fp16", True),
        )

        retriever_cfg = rag_cfg.get("passage", {})
        self.passage_retriever = PassageRetriever(
            long_doc_threshold_chars=retriever_cfg.get("long_doc_threshold_chars", 6000),
            max_passage_chars=retriever_cfg.get("max_passage_chars", 2500),
            top_passages_per_doc=retriever_cfg.get("top_passages_per_doc", 2),
            overlap_paragraphs=retriever_cfg.get("overlap_paragraphs", 0),
        )
        
        context_cfg = rag_cfg.get("context", {})
        self.context_builder = ContextBuilder(
            max_context_chars=context_cfg.get("max_context_chars", 12000)
        )
        
        self.generator = LegalAnswerGenerator(
            model_name=gen_cfg.get("model_name", "gemini-3.1-flash-lite"),
            prompt_version=self.prompt_version,
            temperature=gen_cfg.get("temperature", 0.0),
            max_output_tokens=gen_cfg.get("max_output_tokens", 1024),
            api_retries=gen_cfg.get("max_retries", 3),
        )
        
        self.citation_checker = CitationChecker()
        print("RAG Engine đã sẵn sàng!")

    def ask(self, original_question: str, retrieval_question: str = None) -> dict:
        t_total_start = time.time()
        
        # Nếu không có câu hỏi rewrite, mặc định dùng original_question
        if retrieval_question is None:
            retrieval_question = original_question

        norm_q = normalize_vietnamese_text(retrieval_question)

        # 1. BM25 Search
        t0 = time.time()
        bm25_cids, _ = self.bm25.search(norm_q, k=self.bm25_top_k)
        bm25_cids = [str(c).strip() for c in bm25_cids]
        t_bm25 = time.time() - t0

        # 2. BGE Query Encode
        t0 = time.time()
        q_emb = self.encoder.encode_queries([norm_q], batch_size=1, max_length=512)
        t_bge = time.time() - t0

        # 3. FAISS Search
        t0 = time.time()
        dense_cids_batch, _ = self.faiss_index.search(q_emb, k=self.dense_top_k)
        dense_cids = [str(c).strip() for c in dense_cids_batch[0]]
        t_faiss = time.time() - t0

        # 4. Candidate Union
        t0 = time.time()
        candidate_cids = list(dict.fromkeys(bm25_cids + dense_cids))
        candidate_cids = [str(c).strip() for c in candidate_cids if str(c).strip() in self.doc_lookup]
        candidate_cids = candidate_cids[:20]  # Giới hạn 20 để tối ưu Reranker trên CPU
        t_union = time.time() - t0

        # 5. Reranker
        t0 = time.time()
        sentence_pairs = [(norm_q, self.doc_lookup[cid]) for cid in candidate_cids]
        rerank_scores = self.reranker.compute_scores(
            sentence_pairs,
            batch_size=self.rerank_batch_size,
            max_length=min(self.rerank_max_len, 256),
        )
        ranked_pairs = sorted(
            zip(candidate_cids, rerank_scores), key=lambda x: x[1], reverse=True
        )
        retrieved_cids = [cid for cid, _ in ranked_pairs[:self.top_k]]
        retrieved_scores = [float(score) for _, score in ranked_pairs[:self.top_k]]
        t_reranker = time.time() - t0

        # 6. Passage Retrieval & Context Builder
        t0 = time.time()
        docs_for_retrieval = [
            {"cid": cid, "text": self.doc_lookup[cid], "score": score}
            for cid, score in zip(retrieved_cids, retrieved_scores)
        ]
        processed_docs = self.passage_retriever.process_documents(
            question=retrieval_question,
            documents=docs_for_retrieval,
        )
        context_data = self.context_builder.build_context(processed_docs)
        context_str = context_data.get("context_str", "")
        included_cids = context_data.get("included_cids", [])
        t_passage = time.time() - t0

        # 7. Gemini Generation
        t0 = time.time()
        gen_output = self.generator.generate(
            question=original_question, # Trả lời trực tiếp vào câu hỏi gốc của user
            context_str=context_str,
            allowed_cids=included_cids,
        )
        t_generation = time.time() - t0

        answer = gen_output.get("answer", "")
        answer_status = gen_output.get("answer_status", "unknown")
        declared_cids = gen_output.get("cited_cids", [])

        # 8. Citation Checking
        check_output = self.citation_checker.check(
            answer=answer,
            declared_cids=declared_cids,
            allowed_cids=included_cids,
            answer_status=answer_status,
        )

        return {
            "original_question": original_question,
            "retrieval_question": retrieval_question,
            "candidate_pool_size": len(candidate_cids),
            "retrieved_docs": docs_for_retrieval,
            "answer": answer,
            "answer_status": answer_status,
            "cited_cids": declared_cids,
            "citation_valid": check_output.get("is_valid", False),
            "latency": {
                "bm25": round(t_bm25, 3),
                "dense_encode": round(t_bge, 3),
                "faiss": round(t_faiss, 3),
                "union": round(t_union, 3),
                "reranker": round(t_reranker, 3),
                "passage_context": round(t_passage, 3),
                "generation": round(t_generation, 3),
                "total": round(time.time() - t_total_start, 3)
            }
        }
