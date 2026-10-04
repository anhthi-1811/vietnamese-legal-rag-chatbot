from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PassageScorer = Callable[[str, list[str]], list[float]]

class PassageRetriever:
    """
    Perform local passage selection for long retrieved documents.
    """

    def __init__(
        self,
        long_doc_threshold_chars: int = 6000,
        max_passage_chars: int = 2500,
        top_passages_per_doc: int = 2,
        overlap_paragraphs: int = 0,
        scorer: Optional[PassageScorer] = None,
        strict: bool = True,
    ) -> None:

        if long_doc_threshold_chars <= 0:
            raise ValueError("long_doc_threshold_chars must be > 0")

        if max_passage_chars <= 0:
            raise ValueError("max_passage_chars must be > 0")

        if top_passages_per_doc <= 0:
            raise ValueError("top_passages_per_doc must be > 0")

        if overlap_paragraphs < 0:
            raise ValueError("overlap_paragraphs must be >= 0")

        self.long_doc_threshold_chars = long_doc_threshold_chars
        self.max_passage_chars = max_passage_chars
        self.top_passages_per_doc = top_passages_per_doc
        self.overlap_paragraphs = overlap_paragraphs
        self.scorer = scorer
        self.strict = strict

    def process_documents(
        self,
        question: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        if not isinstance(documents, list):
            raise TypeError("documents must be a list")

        processed_documents: list[dict[str, Any]] = []

        for rank, document in enumerate(documents, start=1):
            try:
                processed = self.process_document(
                    question=question,
                    document=document,
                    document_rank=rank,
                )
            except Exception:
                if self.strict:
                    raise
                logger.exception("Skipping malformed document at rank %d.", rank)
                continue

            if processed is not None:
                processed_documents.append(processed)

        return processed_documents

    def process_document(
        self,
        question: str,
        document: dict[str, Any],
        document_rank: Optional[int] = None,
    ) -> dict[str, Any]:
        if not isinstance(document, dict):
            raise TypeError("document must be a dictionary")

        cid = document.get("cid")
        if cid is None:
            raise ValueError("document is missing 'cid'")

        raw_text = document.get("text", "")
        if raw_text is None:
            raw_text = ""

        text = str(raw_text).strip()
        if not text:
            raise ValueError(f"Document CID={cid} has empty text.")

        result = dict(document)
        original_chars = len(text)

        if original_chars <= self.long_doc_threshold_chars:
            result["text"] = text
            result["passage_selection_applied"] = False
            result["original_chars"] = original_chars
            result["selected_chars"] = original_chars
            result["passage_metadata"] = [
                {
                    "passage_index": 0,
                    "passage_score": None,
                    "chars": original_chars,
                    "selected": True,
                }
            ]
            if document_rank is not None:
                result["document_rank"] = document_rank
            return result

        passages = self._build_passages(text)
        if not passages:
            raise RuntimeError(f"Could not create passages for CID={cid}")

        scores = self._score_passages(question=question, passages=passages)
        self._validate_scores(scores=scores, num_passages=len(passages), cid=cid)

        ranked_indices = sorted(
            range(len(passages)),
            key=lambda idx: (-scores[idx], idx),
        )

        selected_indices = ranked_indices[: self.top_passages_per_doc]
        selected_indices = sorted(selected_indices)

        selected_passages = [passages[idx] for idx in selected_indices]
        selected_text = "\n\n".join(selected_passages).strip()

        passage_metadata = []
        selected_index_set = set(selected_indices)

        for idx, passage in enumerate(passages):
            passage_metadata.append(
                {
                    "passage_index": idx,
                    "passage_score": float(scores[idx]),
                    "chars": len(passage),
                    "selected": idx in selected_index_set,
                }
            )

        result["text"] = selected_text
        result["passage_selection_applied"] = True
        result["original_chars"] = original_chars
        result["selected_chars"] = len(selected_text)
        result["num_passages"] = len(passages)
        result["num_selected_passages"] = len(selected_indices)
        result["selected_passage_indices"] = selected_indices
        result["passage_metadata"] = passage_metadata

        if document_rank is not None:
            result["document_rank"] = document_rank

        return result

    def _build_passages(self, text: str) -> list[str]:
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            return []

        segments: list[str] = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.max_passage_chars:
                segments.append(paragraph)
            else:
                segments.extend(self._split_long_segment(paragraph))

        passages: list[str] = []
        start_idx = 0

        while start_idx < len(segments):
            current_segments: list[str] = []
            current_chars = 0
            end_idx = start_idx

            while end_idx < len(segments):
                segment = segments[end_idx]
                separator_chars = 2 if current_segments else 0
                projected_chars = current_chars + separator_chars + len(segment)

                if current_segments and projected_chars > self.max_passage_chars:
                    break

                current_segments.append(segment)
                current_chars = projected_chars
                end_idx += 1

                if current_chars >= self.max_passage_chars:
                    break

            passage = "\n\n".join(current_segments).strip()
            if passage:
                passages.append(passage)

            if end_idx >= len(segments):
                break

            next_start = end_idx - self.overlap_paragraphs
            start_idx = max(start_idx + 1, next_start)

        return passages

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        raw_parts = re.split(r"\n+", text)
        return [part.strip() for part in raw_parts if part.strip()]

    def _split_long_segment(self, segment: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?;:])\s+", segment)
        pieces: list[str] = []
        current: list[str] = []
        current_chars = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > self.max_passage_chars:
                if current:
                    pieces.append(" ".join(current).strip())
                    current = []
                    current_chars = 0
                pieces.extend(self._hard_split(sentence, self.max_passage_chars))
                continue

            separator_chars = 1 if current else 0
            projected = current_chars + separator_chars + len(sentence)

            if current and projected > self.max_passage_chars:
                pieces.append(" ".join(current).strip())
                current = [sentence]
                current_chars = len(sentence)
            else:
                current.append(sentence)
                current_chars = projected

        if current:
            pieces.append(" ".join(current).strip())

        return [piece for piece in pieces if piece]

    @staticmethod
    def _hard_split(text: str, max_chars: int) -> list[str]:
        chunks: list[str] = []
        remaining = text.strip()

        while len(remaining) > max_chars:
            cut_position = remaining.rfind(" ", 0, max_chars + 1)
            if cut_position <= 0:
                cut_position = max_chars

            chunk = remaining[:cut_position].strip()
            if chunk:
                chunks.append(chunk)

            remaining = remaining[cut_position:].strip()

        if remaining:
            chunks.append(remaining)

        return chunks

    def _score_passages(self, question: str, passages: list[str]) -> list[float]:
        if self.scorer is not None:
            return list(self.scorer(question, passages))
        return self._lexical_scores(question=question, passages=passages)

    def _lexical_scores(self, question: str, passages: list[str]) -> list[float]:
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return [0.0 for _ in passages]

        query_unique = list(dict.fromkeys(query_tokens))
        passage_tokens = [self._tokenize(passage) for passage in passages]
        passage_token_sets = [set(tokens) for tokens in passage_tokens]
        num_passages = len(passages)

        document_frequency: Counter[str] = Counter()
        for token in query_unique:
            document_frequency[token] = sum(1 for token_set in passage_token_sets if token in token_set)

        query_weights = {}
        for token in query_unique:
            df = document_frequency[token]
            query_weights[token] = math.log((num_passages + 1) / (df + 1)) + 1.0

        total_query_weight = sum(query_weights.values())
        query_bigrams = set(zip(query_tokens, query_tokens[1:]))
        scores: list[float] = []

        for tokens, token_set in zip(passage_tokens, passage_token_sets):
            matched_weight = sum(weight for token, weight in query_weights.items() if token in token_set)
            unigram_score = (matched_weight / total_query_weight) if total_query_weight > 0 else 0.0

            if query_bigrams and len(tokens) >= 2:
                passage_bigrams = set(zip(tokens, tokens[1:]))
                bigram_score = len(query_bigrams & passage_bigrams) / len(query_bigrams)
            else:
                bigram_score = 0.0

            score = 0.80 * unigram_score + 0.20 * bigram_score
            scores.append(float(score))

        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _validate_scores(scores: list[float], num_passages: int, cid: Any) -> None:
        if len(scores) != num_passages:
            raise ValueError(f"Passage scorer returned {len(scores)} scores for {num_passages} passages (CID={cid}).")

        for score in scores:
            score_value = float(score)
            if not math.isfinite(score_value):
                raise ValueError(f"Non-finite passage score for CID={cid}: {score_value}") 