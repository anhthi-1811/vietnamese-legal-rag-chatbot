import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Build structured evidence context from already retrieved /
    passage-selected documents.

    This class does NOT perform retrieval or passage selection.
    """

    def __init__(self, max_context_chars: int = 12000):
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be > 0")

        self.max_context_chars = max_context_chars

    def build_context(self, retrieved_docs: list[dict[str, Any]]) -> dict:
        """
        Build context while preserving document ranking.

        Expected input:
            [
                {
                    "cid": ...,
                    "text": "...",
                    "score": ...
                },
                ...
            ]

        Documents should already be ordered from most to least relevant.

        Returns:
            {
                "context_str": str,
                "included_cids": list,
                "included_sources": list,
                "total_chars": int,
                "num_sources": int,
                "budget_exhausted": bool
            }
        """

        blocks = []
        included_cids = []
        included_sources = []

        current_length = 0
        budget_exhausted = False

        for doc in retrieved_docs:
            cid = doc.get("cid")
            text = str(doc.get("text", "")).strip()

            if cid is None or not text:
                logger.warning(
                    "Skipping document with missing CID or empty text."
                )
                continue

            source_number = len(included_cids) + 1

            header = (
                f"[SOURCE {source_number} | CID={cid}]\n"
            )
            footer = "\n[/SOURCE]"

            source_block = f"{header}{text}{footer}"

            separator_length = 2 if blocks else 0

            required_length = (
                current_length
                + separator_length
                + len(source_block)
            )

            if required_length > self.max_context_chars:
                budget_exhausted = True
                logger.debug(
                    "Context budget reached at source %s (CID=%s).",
                    source_number,
                    cid,
                )
                break

            blocks.append(source_block)
            included_cids.append(cid)

            included_sources.append(
                {
                    "source_number": source_number,
                    "cid": cid,
                    "score": doc.get("score"),
                    "chars": len(text),
                }
            )

            current_length = required_length

        context_str = "\n\n".join(blocks)

        return {
            "context_str": context_str,
            "included_cids": included_cids,
            "included_sources": included_sources,
            "total_chars": len(context_str),
            "num_sources": len(included_cids),
            "budget_exhausted": budget_exhausted,
        } 