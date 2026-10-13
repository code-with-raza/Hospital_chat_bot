"""Lightweight keyword reranking for chunks returned by the vector search."""

import re
from typing import Any


_IGNORED_WORDS = {
    "a", "an", "and", "are", "at", "be", "do", "does", "for", "have",
    "how", "i", "is", "it", "of", "on", "the", "to", "what", "when",
    "where", "which", "who", "hospital",
}


def _words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.casefold())
        if word not in _IGNORED_WORDS
    }


def has_query_term_overlap(question: str, chunk: dict[str, Any]) -> bool:
    """Reject candidate context that shares no content words with the query.

    This is a conservative grounding guard in addition to the configurable
    vector/reranker score threshold; it does not alter that threshold.
    """
    query_words = _words(question)
    chunk_words = _words(str(chunk.get("text", "")))
    return bool(query_words & chunk_words)


def rerank_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Rank retrieved chunks by query-term overlap, using retrieval score as a tie-breaker.

    The original chunk fields are kept. ``score`` remains the retriever's score;
    ``rerank_score`` is added for the combined lexical and retrieval ranking.
    """
    if top_n <= 0 or not chunks:
        return []

    query_words = _words(question)
    ranked: list[tuple[float, float, int, dict[str, Any]]] = []

    for position, chunk in enumerate(chunks):
        retrieval_score = float(chunk.get("score") or 0.0)
        chunk_words = _words(str(chunk.get("text", "")))
        overlap = (
            len(query_words & chunk_words) / len(query_words)
            if query_words
            else 0.0
        )
        rerank_score = 0.7 * retrieval_score + 0.3 * overlap

        ranked_chunk = {**chunk, "rerank_score": round(rerank_score, 4)}
        if isinstance(chunk.get("metadata"), dict):
            ranked_chunk["metadata"] = dict(chunk["metadata"])

        ranked.append((rerank_score, retrieval_score, position, ranked_chunk))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in ranked[:top_n]]
