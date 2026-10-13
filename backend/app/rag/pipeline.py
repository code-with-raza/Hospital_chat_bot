"""Retrieve, rerank, and answer questions using approved hospital documents."""

import os
import re
from typing import Any

from app.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from app.rag.prompts import HOSPITAL_RAG_SYSTEM_PROMPT, format_context
from app.rag.reranker import has_query_term_overlap, rerank_chunks
from app.rag.retriever import retrieve
from openai import OpenAI


INFORMATION_UNAVAILABLE = (
    "This information is not available in the hospital guide; the available "
    "information does not specify it. Please contact the hospital directly to confirm."
)


def _normalize_retrieval_query(question: str) -> str:
    """Align common lab-hours wording with the document's terminology."""
    if re.search(r"\blabs?\b", question, re.I) and re.search(
        r"\b(?:hours?|open(?:ing)?|when|what time|timings?)\b", question, re.I
    ):
        return f"{question} diagnostic laboratory testing typical availability"
    return question


def _environment_number(name: str, default: int | float) -> int | float:
    try:
        return type(default)(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _generate_answer(question: str, context: str, model: str) -> str:
    if not LLM_API_KEY:
        raise RuntimeError("Set LLM_API_KEY to enable hospital knowledge answers.")

    from openai import OpenAI

    client_options: dict[str, str] = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        client_options["base_url"] = LLM_BASE_URL

    response = OpenAI(**client_options).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": HOSPITAL_RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Hospital context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _sources_from_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    seen: set[tuple[str, int | None]] = set()

    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        source = str(metadata.get("source", "Unknown source"))
        try:
            page = int(metadata["page"])
        except (KeyError, TypeError, ValueError):
            page = None

        source_key = (source, page)
        if source_key in seen:
            continue
        seen.add(source_key)
        sources.append(
            {
                "source": source,
                "page": page,
                "score": float(chunk.get("score") or 0.0),
            }
        )

    return sources


def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    top_n: int | None = None,
    relevance_threshold: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a hospital question from retrieved documents and return its sources.

    Defaults can be set with ``RAG_TOP_K``, ``RAG_TOP_N``,
    ``RAG_RELEVANCE_THRESHOLD``, and ``RAG_LLM_MODEL`` environment variables.
    """
    # The demo guide is small, so search its full chunk set by default. Larger
    # deployments can lower this with RAG_TOP_K.
    top_k = top_k if top_k is not None else int(_environment_number("RAG_TOP_K", 52))
    top_n = top_n if top_n is not None else int(_environment_number("RAG_TOP_N", 3))
    if relevance_threshold is None:
        relevance_threshold = float(
            _environment_number("RAG_RELEVANCE_THRESHOLD", 0.4)
        )
    model = model or os.getenv("RAG_LLM_MODEL") or LLM_MODEL

    if not question.strip():
        return {"answer": INFORMATION_UNAVAILABLE, "sources": []}

    retrieval_question = _normalize_retrieval_query(question)
    retrieved = retrieve(retrieval_question, top_k=top_k)
    ranked = rerank_chunks(retrieval_question, retrieved, top_n=top_n)
    relevant = [
        chunk
        for chunk in ranked
        if chunk.get("rerank_score", 0.0) >= relevance_threshold
        and has_query_term_overlap(retrieval_question, chunk)
    ]
    if not relevant:
        return {"answer": INFORMATION_UNAVAILABLE, "sources": []}

    context = format_context(relevant)
    answer = _generate_answer(question, context, model)
    if not answer:
        answer = INFORMATION_UNAVAILABLE

    return {"answer": answer, "sources": _sources_from_chunks(relevant)}
