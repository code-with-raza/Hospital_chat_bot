"""
app/rag/retriever.py
--------------------
Retrieves the most relevant document chunks for a user's question.

Flow
----
    user question
        ↓
    embed_query()          (from embeddings.py)
        ↓
    vector_stores.search() (from vector_stores.py)
        ↓
    top-k ranked chunks

Design decisions
----------------
* No embedding logic lives here — delegates entirely to embeddings.py.
* No vector-DB logic lives here — delegates entirely to vector_stores.py.
* No reranking, no prompts, no LangGraph.
* top_k is configurable per call (default read from .env / fallback = 3).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.rag.embeddings import embed_query
from app.rag.vector_stores import search

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

DEFAULT_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "3"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve the *top_k* most relevant chunks for *question*.

    Parameters
    ----------
    question : str
        The user's natural-language question.
    top_k : int
        Number of chunks to return.  Defaults to ``RETRIEVER_TOP_K``
        from ``.env`` (fallback: 3).

    Returns
    -------
    List[Dict[str, Any]]
        Ranked list (most relevant first).  Each item has the shape::

            {
                "text":     "<chunk text>",
                "metadata": {
                    "source":      "<filename>",
                    "page":        <int>,
                    "chunk_index": <int>,
                },
                "score":    <float>   # cosine similarity 0–1 (higher = better)
            }

    Notes
    -----
    * Returns an empty list if the vector store is empty.
    * ``top_k`` is silently clamped to the number of stored documents.
    """
    if not question or not question.strip():
        return []

    query_vector = embed_query(question)
    return search(query_vector, n_results=top_k)
