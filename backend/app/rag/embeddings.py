"""
app/rag/embeddings.py
---------------------
Converts text (chunks or queries) into dense vector embeddings using a
local sentence-transformers model.  Runs completely offline — no API key
required.

Configurable via .env
---------------------
EMBEDDING_MODEL=all-MiniLM-L6-v2   # any sentence-transformers model name

Popular alternatives
--------------------
Model                          Dim   Speed     Notes
-----------------------------  ----  --------  --------------------------------
all-MiniLM-L6-v2  (default)    384   fast      best size/quality trade-off
all-MiniLM-L12-v2              384   moderate  slightly better quality
all-mpnet-base-v2              768   slower    higher quality, larger
paraphrase-multilingual-MiniLM 384   fast      multilingual support

No vector-database or retrieval code lives here.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config — load from .env
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Internal model (lazily loaded once)
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer, downloading it on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_embedding_dimension() -> int:
    """Return the vector dimension produced by the current model."""
    return _get_model().get_embedding_dimension()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings in one batched call.

    Parameters
    ----------
    texts : List[str]
        The strings to embed.  Empty strings are replaced with a single
        space so the model doesn't receive blank input.

    Returns
    -------
    List[List[float]]
        One embedding vector per input string, in the same order.
    """
    if not texts:
        return []

    safe_texts = [t if t.strip() else " " for t in texts]
    model = _get_model()

    # encode() returns a numpy array of shape (N, dim)
    # convert_to_numpy=True is the default; we cast to plain Python lists
    vectors = model.encode(safe_texts, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add an ``"embedding"`` field to each chunk dict (returns new dicts,
    originals are not mutated).

    Parameters
    ----------
    chunks : List[Dict[str, Any]]
        Output of ``chunk_documents()``.

    Returns
    -------
    List[Dict[str, Any]]
        Same chunks with an extra ``"embedding": List[float]`` key.
    """
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    return [
        {**chunk, "embedding": vector}
        for chunk, vector in zip(chunks, vectors)
    ]


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    Parameters
    ----------
    query : str
        The user's question or search term.

    Returns
    -------
    List[float]
        A single embedding vector.
    """
    return embed_texts([query])[0]
