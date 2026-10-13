"""
app/rag/vector_stores.py
------------------------
Stores document chunks (text + metadata + embeddings) in a local
ChromaDB collection and provides cosine-similarity search.

ChromaDB runs fully in-process — no server required.  Data is persisted
to disk so the collection survives restarts.

Configurable via .env
---------------------
CHROMA_DB_DIR        path where ChromaDB persists data
                     default: <backend>/data/chroma
CHROMA_COLLECTION    name of the collection inside ChromaDB
                     default: hospital_chunks

No reranking or LLM generation lives here.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# Where ChromaDB persists its SQLite + binary files
_DEFAULT_DB_DIR = str(
    Path(__file__).resolve().parent.parent / "data" / "chroma"
)

_configured_db_dir = Path(os.getenv("CHROMA_DB_DIR", _DEFAULT_DB_DIR))
if not _configured_db_dir.is_absolute():
    # Resolve configured relative paths against the backend, not the shell's
    # current directory, so ingestion and the API always share one index.
    _configured_db_dir = Path(__file__).resolve().parents[2] / _configured_db_dir

CHROMA_DB_DIR: str = str(_configured_db_dir)
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "hospital_chunks")


# ---------------------------------------------------------------------------
# Internal client (lazily created once)
# ---------------------------------------------------------------------------

_client: chromadb.Client | None = None


def _get_client() -> chromadb.Client:
    """Return a cached persistent ChromaDB client."""
    global _client
    if _client is None:
        Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection(reset: bool = False) -> chromadb.Collection:
    """
    Return (or create) the configured ChromaDB collection.

    Parameters
    ----------
    reset : bool
        If True, delete the collection first so it starts empty.
        Useful when re-ingesting the full document set.
    """
    client = _get_client()
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass  # collection didn't exist yet — fine
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_documents(
    embedded_chunks: List[Dict[str, Any]],
    reset: bool = False,
) -> int:
    """
    Store embedded chunks in the vector store.

    Parameters
    ----------
    embedded_chunks : List[Dict[str, Any]]
        Output of ``embed_chunks()`` — each dict must have:
          - ``"text"``      : str
          - ``"embedding"`` : List[float]
          - ``"metadata"``  : Dict  (source, page, chunk_index)
    reset : bool
        Wipe the collection before inserting (full re-ingest).

    Returns
    -------
    int
        Number of documents stored.
    """
    if not embedded_chunks:
        return 0

    collection = get_collection(reset=reset)

    ids         = [str(uuid.uuid4()) for _ in embedded_chunks]
    texts       = [c["text"]       for c in embedded_chunks]
    embeddings  = [c["embedding"]  for c in embedded_chunks]
    # ChromaDB metadata values must be str / int / float — no nested dicts
    metadatas   = [
        {
            "source":      str(c["metadata"].get("source", "")),
            "page":        int(c["metadata"].get("page", 0)),
            "chunk_index": int(c["metadata"].get("chunk_index", 0)),
        }
        for c in embedded_chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(embedded_chunks)


def search(
    query_embedding: List[float],
    n_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find the *n_results* most similar chunks for a query vector.

    Parameters
    ----------
    query_embedding : List[float]
        Vector produced by ``embed_query()``.
    n_results : int
        Number of top results to return.  Default: 3.

    Returns
    -------
    List[Dict[str, Any]]
        Ranked list (most similar first), each item::

            {
                "text":      "<chunk text>",
                "metadata":  {"source": ..., "page": ..., "chunk_index": ...},
                "score":     <cosine similarity 0–1, higher = more similar>
            }
    """
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    n_results = min(n_results, total)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine space returns distance = 1 - similarity
        similarity = round(1.0 - distance, 4)
        hits.append(
            {
                "text":     text,
                "metadata": meta,
                "score":    similarity,
            }
        )

    return hits


def collection_size() -> int:
    """Return the number of documents currently in the collection."""
    return get_collection().count()
