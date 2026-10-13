"""
scripts/test_vector_store.py
-----------------------------
End-to-end test for app/rag/vector_stores.py.

Run from the backend/ directory:
    python scripts/test_vector_store.py

Steps performed:
    1. Load the PDF
    2. Create chunks
    3. Generate embeddings
    4. Store them in ChromaDB (fresh collection)
    5. Search with a natural-language query
    6. Print top-3 results with score, page, and text preview
"""

import sys
import io
from pathlib import Path

# UTF-8 stdout — safe on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.loader import load_pdf
from app.rag.chunker import chunk_documents
from app.rag.embeddings import embed_chunks, embed_query, EMBEDDING_MODEL
from app.rag.vector_stores import (
    add_documents,
    search,
    collection_size,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION,
)

PDF_PATH     = Path(__file__).resolve().parent.parent / "documents" / "hospital-info.pdf"
QUERY        = "What are the visiting hours at the hospital?"
TOP_N        = 3
PREVIEW_LEN  = 200


def div(title: str = "") -> None:
    if title:
        print(f"\n--- {title} " + "-" * max(0, 50 - len(title)))
    else:
        print("-" * 54)


def main() -> None:
    # ── 1. Load PDF ───────────────────────────────────────────
    div("Step 1 — Load PDF")
    pages = load_pdf(str(PDF_PATH))
    print(f"  Pages loaded : {len(pages)}")

    # ── 2. Chunk ──────────────────────────────────────────────
    div("Step 2 — Chunk documents")
    chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
    print(f"  Chunks created : {len(chunks)}")

    # ── 3. Embed ──────────────────────────────────────────────
    div("Step 3 — Generate embeddings")
    print(f"  Model  : {EMBEDDING_MODEL}")
    print("  Embedding chunks...", end=" ", flush=True)
    embedded = embed_chunks(chunks)
    dim = len(embedded[0]["embedding"])
    print(f"done.  ({len(embedded)} vectors, dim={dim})")

    # ── 4. Store ──────────────────────────────────────────────
    div("Step 4 — Store in ChromaDB")
    print(f"  DB path    : {CHROMA_DB_DIR}")
    print(f"  Collection : {CHROMA_COLLECTION}")
    n_stored = add_documents(embedded, reset=True)   # fresh ingest each run
    print(f"  Stored     : {n_stored} documents")
    print(f"  Collection size (confirmed) : {collection_size()}")

    # ── 5. Search ─────────────────────────────────────────────
    div("Step 5 — Search")
    print(f"  Query : {QUERY!r}")
    query_vec = embed_query(QUERY)
    results   = search(query_vec, n_results=TOP_N)
    print(f"  Results returned : {len(results)}")

    # ── 6 & 7 & 8. Print top results ──────────────────────────
    div(f"Top {TOP_N} results")
    for rank, hit in enumerate(results, start=1):
        meta    = hit["metadata"]
        score   = hit["score"]
        preview = hit["text"][:PREVIEW_LEN].replace("\n", " ").strip()
        print(f"\n  [{rank}]  score={score:.4f}  page={meta['page']}  "
              f"chunk_index={meta['chunk_index']}  source={meta['source']}")
        print(f"       {preview!r}")

    div()
    print("Done.")


if __name__ == "__main__":
    main()
