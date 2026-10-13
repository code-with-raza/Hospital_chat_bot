"""
scripts/test_retriever.py
--------------------------
End-to-end test for app/rag/retriever.py.

Prerequisites:
    The vector store must be populated first.
    Run this script — it auto-ingests if the collection is empty.

Run from the backend/ directory:
    python scripts/test_retriever.py
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
from app.rag.embeddings import embed_chunks
from app.rag.vector_stores import add_documents, collection_size
from app.rag.retriever import retrieve

PDF_PATH    = Path(__file__).resolve().parent.parent / "documents" / "hospital-info.pdf"
TOP_K       = 3
PREVIEW_LEN = 180

QUESTIONS = [
    "What are the visiting hours?",
    "Is the emergency department open 24 hours?",
    "What documents should a patient bring?",
]


def div(title: str = "") -> None:
    width = 60
    if title:
        print(f"\n{'=' * 4}  {title}  {'=' * max(0, width - len(title) - 7)}")
    else:
        print("-" * width)


# ---------------------------------------------------------------------------
# Auto-ingest: populate the vector store if it is empty
# ---------------------------------------------------------------------------

def ensure_populated() -> None:
    if collection_size() > 0:
        print(f"  Vector store already has {collection_size()} documents — skipping ingest.")
        return

    print("  Vector store is empty — ingesting now...")
    pages    = load_pdf(str(PDF_PATH))
    chunks   = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
    embedded = embed_chunks(chunks)
    n        = add_documents(embedded, reset=False)
    print(f"  Ingested {n} chunks into the vector store.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    div("Setup — vector store")
    ensure_populated()
    print(f"  Collection size : {collection_size()} documents")

    for question in QUESTIONS:
        div(f"Query")
        print(f"  Q: {question!r}")
        print(f"  Retrieving top {TOP_K} chunks...\n")

        results = retrieve(question, top_k=TOP_K)

        for rank, hit in enumerate(results, start=1):
            meta    = hit["metadata"]
            score   = hit["score"]
            preview = hit["text"][:PREVIEW_LEN].replace("\n", " ").strip()
            print(
                f"  [{rank}]  score={score:.4f}  "
                f"page={meta['page']}  "
                f"chunk_index={meta['chunk_index']}"
            )
            print(f"       {preview!r}")
            print()

    div()
    print("Done.")


if __name__ == "__main__":
    main()
