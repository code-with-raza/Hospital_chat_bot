"""
scripts/test_embeddings.py
--------------------------
Test app/rag/embeddings.py.

Run from the backend/ directory:
    python scripts/test_embeddings.py
"""

import sys
import io
from pathlib import Path

# UTF-8 stdout for safe printing on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.loader import load_pdf
from app.rag.chunker import chunk_documents
from app.rag.embeddings import (
    EMBEDDING_MODEL,
    embed_chunks,
    embed_query,
    get_embedding_dimension,
)

PDF_PATH  = Path(__file__).resolve().parent.parent / "documents" / "hospital-info.pdf"
PREVIEW_N = 6   # number of vector values to display


def divider(title: str = "") -> None:
    if title:
        print(f"\n--- {title} " + "-" * max(0, 50 - len(title)))
    else:
        print("-" * 54)


def main() -> None:
    # 1. Load and chunk
    divider("Loading chunks")
    pages  = load_pdf(str(PDF_PATH))
    chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=50)
    print(f"  Pages  : {len(pages)}")
    print(f"  Chunks : {len(chunks)}")

    # 2. Generate embeddings
    divider("Generating embeddings")
    print(f"  Model  : {EMBEDDING_MODEL}")
    print("  Embedding all chunks...", end=" ", flush=True)
    embedded = embed_chunks(chunks)
    print("done.")

    # 3. Number of embeddings
    divider("Results")
    print(f"  Embeddings generated : {len(embedded)}")

    # 4. Embedding dimension
    dim = get_embedding_dimension()
    actual_dim = len(embedded[0]["embedding"])
    print(f"  Embedding dimension  : {actual_dim}  (model reports {dim})")

    # 5. First few values of the first embedding
    divider("First embedding — sample values")
    first_vec = embedded[0]["embedding"]
    values    = ", ".join(f"{v:.6f}" for v in first_vec[:PREVIEW_N])
    print(f"  Chunk 0 text  : {embedded[0]['text'][:80].replace(chr(10), ' ').strip()!r}")
    print(f"  Vector[0:{PREVIEW_N}]  : [{values}, ...]")
    print(f"  Full length   : {len(first_vec)} values")

    # Bonus: query embedding
    divider("Query embedding (bonus)")
    query = "What are the visiting hours?"
    q_vec = embed_query(query)
    q_vals = ", ".join(f"{v:.6f}" for v in q_vec[:PREVIEW_N])
    print(f"  Query         : {query!r}")
    print(f"  Vector[0:{PREVIEW_N}]  : [{q_vals}, ...]")
    print(f"  Dimension     : {len(q_vec)}")

    divider()
    print("Done.")


if __name__ == "__main__":
    main()
