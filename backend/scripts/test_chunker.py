"""
scripts/test_chunker.py
-----------------------
Test app/rag/chunker.py.

Run from the backend/ directory:
    python scripts/test_chunker.py
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

PDF_PATH = Path(__file__).resolve().parent.parent / "documents" / "hospital-info.pdf"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
PREVIEW_LEN   = 150   # characters of text to show per chunk


def divider(title: str = "") -> None:
    if title:
        print(f"\n--- {title} " + "-" * max(0, 50 - len(title)))
    else:
        print("-" * 54)


def main() -> None:
    # 1. Load documents
    divider("Loading documents")
    pages = load_pdf(str(PDF_PATH))
    print(f"  Pages loaded : {len(pages)}")

    # 2. Chunk documents
    divider("Chunking")
    chunks = chunk_documents(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 3. Total number of chunks
    divider("Summary")
    print(f"  Total chunks  : {len(chunks)}")
    print(f"  chunk_size    : {CHUNK_SIZE}")
    print(f"  chunk_overlap : {CHUNK_OVERLAP}")

    # 4 & 5 & 6. First 3 chunks — metadata + text preview
    divider("First 3 chunks")
    for i, chunk in enumerate(chunks[:3]):
        meta    = chunk["metadata"]
        preview = chunk["text"][:PREVIEW_LEN].replace("\n", " ").strip()
        print(f"\n  [chunk {i}]")
        print(f"    chunk_index : {meta['chunk_index']}")
        print(f"    page        : {meta['page']}")
        print(f"    source      : {meta['source']}")
        print(f"    text length : {len(chunk['text'])} chars")
        print(f"    preview     : {preview!r}")

    divider()
    print("Done.")


if __name__ == "__main__":
    main()
