"""Build the local vector index from the hospital information PDF."""

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.chunker import chunk_documents
from app.rag.embeddings import embed_chunks
from app.rag.loader import load_pdf
from app.rag.vector_stores import add_documents


def main() -> None:
    pdf_path = BACKEND_DIR / "documents" / "hospital-info.pdf"
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Hospital information PDF not found: {pdf_path}")

    pages = load_pdf(str(pdf_path))
    chunks = chunk_documents(pages)
    embedded_chunks = embed_chunks(chunks)
    stored_count = add_documents(embedded_chunks, reset=True)

    print(f"Pages loaded: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embeddings generated: {len(embedded_chunks)}")
    print(f"Vectors stored: {stored_count}")


if __name__ == "__main__":
    main()
