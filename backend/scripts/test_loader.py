"""
scripts/test_loader.py
----------------------
Quick smoke-test for app/rag/loader.py.

Run from the backend/ directory:
    python scripts/test_loader.py

Output:
    - Total pages extracted
    - First 500 characters of page 1
    - Metadata for every page
"""

import sys
import io
from pathlib import Path

# Reconfigure stdout to UTF-8 so any Unicode in the PDF text is safe to print
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Make sure the backend package is importable when run from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.loader import load_pdf

PDF_PATH = Path(__file__).resolve().parent.parent / "documents" / "hospital-info.pdf"


def main() -> None:
    print("=" * 60)
    print(f"Loading PDF: {PDF_PATH}")
    print("=" * 60)

    pages = load_pdf(str(PDF_PATH))

    # -- Summary -------------------------------------------------------
    print(f"\nPages extracted : {len(pages)}\n")

    # -- First page preview --------------------------------------------
    if pages:
        first = pages[0]
        preview = first["text"][:500].strip()
        print("-- Page 1 text preview (first 500 chars) ----------------")
        print(preview)
        print()

    # -- Metadata for every page ---------------------------------------
    print("-- Metadata per page ------------------------------------")
    for doc in pages:
        print(f"  {doc['metadata']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
