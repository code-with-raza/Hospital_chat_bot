"""
app/rag/loader.py
-----------------
Loads and extracts text from a PDF file page by page using pypdf.

Returns a list of dicts with the extracted text and metadata.
No chunking, embeddings, or retrieval logic is included here.
"""

from pathlib import Path
from typing import List, Dict, Any

import pypdf


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a PDF and extract text from every page.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    List[Dict[str, Any]]
        A list of page documents, each with the shape::

            {
                "text": "<extracted page text>",
                "metadata": {
                    "source": "<filename>",
                    "page": <1-based page number>
                }
            }

    Raises
    ------
    FileNotFoundError
        If the PDF file does not exist at the given path.
    """
    path = Path(file_path)

    # 1. Check the file exists
    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: '{path.resolve()}'. "
            "Make sure the path is correct."
        )

    if not path.is_file():
        raise FileNotFoundError(f"'{path.resolve()}' is not a file.")

    pages: List[Dict[str, Any]] = []

    # 2. Open the PDF
    with open(path, "rb") as pdf_file:
        reader = pypdf.PdfReader(pdf_file)

        # 3 & 4. Extract text page by page, keeping the page number (1-based)
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""  # extract_text() can return None

            # 5. Build the document dict
            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": path.name,     # e.g. "hospital-info.pdf"
                        "page": page_index + 1,  # 1-based
                    },
                }
            )

    return pages
