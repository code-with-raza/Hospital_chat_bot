"""
app/rag/chunker.py
------------------
Splits page-level documents (produced by loader.py) into smaller,
overlapping text chunks suitable for embedding and retrieval.

Strategy — recursive character splitting
-----------------------------------------
We try to break on natural boundaries in this priority order:
    paragraph break  ->  newline  ->  sentence end  ->  word break  ->  character

This keeps semantic units together as long as possible before falling
back to a hard character split.

No embeddings or vector-database code lives here.
"""

from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    """
    Recursively split *text* on the first separator that produces pieces
    short enough to fit within *chunk_size*.  Falls back to the next
    separator when a piece is still too long.
    """
    if not text:
        return []

    # If the entire text already fits, no split needed
    if len(text) <= chunk_size:
        return [text]

    # Try each separator in priority order
    for sep in separators:
        if sep == "":
            # Hard character split — last resort
            return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

        if sep in text:
            parts = text.split(sep)
            chunks: List[str] = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if len(part) <= chunk_size:
                    chunks.append(part)
                else:
                    # Part is still too long — recurse with remaining separators
                    remaining_seps = separators[separators.index(sep) + 1:]
                    chunks.extend(_recursive_split(part, chunk_size, remaining_seps))
            return chunks

    return [text]


def _apply_overlap(pieces: List[str], chunk_size: int, overlap: int) -> List[str]:
    """
    Merge adjacent *pieces* into windows of up to *chunk_size* characters
    with *overlap* characters of carry-over from the previous window.
    """
    if not pieces:
        return []

    chunks: List[str] = []
    current = ""

    for piece in pieces:
        separator = "\n" if current else ""
        candidate = current + separator + piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # Start new window, seeded with the overlap tail of the old one
            tail = current[-overlap:] if overlap and current else ""
            current = (tail + "\n" + piece).strip() if tail else piece

    if current:
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split page-level documents into smaller chunks.

    Parameters
    ----------
    documents : List[Dict[str, Any]]
        Output of ``load_pdf()`` — each item has ``"text"`` and ``"metadata"``.
    chunk_size : int
        Maximum number of characters per chunk.  Default: 500.
    chunk_overlap : int
        Number of characters carried over from the previous chunk so context
        is not lost at boundaries.  Default: 50.

    Returns
    -------
    List[Dict[str, Any]]
        A flat list of chunks, each with the shape::

            {
                "text": "<chunk text>",
                "metadata": {
                    "source":      "<filename>",
                    "page":        <1-based page number>,
                    "chunk_index": <0-based index within this page>
                }
            }

    Notes
    -----
    * Empty pages (blank or whitespace-only text) are silently skipped.
    * Original metadata keys are preserved; ``chunk_index`` is added.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )

    all_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        text: str = doc.get("text", "").strip()
        metadata: Dict[str, Any] = doc.get("metadata", {})

        if not text:
            continue  # skip blank pages

        # Step 1 — recursive split into pieces that respect natural boundaries
        pieces = _recursive_split(text, chunk_size, _SEPARATORS)

        # Step 2 — merge pieces into overlapping windows
        windows = _apply_overlap(pieces, chunk_size, chunk_overlap)

        # Step 3 — build output dicts, injecting chunk_index into metadata
        for idx, chunk_text in enumerate(windows):
            if not chunk_text.strip():
                continue
            all_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **metadata,           # source, page — preserved as-is
                        "chunk_index": idx,   # 0-based within this page
                    },
                }
            )

    return all_chunks
