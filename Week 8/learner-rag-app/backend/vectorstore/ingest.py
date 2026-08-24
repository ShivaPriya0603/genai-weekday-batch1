"""
Shared chunk -> embed -> upsert logic.

Used by both the one-off CLI script (`ingestion/ingest_documents.py`)
and the dynamic upload endpoint (`backend/api/ingest.py`) so a PDF
dropped in through the Streamlit UI at runtime is chunked/embedded
exactly the same way as one ingested up front from disk.
"""

import hashlib
import io
from pathlib import Path
from typing import List

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE
from backend.vectorstore.embeddings import embed_texts
from backend.vectorstore.pinecone_client import upsert_vectors

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple sliding-window character chunker with overlap."""
    text = " ".join(text.split())  # collapse whitespace/newlines
    if not text:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += step
    return chunks


def make_chunk_id(source: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}-{index}".encode("utf-8")).hexdigest()[:10]
    return f"{Path(source).stem}-{index}-{digest}"


def ingest_bytes(filename: str, content: bytes) -> int:
    """
    Extracts text, chunks it, embeds the chunks, and upserts them into
    the shared Pinecone index under `source=filename`. Returns the
    number of chunks upserted (0 if the file had no extractable text).
    """
    text = extract_text(filename, content)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    vectors_values = embed_texts(chunks)
    vectors = [
        {
            "id": make_chunk_id(filename, i),
            "values": values,
            "metadata": {"text": chunk, "source": filename, "chunk_index": i},
        }
        for i, (chunk, values) in enumerate(zip(chunks, vectors_values))
    ]
    return upsert_vectors(vectors)
