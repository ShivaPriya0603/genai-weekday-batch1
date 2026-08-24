"""
One-off ingestion script: chunk docs -> embed -> upsert to Pinecone.

Usage:
    python ingestion/ingest_documents.py --docs-dir ingestion/sample_docs
    python ingestion/ingest_documents.py --docs-dir path/to/your/docs

Supports .txt, .md, and .pdf files. Reuses the exact same chunk/embed/
upsert logic (backend/vectorstore/ingest.py) as the app's dynamic
`POST /ingest` endpoint, and the exact same embedding model and
Pinecone index that both RAG pipelines read from at query time -- so
what you ingest here is what gets retrieved.

This script is for pre-loading a knowledge base in bulk before the app
starts. For adding documents while the app is running (e.g. a user
uploading a PDF from the Streamlit UI), see backend/api/ingest.py.
"""

import argparse
import sys
from pathlib import Path

# Allow `python ingestion/ingest_documents.py` to import the `backend` package
# even though this script lives outside it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.vectorstore.ingest import SUPPORTED_EXTENSIONS, ingest_bytes  # noqa: E402


def ingest_directory(docs_dir: Path) -> int:
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        print(f"No supported documents ({', '.join(SUPPORTED_EXTENSIONS)}) found in {docs_dir}")
        return 0

    total_upserted = 0
    for file_path in files:
        print(f"Reading {file_path} ...")
        count = ingest_bytes(file_path.name, file_path.read_bytes())
        if count == 0:
            print("  (skipped -- no extractable text)")
            continue
        total_upserted += count
        print(f"  upserted {count} vector(s) from {file_path.name}")

    print(f"\nDone. Total vectors upserted: {total_upserted}")
    return total_upserted


def main():
    parser = argparse.ArgumentParser(description="Chunk, embed, and upsert documents into Pinecone.")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(Path(__file__).parent / "sample_docs"),
        help="Directory containing .txt/.md/.pdf files to ingest.",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise SystemExit(f"Docs directory not found: {docs_dir}")

    ingest_directory(docs_dir)


if __name__ == "__main__":
    main()
