"""
POST /ingest -- dynamic document upload.

Lets the Streamlit frontend push a PDF/TXT/MD straight into the shared
Pinecone index at runtime, instead of requiring the standalone CLI
script to be run ahead of time. Both paths share the same chunk/embed/
upsert logic (backend/vectorstore/ingest.py), so a chatbot question can
be answered from a document that was uploaded moments earlier.
"""

import logging

from fastapi import APIRouter, HTTPException, UploadFile

from backend.models.schemas import IngestResponse
from backend.vectorstore.ingest import SUPPORTED_EXTENSIONS, ingest_bytes

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile) -> IngestResponse:
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()
    try:
        chunks_upserted = ingest_bytes(file.filename, content)
    except Exception:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=502, detail="Ingestion failed. See backend logs.")

    if chunks_upserted == 0:
        raise HTTPException(status_code=422, detail="No extractable text found in the uploaded file.")

    return IngestResponse(filename=file.filename, chunks_upserted=chunks_upserted)
