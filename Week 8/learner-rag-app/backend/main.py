"""FastAPI entrypoint. Mounts routers, sets up CORS for the Streamlit frontend."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.chat import router as chat_router
from backend.api.ingest import router as ingest_router
from backend.config import PINECONE_INDEX_NAME
from backend.models.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Learner RAG Chatbot",
    description="RAG chatbot that routes queries to a simple (Llama3.2:3b + tool-calling) "
    "or complex (GPT-4o-mini + multi-query) RAG pipeline over a shared Pinecone index.",
    version="1.0.0",
)

# Streamlit runs on a different port -- allow it (and any local dev origin) through.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(ingest_router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", pinecone_index=PINECONE_INDEX_NAME)
