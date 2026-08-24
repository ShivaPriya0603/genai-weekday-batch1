"""
Central configuration for the RAG chatbot backend.

All environment-driven settings live here so the rest of the codebase
never touches `os.environ` directly. Values are loaded from a `.env`
file (see `.env.example` at the project root) with sane local-dev
defaults for anything that isn't a secret.
"""

import os

from dotenv import load_dotenv

# Load the nearest .env (project root: Week 8/learner-rag-app/.env)
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# API keys / secrets
# ---------------------------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Pinecone (single shared index used by both RAG paths)
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "learner-rag-app")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# Embedding model (shared by both pipelines/ingestion)
# ---------------------------------------------------------------------------
# OpenAI text-embedding-3-small -> keeps embeddings identical for every
# retrieval path regardless of which LLM answers the question.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# ---------------------------------------------------------------------------
# Local model - Ollama (classifier + simple pipeline)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ---------------------------------------------------------------------------
# Cloud model - OpenAI (complex pipeline)
# ---------------------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------------
SIMPLE_TOP_K = int(os.getenv("SIMPLE_TOP_K", "3"))
COMPLEX_TOP_K = int(os.getenv("COMPLEX_TOP_K", "4"))

# Minimum cosine similarity (embeddings are normalized, so scores fall
# roughly in [-1, 1]) a retrieved passage's best match must clear before
# we treat the knowledge base as actually containing an answer. Below
# this, the query is considered "out of context" for whatever is
# currently indexed and the pipeline falls back to the model's own
# general knowledge, clearly flagged as ungrounded rather than silently
# answering off irrelevant chunks. This check always runs AFTER each
# pipeline's own retrieval (simple: single retrieval; complex: pooled
# multi-query retrieval) -- never before, and never as a gate that could
# reject a query ahead of the complex pipeline's decomposition getting a
# chance to find something the raw query alone missed. That matters
# because the knowledge base is built from dynamically uploaded
# documents, so what counts as "in context" changes with every upload;
# a pre-check run before retrieval would be judging against a moving
# target instead of what each pipeline actually found. Tune this by
# watching real scores in the trace.
CONTEXT_RELEVANCE_THRESHOLD = float(os.getenv("CONTEXT_RELEVANCE_THRESHOLD", "0.35"))

# ---------------------------------------------------------------------------
# Ingestion / chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
VERBOSE_TRACE = _get_bool("VERBOSE_TRACE", True)
