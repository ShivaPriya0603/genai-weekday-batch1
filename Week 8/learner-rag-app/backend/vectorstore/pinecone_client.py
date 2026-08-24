"""
Pinecone init + index connection.

There is exactly ONE shared index for the whole app. Both RAG paths
(simple tool-calling and complex multi-query) read from and write to
this same index -- they just query it with different strategies
(single query vs. several decomposed sub-queries).
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from backend.config import (
    EMBEDDING_DIMENSION,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to your .env file."
        )
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index_exists() -> None:
    """Create the shared index if it doesn't exist yet (idempotent)."""
    pc = get_pinecone_client()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


@lru_cache(maxsize=1)
def get_index():
    ensure_index_exists()
    pc = get_pinecone_client()
    return pc.Index(PINECONE_INDEX_NAME)


def upsert_vectors(vectors: List[Dict[str, Any]], batch_size: int = 100) -> int:
    """
    vectors: list of {"id": str, "values": List[float], "metadata": dict}
    Returns the number of vectors upserted.
    """
    index = get_index()
    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        total += len(batch)
    return total


def query_index(
    vector: List[float],
    top_k: int = 5,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Runs a similarity search and returns a simplified list of matches:
    [{"id": ..., "score": ..., "text": ..., "source": ..., "metadata": {...}}]
    """
    index = get_index()
    result = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter,
    )
    matches = []
    for m in result.get("matches", []) if isinstance(result, dict) else result.matches:
        metadata = m["metadata"] if isinstance(m, dict) else (m.metadata or {})
        score = m["score"] if isinstance(m, dict) else m.score
        match_id = m["id"] if isinstance(m, dict) else m.id
        matches.append(
            {
                "id": match_id,
                "score": score,
                "text": metadata.get("text", ""),
                "source": metadata.get("source", "unknown"),
                "metadata": metadata,
            }
        )
    return matches
