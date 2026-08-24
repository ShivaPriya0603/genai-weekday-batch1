"""
Shared embedding model wrapper.

Both the simple pipeline (Llama3.2:3b) and the complex pipeline
(GPT-4o-mini) must embed queries the exact same way they were embedded
during ingestion, otherwise similarity search is meaningless. Using
OpenAI's `text-embedding-3-small` here keeps that guarantee identical
for every caller regardless of which LLM answers the question, and
avoids pulling a local torch/sentence-transformers model.
"""

from functools import lru_cache
from typing import List

from openai import OpenAI

from backend.config import EMBEDDING_MODEL_NAME, OPENAI_API_KEY


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str) -> List[float]:
    """Embed a single string into a dense vector."""
    return embed_texts([text])[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of strings. Shared by ingestion and both RAG pipelines."""
    client = _get_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
    return [item.embedding for item in response.data]
