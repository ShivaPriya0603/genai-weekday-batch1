"""
Shared retrieval logic -- a plain Python function, not an LLM tool.

Both pipelines call this directly:
  - simple_pipeline.py: one call with the raw user query.
  - complex_pipeline.py: one call per decomposed sub-question.

There is no function-calling / tool-use involved anywhere in this app.
The LLM never decides *whether* to retrieve -- the classifier already
made that branching decision, and each pipeline's own code always
retrieves before generating. The only thing that varies is *how many*
retrievals happen and with what query text.

Whether a query is "in context" is decided AFTER each pipeline's own
retrieval runs (see `is_grounded`, used inside simple_pipeline.py and
complex_pipeline.py), not by a pre-check ahead of it. A pre-check based
on one shallow single-query retrieval would risk rejecting a query
before the complex pipeline's decomposition/multi-query retrieval --
which can find relevant content the raw query alone misses -- ever got
a chance to run. That matters especially for a knowledge base built
from dynamically uploaded documents: what counts as "in context"
changes every time someone uploads a new PDF, so it's each pipeline's
own retrieval, run against whatever is currently indexed, that gets to
make the call -- not a static gate ahead of it.
"""

from typing import Any, Dict, List

from backend.config import CONTEXT_RELEVANCE_THRESHOLD, SIMPLE_TOP_K
from backend.vectorstore.embeddings import embed_text
from backend.vectorstore.pinecone_client import query_index


def retrieve(query: str, top_k: int = SIMPLE_TOP_K) -> List[Dict[str, Any]]:
    """Embed `query` and fetch the top_k nearest passages from Pinecone."""
    top_k = max(1, min(int(top_k or SIMPLE_TOP_K), 10))
    vector = embed_text(query)
    return query_index(vector=vector, top_k=top_k)


def is_grounded(matches: List[Dict[str, Any]], threshold: float = CONTEXT_RELEVANCE_THRESHOLD) -> bool:
    """
    Whether the retrieved passages are actually relevant enough to
    ground an answer, vs. the query being out of scope for whatever is
    currently in the (dynamically uploaded) knowledge base.
    """
    if not matches:
        return False
    best_score = max(m.get("score", 0.0) for m in matches)
    return best_score >= threshold


def format_context(matches: List[Dict[str, Any]]) -> str:
    """Render retrieved matches as a numbered context block for the generation prompt."""
    if not matches:
        return "No relevant passages were found in the knowledge base."

    lines = []
    for m in matches:
        lines.append(
            f"(source: {m.get('source', 'unknown')}, score: {m.get('score', 0):.3f})\n"
            f"{m.get('text', '').strip()}"
        )
    return "\n\n".join(lines)
