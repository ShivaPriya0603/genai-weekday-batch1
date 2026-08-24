"""
POST /chat -- orchestrates classify -> route -> respond.

This is the only place that wires the classifier and the two pipelines
together; everything else stays independently testable.

There is deliberately no relevance pre-check gate ahead of classification
here. An earlier version added one (one cheap single-query retrieval
before even classifying, to skip obviously off-topic questions), but
that meant a query could get rejected as "out of context" based on a
shallow single-query retrieval before the complex pipeline's own
decomposition/multi-query retrieval -- which can find relevant content
the raw query alone misses -- ever got a chance to run. That's
especially wrong for a knowledge base that's built from dynamically
uploaded documents: what counts as "in context" changes every time
someone uploads a new PDF, so a static pre-check gate ages badly.
Instead, each pipeline always runs its own retrieval strategy first,
and ONLY THEN decides -- from its own, better-informed results --
whether to answer grounded in retrieved passages or fall back to the
model's own general knowledge (see `retrieval/retriever.py::is_grounded`
and each pipeline's ungrounded system prompt).
"""

import logging

from fastapi import APIRouter, HTTPException

from backend.classifier.query_classifier import classify_query
from backend.config import OLLAMA_MODEL, OPENAI_MODEL
from backend.models.schemas import ChatRequest, ChatResponse
from backend.pipelines.complex_pipeline import run_complex_pipeline
from backend.pipelines.simple_pipeline import run_simple_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    query = request.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="message must not be empty")

    trace = []
    try:
        classification = classify_query(query)
    except Exception:
        logger.exception("Classification failed")
        raise HTTPException(status_code=502, detail="Classifier (Ollama) call failed.")
    trace.append(f"Classifier labeled query as: {classification}")

    try:
        if classification == "complex":
            result = run_complex_pipeline(query)
            model_used = OPENAI_MODEL
        else:
            result = run_simple_pipeline(query)
            model_used = OLLAMA_MODEL
    except Exception:
        logger.exception("Pipeline execution failed for classification=%s", classification)
        raise HTTPException(status_code=502, detail=f"{classification} pipeline failed.")

    trace.extend(result.get("trace", []))

    sources = [
        {"source": m.get("source", "unknown"), "text": m.get("text", ""), "score": float(m.get("score", 0.0))}
        for m in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        classification=classification,
        model_used=model_used,
        sources=sources,
        grounded=result.get("grounded", True),
        trace=trace,
    )
