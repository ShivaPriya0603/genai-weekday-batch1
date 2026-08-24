"""Pydantic request/response models for the /chat endpoint."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's question.")


class SourceItem(BaseModel):
    source: str
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    classification: str  # "simple" | "complex"
    model_used: str  # "llama3.2:3b" | "gpt-4o-mini"
    sources: List[SourceItem] = []
    grounded: bool = Field(
        default=True,
        description="False when no retrieved passage cleared the relevance threshold, "
        "i.e. the answer fell back to the model's own general knowledge instead of the uploaded docs.",
    )
    trace: List[str] = Field(
        default_factory=list,
        description="Step-by-step log of classification/retrieval decisions (for the UI's debug panel).",
    )


class IngestResponse(BaseModel):
    filename: str
    chunks_upserted: int


class HealthResponse(BaseModel):
    status: str
    pinecone_index: Optional[str] = None
