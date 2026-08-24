"""
Wrapper around the local Ollama server (Llama3.2:3b).

Plain chat calls only -- no tool/function-calling. Used by:
  - classifier/query_classifier.py (classification prompt)
  - pipelines/simple_pipeline.py (final answer generation over retrieved context)
"""

from typing import Optional

import ollama

from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL

_client = ollama.Client(host=OLLAMA_BASE_URL)


def chat(prompt: str, system: Optional[str] = None, model: str = OLLAMA_MODEL, temperature: float = 0.2) -> str:
    """One-shot chat call: optional system prompt + a single user turn -> response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature},
    )
    return (response["message"].get("content") or "").strip()
