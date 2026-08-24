"""
Wrapper around the OpenAI API (GPT-4o-mini).

Plain chat calls only -- no tool/function-calling. Used by
pipelines/complex_pipeline.py for query decomposition and for final
synthesis over the combined multi-query retrieved context.
"""

from functools import lru_cache
from typing import Optional

from openai import OpenAI

from backend.config import OPENAI_API_KEY, OPENAI_MODEL


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=OPENAI_API_KEY)


def chat(prompt: str, system: Optional[str] = None, model: str = OPENAI_MODEL, temperature: float = 0.2) -> str:
    """One-shot chat call: optional system prompt + a single user turn -> response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()
