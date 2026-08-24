"""
Query classifier: labels each incoming query "simple" or "complex".

Uses the same Llama3.2:3b model that powers the simple pipeline, so the
classifier costs nothing extra in terms of infrastructure -- just one
extra fast local call before routing.

A 3B model turns out to unreliably judge structure in BOTH directions:
- It conflates "this topic has a lot that COULD be said about it" with
  "this question is structurally complex" -- "What is photosynthesis"
  gets called complex just as often as "Compare the refund and warranty
  policies", even though the first is a plain one-fact lookup.
- It also misses genuinely combined questions ("What are the three
  stages X occurs in, AND how is each one mitigated?") and calls them
  "simple" -- which matters more than it sounds like: the simple
  pipeline retrieves ONCE for the whole question, so when a combined
  question actually needs two different lookups, that single retrieval
  dilutes across both and finds neither well, producing a hedged
  "the passages don't specify this" answer instead of decomposing.

So there are TWO deterministic fast paths, not one, and the LLM call
only happens for whatever's left in the genuinely ambiguous middle:
  1. `_looks_structurally_simple` -- unambiguous single-lookup shapes
     ("what is X", "who was X", "how many X") get "simple" for free.
  2. `_looks_structurally_complex` -- explicit comparison ("compare",
     "X vs Y", "difference between") or explicit combined-question
     markers ("...X, and how does Y...") get "complex" for free.
"""

import logging

from backend.llm.ollama_client import chat as ollama_chat

logger = logging.getLogger(__name__)

# Question shapes that are always a single lookup, regardless of how much
# there is to say about the topic itself ("what is X" is simple whether X
# is "the refund policy" or "quantum entanglement"). Bare interrogative
# starters (not just "what is "/"what are ") so "what CVE is...", "what
# did X expand on...", etc. are covered without enumerating every verb.
_SIMPLE_QUESTION_PREFIXES = (
    "what ", "what's ",
    "who ", "when ", "where ", "why ",
    "define ",
    "how many ", "how much ", "how old ", "how long ", "how far ",
)

# Explicit signals that the question relates/contrasts multiple things or
# combines two questions into one. These force "complex" directly -- they
# don't just withhold the "simple" fast path -- because the LLM call
# turns out to be unreliable on exactly these combined-question cases.
_COMPLEXITY_SIGNALS = (
    "compare", " vs ", " vs.", "versus",
    "difference between", "differences between", "differ from", "differs from",
    "different from", "similarities between", "relationship between",
    "relate to each other", "connect to each other", "pros and cons",
    " and how ", " and which ", " and what ", " and when ", " and where ",
    " and why ", " and who ", "across ",
)


def _normalize(query: str) -> str:
    return " " + query.strip().lower().rstrip("?.! ") + " "


def _has_complexity_signal(query: str) -> bool:
    normalized = _normalize(query)
    return any(signal in normalized for signal in _COMPLEXITY_SIGNALS)


def _looks_structurally_simple(query: str) -> bool:
    if _has_complexity_signal(query):
        return False
    return _normalize(query).lstrip().startswith(_SIMPLE_QUESTION_PREFIXES)


def _looks_structurally_complex(query: str) -> bool:
    return _has_complexity_signal(query)


SYSTEM_PROMPT = """You are a query router for a RAG system. Classify the user's \
question as exactly one word: "simple" or "complex".

DEFAULT TO "simple". Only answer "complex" if AT LEAST ONE of these is true:
1. Answering well genuinely requires breaking the question into two or more \
separate sub-questions that would each need their own lookup.
2. The message combines two or more distinct questions into one (e.g. "What is \
X, and how does Y work?", "What is X and when was it launched?").

If neither is true, the answer is "simple" -- no matter how broad, technical, or \
deep the topic itself is. Judge the question's STRUCTURE, not how much could be \
said about the topic: "What is quantum entanglement?" is one question about one \
thing, so it's "simple", even though a long answer is possible. Only "compare X \
and Y", "what's the difference between X and Y", multi-part questions, and \
questions that name several distinct things to relate to each other count as \
"complex".

Examples:
Q: "What is the refund policy?"
A: simple

Q: "What is photosynthesis?"
A: simple

Q: "What is quantum entanglement?"
A: simple

Q: "Explain how photosynthesis works."
A: simple

Q: "Compare the refund policy with the warranty policy and tell me which is more generous."
A: complex

Q: "When was the product launched?"
A: simple

Q: "What is the refund policy, and when was the warranty policy last updated?"
A: complex

Q: "Explain how onboarding, billing, and support processes connect to each other."
A: complex

When unsure, answer "simple". Respond with ONLY the single word "simple" or \
"complex". No punctuation, no explanation."""


def classify_query(query: str) -> str:
    """Returns "simple" or "complex". Defaults to "simple" on any ambiguity/error."""
    if _looks_structurally_simple(query):
        logger.info("Classified query %r as simple (fast path, no LLM call).", query)
        return "simple"

    if _looks_structurally_complex(query):
        logger.info("Classified query %r as complex (fast path, no LLM call).", query)
        return "complex"

    try:
        raw = ollama_chat(prompt=query, system=SYSTEM_PROMPT, temperature=0.0).lower().strip()
    except Exception:
        logger.exception("Classifier call failed; defaulting to 'simple'.")
        return "simple"

    if "complex" in raw:
        label = "complex"
    elif "simple" in raw:
        label = "simple"
    else:
        logger.warning("Unrecognized classifier output %r; defaulting to 'simple'.", raw)
        label = "simple"

    logger.info("Classified query %r as %s (LLM call)", query, label)
    return label
