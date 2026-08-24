"""
Streamlit chat UI for the Learner RAG Chatbot.

Talks to the FastAPI backend:
  - POST /ingest to dynamically upload a PDF/TXT/MD into the shared
    Pinecone index at runtime (no need to pre-run the CLI script).
  - POST /chat for every question. Each reply comes back with which
    pipeline handled it (simple/complex), which model answered, and
    whether the answer was actually grounded in the knowledge base or
    fell back to the model's own general knowledge. Only that one-line
    status is shown -- no source list, no citations, no step-by-step
    trace in the UI (the API response still carries them, for anyone
    consuming it directly, but the chat view stays just the answer).
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Learner RAG Chatbot", page_icon="🧭", layout="centered")

st.title("🧭 Learner RAG Chatbot")
st.caption(
    "Every question is classified as **simple** (single retrieval → Llama3.2:3b) or "
    "**complex** (decompose → multi-query retrieval → GPT-4o-mini) and routed to the "
    "matching pipeline over the same Pinecone index. If that pipeline's own retrieval "
    "doesn't turn up anything relevant, it answers from general knowledge instead -- "
    "clearly flagged as not based on your documents."
)

with st.sidebar:
    st.header("Settings")
    st.text_input("Backend URL", value=BACKEND_URL, key="backend_url", disabled=True)

    if st.button("Check backend health"):
        try:
            resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
            resp.raise_for_status()
            st.success(resp.json())
        except Exception as exc:
            st.error(f"Backend unreachable: {exc}")

    st.divider()
    st.subheader("Upload a document")
    st.caption("Adds straight into the shared knowledge base -- ask about it right away.")
    uploaded_file = st.file_uploader("PDF, TXT, or MD", type=["pdf", "txt", "md"])
    if uploaded_file is not None and st.button("Upload & ingest"):
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                resp = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Ingested {data['chunks_upserted']} chunk(s) from {data['filename']}.")
            except requests.HTTPError as exc:
                detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                st.error(f"Ingestion failed: {detail}")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


CLASSIFICATION_BADGES = {
    "simple": "🟢 simple",
    "complex": "🔵 complex",
}


def render_meta(meta: dict):
    """Just the one-line status -- no sources, no citations, no trace in the UI."""
    badge = CLASSIFICATION_BADGES.get(meta["classification"], meta["classification"])
    grounded_badge = "📄 grounded in your documents" if meta.get("grounded", True) else "⚠️ not found in your documents -- answered from general knowledge"
    st.caption(f"{badge} · model: `{meta['model_used']}` · {grounded_badge}")


if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "meta": {...}}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            render_meta(msg["meta"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Classifying and retrieving..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/chat", json={"message": user_input}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ Request failed: {exc}"}
                )
                st.stop()

        st.markdown(data["answer"])
        render_meta(data)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "meta": {
                "classification": data["classification"],
                "model_used": data["model_used"],
                "sources": data.get("sources", []),
                "grounded": data.get("grounded", True),
                "trace": data.get("trace", []),
            },
        }
    )
