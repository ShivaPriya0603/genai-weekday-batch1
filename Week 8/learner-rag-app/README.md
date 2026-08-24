# Learner RAG Chatbot -- Query-Classification Routing

A RAG chatbot that classifies every incoming question and diverts the
flow of control to one of two RAG pipelines, both querying the **same
Pinecone index** but with a different retrieval strategy -- and each
finishing with a different generation model.

|                  | Simple Path                        | Complex Path                                                  |
| ---------------- | ---------------------------------- | ------------------------------------------------------------- |
| Retrieval        | One retrieval, on the raw query    | Decompose into 2-4 sub-questions, retrieve each independently |
| Generation model | Llama3.2:3b (Ollama, local)        | GPT-4o-mini (OpenAI)                                          |
| Classifier       | Llama3.2:3b labels the query first | --                                                            |

There is no tool-calling / function-calling anywhere in this app. The
classifier LLM makes exactly one decision -- `simple` or `complex` --
and plain Python code takes it from there: which retrieval strategy
runs, and which model's chat call the retrieved context gets handed to
for the final answer.

```
Streamlit → FastAPI /chat → classifier (Llama3.2:3b labels simple/complex)
          → code branches on the label:
                simple  → retrieve once  → grounded? → Llama3.2:3b answers (from context, or general knowledge if not)
                complex → decompose → retrieve per sub-question → grounded? → GPT-4o-mini answers (from context, or general knowledge if not)
          → response (+ sources + grounded flag + trace) → back to Streamlit
```

## Handling out-of-context questions ("self-answering")

Documents can be uploaded into the knowledge base at any time (see
below), so a question can easily fall outside whatever is currently
indexed. Each pipeline guards against silently answering off irrelevant
chunks, but only **after** it has run its own retrieval -- there is no
relevance check ahead of classification/routing.

That's a deliberate choice, not an oversight: an earlier version of this
app added a cheap pre-check (one shallow retrieval on the raw query,
run before classification) to skip obviously off-topic questions before
doing more work. It backfired for exactly the use case this app is
built around -- a knowledge base fed by dynamically uploaded PDFs. A
static gate judges relevance from a single raw-query retrieval, before
the complex pipeline's decomposition (which can retrieve sub-questions
that find relevant content the raw query alone misses) ever runs, and
before whatever the user just uploaded has had a chance to be searched
properly. What counts as "in context" changes with every upload, so the
call on whether a query is answerable has to be made by each pipeline's
own retrieval against the current index -- not by a gate ahead of it.

So instead:

1. Classification runs on every query, exactly as it decides today
   (`simple` vs `complex`) -- it makes no relevance judgment at all.
2. The routed pipeline runs its own retrieval (simple: one retrieval on
   the raw query; complex: decompose, then one retrieval per
   sub-question, pooled).
3. `retrieval/retriever.py::is_grounded()` checks the **best similarity
   score** among those results against `CONTEXT_RELEVANCE_THRESHOLD`
   (default `0.35`).
4. If it clears the bar, the model answers from the retrieved passages.
   If not, the same model answers from its own general knowledge instead
   -- explicitly told to say so up front, rather than silently guessing
   off irrelevant chunks.

The API returns `grounded: false` whenever step 4 falls back, alongside
the unchanged `classification` (`simple`/`complex`) and `model_used`.
The Streamlit UI shows a `⚠️ not found in your documents -- answered from general knowledge` badge instead of `📄 grounded in your documents`.
Answers never include inline citation markers ([1], [2], etc.) --
retrieved passages are shown only in the UI's "Sources & trace"
expander, not woven into the generated text.

## Project layout

```
learner-rag-app/
├── backend/
│   ├── main.py                  # FastAPI entrypoint, mounts routers
│   ├── api/
│   │   ├── chat.py              # POST /chat -- classify -> route -> respond
│   │   └── ingest.py            # POST /ingest -- dynamic PDF/TXT/MD upload
│   ├── classifier/query_classifier.py
│   ├── pipelines/
│   │   ├── simple_pipeline.py   # 1 retrieval -> groundedness check -> Llama3.2:3b generation
│   │   └── complex_pipeline.py  # decompose -> N retrievals -> groundedness check -> GPT-4o-mini generation
│   ├── retrieval/retriever.py   # plain retrieval function + groundedness check
│   ├── vectorstore/
│   │   ├── pinecone_client.py
│   │   ├── embeddings.py        # shared OpenAI embedding model (text-embedding-3-small)
│   │   └── ingest.py            # shared chunk/embed/upsert logic
│   ├── llm/
│   │   ├── ollama_client.py     # plain chat call, no tools
│   │   └── openai_client.py     # plain chat call, no tools
│   ├── models/schemas.py
│   ├── config.py
│   └── requirements.txt
├── frontend/
│   ├── app.py                   # Streamlit chat UI + document uploader
│   └── requirements.txt
├── ingestion/
│   ├── ingest_documents.py      # CLI: bulk-ingest a folder up front
│   └── sample_docs/             # replace with your own docs
├── .env.example
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the model pulled:
  ```bash
  ollama pull llama3.2:3b
  ollama serve   # if not already running
  ```
- A Pinecone account/API key ([pinecone.io](https://www.pinecone.io)) -- the
  index is created automatically on first run if it doesn't exist.
- An OpenAI API key with access to `gpt-4o-mini`.

### 2. Configure environment

```bash
cd "Week 8/learner-rag-app"
cp .env.example .env
# then edit .env and fill in PINECONE_API_KEY / OPENAI_API_KEY
```

### 3. Install dependencies

From `learner-rag-app/`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 4. Ingest documents

Two ways to get documents into the shared Pinecone index:

- **Bulk, up front** -- drop `.txt` / `.md` / `.pdf` files into
  `ingestion/sample_docs/` (a placeholder doc is already there) and run:
  ```bash
  python ingestion/ingest_documents.py --docs-dir ingestion/sample_docs
  ```
- **Dynamically, at runtime** -- with the backend running, upload a file
  from the Streamlit sidebar ("Upload a document"), or directly:
  ```bash
  curl -F "file=@yourfile.pdf" http://localhost:8000/ingest
  ```

Both paths chunk, embed (with the shared OpenAI `text-embedding-3-small`
model), and upsert into the same index, so a document uploaded through
the UI is queryable on the very next question.

### 5. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Check it's up: `curl http://localhost:8000/health`

### 6. Run the frontend

In a second terminal, from `learner-rag-app/`:

```bash
streamlit run frontend/app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

## How routing works

1. `frontend/app.py` posts `{"message": "..."}` to `backend/api/chat.py`'s `POST /chat`.
2. `chat.py` calls `classifier/query_classifier.py`, which asks Llama3.2:3b
   to label the query `simple` or `complex` (with few-shot examples). This
   is the only decision an LLM makes about control flow, and it's purely
   about the question's shape -- it makes no judgment about whether the
   topic is actually in the knowledge base.
3. **Simple** → `pipelines/simple_pipeline.py`: retrieve once with the raw
   query, check groundedness, then Llama3.2:3b generates the answer --
   from the retrieved passages if grounded, or from general knowledge
   (clearly flagged as such) if not.
4. **Complex** → `pipelines/complex_pipeline.py`: GPT-4o-mini decomposes
   the question into 2-4 sub-questions (a plain prompt, not a tool call),
   each is retrieved independently against the same index, groundedness
   is checked over the pooled results, then GPT-4o-mini answers the same
   way -- grounded synthesis, or general knowledge if nothing relevant
   turned up even after decomposition.
5. Both pipelines return `{answer, sources, grounded, trace}`. The API
   layer adds `classification` and `model_used`. The Streamlit UI shows
   the answer, a classification badge (🟢/🔵), a grounded/ungrounded
   badge, and an expander with sources and a step-by-step trace, so you
   can see exactly how each query was routed, retrieved, and answered.
   Answers are plain prose -- no inline citation markers.

## Notes for extending this

- Swap the embedding model in `backend/config.py`
  (`EMBEDDING_MODEL_NAME` / `EMBEDDING_DIMENSION`) -- just re-ingest
  afterwards, since vectors from different models aren't compatible.
- `CONTEXT_RELEVANCE_THRESHOLD` controls the ungrounded fallback for
  both pipelines. It's checked only after each pipeline's own retrieval
  runs, never before (see the comment in `backend/config.py` for why a
  pre-classification relevance gate is a bad fit here). Watch real
  similarity scores in the trace panel and tune it for your
  documents/embedding model.
- `SIMPLE_TOP_K` / `COMPLEX_TOP_K` tune how many passages each retrieval
  pulls back.

## Quick reference: run commands

Assumes dependencies are already installed and `.env` is filled in (see
[Setup](#setup) above if not). Run these from `learner-rag-app/`, in two
separate terminals.

**0. Make sure Ollama is running with the model pulled** (one-time / as needed):

```bash
ollama pull llama3.2:3b
ollama serve
```

**1. Start the backend** (terminal 1):

```bash
uvicorn backend.main:app --reload --port 8000
```

Verify it's up:

```bash
curl http://localhost:8000/health
# -> {"status":"ok","pinecone_index":"learner-rag-app"}
```

**2. Start the frontend** (terminal 2):

```bash
streamlit run frontend/app.py
```

Open the URL it prints -- default `http://localhost:8501`.

**Stopping either one:** `Ctrl+C` in its terminal. If it was launched in
the background instead, free the port:

```bash
# find the PID listening on the port, then stop it
# backend (8000):
lsof -ti:8000 -sTCP:LISTEN | xargs -r kill      # macOS/Linux
# Windows PowerShell:
Get-NetTCPConnection -LocalPort 8000 -State Listen | Select -Expand OwningProcess | % { Stop-Process -Id $_ -Force }

# frontend (8501): same commands with 8501 instead of 8000
```

Simple (single fact/definition lookup — should route to the local Llama3.2:3b path):

1. What is Prompt Injection (LLM01) according to the OWASP Top 10 for LLM Applications?
2. What is System Prompt Leakage (LLM07)?
3. What are the three root causes of Excessive Agency (LLM06)?
4. What CVE is associated with the "Proof Pudding" attack example under Sensitive Information Disclosure (LLM02)?
5. What did Unbounded Consumption (LLM10) expand upon from the 2023 list?

Complex (comparison / multi-part / requires pulling from multiple sections — should route to GPT-4o-mini with decomposition):

6. Compare Excessive Agency (LLM06) and System Prompt Leakage (LLM07) — what's the root cause of each, and how do their example attack scenarios differ?
7. What are the three stages of the LLM lifecycle where Data and Model Poisoning (LLM04) can occur, and how does the document say each stage should be mitigated?
8. How do the prevention strategies for Prompt Injection (LLM01) differ from the mitigation approach for Vector and Embedding Weaknesses (LLM08) in a multi-tenant RAG setup?
9. Explain how a Prompt Injection attack (LLM01) can lead to Excessive Agency-style damage (LLM06), based on the attack scenarios described in the document.
10. What's new in the 2025 Top 10 compared to the 2023 version, and which categories were expanded or newly added as a result?
