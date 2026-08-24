# Learner RAG Chatbot -- Sample Knowledge Base

This is a placeholder document so `ingest_documents.py` has something to
embed and upsert on a first run. Replace everything in `sample_docs/`
with your own `.txt`, `.md`, or `.pdf` files, then re-run the ingestion
script.

## Architecture recap

The app routes every incoming question through a classifier (Llama3.2:3b)
that labels it "simple" or "complex".

- Simple questions go to a local Llama3.2:3b model that is given a
  `search_knowledge_base` tool. The model decides for itself whether and
  how many times to call it before answering.
- Complex questions go to GPT-4o-mini. The question is first decomposed
  into a small set of focused sub-questions, each sub-question is
  retrieved against the same Pinecone index, and GPT-4o-mini synthesizes
  a final answer from everything retrieved.

Both paths call the exact same tool implementation and read from the
exact same Pinecone index -- they only differ in who decides retrieval
strategy (the model itself vs. orchestration code) and which model
answers.

## Refund policy example

Refunds are available within 30 days of purchase for unused digital
credits. Physical goods can be returned within 14 days in their original
packaging. Refunds are issued to the original payment method within 5-7
business days of approval.

## Warranty policy example

Hardware products carry a 1-year limited warranty covering manufacturing
defects. The warranty does not cover accidental damage, unauthorized
modification, or normal wear and tear. Warranty claims require a proof
of purchase.
