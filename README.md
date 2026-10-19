# docsage

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white) ![RAG](https://img.shields.io/badge/RAG-cited_answers-FF6F00) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Ask questions about your own text and markdown documents and get answers that cite their sources.

docsage is a small, readable retrieval-augmented generation (RAG) service built on FastAPI. It ingests documents, splits them into overlapping chunks, embeds them, stores the vectors in PostgreSQL with pgvector, retrieves with top-k plus MMR re-ranking, and answers through any OpenAI-compatible model, with an offline extractive fallback. A retrieval evaluation harness runs in CI and fails the build if retrieval quality drops.

## Why

Most RAG examples are either a notebook or a framework with ten layers of abstraction. docsage sits in between: every stage is a short module behind a small `Protocol`, so you can read the whole pipeline in an afternoon, swap any backend, and test it without a network connection or an API key.

## Architecture

```mermaid
flowchart LR
    subgraph Ingest
        F["POST /documents<br>text or .md/.txt upload"] --> C["Chunker<br>token windows + overlap"]
        C --> E1["Embedder"]
        E1 --> S[("VectorStore<br>pgvector or in-memory")]
    end
    subgraph Ask
        Q["POST /ask"] --> E2["Embedder"]
        E2 --> R["Retriever<br>fetch_k candidates"]
        S --> R
        R --> M["MMR re-rank<br>top k"]
        M --> L["LLM<br>OpenAI-compatible or extractive"]
        L --> A["Answer + citations"]
    end
```

| Stage | Protocol | Implementations |
|---|---|---|
| Embeddings | `Embedder` | `OpenAIEmbedder` (any `/embeddings` endpoint), `HashingEmbedder` (deterministic, offline) |
| Storage | `VectorStore` | `PgVectorStore` (HNSW, cosine), `InMemoryVectorStore` (exact, tests) |
| Generation | `LLM` | `OpenAIChatLLM` (any `/chat/completions` endpoint), `ExtractiveLLM` (quotes best sentences) |

Code map: `src/docsage/` contains `chunking.py`, `embedders.py`, `store.py`, `retrieval.py`, `llm.py`, `service.py` (orchestration), `config.py` (env settings and wiring), `api.py` (HTTP) and `evaluation.py` (metrics and CLI).

## Quick start

With Docker (app + PostgreSQL/pgvector, offline backends by default):

```bash
docker compose up --build
```

Locally (in-memory store, no database needed):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
uvicorn --factory docsage.api:app_factory --reload
```

To use a real model, set these in the environment or a `.env` file:

```bash
DOCSAGE_EMBEDDER=openai
DOCSAGE_LLM=openai
DOCSAGE_EMBEDDING_DIM=1536        # must match the embedding model
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # or any compatible server
```

Other settings: `DOCSAGE_STORE` (`memory` | `pgvector`), `DATABASE_URL`, `DOCSAGE_EMBEDDING_MODEL`, `DOCSAGE_CHAT_MODEL`, `DOCSAGE_CHUNK_SIZE`, `DOCSAGE_CHUNK_OVERLAP`, `DOCSAGE_FETCH_K`, `DOCSAGE_USE_MMR`, `DOCSAGE_MMR_LAMBDA`.

## API

Interactive docs are served at `http://localhost:8000/docs`.

```bash
# Ingest raw text
curl -s -X POST localhost:8000/documents \
  -H 'content-type: application/json' \
  -d '{"text": "Backups are kept for 35 days. Access logs are deleted after 90 days.", "source": "ops.md"}'
# {"doc_id":"e99f5c086a630194","source":"ops.md","chunks":1,"replaced":0}

# Upload a file (.md, .markdown or .txt, UTF-8, up to 2 MiB)
curl -s -X POST localhost:8000/documents/upload -F file=@evals/corpus/refunds.md

# Ask a question
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "How long are backups kept?", "k": 4}'
# {"question":"How long are backups kept?",
#  "answer":"Backups are kept for 35 days. [1]",
#  "citations":[{"n":1,"chunk_id":"e99f5c086a630194:0","doc_id":"e99f5c086a630194",
#                "source":"ops.md","score":0.3491,"snippet":"Backups are kept for 35 days. ..."}]}

# Retrieval only, no generation
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' -d '{"question": "refund approval limit", "k": 3}'

# List and delete documents
curl -s localhost:8000/documents
curl -s -X DELETE localhost:8000/documents/e99f5c086a630194

curl -s localhost:8000/healthz
```

## Design decisions

- **Protocols, not base classes.** `Embedder`, `VectorStore` and `LLM` are `typing.Protocol`s. A new backend is any class with the right methods; nothing has to inherit from docsage.
- **Token-ish chunking with character offsets.** Tokens are runs of non-whitespace, which tracks model tokens closely enough for sizing without a tokenizer dependency. Each chunk keeps its `start`/`end` offsets and the original formatting.
- **Over-fetch, then MMR.** The store returns `fetch_k` candidates by cosine similarity; MMR then picks `k` that are relevant but not redundant, so overlapping chunks from one paragraph do not fill every slot. `DOCSAGE_MMR_LAMBDA=1` turns it back into plain ranking.
- **Idempotent ingest.** The document id is derived from the source name (or the content when there is no name). Re-ingesting a file replaces its chunks. Embeddings are computed before the old chunks are deleted, so a failed embedding call leaves the previous version in place.
- **Citations are verified.** The model is asked to cite `[n]`; the service keeps only numbers that map to a retrieved chunk and returns that chunk's source, id, score and snippet. A "not found" answer returns no citations.
- **Offline by default.** The hashing embedder and extractive LLM make the whole stack run, and every test pass, with no network and no key. They also act as a lexical baseline when comparing real models in the eval.
- **pgvector without extra adapters.** Vectors are sent as text literals cast to `vector`; the table has an HNSW index with `vector_cosine_ops` and a b-tree on `doc_id` for replace/delete.

What it deliberately does not do: PDF/HTML parsing, auth, multi-tenancy, background ingestion queues or streaming responses. Each is a reasonable next step, and none is needed to show the retrieval pipeline working.

## Tests

```bash
pytest
```

The suite uses the offline backends and `httpx.MockTransport` for the OpenAI-compatible clients, so it needs no network. The pgvector integration test runs when `DOCSAGE_TEST_DATABASE_URL` is set (CI provides a pgvector service container):

```bash
docker compose up -d db
DOCSAGE_TEST_DATABASE_URL=postgresql://docsage:docsage@localhost:5432/docsage pytest
```

Lint and format:

```bash
ruff check . && ruff format --check .
```

## Evaluation

`evals/` holds a small corpus (`evals/corpus/*.md`) and a JSONL dataset of questions with their expected source documents:

```json
{"question": "How long are access logs kept?", "expected_sources": ["data-retention.md"]}
```

The harness ingests the corpus, retrieves for each question and reports **hit-rate@k** (share of questions whose expected source appears in the top k) and **MRR** (mean of 1 / rank of the first relevant result):

```bash
python evals/run_evals.py -k 3 --min-hit-rate 0.9 --min-mrr 0.8
# documents: 5  chunks: 10  questions: 18
# hit_rate@3: 1.000
# mrr:        0.972
```

It exits non-zero when a metric is below its threshold, and CI runs it after the tests, so a change to chunking, embedding or ranking that hurts retrieval fails the build. Add `--json` for machine-readable output.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Ahmed Mahdy.
