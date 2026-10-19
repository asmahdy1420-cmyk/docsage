"""docsage: document Q&A with retrieval, re-ranking and cited answers."""

from __future__ import annotations

from docsage.chunking import Chunk, chunk_text
from docsage.embedders import Embedder, HashingEmbedder, OpenAIEmbedder
from docsage.llm import LLM, ExtractiveLLM, OpenAIChatLLM
from docsage.retrieval import Retriever, mmr
from docsage.service import Answer, Citation, DocSage
from docsage.store import InMemoryVectorStore, PgVectorStore, ScoredChunk, VectorStore

__all__ = [
    "LLM",
    "Answer",
    "Chunk",
    "Citation",
    "DocSage",
    "Embedder",
    "ExtractiveLLM",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "OpenAIChatLLM",
    "OpenAIEmbedder",
    "PgVectorStore",
    "Retriever",
    "ScoredChunk",
    "VectorStore",
    "chunk_text",
    "mmr",
]

__version__ = "0.1.0"
