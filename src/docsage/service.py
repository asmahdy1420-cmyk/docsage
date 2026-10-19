"""The DocSage service: ingest documents, retrieve, answer with citations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

from docsage.chunking import chunk_text
from docsage.embedders import Embedder
from docsage.llm import LLM, NO_ANSWER
from docsage.retrieval import Retriever
from docsage.store import ScoredChunk, VectorStore

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    source: str
    score: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    source: str
    chunks: int
    replaced: int


def make_doc_id(text: str, source: str = "") -> str:
    """Stable id: derived from the source name when given, else from the content.

    Re-ingesting the same source therefore replaces its chunks instead of
    duplicating them.
    """
    basis = f"source:{source}" if source else f"content:{text}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _snippet(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


class DocSage:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        *,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
        fetch_k: int = 20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever = Retriever(
            embedder, store, fetch_k=fetch_k, use_mmr=use_mmr, mmr_lambda=mmr_lambda
        )

    def ingest(self, text: str, source: str = "", doc_id: Optional[str] = None) -> IngestResult:
        if not text.strip():
            raise ValueError("document is empty")
        doc_id = doc_id or make_doc_id(text, source)
        chunks = chunk_text(
            text,
            doc_id,
            source=source,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        # Embed before deleting so a failing embedding call leaves the old version intact.
        embeddings = self.embedder.embed([c.text for c in chunks])
        replaced = self.store.delete_document(doc_id)
        self.store.add(chunks, embeddings)
        return IngestResult(doc_id=doc_id, source=source, chunks=len(chunks), replaced=replaced)

    def search(self, query: str, k: int = 4) -> List[ScoredChunk]:
        return self.retriever.retrieve(query, k)

    def ask(self, question: str, k: int = 4) -> Answer:
        contexts = self.search(question, k)
        text = self.llm.answer(question, contexts)
        return Answer(question=question, answer=text, citations=self._citations(text, contexts))

    @staticmethod
    def _citations(answer: str, contexts: List[ScoredChunk]) -> List[Citation]:
        if answer.strip() == NO_ANSWER:
            return []
        cited = []
        for m in _CITATION.finditer(answer):
            n = int(m.group(1))
            if 1 <= n <= len(contexts) and n not in cited:
                cited.append(n)
        return [
            Citation(
                n=n,
                chunk_id=contexts[n - 1].chunk.id,
                doc_id=contexts[n - 1].chunk.doc_id,
                source=contexts[n - 1].chunk.source,
                score=round(contexts[n - 1].score, 4),
                snippet=_snippet(contexts[n - 1].chunk.text),
            )
            for n in sorted(cited)
        ]
