"""Vector store abstraction with an in-memory and a PostgreSQL + pgvector backend."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

from docsage.chunking import Chunk

Vector = List[float]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    embedding: Tuple[float, ...]


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    source: str
    chunks: int


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Vector]) -> None: ...

    def search(self, query: Vector, k: int) -> List[ScoredChunk]: ...

    def delete_document(self, doc_id: str) -> int: ...

    def list_documents(self) -> List[DocumentInfo]: ...

    def count(self) -> int: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _check_lengths(chunks: Sequence[Chunk], embeddings: Sequence[Vector]) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")


class InMemoryVectorStore:
    """Brute-force cosine search. Exact, thread-safe, and fine up to a few thousand chunks."""

    def __init__(self) -> None:
        self._rows: Dict[str, Tuple[Chunk, Tuple[float, ...]]] = {}
        self._lock = threading.Lock()

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Vector]) -> None:
        _check_lengths(chunks, embeddings)
        with self._lock:
            for chunk, emb in zip(chunks, embeddings):
                self._rows[chunk.id] = (chunk, tuple(emb))

    def search(self, query: Vector, k: int) -> List[ScoredChunk]:
        with self._lock:
            rows = list(self._rows.values())
        scored = [ScoredChunk(c, cosine(query, e), e) for c, e in rows]
        scored.sort(key=lambda s: (-s.score, s.chunk.id))
        return scored[:k]

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            ids = [cid for cid, (c, _) in self._rows.items() if c.doc_id == doc_id]
            for cid in ids:
                del self._rows[cid]
        return len(ids)

    def list_documents(self) -> List[DocumentInfo]:
        with self._lock:
            chunks = [c for c, _ in self._rows.values()]
        docs: Dict[str, DocumentInfo] = {}
        for c in chunks:
            prev = docs.get(c.doc_id)
            docs[c.doc_id] = DocumentInfo(c.doc_id, c.source, (prev.chunks if prev else 0) + 1)
        return sorted(docs.values(), key=lambda d: d.doc_id)

    def count(self) -> int:
        with self._lock:
            return len(self._rows)


def _to_pgvector(vec: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def _from_pgvector(text: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in text.strip("[]").split(",") if x)


class PgVectorStore:
    """PostgreSQL + pgvector store.

    One table holds chunk text and its embedding; an HNSW index with
    ``vector_cosine_ops`` serves the ``<=>`` (cosine distance) operator. Vectors
    are passed as text literals and cast to ``vector`` in SQL, so no extra
    adapter package is needed beyond psycopg 3.
    """

    def __init__(self, dsn: str, dim: int, table: str = "docsage_chunks") -> None:
        if not table.replace("_", "").isalnum():
            raise ValueError("table name must be alphanumeric/underscore")
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "install the 'postgres' extra: pip install docsage[postgres]"
            ) from exc
        self._psycopg = psycopg
        self.dsn = dsn
        self.dim = dim
        self.table = table
        self.ensure_schema()

    def _connect(self) -> Any:
        return self._psycopg.connect(self.dsn, autocommit=True)

    def ensure_schema(self) -> None:
        t = self.table
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {t} (
                    id         text PRIMARY KEY,
                    doc_id     text NOT NULL,
                    idx        integer NOT NULL,
                    source     text NOT NULL DEFAULT '',
                    body       text NOT NULL,
                    start_char integer NOT NULL,
                    end_char   integer NOT NULL,
                    embedding  vector({int(self.dim)}) NOT NULL
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_doc_idx ON {t} (doc_id)")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {t}_hnsw_idx ON {t} "
                "USING hnsw (embedding vector_cosine_ops)"
            )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Vector]) -> None:
        _check_lengths(chunks, embeddings)
        rows = [
            (c.id, c.doc_id, c.index, c.source, c.text, c.start, c.end, _to_pgvector(e))
            for c, e in zip(chunks, embeddings)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self.table}
                    (id, doc_id, idx, source, body, start_char, end_char, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (id) DO UPDATE SET
                    source = EXCLUDED.source, body = EXCLUDED.body,
                    start_char = EXCLUDED.start_char, end_char = EXCLUDED.end_char,
                    embedding = EXCLUDED.embedding
                """,
                rows,
            )

    def search(self, query: Vector, k: int) -> List[ScoredChunk]:
        q = _to_pgvector(query)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT doc_id, idx, body, start_char, end_char, source,
                       1 - (embedding <=> %s::vector) AS score, embedding::text
                FROM {self.table}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q, q, k),
            ).fetchall()
        return [
            ScoredChunk(
                Chunk(doc_id=r[0], index=r[1], text=r[2], start=r[3], end=r[4], source=r[5]),
                float(r[6]),
                _from_pgvector(r[7]),
            )
            for r in rows
        ]

    def delete_document(self, doc_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM {self.table} WHERE doc_id = %s", (doc_id,))
            return cur.rowcount

    def list_documents(self) -> List[DocumentInfo]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT doc_id, max(source), count(*) FROM {self.table} "
                "GROUP BY doc_id ORDER BY doc_id"
            ).fetchall()
        return [DocumentInfo(r[0], r[1], int(r[2])) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row: Optional[Tuple[int]] = conn.execute(
                f"SELECT count(*) FROM {self.table}"
            ).fetchone()
        return int(row[0]) if row else 0
