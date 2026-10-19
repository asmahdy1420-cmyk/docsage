"""Integration test for the pgvector store.

Runs only when DOCSAGE_TEST_DATABASE_URL points at a PostgreSQL with the
pgvector extension available (CI provides one as a service container).
"""

from __future__ import annotations

import os
import uuid

import pytest

from docsage.chunking import Chunk
from docsage.store import _from_pgvector, _to_pgvector

DSN = os.environ.get("DOCSAGE_TEST_DATABASE_URL")


def test_vector_literal_round_trip() -> None:
    vec = [0.25, -1.0, 3e-05]
    assert _from_pgvector(_to_pgvector(vec)) == tuple(vec)


@pytest.mark.skipif(not DSN, reason="DOCSAGE_TEST_DATABASE_URL not set")
def test_pgvector_store_round_trip() -> None:
    from docsage.store import PgVectorStore

    table = f"docsage_test_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(DSN or "", dim=2, table=table)
    try:
        chunks = [Chunk("a", 0, "x", 0, 1, "a.md"), Chunk("b", 0, "y", 0, 1, "b.md")]
        store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
        hits = store.search([0.9, 0.1], k=1)
        assert hits[0].chunk.id == "a:0"
        assert hits[0].chunk.source == "a.md"
        assert hits[0].embedding == (1.0, 0.0)
        assert store.count() == 2
        assert store.delete_document("a") == 1
        assert [d.doc_id for d in store.list_documents()] == ["b"]
    finally:
        with store._connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
