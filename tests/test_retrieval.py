from __future__ import annotations

from docsage.chunking import Chunk
from docsage.retrieval import mmr
from docsage.store import InMemoryVectorStore, ScoredChunk


def _sc(name: str, vec: tuple) -> ScoredChunk:
    return ScoredChunk(Chunk(name, 0, name, 0, len(name)), 0.0, vec)


def test_in_memory_store_ranks_by_cosine_and_deletes_by_document() -> None:
    store = InMemoryVectorStore()
    chunks = [Chunk("a", 0, "x", 0, 1), Chunk("a", 1, "y", 1, 2), Chunk("b", 0, "z", 0, 1)]
    store.add(chunks, [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]])
    hits = store.search([1.0, 0.1], k=2)
    assert [h.chunk.id for h in hits] == ["a:0", "a:1"]
    assert hits[0].score > hits[1].score
    assert store.delete_document("a") == 2
    assert store.count() == 1
    assert [d.doc_id for d in store.list_documents()] == ["b"]


def test_mmr_prefers_diverse_results_over_near_duplicates() -> None:
    query = [1.0, 1.0, 0.0]
    best = _sc("best", (1.0, 0.9, 0.0))
    dup = _sc("dup", (1.0, 0.9, 0.01))
    other = _sc("other", (0.6, 1.0, 0.6))
    plain = mmr(query, [best, dup, other], k=2, lambda_=1.0)
    diverse = mmr(query, [best, dup, other], k=2, lambda_=0.5)
    assert [c.chunk.doc_id for c in plain] == ["best", "dup"]
    assert [c.chunk.doc_id for c in diverse] == ["best", "other"]


def test_mmr_handles_k_larger_than_pool() -> None:
    out = mmr([1.0, 0.0], [_sc("only", (1.0, 0.0))], k=5)
    assert len(out) == 1
