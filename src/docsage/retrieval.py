"""Top-k retrieval with Maximal Marginal Relevance (MMR) re-ranking."""

from __future__ import annotations

from typing import List, Sequence

from docsage.embedders import Embedder
from docsage.store import ScoredChunk, VectorStore, cosine


def mmr(
    query: Sequence[float],
    candidates: Sequence[ScoredChunk],
    k: int,
    lambda_: float = 0.7,
) -> List[ScoredChunk]:
    """Pick ``k`` candidates balancing relevance to the query against redundancy.

    score(c) = lambda * sim(query, c) - (1 - lambda) * max(sim(c, already_selected))

    ``lambda_ = 1`` is plain relevance ranking; lower values push for diversity,
    which matters when overlapping chunks would otherwise fill every slot.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be between 0 and 1")
    pool = list(candidates)
    relevance = [cosine(query, c.embedding) for c in pool]
    selected: List[int] = []
    while pool and len(selected) < min(k, len(pool)):
        best_idx, best_score = -1, float("-inf")
        for i, cand in enumerate(pool):
            if i in selected:
                continue
            redundancy = max(
                (cosine(cand.embedding, pool[j].embedding) for j in selected), default=0.0
            )
            score = lambda_ * relevance[i] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(best_idx)
    return [pool[i] for i in selected]


class Retriever:
    """Embeds the query, over-fetches ``fetch_k`` candidates, then MMR-selects ``k``."""

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        fetch_k: int = 20,
        use_mmr: bool = True,
        mmr_lambda: float = 0.7,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.fetch_k = fetch_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def retrieve(self, query: str, k: int = 4) -> List[ScoredChunk]:
        if k <= 0:
            return []
        qvec = self.embedder.embed([query])[0]
        candidates = self.store.search(qvec, max(k, self.fetch_k))
        if not self.use_mmr:
            return candidates[:k]
        return mmr(qvec, candidates, k, self.mmr_lambda)
