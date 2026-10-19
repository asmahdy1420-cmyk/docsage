"""Retrieval evaluation: hit-rate@k and MRR over a JSONL dataset.

Dataset lines look like::

    {"question": "How long are access logs kept?", "expected_sources": ["data-retention.md"]}

A question is a hit if any of its expected sources appears in the top-k
retrieved chunks; its reciprocal rank is 1 / position of the first such chunk.
The CLI exits non-zero when a metric falls below its threshold, which lets CI
block a change that makes retrieval worse.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from docsage.embedders import HashingEmbedder
from docsage.llm import ExtractiveLLM
from docsage.service import DocSage
from docsage.store import InMemoryVectorStore


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_sources: List[str]


@dataclass(frozen=True)
class EvalReport:
    k: int
    n: int
    hit_rate: float
    mrr: float
    misses: List[str]


def first_relevant_rank(retrieved: Sequence[str], expected: Sequence[str]) -> Optional[int]:
    """1-based rank of the first retrieved source in ``expected``, or None."""
    wanted = set(expected)
    for rank, source in enumerate(retrieved, start=1):
        if source in wanted:
            return rank
    return None


def hit_rate_at_k(ranks: Sequence[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def load_dataset(path: Path) -> List[EvalCase]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "question" not in row or not row.get("expected_sources"):
            raise ValueError(f"{path}:{line_no}: needs 'question' and 'expected_sources'")
        cases.append(EvalCase(row["question"], list(row["expected_sources"])))
    return cases


def ingest_corpus(service: DocSage, corpus_dir: Path) -> int:
    files = sorted(p for p in corpus_dir.iterdir() if p.suffix in {".md", ".txt"})
    for p in files:
        service.ingest(p.read_text(encoding="utf-8"), source=p.name)
    return len(files)


def evaluate(service: DocSage, cases: Sequence[EvalCase], k: int) -> EvalReport:
    ranks: List[Optional[int]] = []
    misses = []
    for case in cases:
        hits = service.search(case.question, k)
        rank = first_relevant_rank([h.chunk.source for h in hits], case.expected_sources)
        ranks.append(rank)
        if rank is None:
            misses.append(case.question)
    return EvalReport(
        k=k,
        n=len(cases),
        hit_rate=hit_rate_at_k(ranks, k),
        mrr=mean_reciprocal_rank(ranks),
        misses=misses,
    )


def default_service() -> DocSage:
    """Offline configuration: hashing embedder + in-memory store. No network."""
    return DocSage(
        HashingEmbedder(dim=512),
        InMemoryVectorStore(),
        ExtractiveLLM(),
        chunk_size=120,
        chunk_overlap=24,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    here = Path.cwd() / "evals"
    parser = argparse.ArgumentParser(description="Evaluate docsage retrieval.")
    parser.add_argument("--dataset", type=Path, default=here / "dataset.jsonl")
    parser.add_argument("--corpus", type=Path, default=here / "corpus")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    service = default_service()
    n_docs = ingest_corpus(service, args.corpus)
    report = evaluate(service, load_dataset(args.dataset), args.k)

    if args.json:
        print(json.dumps(report.__dict__, indent=2))
    else:
        print(f"documents: {n_docs}  chunks: {service.store.count()}  questions: {report.n}")
        print(f"hit_rate@{report.k}: {report.hit_rate:.3f}")
        print(f"mrr:        {report.mrr:.3f}")
        for q in report.misses:
            print(f"  miss: {q}")

    failed = []
    if report.hit_rate < args.min_hit_rate:
        failed.append(f"hit_rate@{report.k} {report.hit_rate:.3f} < {args.min_hit_rate}")
    if report.mrr < args.min_mrr:
        failed.append(f"mrr {report.mrr:.3f} < {args.min_mrr}")
    for msg in failed:
        print(f"FAIL: {msg}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
