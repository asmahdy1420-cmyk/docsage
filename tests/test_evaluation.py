from __future__ import annotations

from pathlib import Path

from docsage.evaluation import (
    first_relevant_rank,
    hit_rate_at_k,
    main,
    mean_reciprocal_rank,
)

EVALS = Path(__file__).resolve().parent.parent / "evals"


def test_metrics() -> None:
    assert first_relevant_rank(["a", "b", "c"], ["c"]) == 3
    assert first_relevant_rank(["a"], ["z"]) is None
    ranks = [1, 2, None, 4]
    assert hit_rate_at_k(ranks, k=2) == 0.5
    assert mean_reciprocal_rank(ranks) == (1 + 0.5 + 0.25) / 4
    assert hit_rate_at_k([], 3) == 0.0


def test_eval_cli_gates_on_thresholds() -> None:
    base = ["--dataset", str(EVALS / "dataset.jsonl"), "--corpus", str(EVALS / "corpus"), "-k", "3"]
    assert main([*base, "--min-hit-rate", "0.9", "--min-mrr", "0.8"]) == 0
    assert main([*base, "--min-mrr", "1.01"]) == 1
