from __future__ import annotations

import json
import math

import httpx
import pytest

from docsage.embedders import Embedder, HashingEmbedder, OpenAIEmbedder
from docsage.store import cosine


def test_hashing_embedder_is_deterministic_normalised_and_semantic_enough() -> None:
    emb = HashingEmbedder(dim=256)
    a, a2, b, c = emb.embed(
        [
            "database backups are kept for 35 days",
            "database backups are kept for 35 days",
            "how long do we keep the database backup?",
            "refunds above 200 euros need approval",
        ]
    )
    assert a == a2
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)
    assert cosine(a, b) > cosine(a, c)
    assert isinstance(emb, Embedder)


def test_openai_embedder_batches_and_restores_order() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body["input"])
        assert request.headers["Authorization"] == "Bearer sk-test"
        # Return items in reverse order to check that `index` is respected.
        data = [
            {"index": i, "embedding": [float(len(t)), 0.0, 1.0]}
            for i, t in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    emb = OpenAIEmbedder(
        dim=3, api_key="sk-test", batch_size=2, transport=httpx.MockTransport(handler)
    )
    out = emb.embed(["a", "bb", "ccc"])
    assert calls == [["a", "bb"], ["ccc"]]
    assert [v[0] for v in out] == [1.0, 2.0, 3.0]


def test_openai_embedder_rejects_wrong_dimension() -> None:
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})
    )
    emb = OpenAIEmbedder(dim=3, transport=transport)
    with pytest.raises(ValueError, match="expected 3"):
        emb.embed(["x"])
