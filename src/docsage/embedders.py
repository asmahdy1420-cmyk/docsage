"""Embedding backends behind one small protocol."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import httpx

Vector = List[float]

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORD_TEXT = """
    a an and are as at be by can do does for from has have how i if in into is it its
    me my of on or our so that the their them then there these this to was we what
    when where which who why will with you your
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into fixed-size vectors. Implementations must be deterministic per text."""

    dim: int

    def embed(self, texts: Sequence[str]) -> List[Vector]: ...


def normalize(vec: Sequence[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return list(vec)
    return [x / norm for x in vec]


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens without stopwords, with a light plural/verb suffix strip."""
    out = []
    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS:
            continue
        out.append(_stem(word))
    return out


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class HashingEmbedder:
    """Feature-hashing embedder: no model, no network, fully deterministic.

    Each unigram and bigram is hashed into one of ``dim`` buckets with a signed
    weight (the signed hashing trick keeps collisions from only adding up).
    Sublinear term frequency keeps one repeated word from dominating. Good
    enough for tests, offline demos and as a lexical baseline in evals.
    """

    def __init__(self, dim: int = 512, use_bigrams: bool = True) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.use_bigrams = use_bigrams

    def _features(self, text: str) -> List[str]:
        tokens = tokenize(text)
        feats = list(tokens)
        if self.use_bigrams:
            feats.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
        return feats

    def _bucket(self, feature: str) -> Tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dim, sign

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        vectors = []
        for text in texts:
            counts: Dict[str, int] = {}
            for feat in self._features(text):
                counts[feat] = counts.get(feat, 0) + 1
            vec = [0.0] * self.dim
            for feat, count in counts.items():
                idx, sign = self._bucket(feat)
                weight = 1.0 + math.log(count)
                if "_" in feat:
                    weight *= 0.5
                vec[idx] += sign * weight
            vectors.append(normalize(vec))
        return vectors


class OpenAIEmbedder:
    """Client for any OpenAI-compatible ``/embeddings`` endpoint.

    Works with OpenAI itself and with self-hosted servers that implement the same
    API (vLLM, Ollama's OpenAI mode, LiteLLM, ...). Requests are batched and the
    response order is restored from the ``index`` field.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        timeout: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def embed(self, texts: Sequence[str]) -> List[Vector]:
        out: List[Vector] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            resp = self._client.post("/embeddings", json={"model": self.model, "input": batch})
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda item: item["index"])
            for item in data:
                vec = item["embedding"]
                if len(vec) != self.dim:
                    raise ValueError(
                        f"embedding has {len(vec)} dimensions, expected {self.dim}; "
                        "set DOCSAGE_EMBEDDING_DIM to match the model"
                    )
                out.append(vec)
        return out

    def close(self) -> None:
        self._client.close()
