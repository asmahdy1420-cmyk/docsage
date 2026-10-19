"""Answer generation behind a small protocol.

Both implementations receive the question plus numbered context chunks and must
cite them as ``[1]``, ``[2]``... so the API can map citations back to sources.
"""

from __future__ import annotations

import re
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import httpx

from docsage.embedders import tokenize
from docsage.store import ScoredChunk

NO_ANSWER = "I could not find this in the indexed documents."

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources provided. "
    "Cite every claim with the source number in square brackets, e.g. [2]. "
    "If the sources do not contain the answer, reply exactly: " + NO_ANSWER
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@runtime_checkable
class LLM(Protocol):
    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> str: ...


def build_prompt(question: str, contexts: Sequence[ScoredChunk]) -> str:
    blocks = []
    for n, sc in enumerate(contexts, start=1):
        label = sc.chunk.source or sc.chunk.doc_id
        blocks.append(f"[{n}] ({label})\n{sc.chunk.text}")
    sources = "\n\n".join(blocks)
    return f"Sources:\n\n{sources}\n\nQuestion: {question}\nAnswer:"


class ExtractiveLLM:
    """No-model fallback: quotes the sentences that best overlap the question.

    Sentences are scored by the share of question terms they contain, so the
    answer is always verbatim text from the sources, with its citation. Useful
    offline, in tests, and as a sanity baseline next to a real model.
    """

    def __init__(self, max_sentences: int = 2, min_overlap: float = 0.2) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> str:
        q_terms = set(tokenize(question))
        if not q_terms or not contexts:
            return NO_ANSWER
        scored: List[Tuple[float, int, int, str]] = []
        for n, sc in enumerate(contexts, start=1):
            for pos, sentence in enumerate(_SENTENCE.split(sc.chunk.text)):
                sentence = " ".join(sentence.split())
                terms = set(tokenize(sentence))
                if not terms:
                    continue
                overlap = len(q_terms & terms) / len(q_terms)
                if overlap >= self.min_overlap:
                    # Prefer higher overlap, then higher-ranked sources, then earlier text.
                    scored.append((overlap, -n, -pos, sentence))
        if not scored:
            return NO_ANSWER
        scored.sort(reverse=True)
        picked: List[str] = []
        seen = set()
        for _, neg_n, _, sentence in scored:
            if sentence in seen:
                continue
            seen.add(sentence)
            picked.append(f"{sentence} [{-neg_n}]")
            if len(picked) == self.max_sentences:
                break
        return " ".join(picked)


class OpenAIChatLLM:
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def answer(self, question: str, contexts: Sequence[ScoredChunk]) -> str:
        if not contexts:
            return NO_ANSWER
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, contexts)},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        self._client.close()
