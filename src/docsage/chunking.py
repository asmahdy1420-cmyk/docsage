"""Token-ish chunking with overlap.

A "token" here is a run of non-whitespace characters. That is close enough to
model tokens for sizing purposes (roughly 0.75 words per token for English) and
needs no tokenizer dependency. Chunks keep the original text between the first
and last token, so markdown formatting and line breaks survive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    index: int
    text: str
    start: int
    end: int
    source: str = ""

    @property
    def id(self) -> str:
        return f"{self.doc_id}:{self.index}"


def chunk_text(
    text: str,
    doc_id: str,
    *,
    source: str = "",
    chunk_size: int = 200,
    overlap: int = 40,
) -> List[Chunk]:
    """Split ``text`` into windows of ``chunk_size`` tokens sharing ``overlap`` tokens.

    Iteration stops as soon as a window reaches the end of the text, so the
    final chunk is never just a copy of the previous chunk's overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    spans = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    if not spans:
        return []

    step = chunk_size - overlap
    chunks: List[Chunk] = []
    start_tok = 0
    while start_tok < len(spans):
        end_tok = min(start_tok + chunk_size, len(spans))
        start_char = spans[start_tok][0]
        end_char = spans[end_tok - 1][1]
        chunks.append(
            Chunk(
                doc_id=doc_id,
                index=len(chunks),
                text=text[start_char:end_char],
                start=start_char,
                end=end_char,
                source=source,
            )
        )
        if end_tok == len(spans):
            break
        start_tok += step
    return chunks
