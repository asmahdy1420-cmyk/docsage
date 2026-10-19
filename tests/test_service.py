from __future__ import annotations

import json

import httpx
import pytest

from docsage import DocSage, HashingEmbedder, InMemoryVectorStore, OpenAIChatLLM
from docsage.llm import NO_ANSWER


def test_ask_returns_extractive_answer_with_citation(loaded: DocSage) -> None:
    result = loaded.ask("How long are backups kept?")
    assert "35 days" in result.answer
    assert result.citations, "answer should cite at least one source"
    assert result.citations[0].source == "infra.md"
    assert f"[{result.citations[0].n}]" in result.answer


def test_unanswerable_question_has_no_citations(loaded: DocSage) -> None:
    result = loaded.ask("quantum chromodynamics lattice")
    assert result.answer == NO_ANSWER
    assert result.citations == []


def test_reingesting_same_source_replaces_chunks(service: DocSage) -> None:
    first = service.ingest("version one " * 30, source="notes.md")
    second = service.ingest("version two " * 50, source="notes.md")
    assert first.doc_id == second.doc_id
    assert second.replaced == first.chunks
    assert service.store.count() == second.chunks
    assert all("two" in h.chunk.text for h in service.search("version", k=10))


def test_empty_document_is_rejected(service: DocSage) -> None:
    with pytest.raises(ValueError):
        service.ingest("   ")


def test_openai_chat_llm_sends_numbered_sources_and_citations_are_mapped() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["prompt"] = body["messages"][1]["content"]
        content = "Backups are kept for 35 days [1]. Unrelated [9]."
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = OpenAIChatLLM(transport=httpx.MockTransport(handler))
    svc = DocSage(HashingEmbedder(dim=128), InMemoryVectorStore(), llm)
    svc.ingest("Backups are kept for 35 days.", source="infra.md")
    result = svc.ask("How long are backups kept?", k=1)
    assert "[1] (infra.md)" in seen["prompt"]
    # [9] does not refer to a retrieved chunk, so it is dropped.
    assert [c.n for c in result.citations] == [1]
    assert result.citations[0].source == "infra.md"
