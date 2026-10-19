"""Settings from environment variables and the factory that wires backends together."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from docsage.embedders import Embedder, HashingEmbedder, OpenAIEmbedder
from docsage.llm import LLM, ExtractiveLLM, OpenAIChatLLM
from docsage.service import DocSage
from docsage.store import InMemoryVectorStore, PgVectorStore, VectorStore


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    embedder: str = "hashing"  # hashing | openai
    store: str = "memory"  # memory | pgvector
    llm: str = "extractive"  # extractive | openai
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 512
    chat_model: str = "gpt-4o-mini"
    database_url: str = "postgresql://docsage:docsage@localhost:5432/docsage"
    chunk_size: int = 200
    chunk_overlap: int = 40
    fetch_k: int = 20
    use_mmr: bool = True
    mmr_lambda: float = 0.7

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Settings:
        e = os.environ if env is None else env
        d = cls()
        return cls(
            embedder=e.get("DOCSAGE_EMBEDDER", d.embedder),
            store=e.get("DOCSAGE_STORE", d.store),
            llm=e.get("DOCSAGE_LLM", d.llm),
            openai_api_key=e.get("OPENAI_API_KEY") or None,
            openai_base_url=e.get("OPENAI_BASE_URL", d.openai_base_url),
            embedding_model=e.get("DOCSAGE_EMBEDDING_MODEL", d.embedding_model),
            embedding_dim=int(e.get("DOCSAGE_EMBEDDING_DIM", d.embedding_dim)),
            chat_model=e.get("DOCSAGE_CHAT_MODEL", d.chat_model),
            database_url=e.get("DATABASE_URL", d.database_url),
            chunk_size=int(e.get("DOCSAGE_CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(e.get("DOCSAGE_CHUNK_OVERLAP", d.chunk_overlap)),
            fetch_k=int(e.get("DOCSAGE_FETCH_K", d.fetch_k)),
            use_mmr=_bool(e.get("DOCSAGE_USE_MMR", str(d.use_mmr))),
            mmr_lambda=float(e.get("DOCSAGE_MMR_LAMBDA", d.mmr_lambda)),
        )


def build_embedder(s: Settings) -> Embedder:
    if s.embedder == "hashing":
        return HashingEmbedder(dim=s.embedding_dim)
    if s.embedder == "openai":
        return OpenAIEmbedder(
            model=s.embedding_model,
            dim=s.embedding_dim,
            api_key=s.openai_api_key,
            base_url=s.openai_base_url,
        )
    raise ValueError(f"unknown DOCSAGE_EMBEDDER: {s.embedder!r}")


def build_store(s: Settings) -> VectorStore:
    if s.store == "memory":
        return InMemoryVectorStore()
    if s.store == "pgvector":
        return PgVectorStore(s.database_url, dim=s.embedding_dim)
    raise ValueError(f"unknown DOCSAGE_STORE: {s.store!r}")


def build_llm(s: Settings) -> LLM:
    if s.llm == "extractive":
        return ExtractiveLLM()
    if s.llm == "openai":
        return OpenAIChatLLM(
            model=s.chat_model, api_key=s.openai_api_key, base_url=s.openai_base_url
        )
    raise ValueError(f"unknown DOCSAGE_LLM: {s.llm!r}")


def build_service(s: Optional[Settings] = None) -> DocSage:
    s = s or Settings.from_env()
    return DocSage(
        build_embedder(s),
        build_store(s),
        build_llm(s),
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        fetch_k=s.fetch_k,
        use_mmr=s.use_mmr,
        mmr_lambda=s.mmr_lambda,
    )
