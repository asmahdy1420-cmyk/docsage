from __future__ import annotations

import pytest

from docsage.config import Settings, build_service
from docsage.embedders import HashingEmbedder
from docsage.store import InMemoryVectorStore


def test_settings_from_env_and_default_wiring() -> None:
    s = Settings.from_env(
        {"DOCSAGE_CHUNK_SIZE": "64", "DOCSAGE_USE_MMR": "false", "DOCSAGE_EMBEDDING_DIM": "128"}
    )
    assert (s.chunk_size, s.use_mmr, s.embedding_dim) == (64, False, 128)
    svc = build_service(s)
    assert isinstance(svc.embedder, HashingEmbedder)
    assert svc.embedder.dim == 128
    assert isinstance(svc.store, InMemoryVectorStore)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="DOCSAGE_STORE"):
        build_service(Settings(store="sqlite"))
