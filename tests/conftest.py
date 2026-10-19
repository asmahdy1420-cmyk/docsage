from __future__ import annotations

import pytest

from docsage import DocSage, ExtractiveLLM, HashingEmbedder, InMemoryVectorStore

PAYMENTS = (
    "Refunds are issued to the original payment method within ten business days. "
    "Support agents may approve refunds up to 200 euros.\n\n"
    "Chargebacks are handled by the finance team, not by support."
)
INFRA = (
    "The database is backed up every night. Backups are kept for 35 days and "
    "restores are tested every quarter.\n\nAccess logs are deleted after 90 days."
)


@pytest.fixture
def service() -> DocSage:
    return DocSage(
        HashingEmbedder(dim=256),
        InMemoryVectorStore(),
        ExtractiveLLM(),
        chunk_size=40,
        chunk_overlap=8,
    )


@pytest.fixture
def loaded(service: DocSage) -> DocSage:
    service.ingest(PAYMENTS, source="payments.md")
    service.ingest(INFRA, source="infra.md")
    return service
