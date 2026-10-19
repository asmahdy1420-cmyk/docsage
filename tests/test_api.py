from __future__ import annotations

from fastapi.testclient import TestClient

from docsage import DocSage
from docsage.api import create_app


def test_ingest_search_ask_and_delete_over_http(service: DocSage) -> None:
    client = TestClient(create_app(service))
    r = client.post(
        "/documents",
        json={"text": "Access logs are deleted after 90 days.", "source": "logs.md"},
    )
    assert r.status_code == 201
    doc_id = r.json()["doc_id"]

    r = client.post(
        "/documents/upload",
        files={
            "file": ("refunds.md", b"Refunds are paid within ten business days.", "text/markdown")
        },
    )
    assert r.status_code == 201
    assert {d["source"] for d in client.get("/documents").json()} == {"logs.md", "refunds.md"}

    hits = client.post("/search", json={"question": "access logs retention", "k": 1}).json()
    assert hits[0]["source"] == "logs.md"

    ask = client.post("/ask", json={"question": "When are access logs deleted?"}).json()
    assert "90 days" in ask["answer"]
    assert ask["citations"][0]["doc_id"] == doc_id

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert client.get("/healthz").json()["status"] == "ok"


def test_validation_and_upload_errors(service: DocSage) -> None:
    client = TestClient(create_app(service))
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "k": 99}).status_code == 422
    assert client.post("/documents", json={"text": "   "}).status_code == 422
    bad = client.post("/documents/upload", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert bad.status_code == 415
    binary = client.post(
        "/documents/upload", files={"file": ("a.txt", b"\xff\xfe\x00", "text/plain")}
    )
    assert binary.status_code == 422
