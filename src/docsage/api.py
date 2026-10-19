"""HTTP API."""

from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from docsage import __version__
from docsage.service import DocSage

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
ALLOWED_SUFFIXES = (".txt", ".md", ".markdown")


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str = ""
    doc_id: Optional[str] = Field(default=None, max_length=128)


class IngestResponse(BaseModel):
    doc_id: str
    source: str
    chunks: int
    replaced: int


class DocumentOut(BaseModel):
    doc_id: str
    source: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)


class Hit(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    score: float
    text: str


class CitationOut(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    source: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationOut]


def _service(request: Request) -> DocSage:
    return request.app.state.service


def create_app(service: Optional[DocSage] = None) -> FastAPI:
    app = FastAPI(title="docsage", version=__version__)
    if service is None:
        from docsage.config import build_service

        service = build_service()
    app.state.service = service

    @app.get("/healthz")
    def healthz(request: Request) -> dict:
        return {"status": "ok", "chunks": _service(request).store.count()}

    @app.post("/documents", response_model=IngestResponse, status_code=201)
    def ingest(body: IngestRequest, request: Request) -> IngestResponse:
        try:
            r = _service(request).ingest(body.text, source=body.source, doc_id=body.doc_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse(**r.__dict__)

    @app.post("/documents/upload", response_model=IngestResponse, status_code=201)
    async def upload(
        request: Request,
        file: UploadFile = File(...),
        doc_id: Optional[str] = Form(default=None),
    ) -> IngestResponse:
        name = file.filename or "upload.txt"
        if not name.lower().endswith(ALLOWED_SUFFIXES):
            raise HTTPException(
                415, detail=f"only {', '.join(ALLOWED_SUFFIXES)} files are accepted"
            )
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="file is larger than 2 MiB")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(422, detail="file must be UTF-8 text") from exc
        try:
            r = _service(request).ingest(text, source=name, doc_id=doc_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return IngestResponse(**r.__dict__)

    @app.get("/documents", response_model=List[DocumentOut])
    def list_documents(request: Request) -> List[DocumentOut]:
        return [DocumentOut(**d.__dict__) for d in _service(request).store.list_documents()]

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, request: Request) -> None:
        if _service(request).store.delete_document(doc_id) == 0:
            raise HTTPException(404, detail="document not found")

    @app.post("/search", response_model=List[Hit])
    def search(body: QueryRequest, request: Request) -> List[Hit]:
        return [
            Hit(
                chunk_id=h.chunk.id,
                doc_id=h.chunk.doc_id,
                source=h.chunk.source,
                score=round(h.score, 4),
                text=h.chunk.text,
            )
            for h in _service(request).search(body.question, body.k)
        ]

    @app.post("/ask", response_model=AskResponse)
    def ask(body: QueryRequest, request: Request) -> AskResponse:
        a = _service(request).ask(body.question, body.k)
        return AskResponse(
            question=a.question,
            answer=a.answer,
            citations=[CitationOut(**c.__dict__) for c in a.citations],
        )

    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory docsage.api:app_factory``."""
    return create_app()
