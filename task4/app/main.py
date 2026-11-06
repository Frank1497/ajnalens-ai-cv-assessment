"""
FastAPI app exposing POST /ask for the pump maintenance RAG assistant.

Run with:
    source .venv/bin/activate
    uvicorn task4.app.main:app --reload --port 8000
  (run from the ai-cv-assessment repo root so the `task4` package resolves)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .agent import answer_question
from .ingest import build_index

app = FastAPI(title="Pump Maintenance RAG Assistant", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    # Build the FAISS index on first run if it doesn't exist yet.
    build_index(force=False)


class AskRequest(BaseModel):
    question: str


class ToolCall(BaseModel):
    name: str
    args: dict
    result: dict


class Citation(BaseModel):
    doc: str
    page: int


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    tool_calls: list[ToolCall]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    # The URL uvicorn prints has no route of its own; the interactive UI is /docs.
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    result = answer_question(req.question)
    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        citations=result["citations"],
        tool_calls=result["tool_calls"],
    )
