"""FastAPI service: the platform's public interface for internal consumers.

POST /ask     — RAG-grounded multi-agent answer with citations + usage block
GET  /healthz — liveness

Wiring (store/gateway/retriever) is behind a single dependency so tests and
the eval harness override one function to run the entire service offline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .agents import ask as run_ask
from .config import get_settings
from .gateway import Gateway
from .observability import init_sentry, log_request
from .retrieval import Retriever
from .store import PgVectorStore


@dataclass
class Runtime:
    retriever: Retriever
    gateway: Gateway
    max_critic_retries: int


@lru_cache
def get_runtime() -> Runtime:
    s = get_settings()
    gateway = Gateway()
    store = PgVectorStore(s.database_url)
    return Runtime(
        retriever=Retriever(store, gateway, top_k=s.top_k),
        gateway=gateway,
        max_critic_retries=s.max_critic_retries,
    )


def require_api_key(authorization: str = Header(default="")) -> None:
    expected = f"Bearer {get_settings().api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class SourceRef(BaseModel):
    source: str
    ordinal: int
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    attempts: int
    sources: list[SourceRef]
    usage: dict


app = FastAPI(title="AskDesk", version=__version__)
init_sentry()


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "version": __version__}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask_endpoint(req: AskRequest, runtime: Runtime = Depends(get_runtime)) -> AskResponse:
    started = time.time()
    result = run_ask(
        req.question,
        retriever=runtime.retriever,
        gateway=runtime.gateway,
        max_retries=runtime.max_critic_retries,
    )
    log_request(req.question, result, started)
    return AskResponse(**result)
