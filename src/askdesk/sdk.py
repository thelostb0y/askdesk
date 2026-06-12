"""AskDesk Python SDK — what internal teams actually consume.

    from askdesk.sdk import Client

    client = Client("https://askdesk.internal", api_key="...")
    result = client.ask("What is our refund policy?")
    print(result.answer, result.sources)

Deliberately tiny: one class, one method, typed results, clear errors.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx


class AskDeskError(RuntimeError):
    """Raised for any non-2xx platform response, with the server detail attached."""


@dataclass(frozen=True)
class Source:
    source: str
    ordinal: int
    score: float


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    grounded: bool
    attempts: int
    sources: list[Source]
    usage: dict


class Client:
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def ask(self, question: str) -> Answer:
        resp = self._http.post("/ask", json={"question": question})
        if resp.status_code != 200:
            raise AskDeskError(f"{resp.status_code}: {resp.text}")
        data = resp.json()
        return Answer(
            question=data["question"],
            answer=data["answer"],
            grounded=data["grounded"],
            attempts=data["attempts"],
            sources=[Source(**s) for s in data["sources"]],
            usage=data["usage"],
        )

    def healthy(self) -> bool:
        try:
            return self._http.get("/healthz").status_code == 200
        except httpx.HTTPError:
            return False
