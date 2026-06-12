"""Model gateway: one provider-agnostic interface to chat + embedding models.

Everything model-shaped goes through this module via LiteLLM, so the rest of
the platform never imports a provider SDK. Swapping Claude for another model
(or pointing at a LiteLLM proxy) is an env-var change, not a code change.

Also the single choke point for usage metering: every call records tokens and
estimated cost, surfaced per-request by the API for chargeback/observability.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache

from .config import get_settings


@lru_cache
def _local_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@dataclass
class Usage:
    """Token/cost accumulator for one request (thread-safe)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int, cost: float) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.cost_usd += cost
            self.calls += 1

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "llm_calls": self.calls,
        }


class Gateway:
    """Chat + embeddings behind one interface. Tests substitute FakeGateway."""

    def __init__(self, chat_model: str | None = None, embed_model: str | None = None):
        s = get_settings()
        self.chat_model = chat_model or s.chat_model
        self.embed_model = embed_model or s.embed_model

    def chat(self, system: str, user: str, usage: Usage | None = None) -> str:
        import litellm

        resp = litellm.completion(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if usage is not None:
            u = resp.usage
            cost = litellm.completion_cost(completion_response=resp) or 0.0
            usage.add(u.prompt_tokens, u.completion_tokens, cost)
        return resp.choices[0].message.content or ""

    def embed(self, texts: list[str], usage: Usage | None = None) -> list[list[float]]:
        # "local/<model>" routes to sentence-transformers on this machine — no
        # vendor, no data egress (install with the [local] extra). Anything else
        # goes through LiteLLM to the configured provider.
        if self.embed_model.startswith("local/"):
            vectors = _local_model(self.embed_model.removeprefix("local/")).encode(
                texts, normalize_embeddings=True
            )
            if usage is not None:
                usage.add(sum(len(t.split()) for t in texts), 0, 0.0)
            return [v.tolist() for v in vectors]

        import litellm

        resp = litellm.embedding(model=self.embed_model, input=texts)
        if usage is not None and getattr(resp, "usage", None):
            usage.add(resp.usage.prompt_tokens, 0, 0.0)
        return [d["embedding"] for d in resp.data]
