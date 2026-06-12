# AskDesk

[![CI](https://github.com/thelostb0y/askdesk/actions/workflows/ci.yml/badge.svg)](https://github.com/thelostb0y/askdesk/actions/workflows/ci.yml)

**An internal AI platform: RAG-grounded, multi-agent Q&A over a private corpus — exposed as an API + Python SDK, with a CI eval gate.**

Teams point AskDesk at their documents and get back a service their apps can call: answers with citations, an explicit groundedness verdict, and per-request token/cost accounting. The platform's promise is *no answer leaves claiming evidence it doesn't have.*

```
                        ┌──────────────────────────────────────────────┐
   POST /ask            │  LangGraph agent graph                       │
  ───────────►  FastAPI │  Researcher ──► Analyst ──► Critic ──► out   │
  ◄───────────          │   (hybrid        (cited      (adversarial    │
   answer + citations   │    retrieval)     draft)      verify, retry) │
   + grounded flag      └──────┬───────────────┬───────────────────────┘
   + usage/cost                │               │
                        pgvector + tsvector   LiteLLM gateway
                        (vector ∪ keyword,    (Claude / any provider,
                         RRF fusion)           usage metering)
```

## Why each piece exists

| Piece | Why |
|---|---|
| **Hybrid retrieval (pgvector + tsvector, RRF)** | Vector search catches paraphrases; keyword catches exact terms (ids, names) embeddings blur. RRF fuses rankings without comparable scores. |
| **Critic agent (adversarial verify)** | LLMs assert; the Critic checks every claim against the cited evidence and bounces ungrounded drafts back (bounded retries). Unparseable verdicts **fail closed**. |
| **LiteLLM gateway** | One interface to any model provider. Models are env config, not code. Single choke point for token/cost metering. Embeddings can also run **fully local** (`ASKDESK_EMBED_MODEL=local/<sentence-transformers model>`, `[local]` extra) — documents never leave your infrastructure. |
| **Python SDK + shipped fakes** (`askdesk.testing`) | Platform-as-a-product: consumers get a typed client *and* deterministic fakes so they can test against AskDesk with zero network. |
| **Eval harness as CI gate** | Treats LLM non-determinism as a testing problem: a golden Q&A set scored on retrieval hit rate, groundedness, and keyword recall — the build fails below floors. |

## Quickstart (offline — no keys, no database)

```bash
uv sync
uv run pytest                      # full offline suite: chunking, RRF, agents, API
uv run python eval/run_eval.py     # eval gate on the sample corpus
```

## Production setup

```bash
cp .env.example .env               # fill in DB url + provider keys
psql "$ASKDESK_DATABASE_URL" -f db/schema.sql
uv run python scripts/ingest_corpus.py corpus/sample   # or your own docs dir
uv run uvicorn askdesk.api:app --port 8000
```

Or containerized: `docker compose up` (local pgvector included). The image is stateless and 12-factor — scale horizontally behind any load balancer.

## Consume it

```python
from askdesk.sdk import Client

client = Client("http://localhost:8000", api_key="...")
result = client.ask("How long do customers have to request a refund?")
result.answer    # "Customers may request a full refund within 30 days... [1]"
result.grounded  # True — Critic-verified against the cited chunks
result.sources   # [Source(source='refund-policy.md', ordinal=0, ...)]
result.usage     # {'prompt_tokens': ..., 'cost_usd': ..., 'llm_calls': 3}
```

## Quality: how non-determinism is handled

- **Offline mode** (CI, default): deterministic `FakeGateway` + in-memory store with the same interface/semantics as pgvector — verifies the *mechanics* (ranking, citation flow, critic loop, fail-closed parsing) on every commit.
- **Live mode** (`eval/run_eval.py --live`): same golden set against real models + pgvector — measures *answer quality* before a release or after a prompt/model change. Live runs (Claude Sonnet analyst/critic, Supabase pgvector) scored **10/10, all metrics 100% under both embedding providers** — OpenAI `text-embedding-3-small` and fully local MiniLM — with the swap being two env vars and a re-ingest.
- Groundedness is enforced structurally (Critic loop), not hoped for.

## Observability

Sentry (errors + traces) via `SENTRY_DSN`; every request logs a structured JSON line (latency, attempts, tokens, cost) and returns its usage block to the caller for chargeback.

## Layout

```
src/askdesk/   config · gateway · chunking · store · retrieval · agents · api · sdk · testing
db/schema.sql  pgvector + tsvector schema with ANN + GIN indexes
scripts/       ingest_corpus.py
eval/          golden.json + run_eval.py (CI gate)
tests/         offline suite (no network, no keys)
```
