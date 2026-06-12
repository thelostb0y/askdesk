# Case study: AskDesk — an internal AI platform in two weekends

## Problem

Internal teams want LLM answers over private documents, but ad-hoc RAG scripts
fail the same three ways: answers hallucinate beyond the retrieved evidence,
quality regressions ship silently because nothing gates them, and every team
re-implements retrieval + provider plumbing.

## What I built

A platform, not a script: a FastAPI service + typed Python SDK that any
internal app can consume, backed by hybrid retrieval (pgvector vector search ∪
Postgres full-text, fused with Reciprocal Rank Fusion) and a LangGraph
multi-agent pipeline — Researcher (retrieve) → Analyst (cited draft) → Critic
(adversarial verification with bounded retry). Provider access goes through a
LiteLLM gateway, so models are env config and every token/cost is metered per
request.

## The design decision that matters most

**Treat LLM non-determinism as a testing problem.** The repo ships
deterministic fakes (`askdesk.testing`) and an in-memory twin of the pgvector
store with identical interface and ranking semantics. That makes the entire
pipeline — chunking, hybrid ranking, citation flow, the critic retry loop,
fail-closed verdict parsing — verifiable in CI with zero network and zero API
keys. A golden Q&A set then runs as a hard CI gate on three metrics
(retrieval hit rate, groundedness rate, keyword recall) with explicit floors;
the same harness re-runs in `--live` mode against real models before releases.

Offline eval on the sample corpus: **10/10 cases pass; all three metrics at
100%** (deterministic by construction — the gate exists to catch regressions
in the platform mechanics, while `--live` measures model quality).

The live runs — Claude Sonnet as Analyst/Critic against Supabase pgvector —
scored **10/10 with all metrics at 100%** under **two different embedding
providers**: fully local sentence-transformers (384-dim, no document egress)
and OpenAI text-embedding-3-small (1536-dim). The provider swap was two env
vars and a re-ingest; the same gate re-ran unchanged — which is the
provider-agnostic design doing its job. A production `/ask` round trip
(SDK → API → agent graph → Claude → pgvector) returns a multi-fact, cited,
Critic-approved answer in ~12s at a metered **$0.005/request**.

## Groundedness is structural, not aspirational

The Critic re-reads every drafted claim against the cited evidence and rejects
unsupported drafts back to the Analyst (bounded retries). Unparseable verdicts
fail closed. Every API response carries an explicit `grounded` flag — an
answer never leaves the platform implying verification it didn't get.

## Stack

Python 3.12 · FastAPI · LangGraph · LiteLLM (Claude) · Postgres/pgvector
(Supabase-compatible) · Sentry · Docker (12-factor, stateless) · GitHub
Actions CI (lint + tests + eval gate).

## What I'd build next at scale

Per-team corpora with row-level security; a message queue for async ingestion;
retrieval caching; DataDog/Jaeger traces across the agent graph; an A/B
harness comparing prompt/model variants on the golden set before promotion.
