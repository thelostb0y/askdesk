"""Vector store: pgvector in production, an in-memory twin for tests/eval.

Both implement the same three-method interface, so the retrieval layer and the
agent graph are store-agnostic. MemoryStore mirrors PgVectorStore's semantics
(cosine similarity + keyword match) closely enough that the offline test suite
exercises the real ranking logic deterministically — no database, no network.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk


@dataclass(frozen=True)
class Hit:
    source: str
    ordinal: int
    text: str
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int: ...
    def vector_search(self, embedding: list[float], k: int) -> list[Hit]: ...
    def keyword_search(self, query: str, k: int) -> list[Hit]: ...


# ── Production store ─────────────────────────────────────────────────────────


class PgVectorStore:
    """Postgres + pgvector. Schema in db/schema.sql (chunks table, ivfflat index,
    tsvector column for keyword search).

    `db_schema` pins the search_path so AskDesk can live in a dedicated
    namespace inside a shared database; `extensions` is included because
    Supabase installs pgvector there."""

    def __init__(self, database_url: str, db_schema: str = "public"):
        import psycopg
        from pgvector.psycopg import register_vector
        from psycopg import sql

        self._conn = psycopg.connect(database_url, autocommit=True)
        self._conn.execute(
            sql.SQL("SET search_path TO {}, extensions, public").format(
                sql.Identifier(db_schema)
            )
        )
        register_vector(self._conn)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        with self._conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings, strict=True):
                cur.execute(
                    """
                    INSERT INTO chunks (source, ordinal, text, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source, ordinal)
                    DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """,
                    (chunk.source, chunk.ordinal, chunk.text, emb),
                )
        return len(chunks)

    def vector_search(self, embedding: list[float], k: int) -> list[Hit]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, ordinal, text, 1 - (embedding <=> %s::vector) AS score
                FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
                """,
                (embedding, embedding, k),
            )
            return [Hit(*row) for row in cur.fetchall()]

    def keyword_search(self, query: str, k: int) -> list[Hit]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, ordinal, text,
                       ts_rank(tsv, websearch_to_tsquery('english', %s)) AS score
                FROM chunks
                WHERE tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC LIMIT %s
                """,
                (query, query, k),
            )
            return [Hit(*row) for row in cur.fetchall()]


# ── Test/eval twin ────────────────────────────────────────────────────────────


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


_WORD = re.compile(r"[a-z0-9]+")


class MemoryStore:
    """In-memory VectorStore with the same interface and ranking semantics."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, int], tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self._rows[(chunk.source, chunk.ordinal)] = (chunk, emb)
        return len(chunks)

    def vector_search(self, embedding: list[float], k: int) -> list[Hit]:
        scored = [
            Hit(c.source, c.ordinal, c.text, _cosine(embedding, emb))
            for c, emb in self._rows.values()
        ]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]

    def keyword_search(self, query: str, k: int) -> list[Hit]:
        terms = set(_WORD.findall(query.lower()))
        scored = []
        for chunk, _ in self._rows.values():
            words = set(_WORD.findall(chunk.text.lower()))
            overlap = len(terms & words)
            if overlap:
                scored.append(Hit(chunk.source, chunk.ordinal, chunk.text, overlap / len(terms)))
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]
